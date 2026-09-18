import asyncio
import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import redis

from ascent_domain.non_omop.schemas.sql_creation_simplified import EntityReferenceSimplified, MedicalConceptSimplified
from ascent_platform.cache import tls_enabled, tls_kwargs
from ascent_platform.config.runtime import get_runtime_settings
from ascent_platform.external.medical_coder_client import (
    execute_medical_coder_full_result,
)


def filter_medical_coding_ontology(json_data):
    """
    Filters tables and columns that contain medical coding ontologies from the given JSON data.

    Args:
        json_data (dict): The JSON data containing table and column information.

    Returns:
        list: A list of dictionaries, each containing the table name and columns with medical coding ontologies.
    """

    filtered_tables = []

    for table in json_data.get("tables", []):
        filtered_columns = [col for col in table["columns"] if col["is_medical_coding_ontology"]]

        if filtered_columns:
            filtered_tables.append({"table_name": table["table_name"], "columns": filtered_columns})

    return filtered_tables


def remove_hyphens(coding_systems):
    # Remove hyphens from each coding system name in the list
    return [code.replace("-", "") for code in coding_systems]


def normalize_ndc(ndc_code: str) -> str:
    """
    Normalize an NDC code to 11-digit no-dash format as stored in claims databases.

    FDA NDC codes come in three formats:
      4-4-2  → pad labeler to 5 digits  → 5+4+2 = 11
      5-3-2  → pad product to 4 digits  → 5+4+2 = 11
      5-4-1  → pad package to 2 digits  → 5+4+2 = 11

    Codes already without dashes or with fewer than 3 segments are returned
    with dashes stripped (no padding applied).
    """
    code = ndc_code.strip()
    parts = code.split("-")
    if len(parts) == 3:
        labeler, product, package = parts
        return labeler.zfill(5) + product.zfill(4) + package.zfill(2)
    # Incomplete or already normalised — just strip dashes
    return code.replace("-", "")


def extract_concepts_medical_coder(res_medical_coder, codes_with_dots=False):
    concepts = []
    for i in res_medical_coder["response_data"]:
        tmp_codes = res_medical_coder["response_data"][i]

        for item in tmp_codes:
            concepts.append(item["CONCEPT_DATA"])

    if not codes_with_dots:
        for concept in concepts:
            concept["CONCEPT_CODE"] = concept["CONCEPT_CODE"].replace(".", "")

    return concepts


_REDIS_CLIENT: redis.Redis | None = None


def _get_redis_client() -> redis.Redis:
    """Return a process-wide Redis client (connection-pooled, thread-safe).

    Reads ``REDIS_HOST`` / ``REDIS_PORT`` from the environment so the same
    package works from the backend container (``cache:6379``) and from
    standalone scripts/notebooks (``localhost:6379``).

    Three settings are deliberate:

    * **TLS** in deployed environments, where the managed Redis enforces
      in-transit encryption — talking plaintext to a TLS endpoint surfaces as
      "Timeout reading from socket".
    * **RESP2 pinned** — redis-py 7.x negotiates RESP3 via ``HELLO`` on connect,
      which Redis servers older than 6.0 reject with "unknown command HELLO".
    * **bounded timeouts + periodic health checks** so a hung read fails fast
      (and falls back to a direct fetch) and stale pooled connections are
      detected before reuse rather than surfacing as read timeouts.
    """
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        runtime = get_runtime_settings()
        host = runtime.REDIS_HOST
        port = runtime.REDIS_PORT
        use_ssl = tls_enabled()
        logging.info("med_coder cache: initialising Redis client at %s:%s (ssl=%s)", host, port, use_ssl)
        _REDIS_CLIENT = redis.Redis(
            host=host,
            port=port,
            db=0,
            **tls_kwargs(),
            protocol=2,
            socket_connect_timeout=5,
            socket_timeout=15,
            retry_on_timeout=True,
            health_check_interval=30,
        )
    return _REDIS_CLIENT


_IS_WITH_DOTS_TTL_SECONDS = 3 * 3600


async def lookup_is_with_dots(database: str) -> bool | None:
    """Return whether ``database`` stores ICD diagnosis codes with decimal dots.

    Reads the curated ``ASCENT.PUBLIC.ASCENT_NON_OMOP_METADATA.IS_WITH_DOTS``
    flag (populated by the ETL). A database whose rows are all NULL — or that
    isn't in the table at all — returns ``None`` ("unknown"), which the caller
    treats as "fall back to the explicitly passed ``codes_with_dots``".

    Result is cached in Redis (incl. the ``None`` outcome) so repeated
    placeholder resolutions don't re-hit Snowflake. Every failure mode
    (Redis down, Snowflake down, no creds in a standalone context) degrades
    to ``None`` so a lookup problem never blocks resolution.
    """
    if not database:
        return None

    cache_key = f"is_with_dots:{database}"
    client = None
    try:
        client = _get_redis_client()
        cached = client.get(cache_key)
        if cached is not None:
            v = cached.decode() if isinstance(cached, bytes) else cached
            return None if v == "null" else (v == "1")
    except redis.RedisError as e:
        logging.warning("is_with_dots cache GET failed for %s: %s", database, e)

    result: bool | None = None
    try:
        from ascent_platform.warehouse.bootstrap import DATABASES

        row = next(
            ((db.codes_with_dots,) for db in DATABASES if db.name == database.upper()),
            None,
        )
        if row and row[0] is not None:
            result = bool(row[0])
    except Exception as e:  # noqa: BLE001 - lookup must never break resolution
        logging.warning("is_with_dots lookup failed for %s: %s", database, e)

    if client is not None:
        try:
            client.setex(
                cache_key,
                _IS_WITH_DOTS_TTL_SECONDS,
                "null" if result is None else ("1" if result else "0"),
            )
        except redis.RedisError as e:
            logging.warning("is_with_dots cache SETEX failed for %s: %s", database, e)

    return result


def get_cached_medical_coder_data(entity, entity_name, coding_systems, ttl_hours=72, top_k=10000):
    """Fetch medical-coder results with TTL caching in Redis.

    Cache key encodes ``(entity, entity_name, vocab, top_k)`` — different
    fetch sizes don't share results. TTL is enforced by Redis ``SETEX``.

    If Redis is unreachable, falls back to direct ``execute_medical_coder_full_result``
    so the pipeline degrades to uncached behaviour instead of failing.
    """
    vocab_list = coding_systems.split(",") if "," in coding_systems else [coding_systems]
    vocabulary = remove_hyphens(vocab_list)

    client = _get_redis_client()
    ttl_seconds = ttl_hours * 3600
    merged_result = {"response_data": {}}

    for vocab in vocabulary:
        cache_key = f"med_coder:{entity.capitalize()}:{entity_name.capitalize()}:{vocab}:top_k={top_k}"

        try:
            cached = client.get(cache_key)
        except redis.RedisError as e:
            logging.warning(
                "med_coder cache GET failed for %s: %s — falling back to direct fetch",
                cache_key,
                e,
            )
            cached = None

        if cached is not None:
            logging.debug("med_coder cache HIT %s", cache_key)
            result_part = json.loads(cached)
        else:
            logging.debug("med_coder cache MISS %s", cache_key)
            result_part = execute_medical_coder_full_result(
                query=entity_name,
                top_k=top_k,
                domain_id=entity.capitalize(),
                vocabulary=[vocab],
            )
            # An empty result is usually the coder being unreachable rather than
            # a concept with no codes, and the service takes minutes to warm its
            # embedders. Caching that answer makes one slow start look like a
            # vocabulary that resolves to nothing for the whole TTL.
            if any((result_part.get("response_data") or {}).values()):
                try:
                    client.setex(cache_key, ttl_seconds, json.dumps(result_part))
                    logging.debug("med_coder cache STORE %s (ttl=%ds)", cache_key, ttl_seconds)
                except redis.RedisError as e:
                    logging.warning("med_coder cache SETEX failed for %s: %s", cache_key, e)
            else:
                logging.info("med_coder returned no codes for %s; not cached", cache_key)

        for key, codes in (result_part.get("response_data") or {}).items():
            if key in merged_result["response_data"]:
                merged_result["response_data"][key].extend(codes)
            else:
                merged_result["response_data"][key] = codes

    return merged_result


def find_entity_references(sql_snippet):
    """
    Find all disctinct entity references in the SQL snippet using regex pattern matching.

    Args:
        sql_snippet (str): The SQL code snippet to search in

    Returns:
        Dictionary: A dictionary where each key represents one entity, entity_name and coding_systems combination
        and the value is a tuple with the properties (entity, entity_name, coding_systems)
    """
    pattern = r"\[(\w+)@([^@]+)@([^\]]+)\]"
    all_tuples = re.findall(pattern, sql_snippet)
    dictionary = {}
    for t in all_tuples:
        key = t[0] + "@" + t[1] + "@" + t[2]
        dictionary[key] = t
    return dictionary


def process_entity_reference(entity, entity_name, coding_systems, codes_with_dots=False, top_k=10000):
    """
    Process a single entity reference by fetching medical codes and formatting them for SQL.

    Args:
        entity (str): The entity type
        entity_name (str): The name of the entity
        coding_systems (str): Comma-separated list of coding systems
        codes_with_dots (bool): Whether to keep dots in codes

    Returns:
        tuple: (formatted_codes_string, list_of_codes)
    """
    logging.debug(f"Processing entity: {entity}, Entity Name: {entity_name}, Coding Systems: {coding_systems}")

    # Use cache layer to fetch data
    data = get_cached_medical_coder_data(entity, entity_name, coding_systems, top_k=top_k)
    concepts = extract_concepts_medical_coder(data, codes_with_dots=codes_with_dots)
    codes = [concept["CONCEPT_CODE"] for concept in concepts]

    # Normalize NDC codes to 11-digit no-dash format (as stored in claims databases)
    if "NDC" in coding_systems.upper():
        codes = [normalize_ndc(c) for c in codes]

    # Format codes as a string for SQL IN clause
    codes_str = "'" + "', '".join(codes) + "'"

    return codes_str, concepts


def replace_entity_placeholders(sql_snippet: str, entity_references_data: Dict):
    """
    Replace entity placeholders in SQL snippet with actual code strings.

    Handles two SQL patterns:
    - ``IN ([placeholder])`` → ``IN (codes)`` (parentheses already present)
    - ``IN [placeholder]``  → ``IN (codes)`` (parentheses added automatically)

    Args:
        sql_snippet (str): The original SQL snippet with placeholders
        entity_references_data (Dict): Dictionary containing entity placeholders and their corresponding codes

    Returns:
        str: Modified SQL snippet with placeholders replaced by actual codes
    """
    modified_sql = sql_snippet
    for placeholder, codes in entity_references_data.items():
        escaped = re.escape(f"[{placeholder}]")
        # First pass: replace ([placeholder]) → (codes) — parens already present
        modified_sql = re.sub(rf"\(\s*{escaped}\s*\)", f"({codes})", modified_sql)
        # Second pass: replace remaining bare [placeholder] → (codes)
        # This handles cases where the LLM omitted parentheses around IN [placeholder]
        modified_sql = modified_sql.replace(f"[{placeholder}]", f"({codes})")

    return modified_sql


def create_placeholder_to_codes_mapping(entity_references: List[EntityReferenceSimplified]):
    mapped_references = {}
    for entity in entity_references:
        # Detect NDC from the placeholder (format: entity@name@CODING_SYSTEM1,CODING_SYSTEM2)
        placeholder_parts = entity.placeholder.split("@")
        is_ndc = len(placeholder_parts) >= 3 and "NDC" in placeholder_parts[2].upper()
        for concept in entity.concepts:
            code = normalize_ndc(concept.CONCEPT_CODE) if is_ndc else concept.CONCEPT_CODE
            mapped_references.setdefault(entity.placeholder, "")
            if mapped_references[entity.placeholder]:
                mapped_references[entity.placeholder] += f", '{code}'"
            else:
                mapped_references[entity.placeholder] = f"'{code}'"

    return mapped_references


def merge_entity_references(sql: str, entity_refs: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Merge entity references that share the same concept into one combined placeholder.

    When the LLM generates separate placeholders for the same clinical concept across
    multiple vocabularies (e.g. ``[procedure@thrombolysis@CPT4]`` and
    ``[procedure@thrombolysis@ICD10PCS]``), this function combines them into a single
    placeholder (``[procedure@thrombolysis@CPT4,ICD10PCS]``) and updates the SQL string
    so that every occurrence of any individual placeholder is replaced with the combined
    one.  Concepts are deduplicated by ``CONCEPT_CODE`` within each merged group.

    Args:
        sql: SQL string potentially containing individual per-vocabulary placeholders.
        entity_refs: List of entity reference dicts as returned by
            :func:`get_medical_concepts_from_sql` or
            :func:`get_medical_concepts_from_sql_async`.

    Returns:
        A 2-tuple ``(updated_sql, merged_entity_refs)`` where ``updated_sql`` has all
        individual placeholders replaced by their combined counterpart and
        ``merged_entity_refs`` contains one entry per unique ``(entity, entity_name)``
        pair.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for ref in entity_refs:
        key = (ref["entity"].lower(), ref["entity_name"].lower())
        groups[key].append(ref)

    merged_refs: List[Dict[str, Any]] = []
    updated_sql = sql

    for (_entity_lower, _name_lower), refs in groups.items():
        # Collect all coding systems preserving order, avoiding duplicates
        seen_vocabs: set = set()
        all_vocabs: List[str] = []
        for ref in refs:
            for vocab in ref["coding_systems"].split(","):
                vocab = vocab.strip()
                if vocab and vocab not in seen_vocabs:
                    seen_vocabs.add(vocab)
                    all_vocabs.append(vocab)

        combined_coding_systems = ",".join(all_vocabs)
        canonical = refs[0]
        combined_placeholder_key = f"{canonical['entity']}@{canonical['entity_name']}@{combined_coding_systems}"
        combined_placeholder = f"[{combined_placeholder_key}]"

        # Replace every individual per-vocab placeholder in the SQL
        for ref in refs:
            old_placeholder = f"[{ref['placeholder']}]"
            if old_placeholder != combined_placeholder:
                updated_sql = updated_sql.replace(old_placeholder, combined_placeholder)

        # Merge concept lists, deduplicating by CONCEPT_CODE
        seen_codes: set = set()
        merged_concepts: List[Dict[str, Any]] = []
        for ref in refs:
            for concept in ref.get("concepts", []):
                code = concept.get("CONCEPT_CODE", "")
                if code not in seen_codes:
                    seen_codes.add(code)
                    merged_concepts.append(concept)

        codes_str = "'" + "', '".join(c["CONCEPT_CODE"] for c in merged_concepts) + "'" if merged_concepts else "''"

        merged_refs.append(
            {
                "entity": canonical["entity"],
                "entity_name": canonical["entity_name"],
                "coding_systems": combined_coding_systems,
                "concepts": merged_concepts,
                "codes_str": codes_str,
                "placeholder": combined_placeholder_key,
                "codes_with_dots": canonical.get("codes_with_dots", False),
            }
        )

    return updated_sql, merged_refs


def get_medical_concepts_from_sql(sql_snippet: str, codes_with_dots=False):
    """
    Retrieve medical entities based on the placeholders in the SQL snipper.

    Args:
        sql_snippet (str): The original SQL snippet with placeholders
        codes_with_dots (bool): Whether to keep dots in codes

    Returns:
        List: A list of dictionaries that contain the information about entities found in the sql snippet
    """
    # Find all entity references in the SQL snippet
    distinct_entity_reference = find_entity_references(sql_snippet)

    # Process each entity reference and collect results
    entity_references = []

    for entity_ref in distinct_entity_reference:
        entity_dictionary = distinct_entity_reference[entity_ref]

        entity = entity_dictionary[0]
        entity_name = entity_dictionary[1]
        coding_systems = entity_dictionary[2]

        codes_str, concepts = process_entity_reference(entity, entity_name, coding_systems, codes_with_dots)

        entity_references.append(
            {
                "entity": entity,
                "entity_name": entity_name,
                "coding_systems": coding_systems,
                "concepts": concepts,
                "codes_str": codes_str,
                "placeholder": entity_ref,
                "codes_with_dots": codes_with_dots,
            }
        )

    return entity_references


async def get_medical_concepts_from_sql_async(
    sql_snippet: str,
    codes_with_dots: bool = False,
    top_k: int = 10000,
    database: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Async version of get_medical_concepts_from_sql that processes entity references
    in parallel using asyncio.gather, avoiding event-loop blocking.

    Args:
        sql_snippet (str): The original SQL snippet with placeholders
        codes_with_dots (bool): Whether to keep dots in codes. Used as the
            fallback when ``database`` has no recorded IS_WITH_DOTS flag.
        top_k (int): Max codes fetched per entity.
        database (str | None): Target non-OMOP database. When provided and the
            ``ASCENT_NON_OMOP_METADATA.IS_WITH_DOTS`` flag is set for it, that
            flag overrides ``codes_with_dots`` so resolved ICD codes match how
            the database actually stores them. A NULL/absent flag leaves the
            caller-supplied ``codes_with_dots`` untouched.

    Returns:
        List: A list of dictionaries that contain the information about entities found in the sql snippet
    """
    if database:
        recorded = await lookup_is_with_dots(database)
        if recorded is not None:
            codes_with_dots = recorded

    distinct_entity_reference = find_entity_references(sql_snippet)

    # Bound the per-entity fan-out. Each entity burns a default-thread-pool
    # worker via to_thread; an unbounded gather across N concurrent requests
    # would demand N x M threads and starve the pool.
    fanout_limit = get_runtime_settings().MEDICAL_CODER_ENTITY_CONCURRENCY
    semaphore = asyncio.Semaphore(max(1, fanout_limit))

    async def _process_one(entity_ref, entity, entity_name, coding_systems):
        async with semaphore:
            codes_str, concepts = await asyncio.to_thread(process_entity_reference, entity, entity_name, coding_systems, codes_with_dots, top_k)
        return {
            "entity": entity,
            "entity_name": entity_name,
            "coding_systems": coding_systems,
            "concepts": concepts,
            "codes_str": codes_str,
            "placeholder": entity_ref,
            "codes_with_dots": codes_with_dots,
        }

    tasks = [_process_one(entity_ref, *distinct_entity_reference[entity_ref]) for entity_ref in distinct_entity_reference]
    entity_references = await asyncio.gather(*tasks)
    return entity_references


def process_sql_with_medical_codes(json_data, codes_with_dots=False, sql_field="sql_code_snippet"):
    """
    Process SQL code snippet in JSON data by replacing medical coder entity placeholders
    with actual medical concept codes.

    Args:
        json_data (dict): JSON data containing SQL code snippet
        codes_with_dots (bool): Whether to keep dots in codes
        sql_field (str): Name of the field containing the SQL code snippet in the JSON data

    Returns:
        dict: Modified JSON data with processed SQL and extracted codes information
    """
    # Get the SQL snippet from the JSON
    sql_snippet = json_data.get(sql_field, "")
    logging.debug("Processing SQL snippet for medical codes")

    processed_entities = get_medical_concepts_from_sql(sql_snippet, codes_with_dots)

    def simplify_entity_reference(entity: Dict):
        placeholder = entity.get("placeholder")
        concepts = [MedicalConceptSimplified(CONCEPT_CODE=concept.get("CONCEPT_CODE")) for concept in entity.get("concepts")]
        return EntityReferenceSimplified(placeholder=placeholder, concepts=concepts)

    entity_references_simplified = []
    for entity in processed_entities:
        entity_references_simplified.append(simplify_entity_reference(entity))

    mapped_references = create_placeholder_to_codes_mapping(entity_references_simplified)

    # Replace placeholders with actual codes
    modified_sql = replace_entity_placeholders(sql_snippet, mapped_references)

    # Update the JSON data
    json_data[sql_field] = modified_sql
    json_data["extracted_concepts"] = {"entity_references": processed_entities}

    return json_data


replace_sql_placeholders_with_medical_coder_concept_list_simplified = process_sql_with_medical_codes
