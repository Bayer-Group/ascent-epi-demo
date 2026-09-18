"""Embed the shipped vocabulary and load it into Qdrant.

Concept search is vector search: a clinical phrase is embedded and matched
against embedded concept names. Without a populated collection the coder
returns nothing for every query -- and returns it successfully, which is the
failure mode worth guarding against.

The synthetic vocabulary is 5,255 concepts, which a CPU embeds in seconds, so
the index is built at first start with no GPU, no
Batch and no external service.

Idempotent: an existing collection with the right number of points is left
alone, so restarting the stack does not re-embed.

Run:  python scripts/seed_qdrant.py [--vocabulary FILE] [--recreate]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("seed-qdrant")

BATCH = 256


def _sparse(text: str):
    """Token-frequency sparse vector, matching the connector's scheme exactly."""
    from ascent_medical_coder.connectors.qdrant import _build_sparse_vector

    return _build_sparse_vector(text)


def _concepts(vocabulary: Path) -> list[dict]:
    import duckdb

    con = duckdb.connect(str(vocabulary), read_only=True)
    rows = con.execute(
        """
        SELECT concept_id, concept_name, vocabulary_id, domain_id,
               concept_code, standard_concept
        FROM CONCEPT
        WHERE concept_name IS NOT NULL AND concept_name <> ''
        ORDER BY concept_id
        """
    ).fetchall()
    con.close()
    # Payload keys are upper-case: the pipeline indexes hits by "CONCEPT_ID",
    # "CONCEPT_CODE", "VOCABULARY_ID" and friends.
    return [
        {
            "CONCEPT_ID": int(r[0]) if str(r[0]).isdigit() else r[0],
            "CONCEPT_NAME": r[1],
            "VOCABULARY_ID": r[2],
            "DOMAIN_ID": r[3],
            "CONCEPT_CODE": r[4],
            "STANDARD_CONCEPT": r[5],
            "IS_VALID": True,
            "PATIENT_COUNT": None,
            "CONCEPT_DATA": f"{r[1]} [{r[2]}] {r[4]}",
        }
        for r in rows
    ]


def _load_model(model_name: str, attempts: int = 3):
    """Load the encoder, retrying the download.

    The first run pulls ~400 MB from HuggingFace, and that drops sometimes: a
    laptop connection stalls, the CDN rate-limits, a proxy resets the transfer.
    This script is a compose dependency, so a single dropped download exits
    non-zero and every service gated on service_completed_successfully never
    starts -- the whole stack fails on a network blip.

    Broad `except Exception` on purpose. The same blip surfaces as a different
    type depending on which hop breaks: a bare ConnectionError raised out of
    hf_xet's Rust client, httpx.ConnectError or httpx.ReadTimeout from
    huggingface_hub's own client, OSError from its size-consistency check, or
    LocalEntryNotFoundError when the metadata call is what failed. Naming four
    of them just lets the fifth through, and a list written today is wrong on
    the next huggingface_hub release. Nothing this call can raise is made worse
    by trying it again; a genuinely bad model name still fails, three times.
    """
    from sentence_transformers import SentenceTransformer

    for attempt in range(1, attempts + 1):
        try:
            return SentenceTransformer(model_name)
        except Exception as exc:
            if attempt == attempts:
                raise
            # Short backoff on purpose: this runs inside `docker compose up`, so
            # a genuinely dead network should fail in seconds with the real
            # traceback rather than hold the terminal open for minutes.
            delay = 2**attempt
            logger.warning(
                "Loading %s failed (attempt %d/%d): %s: %s -- retrying in %ds",
                model_name,
                attempt,
                attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)


async def seed(vocabulary: Path, url: str, collection: str, model_name: str, recreate: bool) -> int:
    from qdrant_client import AsyncQdrantClient, models

    # The connector queries NAMED vectors -- using="dense" and using="sparse".
    # A collection with a single unnamed vector is rejected outright with
    # 'Wrong input: Not existing vector name error: dense', so the names here
    # are contract, not preference. The collection name says "hybrid" for the
    # same reason.

    concepts = _concepts(vocabulary)
    logger.info("Embedding %d concepts with %s", len(concepts), model_name)

    client = AsyncQdrantClient(url=url)
    exists = await client.collection_exists(collection)
    if exists and not recreate:
        count = (await client.count(collection)).count
        if count >= len(concepts):
            logger.info("%s already holds %d points; nothing to do", collection, count)
            return 0
        logger.info("%s holds %d of %d points; rebuilding", collection, count, len(concepts))
        recreate = True

    # Loaded here rather than at the top of the function: the idempotency check
    # above returns early on a warm stack, and a restart should not touch
    # HuggingFace at all.
    model = _load_model(model_name)
    dim = model.get_sentence_embedding_dimension()

    if recreate or not exists:
        if exists:
            await client.delete_collection(collection)
        await client.create_collection(
            collection_name=collection,
            vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        logger.info("Created %s (dense dim=%d cosine + sparse)", collection, dim)

        # A full-text index on CONCEPT_NAME, and it is load-bearing. The drug
        # branch does not use vector search -- it filters with MatchText -- and
        # without this index Qdrant degrades MatchText to case-sensitive
        # substring matching. "Metformin" matched; "metformin", which is what a
        # user types, matched nothing, so every drug lookup came back empty
        # while conditions (which go through vector search) worked fine.
        # lowercase=True on a word tokenizer is what makes the match
        # case-insensitive.
        await client.create_payload_index(
            collection_name=collection,
            field_name="CONCEPT_NAME",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
                min_token_len=2,
            ),
        )
        logger.info("Created full-text index on CONCEPT_NAME")

    for start in range(0, len(concepts), BATCH):
        chunk = concepts[start : start + BATCH]
        vectors = model.encode([c["CONCEPT_NAME"] for c in chunk], show_progress_bar=False)
        await client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=i + start,
                    vector={
                        "dense": v.tolist(),
                        # Same token-frequency scheme the connector uses at
                        # query time; indices must come from the same hash or
                        # nothing ever matches.
                        "sparse": _sparse(c["CONCEPT_NAME"]),
                    },
                    payload=c,
                )
                for i, (c, v) in enumerate(zip(chunk, vectors, strict=True))
            ],
        )
        logger.info("  %d/%d", min(start + BATCH, len(concepts)), len(concepts))

    total = (await client.count(collection)).count
    logger.info("Seeded %s with %d points", collection, total)
    return 0 if total == len(concepts) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocabulary", type=Path, default=Path(os.environ.get("VOCABULARY_DB", "/data/vocabulary.duckdb"))
    )
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://qdrant:6333"))
    parser.add_argument(
        "--collection", default=os.environ.get("BGE_EMBEDDING_COLLECTION_QDRANT", "concepts-bge-hybrid-new")
    )
    parser.add_argument("--model", default=os.environ.get("BGE_EMBED_MODEL", "BAAI/bge-base-en-v1.5"))
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    if not args.vocabulary.exists():
        logger.error("No vocabulary at %s; run build_vocabulary.py first", args.vocabulary)
        return 1
    return asyncio.run(seed(args.vocabulary, args.url, args.collection, args.model, args.recreate))


if __name__ == "__main__":
    sys.exit(main())
