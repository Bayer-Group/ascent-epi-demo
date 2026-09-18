# Medical-coder usage

Turn a clinical phrase into candidate concepts from the **shipped synthetic
vocabulary**. Review concept names, not familiar-looking code strings; these
are not public clinical code lists. See [setup and configuration](README.md).

## What it does

- **General lookup:** embed the query → retrieve concepts → expand descendants
  → optionally filter with an LLM → return concepts and scores.
- **Drug lookup:** explicit `Drug` searches use keyword matching and expansion.
  `Drug_class` first uses an LLM to find drug names. These paths do **not** run
  the general LLM relevance filter.
- **Reasoning:** the reasoning endpoints add per-domain counts and, when a filter
  runs, inclusion/exclusion explanations. Scores are not clinical probabilities.

## Choose an interface

Use the MCP tool **`lookup_medical_codes`** for assistant-driven analysis.
For direct HTTP access, start the [development overlay](../../docs/development.md#run-the-backend-on-the-host).
The base URL is `http://127.0.0.1:18001/v1/medical-coder`;
[interactive API docs](http://127.0.0.1:18001/v1/medical-coder/docs) are available
while it runs. This service does not authenticate; keep it local.

| Endpoint (relative to base URL) | Use it for |
|---|---|
| `POST /get-medical-codes` | Clinical phrase → candidate concepts |
| `POST /get-medical-codes-reasoning` | Same lookup with filtering explanations and stage counts |
| `WS /ws/get-medical-codes-reasoning` | Progress/keepalive messages followed by a result or error |
| `POST /bulk-concept-search` | Comma-separated IDs, codes, or name fragments; direct vocabulary lookup, not semantic search |
| `POST /validate-codes` | Check `(code, vocabulary_id)` pairs for presence and terminology validity |
| `POST /grounded-search` | External web-assisted discovery; **avoid for synthetic coding** |
| `POST /patient-counts` | **Not operational in this distribution:** database routing reaches an unimplemented stub |

WebSocket clients must supply a nonempty `?token=demo` and send one coding
request as JSON. The token check is a local-user stub, **not authentication**.

## First lookup

Ask an MCP assistant:

> Look up hypertension with domain_ids=["Condition"], vocabulary=["DXCODE"],
> encoder="bge", top_k=100, llm_filter=null, and google_search_fallback=false.

Equivalent direct request:

```bash
curl --fail-with-body http://127.0.0.1:18001/v1/medical-coder/get-medical-codes \
  -H 'Content-Type: application/json' \
  -d '{"query":"hypertension","domain_ids":["Condition"],"vocabulary":["DXCODE"],"encoder":"bge","top_k":100,"llm_filter":null,"google_search_fallback":false}'
```

For filtered results, change `llm_filter` to `"gemini"` (requires credentials and
incurs model usage). For explanations, send that payload to
`/get-medical-codes-reasoning`, or use MCP `with_reasoning=true`. Asking for
reasoning alone does not enable filtering.

### Read the response

Both interfaces key basic results by the exact query. Shapes below are sketches,
not observed results:

```text
HTTP: {"<query>": [{"SCORE": number, "CONCEPT_DATA": {...}}]}
MCP:  {"<query>": [{"CONCEPT_ID": integer, ..., "SIMILARITY_SCORE": number}]}
```

HTTP `CONCEPT_DATA` includes `CONCEPT_ID`, `CONCEPT_NAME`, `CONCEPT_CODE`,
`VOCABULARY_ID`, `DOMAIN_ID`, `STANDARD_CONCEPT`, `IS_VALID`, and `PATIENT_COUNT`.
MCP flattens selected fields and renames `SCORE` to `SIMILARITY_SCORE`.
`IS_VALID` concerns terminology deprecation, **not relevance to your question**.

Reasoning responses wrap the concepts under `results` and add
`llm_reasoning.per_domain`. Explanations may be absent when no filter ran or a
result cache was used. No matches look like `{"hypertension": []}`; that is a
concept-search result, **not a patient count**. A null `PATIENT_COUNT` means
unavailable, not zero.

## Key options

Defaults below match the bundled service and MCP lookup unless noted.

| Option | Default | Meaning / advice |
|---|---|---|
| `domain_ids` | `[]` | No domain restriction. Do not mix `Drug`/`Drug_class` with other domains in one request. |
| `vocabulary` | `null` | Unrestricted. Use synthetic names: `DXCODE`/`DXCODE9`, `PHARMLEX`, `DRUGPKG`, `MEDLEX`, `PXCODE`, `LABLEX`/`LABLOCAL`, `ENCTYPE`. |
| `encoder` | `"bge"` | The only seeded index; other choices require separately built collections. |
| `top_k` | `10000` | Retrieval limit, **not a final result cap**: descendants can add concepts. Start with a narrow query and a smaller value. |
| `use_hybrid` | `false` | General search uses dense vectors; `true` adds sparse text matching. |
| `include_descendants` | `true` | Expand related descendants. HTTP only; MCP does not expose this switch. |
| `llm_filter` | `null` | No relevance filtering. `"default"` selects `DEFAULT_LLM_FILTER` (shipped: `"gemini"`). |
| `custom_instructions` | `null` | Additional guidance for the filter; not a database filter. |
| `standard_concept` | `null` | No restriction; `"S"` selects standard concepts, `"C"` classification concepts. |
| `cosine_similarity` | Encoder-specific | Override the general-search threshold; a higher threshold can return fewer concepts. |
| `google_search_fallback` | `false` | Keep off: external discovery can return real codes absent from this synthetic vocabulary. |
| `database` | `null` | Optional precomputed-count enrichment; count tables are **not shipped**. Use backend SQL for demo patient counts. |
| `allow_lts`, `use_lts` | `false` | Different cache controls; see below. |

The service accepts `gemini`, `chatgpt`, `haiku`, and `default` as filter names.
Other connectors need their own credentials. Although MCP also advertises
`llama3`, the bundled HTTP request schema rejects it; do not select it.

## Caching and failures

- **Response cache:** requires the optional service `USR_DB_*` configuration,
  absent from default Compose. When configured, basic lookups cache for 24 hours;
  reasoning/WS lookups use it only with `use_lts=true`. Stored concepts do not
  include the original filtering explanations.
- **Filter-decision reuse:** `allow_lts=true` attempts to reuse previous decisions.
  The shipped read-only vocabulary has no decision-cache table, so this is not
  available by default; failures are logged and filtering continues without it.
- **400:** incompatible domain combination. **422:** invalid request fields or
  unsupported enum values. **404:** the selected embedding collection is missing.
- **502:** requested filtering failed validation after retries. No successful
  unfiltered substitute is returned or cached. This differs from a valid empty list.

The schemas and runtime API docs are the field reference, but some inherited
examples there name public vocabularies. Use the synthetic vocabulary above.
