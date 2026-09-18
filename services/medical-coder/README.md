# Ascent Medical Coder

Transforms free-form clinical queries into curated medical-concept lists using
**Qdrant** vector search, a local **OMOP vocabulary database**, and **LLM**
post-processing. Exposes a FastAPI service plus a reusable coding library, in a
layered `src`-layout package. It does not authenticate; see Authentication
below.

For capabilities, request/response examples, options, and default-deployment
limits, start with the [usage guide](USAGE.md).

---

## Architecture

The main dependency flow is **routes → services → connectors / db**. Services
hold pipeline logic, while connectors handle external systems. Some services
still raise FastAPI `HTTPException`; they are not fully transport-independent.
Tests replace external calls with stubs and mocks.

```
src/ascent_medical_coder/
  main.py              # create_app(): app factory, middleware, lifespan, 500 handler
  routes.py            # aggregates the 5 business routers (auth is stubbed)
  api/
    deps.py            # FastAPI dependencies (current user, settings)
    routers/           # thin HTTP layer: coding, concept_search, grounded_search,
                       #   patient_counts, validate_codes, system
  schemas/             # pydantic request/response models (the public API contract)
  services/            # pipeline orchestration and business logic
    coding.py          #   the CLUES coding pipeline orchestrator
    concept_search.py  grounded_search.py  patient_counts.py  validate_codes.py
    pipeline/          #   enrichment, filtering, fallback, drug_expand,
                       #   counts, lts, vocabulary, tables, encoder, utils
  connectors/          # EXTERNAL-SYSTEM ADAPTERS
    llm/               #   openai (Azure), gemini (google-genai), anthropic (Bedrock) + factory
    embeddings/        #   local (sentence-transformers) + remote (Azure/Gemini) + factory
    qdrant.py  warehouse.py   #   warehouse.py = the local DuckDB vocabulary
  db/                  # POSTGRES: async engine/session, cache + analytics repos, ORM models
  core/                # typed settings, local-user/auth stubs, logging, errors
  prompts/             # LLM prompt templates
scripts/               # build_vocabulary.py, seed_qdrant.py
```

**Endpoints** (all under the `SERVICE_PATH` prefix): `POST /get-medical-codes`,
`POST /get-medical-codes-reasoning`, `WS /ws/get-medical-codes-reasoning`,
`POST /bulk-concept-search`, `POST /grounded-search`, `POST /patient-counts`,
`POST /validate-codes`, plus open system routes (`GET /`, `POST /version`,
`/test-timeout`, `/test-error`).

---

## Run locally

This service is not run on its own. It comes up as the `medical-coder` service
of the stack's `docker-compose.yml` at the repository root:

```bash
docker compose up            # from the repository root, not this directory
```

That builds this directory (`services/medical-coder/Dockerfile`, with the
repository root as build context) and wires up everything it needs:

- **Qdrant** — the `qdrant` service, reached at `QDRANT_HOST=qdrant`.
- **Vocabulary** — `coder-init` populates `/data/vocabulary.duckdb` in a shared
  volume before this service starts (`VOCABULARY_DB` points at it), so there is
  no remote warehouse to connect to or authenticate with.
  `connectors/warehouse.py` transpiles each SQL statement to DuckDB with `sqlglot`.
- **Embedding models** — cached in the shared `models` volume via `HF_HOME`,
  downloaded on first start.
- **LLM keys** — passed through from the root `.env`. The same key the rest of
  the stack uses; see `.env.template` there.

The service answers on `http://medical-coder:8000` inside the Compose network.
It is not published to the host by default. The opt-in
[host-development overlay](../../docs/development.md#run-the-backend-on-the-host)
exposes it at `http://127.0.0.1:18001/v1/medical-coder/`; do not expose this
unauthenticated service remotely.

Only the **`bge` concept index** is seeded by default. SapBERT/BioLORD model
warm-up does not populate additional Qdrant collections. Other encoder choices
require separate indexes. Precomputed patient-count tables are not shipped,
and the standalone `/patient-counts` route is not operational in this build.
Use backend SQL for patient counts; see the [usage guide](USAGE.md).

Requested LLM filtering must produce a complete, disjoint decision over the
candidate indices. Invalid output is retried, then reported as an explicit
error (HTTP 502 on the REST coding path), not an empty or unfiltered success.
Failed filtering is not cached.

### Authentication

None. `core/security.py` returns a fixed local user for REST and WebSocket
paths. The auth-specific `AZURE_CLIENT_ID` and `AZURE_TENANT_ID` fields do not
activate authentication. Other `AZURE_*` fields configure LLM/embedding
connectors and are not auth settings. The backend's `API_AUTH_TOKEN` does not
protect this service.

---

## Develop

From the repository root, install this service's locked dev dependencies and
run its own test suite:

```bash
(cd services/medical-coder && uv sync --frozen --extra dev && uv run --no-sync pytest tests/ -q)
```

The service tests cover filtering, cache isolation, request validation, and
result conversion without running external services or using provider keys.
CI runs them in a dedicated job and builds the medical-coder Docker image.
The root tests also cover the backend's client for this service.

For lint/format checks, use the root toolchain, as CI does:

```bash
uv sync --frozen --extra dev
uv run --no-sync ruff check services/medical-coder/src services/medical-coder/tests
uv run --no-sync ruff format --check services/medical-coder/src services/medical-coder/tests
```

Repository-wide lint debt may still be reported; changed files must pass.
See the [development guide](../../docs/development.md) for backend integration.

---

## Configuration

Settings are typed (`core/settings.py`, pydantic-settings) and bind to the exact
environment-variable names below.

| Group | Env vars |
|-------|----------|
| Service | `SERVICE_PATH`, `RELEASE_VERSION`, `API_VERSION`, `CORS_ORIGINS`, `DEBUG`, `PROFILE_REQUESTS`, `WARM_EMBEDDERS_ON_STARTUP` (default true — set false locally for fast boots) |
| Azure auth | `AZURE_CLIENT_ID`, `AZURE_TENANT_ID` |
| Qdrant | `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_MAX_CONCURRENT_SEARCHES` (default 10), `*_EMBEDDING_COLLECTION_QDRANT` |
| Postgres (optional) | `USR_DB_HOST/PORT/NAME/APP_USR/APP_PWD` — cache + analytics no-op when unset |
| LLM credentials | `GEMINI_API_KEY`/`GOOGLE_API_KEY`, `AZURE_OPENAI_*` (chat + embeddings); Anthropic via Bedrock uses the SDK's default credential chain |
| LLM models (all optional — sane defaults in `settings.py`, the single source of truth for model names) | `GEMINI_MODEL_NAME`, `AZURE_OPENAI_CHAT_MODEL`, `BEDROCK_ANTHROPIC_MODEL_ID`, `GROUNDED_SEARCH_PROVIDER`, `GROUNDED_SEARCH_MODEL_NAME`, `GROUNDED_SEARCH_TEMPERATURE`, `DEFAULT_LLM_FILTER`, `NON_OMOP_MODEL_NAME`, `AZURE_OPENAI_EMBED_MODEL`, `GEMINI_EMBED_MODEL`, `SAP_EMBED_MODEL`, `BGE_EMBED_MODEL`, `BIOLORD_EMBED_MODEL` |

Settings support environment variables and `.env` loading. On the default
Compose path, configure the root `.env` and explicit service environment
mappings; no separate service `.env` is required. Adding a setting to the root
file alone does not pass it into a container. In particular, the coder's Azure
connector settings differ from the backend's generic `OPENAI_*` settings.

`VOCABULARY_DB` (`connectors/warehouse.py`) is read outside the settings model
and defaults to `/data/vocabulary.duckdb`. Optional `USR_DB_*` settings are not
mapped by the default Compose file, so the service's Postgres cache/analytics
are no-ops unless separately configured. This is distinct from the backend's
Postgres store for cohorts and metadata.

The `Azure auth` row is retained for compatibility; authentication is stubbed.

---

## Notes

- **Concurrency:** the service exposes asynchronous interfaces. DuckDB queries,
  local model loading/encoding, and synchronous Bedrock invocation are offloaded
  to threads. Postgres uses an async engine and Qdrant uses an async client.
