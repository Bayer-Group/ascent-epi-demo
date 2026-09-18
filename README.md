# ASCENT

Ask epidemiological questions in natural language on Real-World health data and get back SQL, patient
cohorts, and answers through MCP or the included LibreChat interface.

This distribution ships **synthetic patient data** and a local warehouse. No
hosted warehouse or licensed patient dataset is needed for the default setup.
**An external LLM API key is required**; provider usage can incur charges.

**This is a demonstration, not clinical evidence.** Patients and clinical codes
are generated. Do not enter real patient data, confidential information, or
licensed datasets, and do not use the output to make decisions about patients.
See [Licence and provenance](#licence-and-provenance) for vocabulary exceptions.

## Prerequisites

- Git and Docker with **Compose v2 or newer** and BuildKit support.
- Internet access for the first build/model downloads and for provider calls.
- A Gemini API key for the default end-to-end demo.
- Python **3.12** and [`uv`](https://docs.astral.sh/uv/) only if you want to run
  tests, develop on the host, or generate additional datasets.

Clone into a directory Docker Desktop can share, such as your home directory.
Allow space for images, build caches, and models; see the
[resource notes](docs/operations.md#resources-and-first-start).

## Quickstart

Clone this repository, then run these commands from its root:

```bash
cp .env.template .env          # on a fresh clone; do not overwrite an existing .env
```

Set this value in `.env`:

```dotenv
GEMINI_API_KEY=your-provider-key
```

Then start the stack:

```bash
docker compose up --build
```

The first start builds the backend, medical-coder, and LibreChat images,
constructs the local warehouse, and indexes the synthetic vocabulary. It also
downloads roughly 2.5 GB of embedding models. Build time depends on your machine
and network; LibreChat alone can take around fifteen minutes. Subsequent starts
reuse build/model caches, but still load models into memory.

Check startup in another terminal:

```bash
docker compose ps -a
curl --fail http://localhost:8000/api/public/health
curl --fail http://localhost:8000/mcp/health
```

The first probe reports process liveness. The second checks MCP mounting/tool
registration and Redis—not the warehouse, medical-coder, Qdrant, or LLM provider.
Use the first task below to check that querying works.

### Open the chat UI

Open <http://localhost:3080> and use the seeded local account:

| Setting | Default |
|---|---|
| Email | `demo@ascent.local` |
| Password | `ascentdemo` |

The default **ASCENT** model preset has the main MCP server and skills enabled.
There is also a seeded **ASCENT agent**. Newly seeded agents and skills are
shared for viewing; registration does not inherently make them unavailable.
Existing installations may need their seeded resources refreshed.

Set `CHAT_USER` and `CHAT_PASSWORD` **before first startup** to change the seed
credentials. Changing `.env` later does not reset an existing account's password.
See the [LibreChat guide](librechat/README.md) for sharing, updates, and limits.

### Connect an MCP client instead

The backend exposes three servers:

| URL | Tools | Purpose |
|---|---:|---|
| `http://localhost:8000/mcp/ascent-mcp-v1` | 29 | Main analysis and cohort tools |
| `http://localhost:8000/mcp/ascent-experimental` | 9 | Lower-level tools |
| `http://localhost:8000/mcp/ascent-ui-kit` | 3 | Widgets for compatible MCP-UI hosts |

For Claude Code, using the default unauthenticated local setup:

```bash
claude mcp add --transport http ascent http://localhost:8000/mcp/ascent-mcp-v1
```

If you set `API_AUTH_TOKEN`, use the [bearer-token connection instructions](docs/operations.md#connecting-with-a-bearer-token).
Ports shown here are the defaults; adjust URLs if you change them.

To start without the chat services:

```bash
docker compose up --build postgres redis qdrant medical-coder backend
```

Compose also starts the required `coder-init` dependency.

### Try a first task

In the chat UI or your MCP assistant, ask:

> List the available databases and schemas. Then count the rows in
> SYNTHETIC_EHR_OMOP.CDM.person. Show the SQL you ran and the result.

With the default `DATA_SCALE=1k`, expect **1,000 patients**. This checks metadata
discovery and SQL execution; it does not exercise every tool. For a medical-coder
check, ask it to look up synthetic hypertension concepts using `encoder="bge"`.
See the [medical-coder usage guide](services/medical-coder/USAGE.md) for options and examples.
If a tool fails, inspect its error rather than interpreting an empty result as
zero prevalence. See [troubleshooting](docs/operations.md#troubleshooting).

## Data and limitations

| Dataset | Contents |
|---|---|
| `SYNTHETIC_EHR_OMOP.CDM` | OMOP CDM 5.4; 1,000 patients; 18 clinical/support tables and 4 vocabulary tables |
| `SYNTHETIC_CLAIMS.DATA_202601` | The same patients in a messy source feed; 9 tables |
| Concept vocabulary | 5,255 concepts, with synthetic clinical coding systems |
| Ground truth | [`data/synthetic/ground_truth_1k.json`](data/synthetic/ground_truth_1k.json) |

- Only the **`bge` concept index** is seeded by default. Other encoder choices
  need separately built Qdrant collections; warming an embedding model does not
  create its index.
- Synthetic code strings can resemble real ICD codes but mean something else.
  Use the supplied vocabulary, not public codebooks or remembered codes.
- Source dates use several formats. Warehouse construction normalizes them;
  querying the original SQLite files directly requires equivalent handling.
- Generated SQL and interpretations still require review. Invalid medical-code
  filtering now fails explicitly instead of returning unfiltered results.
- Published literature is real-world evidence, not synthetic ground truth.
  Keep it separate from demo counts and do not treat this population as a sample
  from which real-world prevalence can be estimated.

The 10k archives are not committed, and this repository supplies no default
download URL. [Generate 10k locally or use a maintainer-provided mirror](data/synthetic/README.md#scale).
For custom populations and vocabulary changes, see the
[generator guide](synthetic_data_generation/README.md).

## Configuration and security

The default Compose file publishes only the backend and chat UI, both on
`127.0.0.1`. Postgres, Redis, Qdrant, and medical-coder stay on the internal
network unless you opt into a development override.

Configuration has three owners:

- **Compose:** `.env` interpolation and service environment mappings in
  [`docker-compose.yml`](docker-compose.yml).
- **Backend:** [`ascent_platform.config.runtime.Settings`](src/ascent_platform/config/runtime.py),
  including `.env`/`.env.local` when running on the host.
- **Medical-coder / LibreChat:** their own settings and YAML; see the
  [service guide](services/medical-coder/README.md#configuration) and
  [chat guide](librechat/README.md#configuration-and-build).

Adding a backend setting to `.env` does not automatically pass it into a
container: it must also be mapped in Compose. Start with [`.env.template`](.env.template).

| Variable | Default | Scope / purpose |
|---|---|---|
| `GEMINI_API_KEY` | Unset | Passed to the backend, coder, and chat UI |
| `LLM_PROVIDER_ALIASES` | Unset | Backend assistant-provider substitutions |
| `DATA_SCALE` | `1k` | Compose selects the warehouse scale |
| `API_AUTH_TOKEN` | Unset | Shared bearer for the main and experimental MCP servers |
| `CHAT_USER`, `CHAT_PASSWORD` | Demo credentials above | Account creation, not password rotation |
| `BACKEND_PORT`, `LIBRECHAT_PORT` | `8000`, `3080` | Compose host ports |

Gemini is needed by the default chat and non-OMOP workflows. The OMOP assistant
layer can substitute configured Azure OpenAI (`OPENAI_API_KEY` and
`OPENAI_API_BASE`) or Claude (`ANTHROPIC_API_KEY`, or ambient credentials for Bedrock).
`LLM_PROVIDER_ALIASES` controls that layer, not every service or LibreChat model.
A single non-Gemini key is therefore not a replacement for the default setup.

**Before exposing the stack:** configure the MCP bearer, replace the demo account
credentials and LibreChat's default signing/encryption secrets, and use an
appropriate authenticating TLS proxy. The bearer is shared—not per-user access
control. Health probes and the generic UI-kit server remain public. Follow the
[security checklist](docs/operations.md#before-exposing-the-stack); the default
Compose file is a local demo, not a hardened production deployment.

LibreChat has message limits and a configured $2/week per-user credit allowance.
**That allowance does not cover backend LLM calls made by MCP tools.** Set
provider-side billing controls as well; see [chat limits](librechat/README.md#limits-and-billing).

## Development and tests

Use the [development guide](docs/development.md) for container iteration, the
complete host-backend setup, and migrations. Do not start dependencies with the
default Compose file and assume their ports are available on the host.

From the repository root, run the backend tests:

```bash
uv sync --frozen --extra dev
PYTHONPATH=./src uv run --no-sync pytest tests/ -q
```

The medical-coder has its own lock file and test suite:

```bash
(cd services/medical-coder && uv sync --frozen --extra dev && uv run --no-sync pytest tests/ -q)
```

These suites use local fixtures and mocked transports; they need no running
external services or provider credentials. CI also checks changed-file lint,
formatting, a type-error ratchet, and builds both Python service images.
See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

## Architecture and operations

The backend dependency direction is:

```text
ascent_http / ascent_mcp → ascent_domain → ascent_platform
```

Layering guards live in [`tests/seams/test_layering_rule.py`](tests/seams/test_layering_rule.py).
Medical-coder is a separate Python service. DuckDB runs in-process and reads the
synthetic warehouse; sqlglot translates generated SQL to the local engine's
dialect. Persisted cohorts live in Postgres and can be joined to patient tables
through DuckDB's Postgres attachment.

- [Update, persistence, reset, and troubleshooting](docs/operations.md)
- [Corporate-network certificate setup](docs/corporate_network_setup.md)
- [LibreChat configuration and seeded resources](librechat/README.md)
- [Synthetic data and validation](data/synthetic/README.md)

## Licence and provenance

BSD 3-Clause — see [LICENSE](LICENSE).
LibreChat is MIT-licensed, Copyright (c) Danny Avila and contributors. Its source
is fetched at build time from a pinned commit; the local patches are derivative
works and remain under MIT. See the [chat build notes](librechat/README.md#configuration-and-build).

The synthetic clinical vocabularies contain no licensed terminology codes,
hierarchies, or mappings. Two structural exceptions deliberately reproduce real
OMOP identifiers: gender/race/ethnicity/unit/type concepts (for example, `8532`
for FEMALE), and twenty observation/death concepts that also retain real standard
names under `vocabulary_id = 'SNOMED'`, with invented concept codes. See the
[generator guide](synthetic_data_generation/README.md) for details and replacement instructions.
All patient data is generated; nothing derives from real patients.

The [query library](data/querylib/README.md) accompanies *Generating Patient
Cohorts from Electronic Health Records Using Two-Step Retrieval-Augmented
Text-to-SQL Generation* (ECAI 2025, [arXiv:2502.21107](https://arxiv.org/abs/2502.21107)).
It contains questions and SQL templates, not results or resolved code lists.
The [EpiTrap benchmark](data/epitrap-dataset/README.md) accompanies *ASCENT: An
Agentic System over the Model Context Protocol for Real-World Clinical Data
Analysis* and contains questions and grading rubrics, not executed results or
licensed patient data.

Report security issues privately to the maintainers in
[`.github/CODEOWNERS`](.github/CODEOWNERS), not through a public issue.
