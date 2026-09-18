# Development

Run commands from the repository root in a Bash-compatible shell. Follow the
[root quickstart](../README.md#quickstart) first to configure `.env`; do not
replace an existing file with the template.

Python is pinned to **3.12**. Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --frozen --extra dev
```

## Iterate in containers

This is the simplest way to keep networking and dependencies consistent:

```bash
docker compose up -d --build backend medical-coder
docker compose logs -f backend medical-coder
```

Compose starts the required dependency services, including `coder-init`. Python
source is copied into the images, not mounted for hot reload: repeat the build
command after source changes. The full chat UI is optional; start it using the
root quickstart when needed.

## Run the backend on the host

The default Compose file deliberately exposes **no dependency ports**. Use the
opt-in [development overlay](../docker-compose.dev.yml) rather than trying to
reach its internal hostnames from your laptop.

If the full demo is already running, stop its backend and chat first so they do
not conflict with the host backend or keep calling the wrong process:

```bash
docker compose stop backend librechat chat-init
```

Then start only the containerized dependencies:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build \
  postgres redis qdrant medical-coder
```

Explicit `-f` arguments replace the normal Compose-file selection. If you use
the [corporate TLS overlay](corporate_network_setup.md), also include
`-f docker-compose.tls-proxy.yml`. The host Python process needs its own trusted
CA configuration; container mounts do not configure your laptop.

The development overlay publishes only loopback addresses:

| Dependency | Host endpoint |
|---|---|
| Postgres | `127.0.0.1:15432` |
| Redis | `127.0.0.1:16379` |
| Qdrant HTTP | `127.0.0.1:16333` |
| Medical-coder | `http://127.0.0.1:18001/v1/medical-coder` |

Qdrant's port is available for debugging; code lookup normally goes through
medical-coder. Wait for the coder's model warm-up before continuing:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml ps -a
curl --fail http://127.0.0.1:18001/v1/medical-coder/
```

Export the matching settings **in the shell that will run the backend**:

```bash
export DB_HOST=127.0.0.1 DB_PORT=15432 DB_NAME=ascent
export DB_APP_USR=ascent DB_APP_PWD=ascent
export REDIS_HOST=127.0.0.1 REDIS_PORT=16379 CACHE=redis-local
export MEDICAL_CODER_BASE_URL=http://127.0.0.1:18001/v1/medical-coder
export WAREHOUSE_DATA_DIR=data/synthetic WAREHOUSE_WORK_DIR=.warehouse
```

These values match the shipped local Postgres credentials. A previously
initialized volume with different credentials must use its existing values.
Provider keys and `DATA_SCALE` can remain in `.env`. Exported settings override
that file. Host model downloads use the host's HuggingFace cache, not the
container's `models` volume.

Build the warehouse and run migrations before starting the single host process:

```bash
PYTHONPATH=./src uv run --no-sync python -m ascent_platform.warehouse.bootstrap
PYTHONPATH=./src uv run --no-sync python src/migrate.py
PYTHONPATH=./src uv run --no-sync python -m hypercorn --reload \
  --bind 127.0.0.1:8000 src.main:app
```

Use an MCP client at the usual backend URL. This recipe does not rewire
containerized LibreChat to the host; use the fully containerized stack for chat
integration tests. To switch back, stop the host process, use a fresh shell
without the exported overrides, and run:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
docker compose up -d --build
```

Do not add `--volumes`: saved state should survive this switch.

## Tests and checks

Backend:

```bash
PYTHONPATH=./src uv run --no-sync pytest tests/ -q
```

Medical-coder uses a separate project and lock file:

```bash
(cd services/medical-coder && uv sync --frozen --extra dev && uv run --no-sync pytest tests/ -q)
```

No provider keys or running external services are required by these suites.
Tests may create local/in-memory databases and use shipped synthetic fixtures.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for lint, formatting, and PR checks.

## Database migrations

Postgres holds application state and persisted cohorts. Container startup runs
migrations automatically. With the host environment above, inspect or apply them:

```bash
PYTHONPATH=./src uv run --no-sync alembic current
PYTHONPATH=./src uv run --no-sync alembic upgrade head
```

To create a migration after changing the application models:

```bash
PYTHONPATH=./src uv run --no-sync alembic revision --autogenerate -m "describe the change"
```

Review generated migrations before applying them. Back up persistent state
before upgrading an installation you want to retain; see [operations](operations.md#updates-and-persistence).
