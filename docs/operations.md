# Operations and troubleshooting

These notes are for the local demo. They are not a production deployment or
identity-management design. Commands assume the repository root and default
ports/project name unless stated otherwise.

## Resources and first start

The repository does not yet publish a measured minimum RAM/disk requirement.
The full stack builds three application images and runs several services:

- Backend and medical-coder use Python, Torch, and sentence-transformers.
- Roughly 2.5 GB of embedding weights are cached in the shared `models` volume.
  Loading those weights into memory still happens after a restart.
- LibreChat builds a React bundle. Its Dockerfile allows a 6,144 MiB Node heap;
  that is a build setting, **not a measured minimum for the whole stack**.
- Image layers, package/build caches, model weights, and database volumes all
  consume disk space. Check Docker's allocated resources if a build is killed.

The first run needs internet access for images, packages, DuckDB extensions,
and models. Provider-backed analysis continues to require network access.
MCP-only use avoids the LibreChat build and Mongo services; use the reduced
startup command in the [README](../README.md#connect-an-mcp-client-instead).

## Before exposing the stack

Keep the default loopback bindings unless you have an explicit deployment plan.
Before allowing access from another machine:

1. **Set `API_AUTH_TOKEN`** to an unpredictable value. It protects the main and
   experimental MCP servers, not every HTTP route. Liveness/health probes and
   the generic UI-kit server remain public. The token is shared: it provides
   neither individual identities nor per-caller quotas.
2. **Replace the seeded account credentials.** Set `CHAT_USER` and
   `CHAT_PASSWORD` before creating a new installation. For an existing account,
   use LibreChat's account-management workflow; changing `.env` does not rotate
   its password. Keep seed credentials aligned if you rerun `chat-init`.
3. **Replace LibreChat's known default secrets** before first use on a shared
   deployment:

   | Variable | Generate an independent value with |
   |---|---|
   | `JWT_SECRET` | `openssl rand -hex 32` |
   | `JWT_REFRESH_SECRET` | `openssl rand -hex 32` |
   | `CREDS_KEY` | `openssl rand -hex 32` |
   | `CREDS_IV` | `openssl rand -hex 16` |

   Copy the generated values into `.env`; `.env` does not execute shell
   substitutions. Do not commit or share them. Rotating signing secrets can
   invalidate sessions; changing encryption secrets can make existing stored
   credentials unreadable. Plan rotation rather than resetting a populated
   installation blindly.
4. **Use a properly configured TLS/authentication proxy** and restrict network
   access. Do not publish Postgres, Redis, Qdrant, or medical-coder directly.
   Medical-coder does not authenticate, and `API_AUTH_TOKEN` does not protect it.
   Do not use the development Compose overlay for deployment.
5. **Set provider-side billing controls.** LibreChat's credit allowance excludes
   backend model calls from MCP tools. Do not share the demo login as if it
   provided independent per-user limits.
6. Review registration, account sharing, retention, and backups for your use
   case. Conversations and saved cohorts are persistent, and generated results
   are not clinical evidence. Never enter real patient or confidential data.

Compose already maps the account/token/secret variables above. After editing
`.env`, recreate the affected services with `docker compose up -d`; a plain
`docker compose restart` does not reload container environment configuration.
See [LibreChat](../librechat/README.md) for seeded-resource behavior.

## Connecting with a bearer token

With `API_AUTH_TOKEN` configured, export the same value in the client shell
without committing it to a project configuration. For a new Claude Code entry:

```bash
claude mcp add --scope user --transport http ascent-auth \
  http://localhost:8000/mcp/ascent-mcp-v1 \
  --header "Authorization: Bearer ${API_AUTH_TOKEN:?Export the configured token in this shell first}"
```

The `.env` file is not automatically exported into your shell. Keep the header
in your user configuration, not a shared project file, and protect that
configuration. Remove or update a previous unauthenticated entry to avoid
duplicate servers. Use the deployment's HTTPS URL when connecting remotely.

LibreChat receives the matching token through the Compose
`API_AUTH_TOKEN` → `ASCENT_MCP_TOKEN` mapping. This is a static bearer, not an
interactive OAuth login flow.

## Updates and persistence

Before updating a useful installation, back up Postgres and Mongo using their
normal database backup tools. Copying live database volume files is not a
substitute for a consistent backup. Then, from a clean checkout:

```bash
git pull --ff-only
docker compose up -d --build
docker compose ps -a
```

Backend startup applies migrations. Image builds are needed after source,
dependency, Dockerfile, or chat patch changes. Existing chat agents/skills are
not updated just because images were rebuilt; follow the
[seed refresh instructions](../librechat/README.md#existing-installations-and-updates).

| State | Default storage |
|---|---|
| Application metadata and persisted cohorts | `pgdata` volume (Postgres) |
| Accounts, chat history, agents, skills | `chat_mongo` volume (Mongo) |
| Chat images/uploads/logs | `chat_images`, `chat_uploads`, `chat_logs` volumes |
| Built patient warehouse | `warehouse` volume; host development uses `.warehouse/` |
| Vocabulary database and concept index | `coderdata` and `qdrant` volumes |
| Embedding model downloads | `models` volume |

`docker compose stop` or `docker compose down` preserves named volumes.
**`docker compose down --volumes` destroys named-volume state**, including saved
cohorts and chat accounts/history. Do not use it as a routine repair command.
The default Compose project name prefixes volume names with `ascent_`.

## Troubleshooting

### Locate the failing service

```bash
docker compose ps -a
docker compose logs --tail=100 coder-init medical-coder backend librechat chat-init
```

A successful one-shot `coder-init` or `chat-init` normally shows `Exited (0)`.
`chat-init` is best-effort, so inspect its logs if accounts/resources are missing.

- **No model key:** the backend refuses startup. Configure Gemini for the
  default full demo; provider aliases do not reconfigure every service.
- **Health is green but a tool fails:** `/mcp/health` checks only registration,
  mounts, and Redis. Try the README's functional smoke task and inspect the
  relevant service log.
- **Host backend cannot connect:** default dependencies have no published ports.
  Use the [complete host setup](development.md#run-the-backend-on-the-host).
- **Another encoder gives a missing-collection error:** only `bge` is seeded.
  Model warm-up does not build a Qdrant collection.
- **Filtering fails:** malformed/incomplete concept decisions produce an explicit
  error rather than an unfiltered successful response. Investigate the provider
  or retry; do not interpret the failure as zero matching concepts.
- **Blocking LLM capacity is exhausted:** retry after outstanding calls finish.
  The worker bound is per backend process; cancelling a caller does not stop an
  already-running synchronous SDK call. Advanced settings such as
  `LLM_BLOCKING_EXECUTOR_WORKERS` need an explicit Compose environment mapping
  if you want to override them in a container.
- **No tools or agent in chat:** inspect `chat-init` and LibreChat logs and see
  the [chat guide](../librechat/README.md#existing-installations-and-updates).
- **Unexpected telemetry output:** telemetry is disabled by default through
  `TELEMETRY_ENABLED=False`. Enable it only with a configured collector; it is
  not required to run the demo.

### Certificate or model-download failures

For `CERTIFICATE_VERIFY_FAILED`, HuggingFace CDN trust errors, or LibreChat's
`UNABLE_TO_VERIFY_LEAF_SIGNATURE`, follow the
[corporate-network certificate guide](corporate_network_setup.md). Do not
work around these by disabling TLS verification.

Transient model downloads are retried. If logs indicate corrupted cached model
files rather than a certificate/network problem, stop the stack and remove
**only** its model cache:

```bash
docker compose down
docker volume rm ascent_models
docker compose up --build
```

This forces another model download but leaves the database and Qdrant volumes
alone. Adjust the volume name if you changed the project name. If you replaced
`/models` with a bind mount, this command does not clear that host directory.

To reuse an existing HuggingFace cache instead, add the following to a local
`docker-compose.override.yml` without replacing unrelated settings:

```yaml
services:
  coder-init:    { volumes: ["${HOME}/.cache/huggingface:/models"] }
  medical-coder: { volumes: ["${HOME}/.cache/huggingface:/models"] }
  backend:       { volumes: ["${HOME}/.cache/huggingface:/models"] }
```

`HF_HOME=/models` expects a `hub/` subdirectory. Compose merges mounts by target;
this replaces `/models`, not `/data` or `/warehouse`. The containers write to
this host directory. Include the override explicitly if you use `-f` arguments
or `COMPOSE_FILE` instead of automatic Compose-file discovery.
