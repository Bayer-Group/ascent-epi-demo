# LibreChat in ASCENT

The default Compose stack builds [LibreChat](https://github.com/danny-avila/LibreChat)
and exposes it at <http://localhost:3080>. It connects to ASCENT's main MCP
server over the internal Docker network. The experimental and UI-kit servers
are available separately; they are not in the shipped chat MCP configuration.

## First login and available entry points

Use the account created by `chat-init`:

- Email: `demo@ascent.local`
- Password: `ascentdemo`

Override `CHAT_USER` and `CHAT_PASSWORD` in the root `.env` before first startup.
LibreChat uses login sessions; this account is for local demo convenience, not
a production identity system. Before exposing chat, follow the
[security checklist](../docs/operations.md#before-exposing-the-stack).

There are two configured entry points:

- The default **ASCENT model preset** selects Google/Gemini, attaches
  `ascent-mcp`, enables skills, and supplies the demo system prompt.
- The seeded **ASCENT agent** has instructions from
  [`agent-instructions.md`](agent-instructions.md), the discovered MCP tools,
  and skills enabled.

[`seed.js`](seed.js) attempts to grant public viewer access to newly created
agents and skills. Newly registered accounts can use shared resources; if they
are missing, investigate seeding/permissions instead of assuming a new account
cannot have an agent. Existing resources may predate that sharing step.

Do not enter patient, confidential, or licensed data. The consent notice explains
that patient results are synthetic and literature results describe real sources.

## Configuration and build

The main configuration is [`librechat.yaml`](librechat.yaml): MCP connection,
model preset, capabilities, consent text, and credit allowance. Environment
variables are mapped in the root [`docker-compose.yml`](../docker-compose.yml).
`GEMINI_API_KEY` is passed as `GOOGLE_KEY`, while `API_AUTH_TOKEN` is passed as
`ASCENT_MCP_TOKEN` for the static bearer on backend tool calls.

Model selection has separate settings: the picker (`GOOGLE_MODELS`), the YAML
preset, and the seeded agent's provider/model. Changing the picker alone does
not update the preset or an existing agent. If you customize models, align these
settings and refresh the affected agent. Extra environment variables for the
seed script require an explicit `chat-init.environment` mapping in Compose.

The image is built from a pinned upstream commit in [`Dockerfile`](Dockerfile).
YAML supplies model-spec MCP selection and server-side skills configuration;
client patches handle additional defaults and presentation behavior:

| File | Purpose |
|---|---|
| [`badge-defaults.patch`](badge-defaults.patch) | Skills badge/defaults and MCP selection outside a model spec |
| [`terms-notice.patch`](terms-notice.patch) | Consent notice behavior |
| [`temporary-chat.patch`](temporary-chat.patch) | Temporary-chat behavior |
| [`thought-signatures.patch`](thought-signatures.patch) | Thought-signature handling |
| [`remove-dummy-signature.js`](remove-dummy-signature.js) | Patch the installed agents package after `npm ci` |

Rebuild for Dockerfile, patch, or package changes. Runtime YAML is bind-mounted;
restart LibreChat after editing it. Environment changes require recreation with
`docker compose up -d`, not just `restart`.

Meilisearch and LibreChat's file-RAG services are not included. The config removes
`file_search` and `execute_code` capabilities; this demo is for tool-driven
analysis, not a bundled document-RAG or code-interpreter service.

## Existing installations and updates

`chat-init` is best-effort and does not make the stack fail if seeding fails.
Check its logs:

```bash
docker compose logs --tail=100 chat-init librechat
```

The scripts create missing resources but **skip existing agents and skills by
name**. Rebuilding images or rerunning the seed does not update their bodies,
model choices, tools, or sharing. Updating `.claude/skills/` also does not update
skills already stored in Mongo.

For an installation you want to preserve:

1. Sign in as the owner/administrator and update the existing seeded agent and
   skills through LibreChat, using the checked-in instruction/skill files as
   the source. Check public viewer permissions if other users cannot see them.
2. Prefer editing over deletion when resources are referenced by conversations.
   To recreate an unused seeded resource, remove only that resource through
   LibreChat, then rerun the seed with valid account credentials:

   ```bash
   docker compose run --rm chat-init
   ```

3. Inspect logs and verify the resulting tools, model, skills, and sharing.
   Never delete the Mongo volume merely to refresh an agent or skill.

Changing `CHAT_PASSWORD` in `.env` does not update an existing account. The
user-creation script ignores duplicate-account errors; a mismatched password
then prevents the seed script from logging in. Change the actual account's
password through LibreChat's account-management workflow and keep seed
configuration consistent.

## Limits and billing

The shipped Compose configuration enables message limits of 40 per user per
hour, 60 per IP per hour, and two concurrent messages. The YAML balance block
provides 2,000,000 credits initially and a weekly refill of the same amount
(configured as a $2/week per-user allowance). These are local guardrails, not
individual role/tier policies or a guaranteed total spend ceiling.

**Only LibreChat's own model calls are accounted for in that balance.** The
backend's model calls while executing MCP tools use its own credentials and
are outside this ledger. Set provider-side billing controls too. Users sharing
the demo login also share its account-level allowance.

For certificate problems during the chat build or provider calls, use the
[corporate-network guide](../docs/corporate_network_setup.md). The TLS overlay
selects [`Dockerfile.tls-proxy`](Dockerfile.tls-proxy) and passes the CA bundle as
a build secret, in addition to runtime trust configuration.

## Licence

LibreChat is MIT-licensed, Copyright (c) Danny Avila and contributors. Upstream
source is fetched at image build time, not vendored here. Local patches to it
remain MIT-licensed derivative works. ASCENT's repository licence and data
provenance are described in the [root README](../README.md#licence-and-provenance).
