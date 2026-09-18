# LibreChat, built from upstream source with local patches.
#
# The published image cannot be used as-is: the two defaults this demo needs --
# skills on, MCP servers selected -- live in the React bundle, not in
# librechat.yaml or any environment variable. There is no configuration route
# to them, so the client has to be rebuilt.
#
# Upstream is pinned to a commit rather than a tag or `latest`. The patches
# below carry file context, so a moving base would eventually fail to apply.
#
# This is the default build. A sibling, librechat/Dockerfile.tls-proxy, is this
# file plus the CA handling a TLS-intercepting proxy needs at build time, and
# docker-compose.tls-proxy.yml selects it. The two are kept line-for-line
# identical apart from that; a change here belongs there too.
ARG LIBRECHAT_COMMIT=16e4d1419107a31c00fa23442ef07688e64d6366

FROM alpine/git:2.49.1 AS source
ARG LIBRECHAT_COMMIT
WORKDIR /src
# Fetching the single commit rather than cloning: the full history is ~1 GB and
# nothing here reads it.
RUN git init -q . \
 && git remote add origin https://github.com/danny-avila/LibreChat.git \
 && git fetch -q --depth 1 origin "${LIBRECHAT_COMMIT}" \
 && git checkout -q FETCH_HEAD
COPY badge-defaults.patch terms-notice.patch temporary-chat.patch thought-signatures.patch /tmp/
# `git apply` fails loudly on drift. That matters more than it looks: a patch
# that silently no-ops produces a working image with none of the defaults, and
# the only symptom is a demo that opens with its tools off -- or, for the terms
# notice, one that never appears at all.
RUN git apply --verbose /tmp/badge-defaults.patch \
 && git apply --verbose /tmp/terms-notice.patch \
 && git apply --verbose /tmp/temporary-chat.patch \
 && git apply --verbose /tmp/thought-signatures.patch


FROM node:24.16.0-alpine AS node

RUN apk upgrade --no-cache \
 && apk add --no-cache jemalloc python3 py3-pip uv

ENV LD_PRELOAD=/usr/lib/libjemalloc.so.2

# `uv` for MCP servers launched over stdio. Unused by this stack, which reaches
# its server over HTTP, but removing it would diverge from upstream's runtime
# for no gain.
COPY --from=ghcr.io/astral-sh/uv:0.9.5-python3.12-alpine /usr/local/bin/uv /usr/local/bin/uvx /bin/

ARG NODE_MAX_OLD_SPACE_SIZE=6144

RUN mkdir -p /app && chown node:node /app
WORKDIR /app
USER node

# Manifests first, so a change to the patch does not re-run `npm ci`.
COPY --from=source --chown=node:node /src/package.json /src/package-lock.json ./
COPY --from=source --chown=node:node /src/api/package.json ./api/package.json
COPY --from=source --chown=node:node /src/client/package.json ./client/package.json
COPY --from=source --chown=node:node /src/packages/data-provider/package.json ./packages/data-provider/package.json
COPY --from=source --chown=node:node /src/packages/data-schemas/package.json ./packages/data-schemas/package.json
COPY --from=source --chown=node:node /src/packages/api/package.json ./packages/api/package.json

RUN touch .env \
 && mkdir -p /app/client/public/images /app/logs /app/uploads /app/skill /app/data \
 && chmod 1777 /app/data \
 && npm config set fetch-retry-maxtimeout 600000 \
 && npm config set fetch-retries 5 \
 && npm config set fetch-retry-mintimeout 15000 \
 && npm ci --no-audit

# The DUMMY_SIGNATURE removal cannot ride the git-apply stage: it edits code
# vendored inside the `@librechat/agents` npm package, which only exists after
# `npm ci`. Same fail-loudly contract as the patches -- see the script header
# for why this one line caused the phantom-weather thought summaries.
COPY --chown=node:node remove-dummy-signature.js /tmp/remove-dummy-signature.js
RUN node /tmp/remove-dummy-signature.js

COPY --from=source --chown=node:node /src/ ./

RUN NODE_OPTIONS="--max-old-space-size=${NODE_MAX_OLD_SPACE_SIZE}" npm run frontend \
 && npm prune --production \
 && npm cache clean --force

EXPOSE 3080
ENV HOST=0.0.0.0
CMD ["npm", "run", "backend"]
