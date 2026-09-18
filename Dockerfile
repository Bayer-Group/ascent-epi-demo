FROM python:3.12-slim
# Pinned, not :latest -- an unpinned builder makes the image non-reproducible
# and silently changes resolver behaviour between two builds of the same commit.
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

# UV_HTTP_TIMEOUT: uv defaults to 30s per request, and this image pulls some
# very large wheels (torch, scikit-learn, sentence-transformers). On a slow or
# contended link the download exceeds 30s and the whole build fails with a
# network timeout rather than retrying -- observed on a first build from a
# clean cache. Raised, not removed: a hung mirror should still fail eventually.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_HTTP_TIMEOUT=180

WORKDIR /app

# Manifest and lock alone, so the dependency layer caches on their content
# rather than on the whole source tree.
COPY pyproject.toml uv.lock ./

# Install requirements
# The lock is COPYed and honoured, not regenerated. This used to run
# `uv lock && uv sync --frozen`, which resolves fresh from PyPI and then
# asserts the freshly-written lock is unchanged -- so --frozen guaranteed
# nothing and two builds of the same commit a week apart shipped different
# dependency trees. The committed uv.lock is the pinned set.
#
# No private package index is used: everything resolves from PyPI, except
# torch, which is pinned to PyTorch's public CPU index in pyproject.toml --
# that drops ~4 GB of CUDA packages nothing in this stack can use.
# If you add a dependency from a private index, set UV_EXTRA_INDEX_URL at
# build time rather than defaulting it here -- ARG defaults are visible in
# `docker history`, which is an easy way to leak a token.
#
# --no-install-project: this layer has only pyproject.toml and uv.lock, not
# src/, and the project is an installable package (it has a [build-system]), so
# without this uv tries to build it here and setuptools fails on a missing
# src/. Only the dependencies belong in this layer anyway -- that is what makes
# it cache independently of the source. The application itself is never
# installed: it arrives with `COPY . .` below and runs off PYTHONPATH.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# The former ascent_ai and ascent_non_omop packages were dissolved into
# src/ascent_domain and src/ascent_platform, which arrive with the source copy
# below; PYTHONPATH puts src/ on the path. Nothing is installed from an index.

COPY . .

# Default environment
ARG APP_VERSION
ENV PATH="/app/.venv/bin:$PATH"
# Not appended to an inherited ${PYTHONPATH}: python:3.12-slim does not set one,
# so the reference expanded to empty and left a trailing separator -- and
# BuildKit warns on a variable nothing in this Dockerfile defines. Set it
# outright; compose can still override the whole value.
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV AWS_EMF_ENVIRONMENT="Local"
ENV ENV_LOCAL_PATH="/app"
ENV USE_STRUCTURED_LOGGING="true"
ENV LOG_LEVEL="INFO"
ENV APP_VERSION=${APP_VERSION}

# Guard: the previous ASCENT_*_VERSION build args failed open — when the pins
# were absent the install layers silently no-opped and the image shipped with no
# libraries, surfacing only as an ImportError at container start. Fail the build.
# One package per layer of the monorepo, so a broken import fails here rather
# than at container start. ascent_non_omop was dissolved into ascent_domain and
# ascent_platform; naming a package that no longer exists would fail the build
# for the wrong reason.
RUN python -c "import ascent_platform, ascent_domain, ascent_http, ascent_mcp; \
    import ascent_domain.omop.models.uncertainty.clues_api; \
    import ascent_domain.non_omop.workflows; \
    import torch, sentence_transformers"

EXPOSE 8000

# Perform migrations & launch service. Configuration comes from the
# environment (.env / compose), not from a cloud secret store.
# Build the warehouse before the workers fork: several workers each opening
# AND building it race, and the loser serves empty results without erroring.
# `&&`, not `;`: chained with semicolons, a failed warehouse build or a failed
# migration was ignored and the server started anyway -- serving empty results
# or erroring per request, which is the exact failure the pre-fork build above
# exists to prevent. `exec` hands PID 1 to hypercorn so SIGTERM reaches it and
# `docker stop` is a clean shutdown rather than a 10s wait for SIGKILL.
# JSON form on one line: shell form makes the shell PID 1, so SIGTERM never
# reaches the server and every `docker stop` waits out the 10s grace period for
# SIGKILL (BuildKit warns JSONArgsRecommended for exactly this). `exec` on the
# last command hands PID 1 to hypercorn. Kept as one line because a JSON array
# cannot span lines with backslash continuations.
#
# The configuration check runs first, before the warehouse build, for two
# reasons. It answers in a second where the build takes minutes, so a missing
# provider key is reported immediately rather than after the slowest step. And
# hypercorn exits 0 when a worker fails to load, so a lifespan error reaches
# compose as a clean shutdown; failing here keeps the common case -- no LLM
# provider configured -- a non-zero exit with a named reason.
CMD ["/bin/sh", "-c", "python -c 'from ascent_platform.config.runtime import Settings, require_app_config; require_app_config(Settings())' && python -m ascent_platform.warehouse.bootstrap && python src/migrate.py && exec python -m hypercorn --workers 4 --websocket-ping-interval 20 --bind 0.0.0.0:8000 src.main:app"]
