FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    # Cache the sentence-transformers model here; compose mounts a volume so
    # the ~400 MB download survives a rebuild and is shared with coder-init.
    HF_HOME=/models

WORKDIR /app

# Manifests first for layer caching, then the package.
# No private index: everything resolves from PyPI.
COPY ./services/medical-coder/pyproject.toml /app/pyproject.toml
COPY ./services/medical-coder/uv.lock /app/uv.lock
COPY ./services/medical-coder/README.md /app/README.md
COPY ./services/medical-coder/src /app/src
# Installed from the lock so the image resolves the same way twice. Every
# dependency, duckdb and sqlglot and sentence-transformers included, is
# declared in pyproject and pinned by uv.lock.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
    && uv pip install --system --no-deps "."

COPY ./services/medical-coder/scripts /app/scripts

# Bake the vocabulary in rather than bind-mounting it. It is 96 KB, and a bind
# mount that silently resolves empty (a host path Docker is not sharing) fails
# later as "CONCEPT.csv.zip is missing" while the file sits plainly in the repo.
COPY ./data/synthetic/vocabulary /app/vocabulary

# Fail the build rather than the first request if the package or the two
# bootstrap scripts cannot be imported.
RUN python -c "import ascent_medical_coder, duckdb, sqlglot" && \
    python -m py_compile scripts/build_vocabulary.py scripts/seed_qdrant.py

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "ascent_medical_coder.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
