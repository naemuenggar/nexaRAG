FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir "uv>=0.5,<1"

WORKDIR /app
# Install locked third-party dependencies first so this layer caches across code changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN groupadd --system app && useradd --system --gid app --home-dir /app app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src ./src
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations
# Pre-create the upload dir so a fresh named volume inherits the app user's ownership.
RUN mkdir -p /app/data/uploads && chown -R app:app /app/data

USER app
EXPOSE 8000
CMD ["uvicorn", "--factory", "rag_platform.main:app_factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
