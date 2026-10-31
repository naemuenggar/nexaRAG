# Enterprise RAG Platform

Multi-user knowledge assistant: upload PDFs, ask questions, get answers grounded in retrieved
evidence with citations (document, filename, page, snippet), and an explicit "I don't know"
when the documents don't contain the answer.

> **Status: Milestone 1 of 10 (project setup and infrastructure).** The API currently serves
> health endpoints only. See [docs/architecture.md](docs/architecture.md) for the full design
> and the milestone plan.

## Quick start (local)

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker with Compose v2.

```bash
cp .env.example .env            # then set POSTGRES_PASSWORD (also in DATABASE_URL), QDRANT_API_KEY, JWT_SECRET
docker compose up -d            # Postgres, Redis, Qdrant
uv sync                         # creates .venv from uv.lock
uv run alembic upgrade head     # apply database migrations
uv run uvicorn --factory rag_platform.main:app_factory --reload
```

URLs in `.env.example` use `127.0.0.1` rather than `localhost`: on Windows, `localhost` tries
IPv6 first and each refused attempt costs about two seconds before falling back.

Open http://127.0.0.1:8000/docs (disabled when `ENVIRONMENT=production`).

To run the API in Docker as well: `docker compose --profile app up -d --build`
(runs migrations in a one-shot `migrate` container first).

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest                    # unit tests, no services needed
uv run pytest -m integration     # needs `docker compose up -d`
```

## API (Milestone 1)

| Method | Path            | Description                                                     |
|--------|-----------------|-----------------------------------------------------------------|
| GET    | `/health/live`  | Process is up. Never checks dependencies.                       |
| GET    | `/health/ready` | Postgres, Redis and Qdrant answer within the timeout, else 503. |

```bash
curl -s http://127.0.0.1:8000/health/ready
```

```json
{
  "status": "ready",
  "components": {
    "postgres": {"healthy": true, "latency_ms": 3.1, "error": null},
    "redis": {"healthy": true, "latency_ms": 1.2, "error": null},
    "qdrant": {"healthy": true, "latency_ms": 4.8, "error": null}
  }
}
```

Every error response uses one envelope, and every response carries `X-Request-ID`:

```json
{"error": {"code": "not_found", "message": "Document not found", "request_id": "3f2a..."}}
```
