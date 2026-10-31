# Enterprise RAG Platform

Multi-user knowledge assistant: upload PDFs, ask questions, get answers grounded in retrieved
evidence with citations (document, filename, page, snippet), and an explicit "I don't know"
when the documents don't contain the answer.

> **Status: Milestone 4 of 10 (dense and BM25 retrieval).** Users upload PDFs, the worker
> chunks, embeds and indexes them, and `POST /search` retrieves passages by vector similarity or
> BM25. Hybrid fusion, reranking and answer generation come next. See
> [docs/architecture.md](docs/architecture.md) for the full design
> and the milestone plan.

## Quick start (local)

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker with Compose v2.

```bash
cp .env.example .env            # then set POSTGRES_PASSWORD (also in DATABASE_URL), QDRANT_API_KEY, JWT_SECRET
docker compose up -d            # Postgres, Redis, Qdrant
uv sync                         # creates .venv from uv.lock
uv run alembic upgrade head     # apply database migrations
uv run uvicorn --factory rag_platform.main:app_factory --reload
uv run python -m rag_platform.ingestion.worker   # second terminal; downloads the embedding model on first start
```

URLs in `.env.example` use `127.0.0.1` rather than `localhost`: on Windows, `localhost` tries
IPv6 first and each refused attempt costs about two seconds before falling back.

Open http://127.0.0.1:8000/docs (disabled when `ENVIRONMENT=production`).

To run the API and worker in Docker as well: `docker compose --profile app up -d --build`
(runs migrations in a one-shot `migrate` container first).

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest                    # unit tests, no services needed
uv run pytest -m integration     # needs `docker compose up -d`
```

## API

| Method | Path                         | Auth | Description                                              |
|--------|------------------------------|------|----------------------------------------------------------|
| GET    | `/health/live`               | no   | Process is up. Never checks dependencies.                |
| GET    | `/health/ready`              | no   | Postgres, Redis and Qdrant answer in time, else 503.     |
| POST   | `/auth/register`             | no   | Create an account (password 12 to 128 characters).       |
| POST   | `/auth/login`                | no   | Returns a bearer token.                                  |
| GET    | `/auth/me`                   | yes  | The current user.                                        |
| POST   | `/documents`                 | yes  | Upload a PDF (multipart field `file`). Returns 202.      |
| GET    | `/documents`                 | yes  | Your documents, newest first (`limit`, `offset`).        |
| GET    | `/documents/{id}`            | yes  | Status, page count, latest ingestion job.                |
| GET    | `/documents/{id}/pages`      | yes  | Cleaned text per page (`limit`, `offset`).               |
| DELETE | `/documents/{id}`            | yes  | Delete the document, its pages and the stored file.      |
| POST   | `/search`                    | yes  | Retrieve passages: `mode` is `dense` or `sparse` (BM25). |

Documents belong to the uploader. Another user's document id answers 404, exactly like an id
that does not exist.

```bash
curl -s -X POST http://127.0.0.1:8000/auth/register -H "Content-Type: application/json"   -d '{"email": "ana@example.com", "password": "a-long-enough-password"}'
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/auth/login -H "Content-Type: application/json"   -d '{"email": "ana@example.com", "password": "a-long-enough-password"}' | jq -r .access_token)
curl -s -X POST http://127.0.0.1:8000/documents -H "Authorization: Bearer $TOKEN" -F "file=@report.pdf"
```

```json
{
  "id": "7522735b-5394-4347-abe3-3841ac329368",
  "filename": "report.pdf",
  "size_bytes": 1043,
  "content_sha256": "9c1f...",
  "page_count": null,
  "status": "pending",
  "error_message": null,
  "created_at": "2026-10-31T11:18:58.990Z",
  "updated_at": "2026-10-31T11:18:58.990Z",
  "latest_job": {"id": "8838...", "status": "queued", "attempts": 0, "started_at": null, "finished_at": null}
}
```

Once the worker has run, `GET /documents/{id}` shows `"status": "ready"` and `"page_count": 3`,
and `GET /documents/{id}/pages` returns `[{"page_number": 1, "text": "...", "char_count": 20}, ...]`.

Search your own documents (optionally only some of them):

```bash
curl -s -X POST http://127.0.0.1:8000/search -H "Authorization: Bearer $TOKEN"   -H "Content-Type: application/json"   -d '{"query": "How do PV modules turn light into power?", "mode": "dense", "top_k": 2}'
```

```json
{
  "mode": "dense",
  "latency_ms": 38.7,
  "results": [
    {
      "rank": 1,
      "score": 0.764,
      "chunk_id": "2b0c...",
      "document_id": "7522...",
      "filename": "energy.pdf",
      "page_start": 1,
      "page_end": 1,
      "text": "Solar panels convert sunlight into electricity using photovoltaic cells."
    }
  ]
}
```

(Output from the Docker stack with `bge-small-en-v1.5`. Note that the query shares no words with
the passage.) `document_ids` restricts the search; naming a document that isn't yours, or isn't
ready yet, returns 404.

Upload errors: `400 empty_file`, `409 duplicate_document` (with `details.document_id` of your
existing copy), `413 payload_too_large`, `415 unsupported_media_type` (checked by file signature,
not by extension or Content-Type). PDFs that are scanned (no text), damaged or password-protected
are accepted, then end as `"status": "failed"` with a readable `error_message`.

Every error response uses one envelope, and every response carries `X-Request-ID`:

```json
{"error": {"code": "not_found", "message": "Document not found", "request_id": "3f2a..."}}
```
