# Enterprise RAG Platform

Multi-user knowledge assistant: upload PDFs, ask questions, get answers grounded in retrieved
evidence with citations (document, filename, page, snippet), and an explicit "I don't know"
when the documents don't contain the answer.

> **Status: Milestone 6 of 10 (grounded generation and citations).** `POST /ask` answers from
> your documents with verified citations (document, file, page, snippet), or says it doesn't
> know. Conversations, evaluation and observability come next. See
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
| POST   | `/search`                    | yes  | Retrieve passages (`dense`, `sparse`, `hybrid`, `hybrid_rerank`). |
| POST   | `/ask`                       | yes  | Grounded answer with citations, or an explicit refusal.  |

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
# Default mode is hybrid_rerank. Filters narrow the documents searched:
#   "filters": {"document_ids": [...], "filename_contains": "annual",
#               "uploaded_after": "2026-01-01T00:00:00Z", "uploaded_before": "..."}
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
      "text": "Solar panels convert sunlight into electricity using photovoltaic cells.",
      "scores": {"dense": 0.764}
    }
  ]
}
```

(Output from the Docker stack with `bge-small-en-v1.5`. Note that the query shares no words with
the passage.) `document_ids` restricts the search; naming a document that isn't yours, or isn't
ready yet, returns 404. `scores` shows every stage's score (`dense`, `bm25`, `fused`, `rerank`);
`score` is the one the ranking uses.

### Asking questions

Configure a model first (any OpenAI-compatible endpoint, or Anthropic):

```bash
LLM_PROVIDER=openai            # or: anthropic
LLM_BASE_URL=https://api.openai.com/v1   # vLLM/Ollama/LiteLLM: their /v1 URL; Anthropic: https://api.anthropic.com
LLM_API_KEY=...
LLM_MODEL=...
```

Without it, `/ask` answers `503 llm_not_configured`; search keeps working.

```bash
curl -s -X POST http://127.0.0.1:8000/ask -H "Authorization: Bearer $TOKEN"   -H "Content-Type: application/json" -d '{"question": "What was revenue in Q3 2024?"}'
```

```json
{
  "answer": "Q3 2024 revenue was 4.2 million dollars [1].",
  "answered": true,
  "refusal_reason": null,
  "citations": [
    {
      "number": 1,
      "chunk_id": "…",
      "document_id": "…",
      "filename": "annual-report.pdf",
      "page_start": 2,
      "page_end": 2,
      "snippet": "The company reported quarterly revenue of 4.2 million dollars in Q3 2024."
    }
  ],
  "sources_considered": 3,
  "suspicious_sources": 0,
  "usage": {"model": "…", "prompt_tokens": 412, "completion_tokens": 14, "cost_usd": null},
  "latency": {"retrieval_ms": 85.2, "generation_ms": 910.4, "total_ms": 996.1}
}
```

(Response shape as produced by the integration tests with a scripted model; token and latency
numbers depend on your model.) When the documents don't contain the answer, the response has
`"answered": false`, the fixed answer `"I don't know based on the provided documents."`, no
citations, and a `refusal_reason`: `no_sources`, `low_relevance` (optional threshold on the
reranker score), `model_declined`, or `unsupported` (the model answered without citing a
source it was given; the answer is withheld). Set `LLM_PRICE_INPUT_PER_1K` and
`LLM_PRICE_OUTPUT_PER_1K` to get `cost_usd`.

Upload errors: `400 empty_file`, `409 duplicate_document` (with `details.document_id` of your
existing copy), `413 payload_too_large`, `415 unsupported_media_type` (checked by file signature,
not by extension or Content-Type). PDFs that are scanned (no text), damaged or password-protected
are accepted, then end as `"status": "failed"` with a readable `error_message`.

Every error response uses one envelope, and every response carries `X-Request-ID`:

```json
{"error": {"code": "not_found", "message": "Document not found", "request_id": "3f2a..."}}
```
