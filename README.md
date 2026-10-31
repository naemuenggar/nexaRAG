# Enterprise RAG Platform

Multi-user knowledge assistant: upload PDFs, ask questions, get answers grounded in retrieved
evidence with citations (document, filename, page, snippet), and an explicit "I don't know"
when the documents don't contain the answer.

> **Status: Milestone 8 of 10 (evaluation and benchmarking).** Grounded answers with citations,
> conversations, document sharing, and a reproducible evaluation harness with measured results.
> Observability, hardening and deployment come next. See
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
| GET    | `/documents`                 | yes  | Readable documents, newest first (`ownership`=`all`/`owned`/`shared`). |
| GET    | `/documents/{id}`            | yes  | Status, page count, latest ingestion job.                |
| GET    | `/documents/{id}/pages`      | yes  | Cleaned text per page (`limit`, `offset`).               |
| DELETE | `/documents/{id}`            | owner | Delete the document, its pages, chunks, vectors and file. |
| GET    | `/documents/{id}/shares`     | owner | Who has read access.                                     |
| POST   | `/documents/{id}/shares`     | owner | Share read access: `{"email": "..."}`.                   |
| DELETE | `/documents/{id}/shares/{user_id}` | owner | Revoke read access.                                |
| POST   | `/search`                    | yes  | Retrieve passages (`dense`, `sparse`, `hybrid`, `hybrid_rerank`). |
| POST   | `/ask`                       | yes  | Grounded answer with citations, or an explicit refusal.  |
| POST   | `/conversations`             | yes  | Start a conversation (optional `title`).                 |
| GET    | `/conversations`             | yes  | Your conversations, latest first.                        |
| GET    | `/conversations/{id}`        | yes  | The conversation with every stored turn and citation.    |
| POST   | `/conversations/{id}/messages` | yes | Ask in context: earlier turns are sent to the model.    |
| DELETE | `/conversations/{id}`        | yes  | Delete the conversation.                                 |

Documents belong to the uploader, who can share read access with other users. Recipients can
list, read, search and get answers from a shared document (`"access": "shared"`), but cannot
delete or reshare it (403). For anyone else the document id answers 404, exactly like an id
that does not exist. Revoking a share takes effect on the next request, including search.
Conversations are always private to their owner.

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

## Evaluation

```bash
docker compose up -d
uv run python -m rag_platform.evaluation.runner                 # retrieval only
uv run python -m rag_platform.evaluation.runner --generate      # + answers (needs LLM_* settings)
```

The runner ingests the dataset through the real pipeline (PDF build, parsing, chunking,
embedding, Qdrant) into a throwaway database, collection and upload folder, runs every question,
and writes `evaluation/results/<dataset>-<timestamp>.{md,json}` with per-question rows. Nothing
is simulated; without an LLM the generation section says "not measured".

**Dataset** (`evaluation/datasets/enterprise-v1`): 6 fictional documents (annual report,
datasheet, handbook, incident report, clinical trial summary, transit plan; 20 pages), 32
questions with reference answers and the exact file and page holding the evidence, plus 6
questions the corpus cannot answer. The facts are invented, so a model can't answer from
memory. Questions are tagged `lexical` (shares key terms or codes with the text) or
`paraphrase` (asks the same thing in different words). Use your own set with
`--dataset path/to/dir` (same two files).

**Metrics.** Retrieval relevance is per page: a result counts if it comes from an expected file
and covers an expected page. Recall@5, MRR and nDCG@5 (binary gains, duplicates of a page
count once). Generation: answer rate on answerable questions, refusal rate on unanswerable ones,
citation precision/recall against the expected pages (deterministic), and LLM-as-judge
faithfulness (share of claims supported by the cited passages), relevancy and correctness vs
the reference (0 / 0.5 / 1). Latency is wall-clock per question after one untimed warm-up;
tokens and cost come from the provider's usage report.

**Measured results** (2026-10-31, Windows laptop CPU, no GPU; `evaluation/results/enterprise-v1-20261031T132037Z.md`):

| Mode | Recall@5 | MRR | nDCG@5 | Recall@5 lexical | Recall@5 paraphrase | Latency mean / p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| dense | 1.000 | 0.924 | 0.944 | 1.000 | 1.000 | 53.6 / 51.9 / 63.0 |
| sparse (BM25) | 0.906 | 0.734 | 0.778 | 1.000 | 0.850 | 10.0 / 8.1 / 19.2 |
| hybrid (RRF) | 1.000 | 0.839 | 0.880 | 1.000 | 1.000 | 52.2 / 51.2 / 60.0 |
| hybrid + rerank | 1.000 | 0.943 | 0.957 | 1.000 | 1.000 | 228.6 / 218.7 / 278.5 |

What this shows, and what it doesn't:
* BM25 is the fastest and perfect on lexical questions, but misses 15% of paraphrased ones.
* RRF fusion recovers BM25's misses but ranks the right page lower than dense alone (MRR 0.839
  vs 0.924): BM25 promotes keyword matches from the wrong page.
* The cross-encoder fixes that ordering (best MRR and nDCG) at ~4x the latency.
* The corpus is small (20 pages), so Recall@5 saturates at 1.0 for three modes; MRR and nDCG
  are the discriminating metrics here. A larger corpus with near-duplicate documents is
  needed to separate the modes on recall. Treat these numbers as a regression baseline, not as
  a general claim about the methods.
* **Generation metrics have not been measured yet**: no LLM is configured on the machine that
  produced this run. Configure `LLM_*` and run with `--generate`.

`tests/integration/test_retrieval_benchmark.py` runs the same evaluation in CI and fails if
any mode drops below a floor set slightly under these numbers.

