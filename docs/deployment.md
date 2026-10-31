# Deployment

The supported deployment is one Linux host (VM or bare metal) running Docker Compose, with
Caddy terminating TLS. Each piece maps to a managed service if you outgrow one host (see the
end of this page).

## Sizing

| | Minimum | Comfortable for ~100-1,000 PDFs |
|---|---|---|
| CPU | 4 vCPU | 8 vCPU (embedding and reranking run on CPU) |
| RAM | 8 GB | 16 GB |
| Disk | 40 GB SSD | uploads + ~2x their text size for Postgres and Qdrant |

The API and the worker each load the embedding model (and the API also loads the reranker),
which takes about 1 GB of RAM per process.

## 1. Build or pull the image

Tag a release to build and publish the image with GitHub Actions
(`.github/workflows/release.yml`):

```bash
git tag v1.0.0 && git push origin v1.0.0
# -> ghcr.io/<owner>/<repo>:1.0.0 and :latest (repository name lowercased)
```

For a private repository, log the host in once:
`echo <PAT with read:packages> | docker login ghcr.io -u <user> --password-stdin`.

## 2. Prepare the host

```bash
git clone https://github.com/<owner>/<repo>.git /opt/rag && cd /opt/rag
cp .env.example .env
```

Edit `.env`:

* `ENVIRONMENT=production` (disables `/docs`, enables HSTS).
* Generate fresh values for `POSTGRES_PASSWORD` (also inside `DATABASE_URL`, with host
  `postgres`), `QDRANT_API_KEY` and `JWT_SECRET`:
  `python3 -c "import secrets; print(secrets.token_urlsafe(48))"`.
* `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` for answers; optionally the price
  settings for cost reporting.
* `METRICS_TOKEN` if anything outside the Docker network will scrape `/metrics` (Caddy blocks
  `/metrics` from the internet either way).
* Optional tracing: `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS`.

Point a DNS record at the host and open ports 80 and 443 only.

## 3. Start

```bash
export APP_IMAGE=ghcr.io/<owner>/<repo>:1.0.0
export DOMAIN=rag.example.com
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile app up -d
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile app ps
curl -s https://$DOMAIN/health/ready
```

The `migrate` container applies migrations and exits before the API and worker start. The
first start downloads the models into the `hf-cache` volume (a few minutes); the API reports
healthy as soon as it serves `/health/live`, and loads models in the background.

What the production override changes: the image comes from the registry, nothing but Caddy
publishes a port, Caddy obtains and renews certificates for `DOMAIN` automatically, every
service restarts on failure and has a memory limit, the API runs two workers behind the proxy
and trusts `X-Forwarded-For` (it is reachable only from the Docker network), so rate limits see
real client addresses.

## 4. Upgrade

```bash
export APP_IMAGE=ghcr.io/<owner>/<repo>:1.1.0
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile app up -d
```

Migrations run first, as part of the same command. Migrations in this repository are additive
and tested for a clean downgrade; still, back up first (below). Changing `EMBEDDING_MODEL` to
one with another vector size needs a new `QDRANT_COLLECTION` and a re-ingest: the worker refuses
to start against a collection of the wrong size.

## 5. Back up and restore

State lives in four volumes: `postgres-data` (source of truth: users, documents, pages, chunks,
shares, conversations), `uploads` (original PDFs), `qdrant-data` (vectors, rebuildable from
Postgres by re-ingesting) and `redis-data` (rate-limit counters only, safe to lose).

```bash
# Postgres: consistent logical dump
docker compose exec -T postgres pg_dump -U $POSTGRES_USER -Fc $POSTGRES_DB > rag-$(date +%F).dump
# Uploads
docker run --rm -v enterprise-rag-platform_uploads:/data -v "$PWD":/backup alpine tar czf /backup/uploads-$(date +%F).tgz -C /data .
# Qdrant: built-in snapshot
curl -s -X POST -H "api-key: $QDRANT_API_KEY" http://127.0.0.1:6333/collections/document_chunks/snapshots
```

(The volume name prefix is the Compose project name, `enterprise-rag-platform` by default.
Inside the production override Qdrant publishes no port; run the snapshot request with
`docker compose exec qdrant` or from another container on the network.)

Restore Postgres with `pg_restore --clean -d $POSTGRES_DB`, untar the uploads into the volume,
and restore the Qdrant snapshot or re-queue ingestion for every document.

## 6. Operate

* **Health**: `/health/live` (process) and `/health/ready` (Postgres, Redis, Qdrant). The worker
  container is healthy while its loop runs (heartbeat file checked every 30 s).
* **Logs**: `docker compose logs -f api worker`, JSON lines with `request_id` and `trace_id`.
* **Metrics**: scrape `api:8000/metrics` from a Prometheus container on the same network.
  Useful alerts: 5xx rate, `rag_ask_outcomes_total` with reason `unsupported` rising (the model
  answers without evidence), `rag_suspicious_sources_total`, p95 of
  `rag_search_duration_seconds`, failed ingestion jobs in the logs.
* **Scaling on one host**: raise `--workers` for the API (each loads the models), or run more
  worker containers (`docker compose ... up -d --scale worker=3`); jobs are claimed with
  `SKIP LOCKED`, so workers never process the same document twice.

## Beyond one host

| Component | Managed equivalent | Change needed |
|---|---|---|
| PostgreSQL | RDS, Cloud SQL, Azure Database | `DATABASE_URL` |
| Qdrant | Qdrant Cloud | `QDRANT_URL`, `QDRANT_API_KEY` |
| Redis | ElastiCache, Memorystore | `REDIS_URL` |
| Uploads volume | S3-compatible object storage | a storage backend implementing `LocalFileStorage`'s methods |
| API, worker | ECS, Cloud Run, Kubernetes Deployments | same image; `alembic upgrade head` as a pre-deploy job |
| Caddy | the platform's load balancer / ingress | TLS and forwarding there |

Several API replicas each keep their own BM25 cache; that's correct, just memory per replica.
