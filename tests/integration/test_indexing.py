"""Chunking + embedding + Qdrant indexing, end to end through the worker."""

import math
import uuid

import pytest
from fastapi.testclient import TestClient
from qdrant_client import models as qm
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from rag_platform.core.config import Settings
from rag_platform.db.models import Chunk, IngestionJob, JobStatus
from rag_platform.embeddings.sentence_transformers import SentenceTransformerEmbedder
from rag_platform.infrastructure.qdrant import create_qdrant_client
from rag_platform.infrastructure.vector_index import QdrantVectorIndex, VectorIndexError
from rag_platform.ingestion.worker import IngestionWorker, create_embedder
from tests.fakes import HashingEmbedder
from tests.integration.test_documents_api import drain, signup, upload
from tests.pdf_factory import make_pdf

pytestmark = pytest.mark.integration


def long_page(topic: str, sentences: int = 12) -> str:
    return "\n".join(
        f"Sentence {i} about {topic} with some extra filler words to pad it."
        for i in range(sentences)
    )


async def db_chunks(engine: AsyncEngine, document_id: str) -> list[Chunk]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            select(Chunk).where(Chunk.document_id == uuid.UUID(document_id)).order_by(Chunk.ordinal)
        )
        return list(rows.all())  # type: ignore[arg-type]


async def test_chunks_are_stored_and_indexed_with_owner_payload(
    api: TestClient,
    worker: IngestionWorker,
    worker_engine: AsyncEngine,
    vector_index: QdrantVectorIndex,
    app_settings: Settings,
) -> None:
    headers = signup(api)
    owner_id = api.get("/auth/me", headers=headers).json()["id"]
    pdf = make_pdf([long_page(f"solar power {uuid.uuid4()}"), "", long_page("wind farms")])
    document_id = upload(api, headers, pdf).json()["id"]
    await drain(worker)
    assert api.get(f"/documents/{document_id}", headers=headers).json()["status"] == "ready"

    chunks = await db_chunks(worker_engine, document_id)
    assert len(chunks) >= 4  # ~144 words per page, 40-word chunks
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert {c.page_start for c in chunks} == {1, 3}  # blank page 2 yields nothing
    assert all(c.page_start == c.page_end for c in chunks)
    assert all(c.word_count <= app_settings.chunk_size_words for c in chunks)
    assert all(str(c.owner_id) == owner_id for c in chunks)

    assert await vector_index.count_document(uuid.UUID(document_id)) == len(chunks)
    client = create_qdrant_client(app_settings)
    try:
        points = await client.retrieve(
            app_settings.qdrant_collection, ids=[str(chunks[0].id)], with_vectors=True
        )
    finally:
        await client.close()
    assert points[0].payload == {
        "owner_id": owner_id,
        "document_id": document_id,
        "ordinal": 0,
        "page_start": 1,
        "page_end": 1,
    }
    vector = points[0].vector
    assert isinstance(vector, list)
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-4)


async def test_reprocessing_replaces_instead_of_duplicating(
    api: TestClient,
    worker: IngestionWorker,
    worker_engine: AsyncEngine,
    vector_index: QdrantVectorIndex,
) -> None:
    headers = signup(api)
    document_id = upload(api, headers, make_pdf([long_page(f"rerun {uuid.uuid4()}")])).json()["id"]
    await drain(worker)
    first = await db_chunks(worker_engine, document_id)

    async with worker_engine.begin() as conn:
        await conn.execute(
            update(IngestionJob)
            .where(IngestionJob.document_id == uuid.UUID(document_id))
            .values(status=JobStatus.QUEUED)
        )
    await drain(worker)

    second = await db_chunks(worker_engine, document_id)
    assert len(second) == len(first)
    assert {c.id for c in second}.isdisjoint({c.id for c in first})
    assert await vector_index.count_document(uuid.UUID(document_id)) == len(second)


async def test_delete_removes_chunks_and_vectors(
    api: TestClient,
    worker: IngestionWorker,
    worker_engine: AsyncEngine,
    vector_index: QdrantVectorIndex,
) -> None:
    headers = signup(api)
    document_id = upload(api, headers, make_pdf([long_page(f"gone {uuid.uuid4()}")])).json()["id"]
    await drain(worker)
    assert await vector_index.count_document(uuid.UUID(document_id)) > 0

    assert api.delete(f"/documents/{document_id}", headers=headers).status_code == 204

    assert await vector_index.count_document(uuid.UUID(document_id)) == 0
    async with worker_engine.connect() as conn:
        remaining = await conn.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_id == uuid.UUID(document_id))
        )
    assert remaining == 0


async def test_owner_filter_isolates_users_in_vector_search(
    api: TestClient, worker: IngestionWorker, app_settings: Settings
) -> None:
    alice, bob = signup(api), signup(api)
    alice_id = api.get("/auth/me", headers=alice).json()["id"]
    marker = f"zebra{uuid.uuid4().hex[:6]}"
    upload(api, alice, make_pdf([long_page(f"{marker} migration alice")]))
    upload(api, bob, make_pdf([long_page(f"{marker} migration bob")]))
    await drain(worker)

    query = await HashingEmbedder().embed_query(f"{marker} migration")
    client = create_qdrant_client(app_settings)
    try:
        hits = await client.query_points(
            app_settings.qdrant_collection,
            query=query,
            query_filter=qm.Filter(
                must=[qm.FieldCondition(key="owner_id", match=qm.MatchValue(value=alice_id))]
            ),
            limit=50,
        )
    finally:
        await client.close()
    assert hits.points
    assert {p.payload["owner_id"] for p in hits.points if p.payload} == {alice_id}


async def test_collection_with_other_dimension_is_refused(app_settings: Settings) -> None:
    client = create_qdrant_client(app_settings)
    name = f"test_dim_{uuid.uuid4().hex[:8]}"
    try:
        index = QdrantVectorIndex(client, name)
        await index.ensure_collection(8)
        await index.ensure_collection(8)  # idempotent
        with pytest.raises(VectorIndexError, match="vector size 8"):
            await index.ensure_collection(16)
    finally:
        await client.delete_collection(name)
        await client.close()


async def test_sentence_transformer_embedder_produces_meaningful_vectors(
    live_settings: Settings,
) -> None:
    """Runs the real configured model (downloaded on first run, ~130 MB for bge-small)."""
    embedder: SentenceTransformerEmbedder = create_embedder(live_settings)
    passages = await embedder.embed_passages(
        [
            "Paris is the capital and largest city of France.",
            "Bananas are yellow when they are ripe.",
        ]
    )
    query = await embedder.embed_query("What is the capital of France?")

    assert embedder.dimension == len(query) == len(passages[0])
    for vector in [*passages, query]:
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-4)

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cosine(query, passages[0]) > cosine(query, passages[1]) + 0.1
    assert await embedder.embed_passages([]) == []
