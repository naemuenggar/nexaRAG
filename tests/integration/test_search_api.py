"""Dense and BM25 search through the API, on a small fixed corpus."""

import uuid

import pytest
from fastapi.testclient import TestClient

from rag_platform.ingestion.worker import IngestionWorker
from tests.integration.test_documents_api import drain, signup, upload
from tests.pdf_factory import make_pdf

pytestmark = pytest.mark.integration

CORPUS = {
    "energy.pdf": [
        "Solar panels convert sunlight into electricity using photovoltaic cells.",
        "Wind turbines generate power when the rotor blades are turned by moving air.",
    ],
    "finance.pdf": [
        "The company reported quarterly revenue of 4.2 million dollars in Q3 2024.",
        "Operating expenses decreased because the office lease was renegotiated.",
    ],
    "health.pdf": [
        "Regular aerobic exercise lowers resting heart rate and blood pressure.",
        "Vitamin D is produced in the skin after exposure to ultraviolet sunlight.",
    ],
}

# Retrieval regression set: query -> (filename, page) that must rank first.
ALL_MODES = ("sparse", "dense", "hybrid", "hybrid_rerank")

EXPECTED_TOP_HIT = {
    "photovoltaic cells electricity": ("energy.pdf", 1),
    "rotor blades turbines": ("energy.pdf", 2),
    "quarterly revenue Q3 2024": ("finance.pdf", 1),
    "office lease renegotiated expenses": ("finance.pdf", 2),
    "aerobic exercise heart rate": ("health.pdf", 1),
    "vitamin D skin ultraviolet": ("health.pdf", 2),
}


@pytest.fixture
async def corpus_user(api: TestClient, worker: IngestionWorker) -> dict[str, object]:
    headers = signup(api)
    ids = {}
    for filename, pages in CORPUS.items():
        # Fixed content (no random nonce) keeps hashed test embeddings, and so rankings,
        # identical on every run. Dedupe is per user, so each test user can upload the same bytes.
        response = upload(api, headers, make_pdf(pages), name=filename)
        assert response.status_code == 202
        ids[filename] = response.json()["id"]
    await drain(worker)
    return {"headers": headers, "ids": ids}


def search(api: TestClient, headers: object, **body: object) -> dict:  # type: ignore[type-arg]
    response = api.post("/search", headers=headers, json=body)  # type: ignore[arg-type]
    assert response.status_code == 200, response.text
    return response.json()  # type: ignore[no-any-return]


@pytest.mark.parametrize("mode", ["sparse", "dense", "hybrid", "hybrid_rerank"])
async def test_retrieval_regression(
    api: TestClient, corpus_user: dict[str, object], mode: str
) -> None:
    misses = {}
    for query, expected in EXPECTED_TOP_HIT.items():
        results = search(api, corpus_user["headers"], query=query, mode=mode, top_k=3)["results"]
        top = (results[0]["filename"], results[0]["page_start"]) if results else None
        if top != expected:
            misses[query] = top
    assert misses == {}


async def test_result_shape_carries_citation_fields(
    api: TestClient, corpus_user: dict[str, object]
) -> None:
    body = search(api, corpus_user["headers"], query="quarterly revenue", mode="sparse", top_k=2)
    assert body["mode"] == "sparse"
    assert body["latency_ms"] >= 0
    first = body["results"][0]
    assert first["rank"] == 1
    assert first["document_id"] == corpus_user["ids"]["finance.pdf"]  # type: ignore[index]
    assert first["filename"] == "finance.pdf"
    assert first["page_start"] == first["page_end"] == 1
    assert "4.2 million" in first["text"]
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


async def test_document_filter_restricts_results(
    api: TestClient, corpus_user: dict[str, object]
) -> None:
    health_id = corpus_user["ids"]["health.pdf"]  # type: ignore[index]
    for mode in ALL_MODES:
        results = search(
            api,
            corpus_user["headers"],
            query="sunlight",
            mode=mode,
            top_k=10,
            filters={"document_ids": [health_id]},
        )["results"]
        assert results
        assert {r["document_id"] for r in results} == {health_id}


async def test_users_never_retrieve_each_others_chunks(
    api: TestClient, worker: IngestionWorker, corpus_user: dict[str, object]
) -> None:
    bob = signup(api)
    upload(api, bob, make_pdf([f"Bob writes about gardening tomatoes {uuid.uuid4()}"]))
    await drain(worker)
    alice_ids = set(corpus_user["ids"].values())  # type: ignore[attr-defined]

    for mode in ALL_MODES:
        results = search(api, bob, query="quarterly revenue solar exercise", mode=mode, top_k=50)[
            "results"
        ]
        assert not {r["document_id"] for r in results} & alice_ids
        alice_results = search(
            api, corpus_user["headers"], query="gardening tomatoes", mode=mode, top_k=50
        )["results"]
        assert all(r["filename"] != "report.pdf" for r in alice_results)

    # Naming another user's document in the filter is a 404, never a silent empty result.
    response = api.post(
        "/search",
        headers=bob,
        json={
            "query": "revenue",
            "filters": {"document_ids": [corpus_user["ids"]["finance.pdf"]]},
        },  # type: ignore[index]
    )
    assert response.status_code == 404


async def test_documents_not_ready_or_deleted_are_not_searchable(
    api: TestClient, worker: IngestionWorker, corpus_user: dict[str, object]
) -> None:
    headers = corpus_user["headers"]
    marker = f"kangaroo{uuid.uuid4().hex[:6]}"
    pending = upload(api, headers, make_pdf([f"{marker} migration patterns"]))  # type: ignore[arg-type]
    pending_id = pending.json()["id"]
    assert search(api, headers, query=marker, mode="sparse")["results"] == []
    pending_filter = api.post(
        "/search",
        headers=headers,
        json={"query": marker, "filters": {"document_ids": [pending_id]}},  # type: ignore[arg-type]
    )
    assert pending_filter.status_code == 404

    await drain(worker)
    assert search(api, headers, query=marker, mode="sparse")["results"]

    api.delete(f"/documents/{pending_id}", headers=headers)  # type: ignore[arg-type]
    for mode in ALL_MODES:
        results = search(api, headers, query=marker, mode=mode, top_k=50)["results"]
        assert pending_id not in {r["document_id"] for r in results}


async def test_bm25_index_is_rebuilt_only_when_corpus_changes(
    api: TestClient, worker: IngestionWorker, corpus_user: dict[str, object]
) -> None:
    headers = corpus_user["headers"]
    cache = api.app.state.container.bm25_cache  # type: ignore[attr-defined]
    search(api, headers, query="revenue", mode="sparse")
    builds = cache.builds
    search(api, headers, query="exercise", mode="sparse")
    assert cache.builds == builds

    upload(api, headers, make_pdf([f"brand new topic {uuid.uuid4()}"]))  # type: ignore[arg-type]
    await drain(worker)
    search(api, headers, query="topic", mode="sparse")
    assert cache.builds == builds + 1


def test_search_validation(api: TestClient) -> None:
    headers = signup(api)
    assert api.post("/search", json={"query": "x"}).status_code == 401
    assert api.post("/search", headers=headers, json={"query": ""}).status_code == 422
    assert (
        api.post("/search", headers=headers, json={"query": "x", "mode": "magic"}).status_code
        == 422
    )
    too_many = api.post("/search", headers=headers, json={"query": "x", "top_k": 1000})
    assert too_many.status_code == 400
    assert too_many.json()["error"]["code"] == "invalid_top_k"
    stopwords_only = search(api, headers, query="the of and", mode="sparse")
    assert stopwords_only["results"] == []


async def test_hybrid_results_report_each_stage_score(
    api: TestClient, corpus_user: dict[str, object]
) -> None:
    headers = corpus_user["headers"]
    hybrid = search(api, headers, query="quarterly revenue", mode="hybrid", top_k=3)["results"]
    assert {"dense", "bm25", "fused"} <= set(hybrid[0]["scores"])
    assert hybrid[0]["score"] == hybrid[0]["scores"]["fused"]

    reranked = search(api, headers, query="quarterly revenue", mode="hybrid_rerank", top_k=3)
    first = reranked["results"][0]
    assert first["score"] == first["scores"]["rerank"]
    assert "fused" in first["scores"]
    scores = [r["score"] for r in reranked["results"]]
    assert scores == sorted(scores, reverse=True)


async def test_default_mode_is_hybrid_rerank(
    api: TestClient, corpus_user: dict[str, object]
) -> None:
    body = search(api, corpus_user["headers"], query="vitamin D")
    assert body["mode"] == "hybrid_rerank"
    reranker = api.app.state.container.reranker  # type: ignore[attr-defined]
    query, candidates = reranker.calls[-1]
    assert query == "vitamin D"
    assert candidates <= api.app.state.container.settings.rerank_candidates  # type: ignore[attr-defined]


async def test_metadata_filters(api: TestClient, corpus_user: dict[str, object]) -> None:
    headers = corpus_user["headers"]
    by_name = search(
        api,
        headers,
        query="sunlight",
        mode="hybrid",
        top_k=10,
        filters={"filename_contains": "HEALTH"},
    )["results"]
    assert by_name
    assert {r["filename"] for r in by_name} == {"health.pdf"}

    # LIKE wildcards in the filter are matched literally, not as patterns.
    assert (
        search(api, headers, query="sunlight", mode="hybrid", filters={"filename_contains": "%"})[
            "results"
        ]
        == []
    )

    future = search(
        api,
        headers,
        query="sunlight",
        mode="hybrid",
        filters={"uploaded_after": "2999-01-01T00:00:00Z"},
    )
    assert future["results"] == []
    past = search(
        api,
        headers,
        query="sunlight",
        mode="hybrid",
        filters={
            "uploaded_after": "2000-01-01T00:00:00Z",
            "uploaded_before": "2999-01-01T00:00:00Z",
        },
    )
    assert past["results"]

    combined = search(
        api,
        headers,
        query="sunlight",
        mode="sparse",
        top_k=10,
        filters={
            "document_ids": [corpus_user["ids"]["energy.pdf"], corpus_user["ids"]["health.pdf"]],  # type: ignore[index]
            "filename_contains": "energy",
        },
    )["results"]
    assert {r["filename"] for r in combined} == {"energy.pdf"}

    naive_date = api.post(
        "/search",
        headers=headers,  # type: ignore[arg-type]
        json={"query": "x", "filters": {"uploaded_after": "2024-01-01T00:00:00"}},
    )
    assert naive_date.status_code == 422  # timezone required: no ambiguous dates
    unknown = api.post("/search", headers=headers, json={"query": "x", "colour": "red"})  # type: ignore[arg-type]
    assert unknown.status_code == 422
