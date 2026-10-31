"""Document-level authorization: owners, share recipients and everyone else."""

import uuid

import pytest
from fastapi.testclient import TestClient

from rag_platform.ingestion.worker import IngestionWorker
from tests.fakes import ScriptedLLM
from tests.integration.test_documents_api import PASSWORD, drain, upload
from tests.pdf_factory import make_pdf

pytestmark = pytest.mark.integration


def account(api: TestClient) -> tuple[dict[str, str], str, str]:
    """(auth headers, user id, email) for a fresh user."""
    email = f"share-{uuid.uuid4().hex[:10]}@example.com"
    api.post("/auth/register", json={"email": email, "password": PASSWORD})
    token = api.post("/auth/login", json={"email": email, "password": PASSWORD}).json()[
        "access_token"
    ]
    headers = {"Authorization": f"Bearer {token}"}
    return headers, api.get("/auth/me", headers=headers).json()["id"], email


@pytest.fixture
async def shared_setup(api: TestClient, worker: IngestionWorker) -> dict[str, object]:
    owner, _, _ = account(api)
    reader, reader_id, reader_email = account(api)
    stranger, _, _ = account(api)
    marker = f"okapi{uuid.uuid4().hex[:6]}"
    document_id = upload(
        api, owner, make_pdf([f"The {marker} project budget is 12 million euros."]), "budget.pdf"
    ).json()["id"]
    await drain(worker)
    return {
        "owner": owner,
        "reader": reader,
        "reader_id": reader_id,
        "reader_email": reader_email,
        "stranger": stranger,
        "document_id": document_id,
        "marker": marker,
    }


def share(api: TestClient, setup: dict[str, object]) -> None:
    response = api.post(
        f"/documents/{setup['document_id']}/shares",
        headers=setup["owner"],  # type: ignore[arg-type]
        json={"email": setup["reader_email"]},
    )
    assert response.status_code == 201, response.text


def search_ids(api: TestClient, headers: object, query: str, mode: str) -> set[str]:
    response = api.post("/search", headers=headers, json={"query": query, "mode": mode})  # type: ignore[arg-type]
    assert response.status_code == 200, response.text
    return {r["document_id"] for r in response.json()["results"]}


async def test_unshared_document_is_invisible_to_others(
    api: TestClient, shared_setup: dict[str, object]
) -> None:
    doc = shared_setup["document_id"]
    for who in ("reader", "stranger"):
        headers = shared_setup[who]
        assert api.get(f"/documents/{doc}", headers=headers).status_code == 404  # type: ignore[arg-type]
        assert api.get("/documents", headers=headers).json()["total"] == 0  # type: ignore[arg-type]
        for mode in ("dense", "sparse", "hybrid_rerank"):
            assert doc not in search_ids(api, headers, str(shared_setup["marker"]), mode)


async def test_share_grants_read_and_search_but_not_ownership(
    api: TestClient, shared_setup: dict[str, object]
) -> None:
    share(api, shared_setup)
    doc = shared_setup["document_id"]
    reader = shared_setup["reader"]

    detail = api.get(f"/documents/{doc}", headers=reader)  # type: ignore[arg-type]
    assert detail.status_code == 200
    assert detail.json()["access"] == "shared"
    assert api.get(f"/documents/{doc}/pages", headers=reader).status_code == 200  # type: ignore[arg-type]
    listing = api.get("/documents", headers=reader, params={"ownership": "shared"}).json()  # type: ignore[arg-type]
    assert [d["id"] for d in listing["items"]] == [doc]
    assert api.get("/documents", headers=reader, params={"ownership": "owned"}).json()["total"] == 0  # type: ignore[arg-type]
    for mode in ("dense", "sparse", "hybrid", "hybrid_rerank"):
        assert doc in search_ids(api, reader, str(shared_setup["marker"]), mode)

    # Recipients can read, but cannot delete or reshare.
    assert api.delete(f"/documents/{doc}", headers=reader).status_code == 403  # type: ignore[arg-type]
    reshare = api.post(
        f"/documents/{doc}/shares",
        headers=reader,
        json={"email": "x@example.com"},  # type: ignore[arg-type]
    )
    assert reshare.status_code == 403
    assert api.get(f"/documents/{doc}/shares", headers=reader).status_code == 403  # type: ignore[arg-type]

    # Strangers still see nothing.
    assert api.get(f"/documents/{doc}", headers=shared_setup["stranger"]).status_code == 404  # type: ignore[arg-type]


async def test_answers_can_cite_shared_documents(
    api: TestClient, shared_setup: dict[str, object]
) -> None:
    share(api, shared_setup)
    api.app.state.container.llm = ScriptedLLM("The budget is 12 million euros [1].")  # type: ignore[attr-defined]
    body = api.post(
        "/ask",
        headers=shared_setup["reader"],  # type: ignore[arg-type]
        json={"question": f"What is the {shared_setup['marker']} project budget?"},
    ).json()
    assert body["answered"] is True
    assert body["citations"][0]["document_id"] == shared_setup["document_id"]
    assert body["citations"][0]["filename"] == "budget.pdf"


async def test_revoking_a_share_removes_access_immediately(
    api: TestClient, shared_setup: dict[str, object]
) -> None:
    share(api, shared_setup)
    doc, reader, marker = (
        shared_setup["document_id"],
        shared_setup["reader"],
        str(shared_setup["marker"]),
    )
    assert doc in search_ids(api, reader, marker, "sparse")  # warms the reader's BM25 cache

    shares = api.get(f"/documents/{doc}/shares", headers=shared_setup["owner"]).json()  # type: ignore[arg-type]
    assert [s["email"] for s in shares] == [shared_setup["reader_email"]]
    revoked = api.delete(
        f"/documents/{doc}/shares/{shared_setup['reader_id']}",
        headers=shared_setup["owner"],  # type: ignore[arg-type]
    )
    assert revoked.status_code == 204

    assert api.get(f"/documents/{doc}", headers=reader).status_code == 404  # type: ignore[arg-type]
    for mode in ("dense", "sparse", "hybrid_rerank"):
        assert doc not in search_ids(api, reader, marker, mode)
    again = api.delete(
        f"/documents/{doc}/shares/{shared_setup['reader_id']}",
        headers=shared_setup["owner"],  # type: ignore[arg-type]
    )
    assert again.status_code == 404


async def test_share_validation(api: TestClient, shared_setup: dict[str, object]) -> None:
    doc, owner = shared_setup["document_id"], shared_setup["owner"]
    unknown = api.post(
        f"/documents/{doc}/shares",
        headers=owner,
        json={"email": "nobody-here@example.com"},  # type: ignore[arg-type]
    )
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "user_not_found"
    me = api.get("/auth/me", headers=owner).json()["email"]  # type: ignore[arg-type]
    self_share = api.post(f"/documents/{doc}/shares", headers=owner, json={"email": me})  # type: ignore[arg-type]
    assert self_share.status_code == 400

    share(api, shared_setup)
    share(api, shared_setup)  # sharing twice is idempotent
    assert len(api.get(f"/documents/{doc}/shares", headers=owner).json()) == 1  # type: ignore[arg-type]

    # A stranger cannot manage shares of a document they can't see: 404, not 403.
    stranger_share = api.post(
        f"/documents/{doc}/shares",
        headers=shared_setup["stranger"],
        json={"email": me},  # type: ignore[arg-type]
    )
    assert stranger_share.status_code == 404


async def test_deleting_a_shared_document_removes_it_for_recipients(
    api: TestClient, shared_setup: dict[str, object]
) -> None:
    share(api, shared_setup)
    doc = shared_setup["document_id"]
    assert api.delete(f"/documents/{doc}", headers=shared_setup["owner"]).status_code == 204  # type: ignore[arg-type]
    reader = shared_setup["reader"]
    assert api.get(f"/documents/{doc}", headers=reader).status_code == 404  # type: ignore[arg-type]
    assert doc not in search_ids(api, reader, str(shared_setup["marker"]), "sparse")
