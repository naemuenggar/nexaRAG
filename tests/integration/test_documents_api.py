"""End-to-end: upload, background ingestion, then reading parsed pages, on real Postgres."""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncEngine

from rag_platform.core.config import Settings
from rag_platform.db.models import IngestionJob, JobStatus
from rag_platform.ingestion.worker import IngestionWorker
from tests.pdf_factory import make_encrypted_pdf, make_malformed_pdf, make_pdf

pytestmark = pytest.mark.integration

PASSWORD = "a-long-enough-password"


def signup(api: TestClient, email: str | None = None) -> dict[str, str]:
    email = email or f"user-{uuid.uuid4().hex[:10]}@example.com"
    assert (
        api.post("/auth/register", json={"email": email, "password": PASSWORD}).status_code == 201
    )
    response = api.post("/auth/login", json={"email": email, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def upload(
    api: TestClient, headers: dict[str, str], data: bytes, name: str = "report.pdf"
) -> httpx.Response:
    return api.post("/documents", headers=headers, files={"file": (name, data, "application/pdf")})


async def drain(worker: IngestionWorker) -> int:
    processed = 0
    while await worker.run_once():
        processed += 1
    return processed


# --- auth ----------------------------------------------------------------------------------


def test_register_login_me(api: TestClient) -> None:
    email = f"Mixed.Case-{uuid.uuid4().hex[:6]}@Example.com"
    headers = signup(api, email)
    me = api.get("/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["email"] == email.lower()
    assert "hashed_password" not in me.text


def test_duplicate_registration_is_rejected(api: TestClient) -> None:
    email = f"dup-{uuid.uuid4().hex[:6]}@example.com"
    signup(api, email)
    response = api.post("/auth/register", json={"email": email.upper(), "password": PASSWORD})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "email_taken"


def test_login_failures_are_indistinguishable(api: TestClient) -> None:
    email = f"login-{uuid.uuid4().hex[:6]}@example.com"
    signup(api, email)
    wrong_password = api.post("/auth/login", json={"email": email, "password": "nope-nope-nope"})
    unknown_user = api.post(
        "/auth/login", json={"email": "nobody@example.com", "password": "nope-nope-nope"}
    )
    assert wrong_password.status_code == unknown_user.status_code == 401
    assert wrong_password.json()["error"]["message"] == unknown_user.json()["error"]["message"]


# --- happy path ----------------------------------------------------------------------------


async def test_upload_is_parsed_into_pages(api: TestClient, worker: IngestionWorker) -> None:
    headers = signup(api)
    response = upload(
        api, headers, make_pdf(["Intro page", "Revenue was 4.2M in Q3.", "Conclusion"])
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["latest_job"]["status"] == "queued"
    assert body["filename"] == "report.pdf"
    assert "storage_path" not in body
    document_id = body["id"]

    assert await drain(worker) >= 1

    document = api.get(f"/documents/{document_id}", headers=headers).json()
    assert document["status"] == "ready"
    assert document["page_count"] == 3
    assert document["latest_job"]["status"] == "succeeded"
    assert document["latest_job"]["attempts"] == 1

    pages = api.get(f"/documents/{document_id}/pages", headers=headers).json()
    assert [(p["page_number"], p["text"]) for p in pages] == [
        (1, "Intro page"),
        (2, "Revenue was 4.2M in Q3."),
        (3, "Conclusion"),
    ]
    second_page = api.get(
        f"/documents/{document_id}/pages", headers=headers, params={"offset": 1, "limit": 1}
    ).json()
    assert [p["page_number"] for p in second_page] == [2]

    listing = api.get("/documents", headers=headers).json()
    assert listing["total"] == 1
    assert listing["items"][0]["id"] == document_id


async def test_nothing_to_do_returns_false(worker: IngestionWorker) -> None:
    await drain(worker)
    assert await worker.run_once() is False


# --- duplicates ----------------------------------------------------------------------------


def test_duplicate_upload_is_rejected_per_user(api: TestClient) -> None:
    alice, bob = signup(api), signup(api)
    data = make_pdf([f"unique content {uuid.uuid4()}"])

    first = upload(api, alice, data)
    assert first.status_code == 202
    again = upload(api, alice, data, name="renamed.pdf")
    assert again.status_code == 409
    error = again.json()["error"]
    assert error["code"] == "duplicate_document"
    assert error["details"]["document_id"] == first.json()["id"]

    # Another user uploading the same bytes gets their own copy and learns nothing about Alice's.
    assert upload(api, bob, data).status_code == 202


# --- authorization -------------------------------------------------------------------------


def test_users_cannot_see_or_touch_each_others_documents(api: TestClient) -> None:
    alice, bob = signup(api), signup(api)
    document_id = upload(api, alice, make_pdf([f"alice private {uuid.uuid4()}"])).json()["id"]

    for method, path in [
        ("GET", f"/documents/{document_id}"),
        ("GET", f"/documents/{document_id}/pages"),
        ("DELETE", f"/documents/{document_id}"),
    ]:
        response = api.request(method, path, headers=bob)
        assert response.status_code == 404, (method, path)
        assert response.json()["error"]["code"] == "not_found"

    # Same response as for an id that doesn't exist at all: existence is not revealed.
    missing = api.get(f"/documents/{uuid.uuid4()}", headers=bob)
    assert missing.status_code == 404
    assert missing.json()["error"]["message"] == "Document not found"

    assert api.get("/documents", headers=bob).json()["total"] == 0
    assert api.get(f"/documents/{document_id}", headers=alice).status_code == 200


def test_requests_without_token_are_rejected(api: TestClient) -> None:
    assert api.get("/documents").status_code == 401
    assert upload(api, {}, make_pdf(["x"])).status_code == 401


# --- invalid uploads -----------------------------------------------------------------------


def _incoming(settings: Settings) -> list[Path]:
    incoming = settings.storage_dir / ".incoming"
    return list(incoming.iterdir()) if incoming.exists() else []


def test_non_pdf_is_rejected(api: TestClient, app_settings: Settings) -> None:
    headers = signup(api)
    response = upload(api, headers, b"MZ\x90\x00 an executable", name="evil.pdf")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"
    assert _incoming(app_settings) == []


def test_empty_file_is_rejected(api: TestClient) -> None:
    response = upload(api, signup(api), b"")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"


def test_oversized_file_is_rejected(api: TestClient, app_settings: Settings) -> None:
    headers = signup(api)
    too_big = b"%PDF-1.4\n" + b"0" * (app_settings.max_upload_size_bytes + 1)
    response = upload(api, headers, too_big)
    assert response.status_code == 413
    assert _incoming(app_settings) == []
    assert api.get("/documents", headers=headers).json()["total"] == 0


@pytest.mark.parametrize(
    ("data", "message_part"),
    [
        (make_pdf(["", ""]), "No extractable text"),
        (make_malformed_pdf(), "damaged"),
        (make_encrypted_pdf(["secret"], user_password="pw"), "password-protected"),
    ],
    ids=["no-text", "malformed", "encrypted"],
)
async def test_unprocessable_pdfs_fail_without_retry(
    api: TestClient, worker: IngestionWorker, data: bytes, message_part: str
) -> None:
    headers = signup(api)
    response = upload(api, headers, data)
    assert response.status_code == 202
    await drain(worker)

    document = api.get(f"/documents/{response.json()['id']}", headers=headers).json()
    assert document["status"] == "failed"
    assert message_part in document["error_message"]
    assert document["latest_job"]["status"] == "failed"
    assert document["latest_job"]["attempts"] == 1  # permanent errors are not retried
    assert api.get(f"/documents/{document['id']}/pages", headers=headers).json() == []


# --- worker robustness ---------------------------------------------------------------------


async def test_transient_errors_are_retried_then_fail(
    api: TestClient, worker: IngestionWorker, app_settings: Settings
) -> None:
    headers = signup(api)
    owner_id = api.get("/auth/me", headers=headers).json()["id"]
    document_id = upload(api, headers, make_pdf([f"x {uuid.uuid4()}"])).json()["id"]
    # Simulate an infrastructure fault: the stored file disappears.
    (app_settings.storage_dir / owner_id / f"{document_id}.pdf").unlink()

    await drain(worker)

    document = api.get(f"/documents/{document_id}", headers=headers).json()
    assert document["status"] == "failed"
    assert document["error_message"] == "Processing failed due to an internal error."
    assert document["latest_job"]["attempts"] == app_settings.ingestion_max_attempts


async def test_job_with_expired_lease_is_reclaimed(
    api: TestClient, worker: IngestionWorker, worker_engine: AsyncEngine, app_settings: Settings
) -> None:
    headers = signup(api)
    document_id = upload(api, headers, make_pdf([f"lease {uuid.uuid4()}"])).json()["id"]
    # Pretend a worker claimed the job long ago and then crashed.
    async with worker_engine.begin() as conn:
        await conn.execute(
            update(IngestionJob)
            .where(IngestionJob.document_id == uuid.UUID(document_id))
            .values(
                status=JobStatus.RUNNING,
                attempts=1,
                started_at=datetime.now(UTC)
                - timedelta(seconds=app_settings.ingestion_lease_seconds + 5),
            )
        )

    await drain(worker)

    document = api.get(f"/documents/{document_id}", headers=headers).json()
    assert document["status"] == "ready"
    assert document["latest_job"]["attempts"] == 2


async def test_delete_removes_document_pages_and_file(
    api: TestClient, worker: IngestionWorker, app_settings: Settings
) -> None:
    headers = signup(api)
    owner_id = api.get("/auth/me", headers=headers).json()["id"]
    document_id = upload(api, headers, make_pdf([f"to delete {uuid.uuid4()}"])).json()["id"]
    await drain(worker)
    stored = app_settings.storage_dir / owner_id / f"{document_id}.pdf"
    assert stored.exists()

    assert api.delete(f"/documents/{document_id}", headers=headers).status_code == 204
    assert not stored.exists()
    assert api.get(f"/documents/{document_id}", headers=headers).status_code == 404
    assert api.get(f"/documents/{document_id}/pages", headers=headers).status_code == 404
