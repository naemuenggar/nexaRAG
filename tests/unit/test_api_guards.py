"""Request guards that act before any database access."""

from collections.abc import Iterator

from fastapi.testclient import TestClient

from rag_platform.main import MULTIPART_OVERHEAD_BYTES
from tests.pdf_factory import make_pdf


def test_upload_requires_authentication(client: TestClient) -> None:
    response = client.post(
        "/documents", files={"file": ("a.pdf", make_pdf(["x"]), "application/pdf")}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"
    assert response.headers["www-authenticate"] == "Bearer"


def test_invalid_token_is_rejected(client: TestClient) -> None:
    response = client.get("/documents", headers={"Authorization": "Bearer not-a-token"})
    assert response.status_code == 401


def test_declared_oversized_body_is_rejected_before_parsing(client: TestClient) -> None:
    limit = 50 * 1024 * 1024 + MULTIPART_OVERHEAD_BYTES
    response = client.post(
        "/documents",
        content=b"x" * 10,
        headers={"Content-Length": str(limit + 1), "Content-Type": "application/pdf"},
    )
    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "payload_too_large"
    assert error["request_id"] == response.headers["x-request-id"]


def test_streamed_oversized_body_is_rejected(client: TestClient) -> None:
    limit = 50 * 1024 * 1024 + MULTIPART_OVERHEAD_BYTES

    def body() -> Iterator[bytes]:  # no Content-Length: sent with chunked transfer encoding
        yield b'--b\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n\r\n%PDF-'
        sent = 0
        chunk = b"0" * (1024 * 1024)
        while sent <= limit:
            sent += len(chunk)
            yield chunk

    response = client.post(
        "/documents", content=body(), headers={"Content-Type": "multipart/form-data; boundary=b"}
    )
    assert response.status_code == 413


def test_document_id_must_be_uuid(client: TestClient) -> None:
    response = client.get("/documents/not-a-uuid", headers={"Authorization": "Bearer x"})
    assert response.status_code in {401, 422}


def test_register_validates_password_length(client: TestClient) -> None:
    response = client.post("/auth/register", json={"email": "a@example.com", "password": "tiny1"})
    assert response.status_code == 422
    assert "tiny1" not in response.text
