from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from rag_platform.core.errors import NotFoundError, PermissionDeniedError


class Payload(BaseModel):
    name: str = Field(max_length=5)


@pytest.fixture
def error_client(app: FastAPI) -> Iterator[TestClient]:
    async def not_found() -> None:
        raise NotFoundError("Document not found")

    async def forbidden() -> None:
        raise PermissionDeniedError("You cannot access this document")

    async def boom() -> None:
        raise RuntimeError("db password=hunter2 at 10.0.0.5")

    async def validate(payload: Payload) -> Payload:
        return payload

    app.add_api_route("/_test/not-found", not_found)
    app.add_api_route("/_test/forbidden", forbidden)
    app.add_api_route("/_test/boom", boom)
    app.add_api_route("/_test/validate", validate, methods=["POST"])
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_app_error_maps_to_status_and_code(error_client: TestClient) -> None:
    response = error_client.get("/_test/not-found", headers={"X-Request-ID": "req-123"})
    assert response.status_code == 404
    assert response.json() == {
        "error": {"code": "not_found", "message": "Document not found", "request_id": "req-123"}
    }


def test_permission_denied_is_403(error_client: TestClient) -> None:
    response = error_client.get("/_test/forbidden")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "permission_denied"


def test_unhandled_error_is_generic_500(error_client: TestClient) -> None:
    response = error_client.get("/_test/boom")
    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "internal_error"
    assert error["message"] == "An internal error occurred"
    assert error["request_id"] == response.headers["x-request-id"]
    assert "hunter2" not in response.text


def test_validation_error_does_not_echo_input(error_client: TestClient) -> None:
    response = error_client.post("/_test/validate", json={"name": "<script>too-long</script>"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["loc"] == ["body", "name"]
    assert "<script>" not in response.text


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_error"
