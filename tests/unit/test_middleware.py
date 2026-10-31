from fastapi.testclient import TestClient

from rag_platform.main import create_app
from tests.conftest import make_settings


def test_generates_request_id(client: TestClient) -> None:
    response = client.get("/health/live")
    request_id = response.headers["x-request-id"]
    assert len(request_id) == 32
    assert request_id.isalnum()


def test_propagates_valid_request_id(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Request-ID": "abc-123.DEF_4"})
    assert response.headers["x-request-id"] == "abc-123.DEF_4"


def test_replaces_unsafe_request_id(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Request-ID": "bad id\twith spaces"})
    assert response.headers["x-request-id"] != "bad id\twith spaces"
    assert len(response.headers["x-request-id"]) == 32


def test_security_headers(client: TestClient) -> None:
    headers = client.get("/health/live").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "no-referrer"


def test_docs_disabled_in_production() -> None:
    app = create_app(make_settings(environment="production"))
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/docs" not in paths
    assert "/openapi.json" not in paths


def test_docs_enabled_outside_production(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 200
