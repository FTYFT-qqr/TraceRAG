from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_endpoint_returns_runtime_metadata() -> None:
    app = create_app(
        Settings(
            app_name="TraceRAG Test",
            environment="test",
            version="0.1.0-test",
        )
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "TraceRAG Test",
        "environment": "test",
        "version": "0.1.0-test",
    }
