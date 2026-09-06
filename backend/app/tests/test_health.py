"""API tests for the /health endpoint and error handling baseline."""

from fastapi import status


def test_get_health_root(client):
    """Test GET /health returns expected structured response."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data
    assert "environment" in data
    assert data["service"] == "finpilot-backend"


def test_get_health_api_v1(client):
    """Test GET /api/v1/health returns expected structured response."""
    response = client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"


def test_http_exception_format(client):
    """Test HTTP errors return structured JSON error format."""
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == status.HTTP_404_NOT_FOUND

    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "HTTP_ERROR"
    assert "message" in data["error"]


def test_validation_error_sanitized(client):
    """Test validation errors return sanitized response without input values."""
    from fastapi import APIRouter
    from pydantic import BaseModel, Field

    from app.main import app

    test_router = APIRouter()

    class SecretPayload(BaseModel):
        secret_token: str = Field(..., min_length=10)

    @test_router.post("/_test_validation")
    def dummy_endpoint(payload: SecretPayload):
        return {"status": "ok"}

    app.include_router(test_router)

    # Post invalid payload with sensitive value that fails validation
    response = client.post(
        "/_test_validation",
        json={"secret_token": "short"},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert data["error"]["message"] == "Request validation failed"
    assert isinstance(data["error"]["details"], list)
    assert len(data["error"]["details"]) > 0

    first_err = data["error"]["details"][0]
    assert "loc" in first_err
    assert "msg" in first_err
    assert "type" in first_err
    assert "input" not in first_err
