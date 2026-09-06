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
