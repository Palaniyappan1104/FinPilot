"""Pytest fixtures for backend tests."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


@pytest.fixture
def client():
    """Test client fixture for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def clean_settings():
    """Fixture to ensure fresh settings without cache."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
