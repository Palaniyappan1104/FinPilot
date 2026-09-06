"""Health check API endpoint."""

from typing import Dict

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    service: str
    version: str
    environment: str


@router.get("", response_model=HealthResponse)
def get_health() -> Dict[str, str]:
    """Return application health status."""
    settings = get_settings()
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME.lower() + "-backend",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }
