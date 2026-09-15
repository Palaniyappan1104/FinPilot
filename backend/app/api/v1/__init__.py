"""API v1 router configuration."""

from fastapi import APIRouter

from app.api.v1.demo import router as demo_router
from app.api.v1.documents import router as documents_router
from app.api.v1.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
api_router.include_router(demo_router, prefix="/demo", tags=["demo"])

# Full LangGraph pipeline — guarded import so server still boots during development
try:
    from app.api.v1.analysis import router as analysis_router
    api_router.include_router(analysis_router, tags=["analysis"])
except Exception as _e:  # noqa: BLE001
    import logging
    logging.getLogger("app.api.v1").warning(
        "Full analysis router failed to load (expected during demo mode): %s", _e
    )
