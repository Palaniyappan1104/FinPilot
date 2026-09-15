"""Main FastAPI application entrypoint for FinPilot."""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1 import api_router
from app.api.v1.health import HealthResponse, get_health
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestLoggingMiddleware

# Load backend/.env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# Initialize application logging in a reliable order before components depend on it
setup_logging()

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown."""
    logger.info("Starting FinPilot backend service...")
    yield
    logger.info("Shutting down FinPilot backend service...")


API_DESCRIPTION = """
FinPilot is an autonomous multi-agent financial research and equity analysis engine.

### Core Capabilities:
- **Autonomous Multi-Agent Analysis**: Orchestrates LangGraph specialized agents.
- **Conversational Workflows**: Proactively identifies underspecified financial goals.
- **Synchronous & Asynchronous Execution**: Real-time or background execution.
- **Status & Progress Tracking**: Real-time lifecycle polling for active analyses.
- **Document-Grounded Research**: Upload and query corporate filings and SEC docs.
- **Multi-Format Report Delivery**: Structured JSON, Markdown, and Summaries.
- **Safe Error Envelopes**: Structured error responses without leaking internals.
"""

TAGS_METADATA = [
    {
        "name": "analysis",
        "description": (
            "Multi-agent equity analysis, conversational queries, "
            "clarification workflows, and in-progress status tracking."
        ),
    },
    {
        "name": "reports",
        "description": (
            "Retrieval and multi-format rendering (JSON, Markdown, Executive Summary) "
            "of completed investment research reports."
        ),
    },
    {
        "name": "documents",
        "description": (
            "Document upload and research vault storage for corporate filings."
        ),
    },
    {
        "name": "health",
        "description": "Service health, versioning, and environment diagnostics.",
    },
]


def create_application() -> FastAPI:
    """Application factory for FinPilot backend."""
    settings = get_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=API_DESCRIPTION.strip(),
        openapi_tags=TAGS_METADATA,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # CORS configuration
    origins = (
        settings.CORS_ORIGINS
        if isinstance(settings.CORS_ORIGINS, list)
        else [settings.CORS_ORIGINS]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Global Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Strip submitted input values from error details to avoid exposing
        # sensitive inputs, and stringify any exception objects in ctx
        sanitized = []
        for err in exc.errors():
            clean_err = {}
            for k, v in err.items():
                if k == "input":
                    continue
                if k == "ctx" and isinstance(v, dict):
                    clean_err[k] = {
                        sub_k: str(sub_v) if isinstance(sub_v, Exception) else sub_v
                        for sub_k, sub_v in v.items()
                    }
                else:
                    clean_err[k] = v
            sanitized.append(clean_err)
        logger.warning(
            "Validation error on %s %s (%d field(s))",
            request.method,
            request.url.path,
            len(sanitized),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": sanitized,
                }
            },
        )

    @app.exception_handler(HTTPException)
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        logger.warning(
            "HTTP error on %s %s: %d - %s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": exc.detail,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(
            "Unhandled error processing %s %s", request.method, request.url.path
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred. Please try again later.",
                }
            },
        )

    # Top-level GET /health
    app.add_api_route(
        "/health",
        get_health,
        methods=["GET"],
        response_model=HealthResponse,
        tags=["health"],
        summary="Top-level service health check",
        description=(
            "Top-level health check probe for load balancers and orchestrators."
        ),
        operation_id="getHealthRoot",
    )

    # API v1 routes
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_application()
