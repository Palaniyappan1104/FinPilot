"""Main FastAPI application entrypoint for FinPilot."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1 import api_router
from app.api.v1.health import get_health
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestLoggingMiddleware

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown."""
    setup_logging()
    logger.info("Starting FinPilot backend service...")
    yield
    logger.info("Shutting down FinPilot backend service...")


def create_application() -> FastAPI:
    """Application factory for FinPilot backend."""
    settings = get_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
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
        logger.warning(
            "Validation error on %s %s: %s",
            request.method,
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": exc.errors(),
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
    app.add_api_route("/health", get_health, methods=["GET"], tags=["health"])

    # API v1 routes
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_application()
