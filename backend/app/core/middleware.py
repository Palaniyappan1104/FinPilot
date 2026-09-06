"""HTTP middleware for request logging and monitoring."""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger("app.middleware.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log incoming requests and response latency safely."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        method = request.method
        path = request.url.path

        try:
            response = await call_next(request)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "%s %s -> %d (%.2fms)",
                method,
                path,
                response.status_code,
                latency_ms,
            )
            return response
        except Exception:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "%s %s -> unhandled exception (%.2fms)",
                method,
                path,
                latency_ms,
            )
            raise
