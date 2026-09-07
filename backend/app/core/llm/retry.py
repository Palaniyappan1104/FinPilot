"""Retry mechanism with exponential backoff for LLM provider API calls."""

import time
from typing import Callable, Set, Type, TypeVar

from app.core.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMRetryExhaustedError,
    LLMTransientError,
)
from app.core.logging import get_logger

logger = get_logger("app.core.llm.retry")

T = TypeVar("T")

# Non-retryable exception types
NON_RETRYABLE_EXCEPTIONS: Set[Type[Exception]] = {
    LLMConfigurationError,
    LLMAuthenticationError,
}


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    provider_name: str = "unknown",
) -> T:
    """Execute a callable with bounded exponential backoff on transient failures.

    Args:
        func: Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before the first retry.
        backoff_factor: Multiplier applied to delay after each failed attempt.
        provider_name: Provider name for structured error attribution.

    Returns:
        The return value of func().

    Raises:
        LLMRetryExhaustedError: When max_retries attempts fail due to transient errors.
        LLMAuthenticationError: Immediately when authentication failure occurs.
        LLMConfigurationError: Immediately when configuration failure occurs.
        Exception: Any non-transient, unhandled exception without retrying.
    """
    delay = initial_delay
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except tuple(NON_RETRYABLE_EXCEPTIONS) as non_retryable:
            logger.warning(
                "Non-retryable LLM error on attempt %d/%d from provider '%s': %s",
                attempt,
                max_retries,
                provider_name,
                type(non_retryable).__name__,
            )
            raise
        except LLMTransientError as transient_err:
            last_error = transient_err
            if attempt == max_retries:
                logger.error(
                    "All %d retry attempts exhausted for provider '%s'. Last error: %s",
                    max_retries,
                    provider_name,
                    type(transient_err).__name__,
                )
                break
            logger.warning(
                "Transient error on attempt %d/%d from provider '%s'. "
                "Retrying in %.2fs...",
                attempt,
                max_retries,
                provider_name,
                delay,
            )
            time.sleep(delay)
            delay *= backoff_factor
        except Exception as unexpected_err:
            logger.error(
                "Unexpected non-transient error from provider '%s': %s",
                provider_name,
                type(unexpected_err).__name__,
            )
            raise

    raise LLMRetryExhaustedError(
        message=f"LLM call failed after {max_retries} attempts: {last_error}",
        provider=provider_name,
        attempts=max_retries,
        last_error=last_error,
    )
