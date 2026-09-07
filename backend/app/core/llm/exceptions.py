"""LLM provider custom exceptions for FinPilot."""


class LLMError(Exception):
    """Base exception for all LLM provider errors."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider

    def __str__(self) -> str:
        return f"[{self.provider}] {self.message}"


class LLMConfigurationError(LLMError):
    """Raised when LLM configuration or credentials are missing or invalid."""


class LLMAuthenticationError(LLMError):
    """Raised when LLM authentication fails (invalid or unauthorized API key)."""


class LLMTransientError(LLMError):
    """Raised for transient provider failures that can be safely retried."""


class LLMRetryExhaustedError(LLMError):
    """Raised when all retry attempts for a transient LLM error have been exhausted."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        attempts: int = 0,
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.attempts = attempts
        self.last_error = last_error


class LLMResponseError(LLMError):
    """Raised when LLM returns an invalid or empty response."""


class LLMStructuredOutputError(LLMError):
    """Raised when LLM output cannot be parsed or validated against a schema."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        schema_name: str | None = None,
        attempts: int = 2,
        validation_failure: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.schema_name = schema_name
        self.attempts = attempts
        self.validation_failure = validation_failure
