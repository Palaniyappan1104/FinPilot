"""Embedding Provider interface and factory for FinPilot (Phase 9.6).

Establishes the provider-agnostic interface for generating dense text
embeddings for document chunks and the factory for instantiating
the configured provider.

The embedding layer is fully independent of ChromaDB and any vector store.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.embeddings import (
    ChunkEmbeddingRequest,
    ChunkEmbeddingResult,
    EmbeddingModelInfo,
)

logger = get_logger("app.providers.embedding")


# ===========================================================================
# TYPED EXCEPTIONS (Phase 9.6)
# ===========================================================================


class EmbeddingError(Exception):
    """Base exception for all embedding provider errors."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        code: str = "EMBEDDING_ERROR",
    ) -> None:
        self.message = message
        self.provider = provider
        self.code = code
        super().__init__(f"[{provider}][{code}] {message}")


class EmbeddingProviderNotConfiguredError(EmbeddingError):
    """Raised when the embedding provider is not configured or missing settings."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(
            message=message,
            provider=provider,
            code="PROVIDER_NOT_CONFIGURED",
        )


class EmbeddingProviderUnavailableError(EmbeddingError):
    """Raised when the embedding provider is unreachable or times out."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(
            message=message,
            provider=provider,
            code="PROVIDER_UNAVAILABLE",
        )


class EmbeddingProviderRateLimitError(EmbeddingError):
    """Raised when the embedding provider explicitly rate-limits requests."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(
            message=message,
            provider=provider,
            code="RATE_LIMITED",
        )


class EmbeddingAuthenticationError(EmbeddingError):
    """Raised when the embedding provider authentication fails."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(
            message=message,
            provider=provider,
            code="AUTHENTICATION_FAILED",
        )


class EmbeddingInvalidResponseError(EmbeddingError):
    """Raised when the provider returns a malformed or unusable embedding response."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(
            message=message,
            provider=provider,
            code="INVALID_RESPONSE",
        )


class EmbeddingDimensionMismatchError(EmbeddingInvalidResponseError):
    """Raised when returned vector dimension does not match expected dimension."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        expected_dimensions: Optional[int] = None,
        actual_dimensions: Optional[int] = None,
    ) -> None:
        super().__init__(message=message, provider=provider)
        self.code = "DIMENSION_MISMATCH"
        self.expected_dimensions = expected_dimensions
        self.actual_dimensions = actual_dimensions


class EmbeddingEmptyInputError(EmbeddingError):
    """Raised when empty or invalid text is submitted for embedding."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(
            message=message,
            provider=provider,
            code="EMPTY_INPUT",
        )


# ===========================================================================
# ABSTRACT EMBEDDING PROVIDER INTERFACE
# ===========================================================================


class EmbeddingProvider(ABC):
    """Abstract interface defining the contract for embedding providers.

    Implementations must:
    - Use an API/model that returns a real dense numeric embedding vector.
    - Never fabricate or return zero/random vectors on failure.
    - Raise typed EmbeddingError subclasses on all failure modes.
    - Keep API keys and secrets out of logs and return values.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g. 'gemini')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the embedding model identifier (e.g. 'gemini-embedding-001')."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors produced."""

    @abstractmethod
    def get_model_info(self) -> EmbeddingModelInfo:
        """Return metadata about this provider and model configuration."""

    @abstractmethod
    def embed_single(self, request: ChunkEmbeddingRequest) -> ChunkEmbeddingResult:
        """Generate an embedding for a single ChunkEmbeddingRequest.

        Args:
            request: Validated ChunkEmbeddingRequest containing text and provenance.

        Returns:
            ChunkEmbeddingResult: Provenance-coupled embedding result.

        Raises:
            EmbeddingEmptyInputError: If request text is blank.
            EmbeddingAuthenticationError: If API key is invalid.
            EmbeddingProviderUnavailableError: If provider is unreachable.
            EmbeddingProviderRateLimitError: If rate-limited.
            EmbeddingInvalidResponseError: If response is malformed.
            EmbeddingError: For any other provider-level failure.
        """

    @abstractmethod
    def embed_batch(
        self, requests: List[ChunkEmbeddingRequest]
    ) -> List[ChunkEmbeddingResult]:
        """Generate embeddings for a list of ChunkEmbeddingRequests.

        Args:
            requests: List of validated ChunkEmbeddingRequest objects.

        Returns:
            List[ChunkEmbeddingResult]: Ordered list of provenance-coupled results.

        Raises:
            EmbeddingEmptyInputError: If any request text is blank.
            EmbeddingAuthenticationError: If API key is invalid.
            EmbeddingProviderUnavailableError: If provider is unreachable.
            EmbeddingProviderRateLimitError: If rate-limited.
            EmbeddingInvalidResponseError: If any response is malformed.
            EmbeddingError: For any other provider-level failure.
        """


# ===========================================================================
# PROVIDER FACTORY
# ===========================================================================


def get_embedding_provider(
    settings: Optional[Settings] = None,
) -> EmbeddingProvider:
    """Instantiate and return the configured EmbeddingProvider.

    Reads provider from settings.EMBEDDING_PROVIDER (default: 'gemini').

    Args:
        settings: Optional application settings instance.

    Returns:
        EmbeddingProvider: Concrete provider implementation.

    Raises:
        EmbeddingProviderNotConfiguredError: If provider name is missing.
        EmbeddingError: If provider name is unsupported.
    """
    app_settings = settings or get_settings()
    provider_name = (
        (getattr(app_settings, "EMBEDDING_PROVIDER", "gemini") or "").strip().lower()
    )

    if not provider_name:
        raise EmbeddingProviderNotConfiguredError(
            "EMBEDDING_PROVIDER configuration cannot be empty.",
            provider="unknown",
        )

    if provider_name == "gemini":
        logger.info(
            "Initializing embedding provider: Gemini (%s)",
            getattr(app_settings, "EMBEDDING_MODEL", "gemini-embedding-001"),
        )
        from app.providers.gemini_embedding import GeminiEmbeddingProvider

        return GeminiEmbeddingProvider(settings=app_settings)

    logger.error("Unsupported EMBEDDING_PROVIDER configured: '%s'", provider_name)
    raise EmbeddingError(
        message=(
            f"Unsupported EMBEDDING_PROVIDER '{provider_name}'. "
            "Supported providers are: ['gemini']."
        ),
        provider=provider_name,
        code="UNSUPPORTED_PROVIDER",
    )
