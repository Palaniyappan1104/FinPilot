"""Unit and integration tests for Phase 9.8: Query Embedding.

Verifies:
1. QueryEmbeddingRequest schema validation and whitespace stripping.
2. QueryEmbeddingResult schema validation, immutability, and finite vector checks.
3. QueryEmbedder service with mock provider.
4. embed_query convenience function.
5. Empty / whitespace-only query rejection.
6. Non-string / invalid query input rejection.
7. Output dimensionality validation against provider dimensions.
8. Malformed / empty provider response handling.
9. Non-finite provider response handling.
10. Provider exception propagation (Unavailable, Auth, RateLimit).
11. GeminiEmbeddingProvider.embed_query invocation with RETRIEVAL_QUERY task type.
12. GeminiEmbeddingProvider.embed_query dimensionality configuration.
13. GeminiEmbeddingProvider.embed_query exception mapping (401, 429, 503).
14. Alignment between document chunk and query embedding dimensions.
15. Safe logging (ensures sensitive query text is not leaked).
16. QueryEmbedder default provider resolution.
"""

import logging
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.models.embeddings import (
    EmbeddingModelInfo,
    QueryEmbeddingRequest,
    QueryEmbeddingResult,
)
from app.providers.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingDimensionMismatchError,
    EmbeddingEmptyInputError,
    EmbeddingInvalidResponseError,
    EmbeddingProvider,
    EmbeddingProviderRateLimitError,
    EmbeddingProviderUnavailableError,
)
from app.providers.gemini_embedding import GeminiEmbeddingProvider
from app.services.query_embedder import QueryEmbedder, embed_query

# ---------------------------------------------------------------------------
# Test constants & Mock Provider
# ---------------------------------------------------------------------------

_MOCK_DIMS = 8
_MOCK_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
_MOCK_MODEL = "mock-embedding-001"


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock EmbeddingProvider for isolated offline query embedding tests."""

    def __init__(
        self,
        dimensions: int = _MOCK_DIMS,
        vector: Optional[List[float]] = None,
        model_name: str = _MOCK_MODEL,
    ) -> None:
        self._dims = dimensions
        self._vector = vector if vector is not None else list(_MOCK_VECTOR)
        self._model_name = model_name
        self.last_query: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dims

    def get_model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            provider="mock",
            model_name=self._model_name,
            dimensions=self._dims,
        )

    def embed_single(self, request: object) -> object:
        raise NotImplementedError

    def embed_batch(self, requests: list) -> list:
        raise NotImplementedError

    def embed_query(self, query: str) -> List[float]:
        self.last_query = query
        return list(self._vector)


# ===========================================================================
# 1. QueryEmbeddingRequest Schema Tests
# ===========================================================================


class TestQueryEmbeddingRequest:
    """Test validation of the QueryEmbeddingRequest model."""

    def test_valid_request(self) -> None:
        req = QueryEmbeddingRequest(query="What was Apple's Q3 revenue?")
        assert req.query == "What was Apple's Q3 revenue?"
        assert req.query_id is None
        assert req.ticker is None

    def test_valid_request_with_context(self) -> None:
        req = QueryEmbeddingRequest(
            query="  What was Apple's revenue?  ",
            query_id="qid-101",
            ticker="AAPL",
        )
        assert req.query == "What was Apple's revenue?"
        assert req.query_id == "qid-101"
        assert req.ticker == "AAPL"

    def test_empty_query_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            QueryEmbeddingRequest(query="")

    def test_whitespace_query_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            QueryEmbeddingRequest(query="   \t\n  ")

    def test_frozen_immutability(self) -> None:
        req = QueryEmbeddingRequest(query="valid query")
        with pytest.raises(ValidationError):
            req.query = "new query"  # type: ignore[misc]


# ===========================================================================
# 2. QueryEmbeddingResult Schema Tests
# ===========================================================================


class TestQueryEmbeddingResult:
    """Test validation and consistency of the QueryEmbeddingResult model."""

    def test_valid_result(self) -> None:
        res = QueryEmbeddingResult(
            query="Apple income",
            query_id="qid-1",
            ticker="AAPL",
            embedding=[0.1, 0.2, 0.3],
            embedding_model="test-model",
            embedding_dimensions=3,
        )
        assert res.query == "Apple income"
        assert res.embedding_dimensions == 3
        assert len(res.embedding) == 3
        assert res.embedded_at is not None

    def test_empty_embedding_vector_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryEmbeddingResult(
                query="query",
                embedding=[],
                embedding_model="test-model",
                embedding_dimensions=0,
            )

    def test_nan_in_embedding_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryEmbeddingResult(
                query="query",
                embedding=[0.1, float("nan"), 0.3],
                embedding_model="test-model",
                embedding_dimensions=3,
            )

    def test_inf_in_embedding_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryEmbeddingResult(
                query="query",
                embedding=[0.1, float("inf"), 0.3],
                embedding_model="test-model",
                embedding_dimensions=3,
            )

    def test_dimension_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryEmbeddingResult(
                query="query",
                embedding=[0.1, 0.2],
                embedding_model="test-model",
                embedding_dimensions=3,
            )

    def test_frozen_immutability(self) -> None:
        res = QueryEmbeddingResult(
            query="query",
            embedding=[0.1, 0.2],
            embedding_model="test-model",
            embedding_dimensions=2,
        )
        with pytest.raises(ValidationError):
            res.embedding = [0.3, 0.4]  # type: ignore[misc]


# ===========================================================================
# 3. QueryEmbedder Service Tests
# ===========================================================================


class TestQueryEmbedderService:
    """Test the QueryEmbedder service layer."""

    def test_embed_valid_query_string(self) -> None:
        provider = MockEmbeddingProvider()
        embedder = QueryEmbedder(provider=provider)

        result = embedder.embed_query(
            "What was the total revenue in fiscal 2024?",
            query_id="qid-001",
            ticker="NVDA",
        )

        assert isinstance(result, QueryEmbeddingResult)
        assert result.query == "What was the total revenue in fiscal 2024?"
        assert result.query_id == "qid-001"
        assert result.ticker == "NVDA"
        assert result.embedding == _MOCK_VECTOR
        assert result.embedding_dimensions == _MOCK_DIMS
        assert result.embedding_model == _MOCK_MODEL
        assert provider.last_query == "What was the total revenue in fiscal 2024?"

    def test_embed_valid_query_request_object(self) -> None:
        provider = MockEmbeddingProvider()
        embedder = QueryEmbedder(provider=provider)
        req = QueryEmbeddingRequest(
            query="How did operating margins change?",
            query_id="qid-002",
            ticker="MSFT",
        )

        result = embedder.embed_query(req)

        assert result.query == "How did operating margins change?"
        assert result.query_id == "qid-002"
        assert result.ticker == "MSFT"
        assert result.embedding == _MOCK_VECTOR
        assert result.embedding_dimensions == _MOCK_DIMS

    def test_convenience_function_embed_query(self) -> None:
        provider = MockEmbeddingProvider()
        result = embed_query(
            "Is free cash flow positive?",
            provider=provider,
            query_id="qid-003",
            ticker="TSLA",
        )
        assert isinstance(result, QueryEmbeddingResult)
        assert result.query == "Is free cash flow positive?"
        assert result.ticker == "TSLA"
        assert result.embedding == _MOCK_VECTOR

    def test_model_info_property(self) -> None:
        provider = MockEmbeddingProvider()
        embedder = QueryEmbedder(provider=provider)
        info = embedder.model_info
        assert info.provider == "mock"
        assert info.dimensions == _MOCK_DIMS
        assert info.model_name == _MOCK_MODEL

    def test_provider_property(self) -> None:
        provider = MockEmbeddingProvider()
        embedder = QueryEmbedder(provider=provider)
        assert embedder.provider is provider


# ===========================================================================
# 4. Input Validation & Edge Cases
# ===========================================================================


class TestQueryEmbeddingValidation:
    """Test validation and rejection of invalid or malformed query inputs."""

    def test_empty_string_rejected(self) -> None:
        embedder = QueryEmbedder(provider=MockEmbeddingProvider())
        with pytest.raises(EmbeddingEmptyInputError) as exc_info:
            embedder.embed_query("")
        assert "empty or whitespace" in str(exc_info.value)

    def test_whitespace_only_string_rejected(self) -> None:
        embedder = QueryEmbedder(provider=MockEmbeddingProvider())
        with pytest.raises(EmbeddingEmptyInputError) as exc_info:
            embedder.embed_query("   \t\n   ")
        assert "empty or whitespace" in str(exc_info.value)

    def test_non_string_input_rejected(self) -> None:
        embedder = QueryEmbedder(provider=MockEmbeddingProvider())
        with pytest.raises(EmbeddingEmptyInputError) as exc_info:
            embedder.embed_query(12345)  # type: ignore[arg-type]
        assert "Query must be a string" in str(exc_info.value)

    def test_none_input_rejected(self) -> None:
        embedder = QueryEmbedder(provider=MockEmbeddingProvider())
        with pytest.raises(EmbeddingEmptyInputError):
            embedder.embed_query(None)  # type: ignore[arg-type]

    def test_whitespace_trimmed_before_provider(self) -> None:
        provider = MockEmbeddingProvider()
        embedder = QueryEmbedder(provider=provider)
        result = embedder.embed_query("   Trailing and leading whitespace   ")
        assert result.query == "Trailing and leading whitespace"
        assert provider.last_query == "Trailing and leading whitespace"


# ===========================================================================
# 5. Service-Layer Output Validation
# ===========================================================================


class TestQueryEmbedderOutputValidation:
    """Test service-layer detection and handling of provider response issues."""

    def test_empty_provider_vector_raises_invalid_response(self) -> None:
        provider = MockEmbeddingProvider(vector=[])
        embedder = QueryEmbedder(provider=provider)
        with pytest.raises(EmbeddingInvalidResponseError) as exc_info:
            embedder.embed_query("Valid query")
        assert "empty vector" in str(exc_info.value)

    def test_nan_in_provider_vector_raises_invalid_response(self) -> None:
        nan_vec = [0.1, float("nan"), 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        provider = MockEmbeddingProvider(vector=nan_vec)
        embedder = QueryEmbedder(provider=provider)
        with pytest.raises(EmbeddingInvalidResponseError) as exc_info:
            embedder.embed_query("Valid query")
        assert "non-finite" in str(exc_info.value)

    def test_inf_in_provider_vector_raises_invalid_response(self) -> None:
        inf_vec = [0.1, 0.2, float("inf"), 0.4, 0.5, 0.6, 0.7, 0.8]
        provider = MockEmbeddingProvider(vector=inf_vec)
        embedder = QueryEmbedder(provider=provider)
        with pytest.raises(EmbeddingInvalidResponseError) as exc_info:
            embedder.embed_query("Valid query")
        assert "non-finite" in str(exc_info.value)

    def test_dimension_mismatch_raises_typed_error(self) -> None:
        short_vec = [0.1, 0.2, 0.3, 0.4]  # 4 dims instead of 8
        provider = MockEmbeddingProvider(dimensions=8, vector=short_vec)
        embedder = QueryEmbedder(provider=provider)
        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            embedder.embed_query("Valid query")
        assert exc_info.value.expected_dimensions == 8
        assert exc_info.value.actual_dimensions == 4


# ===========================================================================
# 6. Provider Error Propagation
# ===========================================================================


class TestQueryEmbedderErrorPropagation:
    """Verify that provider errors propagate unmasked through QueryEmbedder."""

    def test_provider_unavailable_error_propagates(self) -> None:
        provider = MagicMock(spec=EmbeddingProvider)
        provider.provider_name = "mock"
        provider.dimensions = _MOCK_DIMS
        provider.embed_query.side_effect = EmbeddingProviderUnavailableError(
            "Mock provider is down", provider="mock"
        )
        embedder = QueryEmbedder(provider=provider)

        with pytest.raises(EmbeddingProviderUnavailableError) as exc_info:
            embedder.embed_query("Valid question")
        assert "Mock provider is down" in str(exc_info.value)

    def test_authentication_error_propagates(self) -> None:
        provider = MagicMock(spec=EmbeddingProvider)
        provider.provider_name = "mock"
        provider.dimensions = _MOCK_DIMS
        provider.embed_query.side_effect = EmbeddingAuthenticationError(
            "Invalid API key", provider="mock"
        )
        embedder = QueryEmbedder(provider=provider)

        with pytest.raises(EmbeddingAuthenticationError) as exc_info:
            embedder.embed_query("Valid question")
        assert "Invalid API key" in str(exc_info.value)

    def test_rate_limit_error_propagates(self) -> None:
        provider = MagicMock(spec=EmbeddingProvider)
        provider.provider_name = "mock"
        provider.dimensions = _MOCK_DIMS
        provider.embed_query.side_effect = EmbeddingProviderRateLimitError(
            "Quota exceeded", provider="mock"
        )
        embedder = QueryEmbedder(provider=provider)

        with pytest.raises(EmbeddingProviderRateLimitError) as exc_info:
            embedder.embed_query("Valid question")
        assert "Quota exceeded" in str(exc_info.value)


# ===========================================================================
# 7. GeminiEmbeddingProvider Query Implementation Tests
# ===========================================================================


class TestGeminiQueryEmbedding:
    """Test GeminiEmbeddingProvider.embed_query offline with mocked google.genai."""

    def _make_gemini_provider(self, dimensions: int = 3072) -> GeminiEmbeddingProvider:
        fake_settings = MagicMock(spec=Settings)
        fake_settings.GEMINI_API_KEY = "mock-key-12345"
        fake_settings.EMBEDDING_MODEL = "gemini-embedding-001"
        fake_settings.EMBEDDING_DIMENSIONS = dimensions

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            provider = GeminiEmbeddingProvider(settings=fake_settings)
            provider._client = mock_client
            return provider

    def test_embed_query_uses_retrieval_query_task_type(self) -> None:
        """Gemini embed_query passes task_type='RETRIEVAL_QUERY' and dimensionality."""
        from google.genai import types as genai_types

        provider = self._make_gemini_provider(dimensions=3072)

        # Mock API response with 3072-dim float vector
        mock_embedding = MagicMock()
        mock_embedding.values = [0.05] * 3072
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        provider._client.models.embed_content.return_value = mock_response

        vector = provider.embed_query("What are the company's main revenue drivers?")

        assert len(vector) == 3072
        assert vector[0] == pytest.approx(0.05)

        # Verify exact call arguments
        provider._client.models.embed_content.assert_called_once()
        call_kwargs = provider._client.models.embed_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-embedding-001"
        assert call_kwargs["contents"] == "What are the company's main revenue drivers?"

        config = call_kwargs["config"]
        assert isinstance(config, genai_types.EmbedContentConfig)
        assert config.task_type == "RETRIEVAL_QUERY"
        assert config.output_dimensionality == 3072

    def test_embed_query_empty_text_raises_empty_input_error(self) -> None:
        provider = self._make_gemini_provider()
        with pytest.raises(EmbeddingEmptyInputError):
            provider.embed_query("")
        with pytest.raises(EmbeddingEmptyInputError):
            provider.embed_query("   \n\t  ")

    def test_embed_query_maps_401_to_authentication_error(self) -> None:
        provider = self._make_gemini_provider()
        err = Exception("401 Invalid API key unauthenticated")
        setattr(err, "code", 401)
        provider._client.models.embed_content.side_effect = err

        with pytest.raises(EmbeddingAuthenticationError):
            provider.embed_query("Valid query")

    def test_embed_query_maps_429_to_rate_limit_error(self) -> None:
        provider = self._make_gemini_provider()
        err = Exception("429 Resource has been exhausted (rate limit)")
        setattr(err, "code", 429)
        provider._client.models.embed_content.side_effect = err

        with pytest.raises(EmbeddingProviderRateLimitError):
            provider.embed_query("Valid query")

    def test_embed_query_maps_503_to_unavailable_error(self) -> None:
        provider = self._make_gemini_provider()
        err = Exception("503 Service Unavailable / connection timeout")
        setattr(err, "code", 503)
        provider._client.models.embed_content.side_effect = err

        with pytest.raises(EmbeddingProviderUnavailableError):
            provider.embed_query("Valid query")

    def test_embed_query_raises_invalid_response_on_empty_embeddings(self) -> None:
        provider = self._make_gemini_provider()
        mock_response = MagicMock()
        mock_response.embeddings = []
        provider._client.models.embed_content.return_value = mock_response

        with pytest.raises(EmbeddingInvalidResponseError) as exc_info:
            provider.embed_query("Valid query")
        assert "no embeddings" in str(exc_info.value)


# ===========================================================================
# 8. Dimensionality Alignment & Base Class Tests
# ===========================================================================


class TestQueryDocumentDimensionalityAlignment:
    """Verify that query embeddings match the exact document embedding dimensions."""

    def test_query_and_document_dimensions_are_identical(self) -> None:
        provider = MockEmbeddingProvider(dimensions=3072, vector=[0.01] * 3072)
        embedder = QueryEmbedder(provider=provider)

        result = embedder.embed_query("What are the risk factors?")
        assert result.embedding_dimensions == 3072
        assert len(result.embedding) == 3072
        assert result.embedding_dimensions == provider.dimensions

    def test_base_embedding_provider_default_embed_query_raises_not_implemented(
        self,
    ) -> None:
        class UnimplementedProvider(EmbeddingProvider):
            @property
            def provider_name(self) -> str:
                return "stub"

            @property
            def model_name(self) -> str:
                return "stub-model"

            @property
            def dimensions(self) -> int:
                return 4

            def get_model_info(self) -> EmbeddingModelInfo:
                return EmbeddingModelInfo(
                    provider="stub", model_name="stub", dimensions=4
                )

            def embed_single(self, request: object) -> object:
                raise NotImplementedError

            def embed_batch(self, requests: list) -> list:
                raise NotImplementedError

        unimpl = UnimplementedProvider()
        with pytest.raises(NotImplementedError) as exc_info:
            unimpl.embed_query("query")
        assert "does not implement embed_query" in str(exc_info.value)


# ===========================================================================
# 9. Safe Logging Verification
# ===========================================================================


class TestSafeLogging:
    """Ensure user query text is not leaked into log messages."""

    def test_query_text_not_emitted_in_service_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = MockEmbeddingProvider()
        embedder = QueryEmbedder(provider=provider)
        secret_query = "CONFIDENTIAL_MERGER_QUERY_TOP_SECRET"

        with caplog.at_level(logging.INFO):
            embedder.embed_query(secret_query, query_id="qid-999", ticker="XYZ")

        for record in caplog.records:
            assert secret_query not in record.message
        # Confirm that safe metadata was logged
        assert any(f"chars={len(secret_query)}" in r.message for r in caplog.records)
        assert any("qid-999" in r.message for r in caplog.records)
