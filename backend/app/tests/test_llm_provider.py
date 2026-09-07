"""Unit tests for the LLMProvider abstraction and GeminiProvider."""

from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.core.llm.base import LLMProvider, LLMResponse
from app.core.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMResponseError,
    LLMRetryExhaustedError,
)
from app.core.llm.factory import get_llm_provider
from app.core.llm.gemini import GeminiProvider


# Helper to build test Settings without real external resources
def make_test_settings(**kwargs) -> Settings:
    defaults = {
        "GEMINI_API_KEY": "test-gemini-key-123",
        "LLM_PROVIDER": "gemini",
        "LLM_MODEL": "gemini-2.5-flash",
        "LLM_MAX_RETRIES": 3,
        "LLM_INITIAL_RETRY_DELAY": 0.01,  # Fast retries for unit tests
        "LLM_BACKOFF_FACTOR": 1.0,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


# Helper to create a fake Gemini response object
def make_fake_gemini_response(
    text: str = "Analysis completed.",
    finish_reason: str = "STOP",
    prompt_tokens: int = 15,
    candidate_tokens: int = 25,
    total_tokens: int = 40,
):
    fake_candidate = MagicMock()
    fake_candidate.finish_reason = finish_reason

    fake_usage = MagicMock()
    fake_usage.prompt_token_count = prompt_tokens
    fake_usage.candidates_token_count = candidate_tokens
    fake_usage.total_token_count = total_tokens

    fake_response = MagicMock()
    fake_response.text = text
    fake_response.candidates = [fake_candidate]
    fake_response.usage_metadata = fake_usage
    return fake_response


# ==============================================================================
# 1. LLMProvider Interface Contract
# ==============================================================================


def test_gemini_provider_satisfies_llm_provider_contract():
    """Verify GeminiProvider is an instance of the abstract LLMProvider interface."""
    fake_client = MagicMock()
    settings = make_test_settings()
    provider = GeminiProvider(settings=settings, client=fake_client)

    assert isinstance(provider, LLMProvider)
    assert hasattr(provider, "generate")
    assert callable(provider.generate)


# ==============================================================================
# 2. Mocked Successful Gemini Response
# ==============================================================================


def test_gemini_generate_success():
    """Verify a mocked successful Gemini response converts into LLMResponse."""
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = make_fake_gemini_response(
        text="Bullish sentiment observed.",
        finish_reason="STOP",
        prompt_tokens=10,
        candidate_tokens=20,
        total_tokens=30,
    )

    settings = make_test_settings(LLM_MODEL="gemini-2.5-flash")
    provider = GeminiProvider(settings=settings, client=fake_client)

    response = provider.generate("Analyze AAPL earnings")

    assert isinstance(response, LLMResponse)
    assert response.content == "Bullish sentiment observed."
    assert response.model == "gemini-2.5-flash"
    assert response.provider == "gemini"
    assert response.metadata["finish_reason"] == "STOP"
    assert response.metadata["total_token_count"] == 30

    fake_client.models.generate_content.assert_called_once_with(
        model="gemini-2.5-flash",
        contents="Analyze AAPL earnings",
    )


def test_gemini_generate_empty_prompt_raises_error():
    """Verify an empty or whitespace prompt raises LLMResponseError."""
    fake_client = MagicMock()
    settings = make_test_settings()
    provider = GeminiProvider(settings=settings, client=fake_client)

    with pytest.raises(LLMResponseError) as exc_info:
        provider.generate("   ")

    assert "empty" in str(exc_info.value).lower()


# ==============================================================================
# 3. Provider Factory
# ==============================================================================


def test_provider_factory_returns_gemini_provider(monkeypatch):
    """Verify provider factory returns GeminiProvider when LLM_PROVIDER=gemini."""
    # Provide a dummy API key so initialization succeeds
    settings = make_test_settings(LLM_PROVIDER="gemini", GEMINI_API_KEY="dummy-key")
    provider = get_llm_provider(settings=settings)

    assert isinstance(provider, GeminiProvider)
    assert isinstance(provider, LLMProvider)


# ==============================================================================
# 4. Unsupported Provider Configuration
# ==============================================================================


def test_provider_factory_unsupported_provider_raises_error():
    """Verify an unsupported LLM_PROVIDER value raises LLMConfigurationError."""
    settings = make_test_settings(LLM_PROVIDER="unsupported_provider")

    with pytest.raises(LLMConfigurationError) as exc_info:
        get_llm_provider(settings=settings)

    assert "unsupported llm_provider" in str(exc_info.value).lower()
    assert exc_info.value.provider == "unsupported_provider"


def test_gemini_missing_api_key_raises_configuration_error():
    """Verify missing GEMINI_API_KEY raises LLMConfigurationError."""
    settings = make_test_settings(GEMINI_API_KEY="")

    with pytest.raises(LLMConfigurationError) as exc_info:
        GeminiProvider(settings=settings)

    assert "gemini_api_key" in str(exc_info.value).lower()


# ==============================================================================
# 5. Transient Failure Triggers Retries
# ==============================================================================


def test_gemini_transient_failure_retries_and_succeeds():
    """Verify a transient failure (e.g. 429) triggers retries then succeeds."""
    fake_client = MagicMock()

    class TransientRateLimitError(Exception):
        status_code = 429

    success_resp = make_fake_gemini_response(text="Recovered successfully")

    # Fail on first 2 calls, succeed on the 3rd
    fake_client.models.generate_content.side_effect = [
        TransientRateLimitError("Resource exhausted"),
        TransientRateLimitError("Resource exhausted"),
        success_resp,
    ]

    settings = make_test_settings(LLM_MAX_RETRIES=3, LLM_INITIAL_RETRY_DELAY=0.01)
    provider = GeminiProvider(settings=settings, client=fake_client)

    response = provider.generate("Give me market outlook")

    assert response.content == "Recovered successfully"
    assert fake_client.models.generate_content.call_count == 3


# ==============================================================================
# 6. Retry Exhaustion Produces Application-Level Error
# ==============================================================================


def test_gemini_retry_exhaustion_raises_application_error():
    """Verify persistent transient failures exhaust retries and raise error."""
    fake_client = MagicMock()

    class PersistentUnavailableError(Exception):
        status_code = 503

    fake_client.models.generate_content.side_effect = PersistentUnavailableError(
        "Service unavailable"
    )

    settings = make_test_settings(LLM_MAX_RETRIES=3, LLM_INITIAL_RETRY_DELAY=0.01)
    provider = GeminiProvider(settings=settings, client=fake_client)

    with pytest.raises(LLMRetryExhaustedError) as exc_info:
        provider.generate("Analyze inflation trends")

    assert exc_info.value.attempts == 3
    assert exc_info.value.provider == "gemini"
    assert fake_client.models.generate_content.call_count == 3


# ==============================================================================
# 7. Authentication Errors Handled Without Retries
# ==============================================================================


def test_gemini_authentication_error_fails_immediately_without_retries():
    """Verify an auth error (401/403) fails immediately without unnecessary retries."""
    fake_client = MagicMock()

    class AuthError(Exception):
        status_code = 401

    fake_client.models.generate_content.side_effect = AuthError("API_KEY_INVALID")

    settings = make_test_settings(LLM_MAX_RETRIES=3, LLM_INITIAL_RETRY_DELAY=0.01)
    provider = GeminiProvider(settings=settings, client=fake_client)

    with pytest.raises(LLMAuthenticationError) as exc_info:
        provider.generate("Analyze portfolio risk")

    # MUST be called only once — no retries on authentication error
    assert fake_client.models.generate_content.call_count == 1
    assert "authentication failed" in str(exc_info.value).lower()


# ==============================================================================
# 8. No Secrets or API Keys Leaked in Errors
# ==============================================================================


def test_no_secret_key_leaked_in_errors():
    """Verify errors and exception messages do not contain the secret API key."""
    fake_client = MagicMock()
    secret_key = "AIzaSySecretRealKey999"

    class CustomApiError(Exception):
        status_code = 403

    fake_client.models.generate_content.side_effect = CustomApiError("Access forbidden")

    settings = make_test_settings(GEMINI_API_KEY=secret_key, LLM_MAX_RETRIES=1)
    provider = GeminiProvider(settings=settings, client=fake_client)

    with pytest.raises(LLMError) as exc_info:
        provider.generate("Run market scan")

    error_str = str(exc_info.value)
    assert secret_key not in error_str
