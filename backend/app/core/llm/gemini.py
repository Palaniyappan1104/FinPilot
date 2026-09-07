"""Gemini LLM Provider implementation using the official google-genai SDK."""

from typing import Any, Dict, Optional

from app.core.config import Settings, get_settings
from app.core.llm.base import LLMProvider, LLMResponse
from app.core.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMResponseError,
    LLMTransientError,
)
from app.core.llm.retry import retry_with_backoff
from app.core.logging import get_logger

logger = get_logger("app.core.llm.gemini")


class GeminiProvider(LLMProvider):
    """Concrete LLMProvider implementation for Google Gemini."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[Any] = None,
    ) -> None:
        """Initialize the Gemini provider.

        Args:
            settings: Application Settings instance. Defaults to get_settings().
            client: Pre-instantiated genai.Client instance for testing or
                dependency injection.
        """
        self._settings = settings or get_settings()
        self._model = self._settings.LLM_MODEL
        self._provider_name = "gemini"

        if client is not None:
            self._client = client
        else:
            api_key = self._settings.GEMINI_API_KEY.strip()
            if not api_key:
                raise LLMConfigurationError(
                    message="GEMINI_API_KEY is not configured or empty. "
                    "Please set GEMINI_API_KEY in your environment or .env file.",
                    provider=self._provider_name,
                )
            try:
                from google import genai

                self._client = genai.Client(api_key=api_key)
            except Exception as exc:
                raise LLMConfigurationError(
                    message=f"Failed to initialize Gemini client: {exc}",
                    provider=self._provider_name,
                ) from exc

    @property
    def provider_name(self) -> str:
        """Return the provider identifier name."""
        return self._provider_name

    @property
    def model_name(self) -> str:
        """Return the active model name."""
        return self._model

    def generate(
        self,
        prompt: str,
        schema: Optional[Any] = None,
    ) -> LLMResponse:
        """Generate a response using Google Gemini.

        Args:
            prompt: Text prompt to submit to the model.
            schema: Reserved for Phase 2.2 structured output handling.

        Returns:
            LLMResponse: Application-level response with generated content and metadata.
        """
        if not prompt or not prompt.strip():
            raise LLMResponseError(
                message="Prompt cannot be empty or whitespace.",
                provider=self._provider_name,
            )

        create_kwargs: Dict[str, Any] = {
            "model": self._model,
            "input": prompt,
        }

        # If a structured schema is provided, configure the response_format
        if schema is not None:
            json_schema: Dict[str, Any]
            if hasattr(schema, "model_json_schema") and callable(
                schema.model_json_schema
            ):
                json_schema = schema.model_json_schema()
            elif isinstance(schema, dict):
                json_schema = schema
            else:
                raise LLMResponseError(
                    message=f"Unsupported schema type: {type(schema).__name__}",
                    provider=self._provider_name,
                )

            create_kwargs["response_format"] = {
                "type": "text",
                "mime_type": "application/json",
                "schema": json_schema,
            }

        def _call_gemini() -> LLMResponse:
            try:
                response = self._client.interactions.create(**create_kwargs)
            except Exception as exc:
                self._map_and_raise_exception(exc)

            # Retrieve text from output_text property or fallback to text/content
            text_output = getattr(response, "output_text", None)
            if text_output is None and hasattr(response, "text"):
                text_output = response.text

            if response is None or not text_output:
                raise LLMResponseError(
                    message="Gemini returned an empty or invalid response.",
                    provider=self._provider_name,
                )

            metadata: Dict[str, Any] = {}
            # Capture interaction identifier if available
            interaction_id = getattr(response, "id", None)
            if interaction_id:
                metadata["interaction_id"] = interaction_id

            # Capture usage token metrics from Interaction.usage
            usage = getattr(response, "usage", None)
            if usage is not None:
                metadata["input_tokens"] = getattr(usage, "total_input_tokens", None)
                metadata["output_tokens"] = getattr(usage, "total_output_tokens", None)
                metadata["total_tokens"] = getattr(usage, "total_tokens", None)
            elif hasattr(response, "usage_metadata") and response.usage_metadata:
                # Backward-compatibility fallback if legacy usage_metadata is present
                legacy_usage = response.usage_metadata
                metadata["input_tokens"] = getattr(
                    legacy_usage, "prompt_token_count", None
                )
                metadata["output_tokens"] = getattr(
                    legacy_usage, "candidates_token_count", None
                )
                metadata["total_tokens"] = getattr(
                    legacy_usage, "total_token_count", None
                )

            return LLMResponse(
                content=text_output,
                model=self._model,
                provider=self._provider_name,
                metadata=metadata,
            )

        return retry_with_backoff(
            func=_call_gemini,
            max_retries=self._settings.LLM_MAX_RETRIES,
            initial_delay=self._settings.LLM_INITIAL_RETRY_DELAY,
            backoff_factor=self._settings.LLM_BACKOFF_FACTOR,
            provider_name=self._provider_name,
        )

    def _map_and_raise_exception(self, exc: Exception) -> None:
        """Map SDK/HTTP exceptions into FinPilot LLM exception hierarchy.

        Ensures no secrets or API keys are leaked into exception messages.
        """
        err_msg = str(exc)
        status_code = getattr(exc, "status_code", getattr(exc, "code", None))

        # Check for authentication/permission issues (401, 403, API_KEY_INVALID)
        if (
            status_code in (401, 403)
            or "API_KEY_INVALID" in err_msg
            or "unauthorized" in err_msg.lower()
            or "permission denied" in err_msg.lower()
        ):
            raise LLMAuthenticationError(
                message=(
                    "Gemini API authentication failed. "
                    "Verify that GEMINI_API_KEY is valid."
                ),
                provider=self._provider_name,
            ) from exc

        # Check for transient errors (429, 500, 502, 503, 504, network errors)
        if (
            status_code in (429, 500, 502, 503, 504)
            or "rate limit" in err_msg.lower()
            or "quota" in err_msg.lower()
            or "resource exhausted" in err_msg.lower()
            or "unavailable" in err_msg.lower()
            or "timeout" in err_msg.lower()
            or "connection" in err_msg.lower()
        ):
            raise LLMTransientError(
                message=f"Gemini transient failure: {err_msg}",
                provider=self._provider_name,
            ) from exc

        # Other errors: client errors, bad request, invalid arg, etc.
        raise LLMError(
            message=f"Gemini API error: {err_msg}",
            provider=self._provider_name,
        ) from exc
