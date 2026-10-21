"""Groq LLM Provider implementation using the official groq SDK."""

import os
from typing import Any, Dict, List, Optional

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

logger = get_logger("app.core.llm.groq")


class GroqProvider(LLMProvider):
    """Concrete LLMProvider implementation for Groq."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[Any] = None,
    ) -> None:
        """Initialize the Groq provider.

        Args:
            settings: Application Settings instance. Defaults to get_settings().
            client: Pre-instantiated Groq client instance for testing or DI.
        """
        self._settings = settings or get_settings()
        self._model = self._settings.LLM_MODEL or "openai/gpt-oss-20b"
        self._provider_name = "groq"

        if client is not None:
            self._client = client
        else:
            api_key = (
                getattr(self._settings, "GROQ_API_KEY", "")
                or os.getenv("GROQ_API_KEY", "")
            ).strip()

            if not api_key:
                raise LLMConfigurationError(
                    message="GROQ_API_KEY is not configured or empty. "
                    "Please set GROQ_API_KEY in your environment or .env file.",
                    provider=self._provider_name,
                )
            try:
                from groq import Groq

                self._client = Groq(api_key=api_key)
            except Exception as exc:
                raise LLMConfigurationError(
                    message=f"Failed to initialize Groq client: {exc}",
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
        """Generate a response using Groq.

        Args:
            prompt: Text prompt to submit to the model.
            schema: Optional schema definition for structured output.

        Returns:
            LLMResponse: Application-level response with generated content and metadata.
        """
        if not prompt or not prompt.strip():
            raise LLMResponseError(
                message="Prompt cannot be empty or whitespace.",
                provider=self._provider_name,
            )

        messages: List[Dict[str, str]] = []
        user_prompt = prompt

        create_kwargs: Dict[str, Any] = {
            "model": self._model,
        }

        # If structured schema is requested, enable JSON mode
        if schema is not None:
            create_kwargs["response_format"] = {"type": "json_object"}
            if "json" not in user_prompt.lower():
                user_prompt += "\n\nRespond with valid JSON only."

        messages.append({"role": "user", "content": user_prompt})
        create_kwargs["messages"] = messages

        def _call_groq() -> LLMResponse:
            try:
                response = self._client.chat.completions.create(**create_kwargs)
            except Exception as exc:
                self._map_and_raise_exception(exc)

            if (
                not response
                or not response.choices
                or not response.choices[0].message
                or response.choices[0].message.content is None
            ):
                raise LLMResponseError(
                    message="Groq returned an empty or invalid response.",
                    provider=self._provider_name,
                )

            text_output = response.choices[0].message.content

            metadata: Dict[str, Any] = {}
            if getattr(response, "id", None):
                metadata["interaction_id"] = response.id

            usage = getattr(response, "usage", None)
            if usage is not None:
                metadata["input_tokens"] = getattr(usage, "prompt_tokens", None)
                metadata["output_tokens"] = getattr(usage, "completion_tokens", None)
                metadata["total_tokens"] = getattr(usage, "total_tokens", None)

            return LLMResponse(
                content=text_output,
                model=self._model,
                provider=self._provider_name,
                metadata=metadata,
            )

        return retry_with_backoff(
            func=_call_groq,
            max_retries=self._settings.LLM_MAX_RETRIES,
            initial_delay=self._settings.LLM_INITIAL_RETRY_DELAY,
            backoff_factor=self._settings.LLM_BACKOFF_FACTOR,
            provider_name=self._provider_name,
        )

    def _map_and_raise_exception(self, exc: Exception) -> None:
        """Map Groq/HTTP exceptions into FinPilot LLM exception hierarchy."""
        err_msg = str(exc)
        status_code = getattr(exc, "status_code", getattr(exc, "code", None))

        if (
            status_code in (401, 403)
            or "invalid_api_key" in err_msg.lower()
            or "unauthorized" in err_msg.lower()
            or "permission denied" in err_msg.lower()
        ):
            raise LLMAuthenticationError(
                message="Groq API authentication failed. Verify that GROQ_API_KEY is valid.",
                provider=self._provider_name,
            ) from exc

        err_msg_lower = err_msg.lower()
        if (
            status_code in (429, 500, 502, 503, 504)
            or "rate limit" in err_msg_lower
            or "rate_limit_exceeded" in err_msg_lower
            or "quota" in err_msg_lower
            or "timeout" in err_msg_lower
            or "connection" in err_msg_lower
            or "unavailable" in err_msg_lower
        ):
            raise LLMTransientError(
                message=f"Groq transient failure: {err_msg}",
                provider=self._provider_name,
            ) from exc

        raise LLMError(
            message=f"Groq API error: {err_msg}",
            provider=self._provider_name,
        ) from exc
