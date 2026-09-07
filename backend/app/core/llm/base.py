"""LLMProvider base interface and response models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Application-level standard response returned by any LLMProvider."""

    content: str = Field(..., description="The generated text content")
    model: str = Field(..., description="The model name that generated the response")
    provider: str = Field(..., description="The provider name, e.g. gemini")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-agnostic metadata such as finish reason and token counts",
    )


class LLMProvider(ABC):
    """Abstract interface defining the common contract for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        schema: Optional[Any] = None,
    ) -> LLMResponse:
        """Generate a response for the given prompt.

        Args:
            prompt: Text prompt for the language model.
            schema: Optional schema definition for structured output.
                    Reserved for Phase 2.2 structured-output implementation.

        Returns:
            LLMResponse containing the generated text, model, and metadata.

        Raises:
            LLMConfigurationError: If provider configuration is invalid or missing.
            LLMAuthenticationError: If authentication with the provider fails.
            LLMRetryExhaustedError: If transient failures persist after all retries.
            LLMResponseError: If the provider returns an invalid or empty response.
            LLMError: For any other unhandled provider errors.
        """
        pass
