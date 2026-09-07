"""Structured output generation helper using Pydantic models."""

import json
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.llm.base import LLMProvider
from app.core.llm.exceptions import LLMStructuredOutputError
from app.core.logging import get_logger

logger = get_logger("app.core.llm.structured")

T = TypeVar("T", bound=BaseModel)


def generate_structured(
    provider: LLMProvider,
    prompt: str,
    schema: Type[T],
) -> T:
    """Request structured output from an LLMProvider and validate with Pydantic.

    Retries ONCE if the model returns malformed JSON or invalid schema fields.
    If the second attempt also fails, raises a typed LLMStructuredOutputError.

    Args:
        provider: An instance of LLMProvider.
        prompt: The text prompt for the model.
        schema: A Pydantic BaseModel class specifying the target schema.

    Returns:
        An instance of the requested Pydantic model (schema).

    Raises:
        LLMStructuredOutputError: If output cannot be validated after 2 attempts.
        LLMAuthenticationError: Propagated without retry if authentication fails.
        LLMConfigurationError: Propagated without retry if configuration is invalid.
        LLMRetryExhaustedError: Propagated if transient API retries exhaust.
    """
    last_error: Exception | None = None
    validation_failure_summary: str | None = None

    # Max 2 attempts: 1 initial attempt + exactly 1 retry on validation failure
    for attempt in range(1, 3):
        try:
            response = provider.generate(prompt=prompt, schema=schema)
            clean_content = response.content.strip()

            # Strip markdown code blocks if the model wrapped the JSON
            if clean_content.startswith("```"):
                lines = clean_content.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                clean_content = "\n".join(lines).strip()

            parsed_data = json.loads(clean_content)
            return schema.model_validate(parsed_data)

        except (json.JSONDecodeError, ValidationError) as err:
            last_error = err
            validation_failure_summary = (
                str(err.errors()) if isinstance(err, ValidationError) else str(err)
            )
            if attempt == 1:
                logger.warning(
                    "Structured validation failed on attempt 1 for schema %s: %s. "
                    "Retrying once...",
                    schema.__name__,
                    type(err).__name__,
                )
            else:
                logger.error(
                    "Structured validation failed on attempt 2 for schema %s: %s.",
                    schema.__name__,
                    type(err).__name__,
                )

    raise LLMStructuredOutputError(
        message=(
            f"Structured output validation failed for schema '{schema.__name__}' "
            f"after exactly 2 attempts. Failure: {last_error}"
        ),
        provider=provider.provider_name,
        schema_name=schema.__name__,
        attempts=2,
        validation_failure=validation_failure_summary,
    )
