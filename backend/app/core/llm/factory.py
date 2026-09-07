"""Factory for instantiating LLMProvider instances based on configuration."""

from typing import Optional

from app.core.config import Settings, get_settings
from app.core.llm.base import LLMProvider
from app.core.llm.exceptions import LLMConfigurationError
from app.core.llm.gemini import GeminiProvider
from app.core.logging import get_logger

logger = get_logger("app.core.llm.factory")


def get_llm_provider(settings: Optional[Settings] = None) -> LLMProvider:
    """Instantiate and return the configured LLMProvider.

    Reads the provider type from settings.LLM_PROVIDER. Supported providers:
    - 'gemini' -> GeminiProvider

    Args:
        settings: Optional Settings instance. Defaults to get_settings().

    Returns:
        LLMProvider: Configured concrete provider implementation.

    Raises:
        LLMConfigurationError: When an unsupported or empty provider name is configured.
    """
    app_settings = settings or get_settings()
    provider_name = (app_settings.LLM_PROVIDER or "").strip().lower()

    if not provider_name:
        raise LLMConfigurationError(
            message="LLM_PROVIDER configuration cannot be empty.",
            provider="unknown",
        )

    if provider_name == "gemini":
        logger.info(
            "Initializing LLM provider: Gemini (model: %s)", app_settings.LLM_MODEL
        )
        return GeminiProvider(settings=app_settings)

    # Extension point for future providers (e.g. openai, anthropic)
    logger.error("Unsupported LLM_PROVIDER configured: '%s'", provider_name)
    raise LLMConfigurationError(
        message=f"Unsupported LLM_PROVIDER '{provider_name}'. "
        "Supported providers are: ['gemini'].",
        provider=provider_name,
    )
