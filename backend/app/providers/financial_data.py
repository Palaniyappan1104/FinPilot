"""Financial Data Provider interface and factory for FinPilot.

Phase 6.1 establishes the provider-agnostic interface for retrieving fundamental
company data and the factory for instantiating the configured provider.
"""

from abc import ABC, abstractmethod
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.financial_data import CompanyFundamentals
from app.providers.exceptions import FinancialDataError

logger = get_logger("app.providers.financial_data")


class FinancialDataProvider(ABC):
    """Abstract interface defining the common contract for financial data providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier name (e.g. 'yahoo')."""
        pass

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> CompanyFundamentals:
        """Retrieve and normalize fundamental financial data for the given ticker.

        Args:
            ticker: Stock symbol (e.g. 'AAPL', 'MSFT').

        Returns:
            CompanyFundamentals: Strongly typed normalized domain model with
            profile, financial statement periods, and provider-reported raw snapshot.

        Raises:
            TickerNotFoundError: If the ticker is invalid or returns no data.
            ProviderUnavailableError: If the provider is unreachable or times out.
            ProviderRateLimitError: If rate limiting is explicitly returned.
            ProviderMalformedDataError: If the response is corrupted or unparseable.
            FinancialDataError: For any other unhandled provider errors.
        """
        pass


def get_financial_data_provider(
    settings: Optional[Settings] = None,
) -> FinancialDataProvider:
    """Instantiate and return the configured FinancialDataProvider.

    Reads provider type from settings.FINANCIAL_DATA_PROVIDER (default: 'yahoo').

    Args:
        settings: Optional application settings instance.

    Returns:
        FinancialDataProvider: Configured concrete provider implementation.

    Raises:
        FinancialDataError: If the configured provider name is unsupported or invalid.
    """
    app_settings = settings or get_settings()
    provider_name = (
        (getattr(app_settings, "FINANCIAL_DATA_PROVIDER", "yahoo") or "")
        .strip()
        .lower()
    )

    if not provider_name:
        raise FinancialDataError(
            message="FINANCIAL_DATA_PROVIDER configuration cannot be empty.",
            provider="unknown",
        )

    if provider_name == "yahoo":
        logger.info("Initializing financial data provider: Yahoo Finance")
        from app.providers.yahoo_finance import YahooFinanceProvider

        return YahooFinanceProvider()

    logger.error("Unsupported FINANCIAL_DATA_PROVIDER configured: '%s'", provider_name)
    raise FinancialDataError(
        message=(
            f"Unsupported FINANCIAL_DATA_PROVIDER '{provider_name}'. "
            "Supported providers are: ['yahoo']."
        ),
        provider=provider_name,
    )
