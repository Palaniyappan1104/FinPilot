"""Market Data Provider interface and factory for FinPilot.

Phase 7.1 establishes the provider-agnostic interface for retrieving historical
OHLCV market price and volume data and the factory for instantiating the provider.
"""

from abc import ABC, abstractmethod
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.market_data import HistoricalMarketData
from app.providers.exceptions import MarketDataError

logger = get_logger("app.providers.market_data")


class MarketDataProvider(ABC):
    """Abstract interface defining contract for historical market data providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier name (e.g. 'yahoo')."""
        pass

    @abstractmethod
    def get_ohlcv(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> HistoricalMarketData:
        """Retrieve and normalize historical OHLCV market data for the given ticker.

        Args:
            ticker: Stock symbol (e.g. 'AAPL', 'MSFT').
            period: Requested lookback window (e.g. '1mo', '3mo', '6mo', '1y', 'max').
            interval: Bar aggregation frequency (e.g. '1d', '1wk', '1mo').

        Returns:
            HistoricalMarketData: Strongly typed normalized domain model with
            chronological candles, metadata, and data validation guarantees.

        Raises:
            MarketDataTickerNotFoundError: If the ticker is invalid or returns no data.
            MarketDataProviderUnavailableError: If provider is unreachable or times out.
            MarketDataProviderRateLimitError: If rate limiting is explicitly returned.
            MarketDataMalformedError: If response is corrupt or violates constraints.
            EmptyMarketDataError: If provider returns empty series or no candles.
            MarketDataError: For any other unhandled market data provider errors.
        """
        pass


def get_market_data_provider(
    settings: Optional[Settings] = None,
) -> MarketDataProvider:
    """Instantiate and return the configured MarketDataProvider.

    Reads provider type from settings.MARKET_DATA_PROVIDER (default: 'yahoo').

    Args:
        settings: Optional application settings instance.

    Returns:
        MarketDataProvider: Configured concrete provider implementation.

    Raises:
        MarketDataError: If the configured provider name is unsupported or invalid.
    """
    app_settings = settings or get_settings()
    provider_name = (
        (getattr(app_settings, "MARKET_DATA_PROVIDER", "yahoo") or "").strip().lower()
    )

    if not provider_name:
        raise MarketDataError(
            message="MARKET_DATA_PROVIDER configuration cannot be empty.",
            provider="unknown",
        )

    if provider_name == "yahoo":
        logger.info("Initializing market data provider: Yahoo Finance")
        from app.providers.yahoo_market_data import YahooFinanceMarketDataProvider

        return YahooFinanceMarketDataProvider()

    logger.error("Unsupported MARKET_DATA_PROVIDER configured: '%s'", provider_name)
    raise MarketDataError(
        message=(
            f"Unsupported MARKET_DATA_PROVIDER '{provider_name}'. "
            "Supported providers are: ['yahoo']."
        ),
        provider=provider_name,
    )
