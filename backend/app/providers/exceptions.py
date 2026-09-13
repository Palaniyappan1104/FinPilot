"""Financial data provider domain exceptions for FinPilot.

Phase 6.1 defines typed exceptions for all external financial data operations:
- FinancialDataError: Base exception for all data provider errors.
- TickerNotFoundError: Ticker is invalid, unknown, or has no available data.
- ProviderUnavailableError: Provider/network/connection failure or timeout.
- ProviderRateLimitError: Provider explicit rate limit (HTTP 429).
- ProviderMalformedDataError: Unexpected or corrupted data structure returned.
"""


class FinancialDataError(Exception):
    """Base exception for all financial data provider errors."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider

    def __str__(self) -> str:
        return f"[{self.provider}] {self.message}"


class TickerNotFoundError(FinancialDataError):
    """Raised when a ticker symbol is unknown, invalid, or returns no data."""


class ProviderUnavailableError(FinancialDataError):
    """Raised when the provider is unreachable due to network or timeout."""


class ProviderRateLimitError(FinancialDataError):
    """Raised when the provider explicitly rejects requests (HTTP 429)."""


class ProviderMalformedDataError(FinancialDataError):
    """Raised when the provider response is unexpected or corrupted."""


# ---------------------------------------------------------------------------
# Market Data Domain Exceptions (Phase 7.1)
# ---------------------------------------------------------------------------


class MarketDataError(Exception):
    """Base exception for all market data provider errors."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider

    def __str__(self) -> str:
        return f"[{self.provider}] {self.message}"


class MarketDataTickerNotFoundError(MarketDataError, TickerNotFoundError):
    """Raised when a ticker symbol is unknown, invalid, or has no market data."""


class MarketDataProviderUnavailableError(MarketDataError, ProviderUnavailableError):
    """Raised when the market data provider is unreachable due to network/timeout."""


class MarketDataProviderRateLimitError(MarketDataError, ProviderRateLimitError):
    """Raised when the market data provider explicitly rejects requests (HTTP 429)."""


class MarketDataMalformedError(MarketDataError, ProviderMalformedDataError):
    """Raised when market data is corrupted or violates integrity constraints."""


class EmptyMarketDataError(MarketDataError):
    """Raised when provider returns an empty series or zero valid candles."""


# ---------------------------------------------------------------------------
# News Data Domain Exceptions (Phase 8.1)
# ---------------------------------------------------------------------------


class NewsDataError(Exception):
    """Base exception for all news data provider errors."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider

    def __str__(self) -> str:
        return f"[{self.provider}] {self.message}"


class NewsProviderNotConfiguredError(NewsDataError):
    """Raised when the news provider is not configured or missing settings."""


class NewsAuthenticationError(NewsDataError):
    """Raised when news provider authentication fails (e.g. invalid API key)."""


class NewsQueryError(NewsDataError):
    """Raised when news query parameters or date range are invalid."""


class NewsProviderUnavailableError(NewsDataError):
    """Raised when the news provider is unreachable due to network/timeout."""


class NewsProviderRateLimitError(NewsDataError):
    """Raised when the news provider explicitly rejects requests (HTTP 429)."""


class NewsMalformedDataError(NewsDataError):
    """Raised when news provider data is corrupted or violates constraints."""


class EmptyNewsDataError(NewsDataError):
    """Raised when provider returns zero articles for the requested query."""
