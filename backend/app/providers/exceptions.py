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
