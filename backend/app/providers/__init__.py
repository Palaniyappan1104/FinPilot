"""Providers package for FinPilot.

Contains external data and service provider abstractions and concrete implementations.
"""

from app.providers.exceptions import (
    EmptyMarketDataError,
    EmptyNewsDataError,
    FinancialDataError,
    MarketDataError,
    MarketDataMalformedError,
    MarketDataProviderRateLimitError,
    MarketDataProviderUnavailableError,
    MarketDataTickerNotFoundError,
    NewsAuthenticationError,
    NewsDataError,
    NewsMalformedDataError,
    NewsProviderNotConfiguredError,
    NewsProviderRateLimitError,
    NewsProviderUnavailableError,
    NewsQueryError,
    ProviderMalformedDataError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    TickerNotFoundError,
)
from app.providers.financial_data import (
    FinancialDataProvider,
    get_financial_data_provider,
)
from app.providers.finnhub_news import (
    FinnhubNewsProvider,
)
from app.providers.market_data import (
    MarketDataProvider,
    get_market_data_provider,
)
from app.providers.news import (
    NewsProvider,
    get_news_provider,
)
from app.providers.yahoo_market_data import (
    YahooFinanceMarketDataProvider,
)

__all__ = [
    "EmptyMarketDataError",
    "EmptyNewsDataError",
    "FinancialDataError",
    "FinancialDataProvider",
    "FinnhubNewsProvider",
    "MarketDataError",
    "MarketDataMalformedError",
    "MarketDataProvider",
    "MarketDataProviderRateLimitError",
    "MarketDataProviderUnavailableError",
    "MarketDataTickerNotFoundError",
    "NewsAuthenticationError",
    "NewsDataError",
    "NewsMalformedDataError",
    "NewsProvider",
    "NewsProviderNotConfiguredError",
    "NewsProviderRateLimitError",
    "NewsProviderUnavailableError",
    "NewsQueryError",
    "ProviderMalformedDataError",
    "ProviderRateLimitError",
    "ProviderUnavailableError",
    "TickerNotFoundError",
    "YahooFinanceMarketDataProvider",
    "get_financial_data_provider",
    "get_market_data_provider",
    "get_news_provider",
]
