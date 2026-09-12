"""Unit tests for Financial Data Provider abstraction, models, and Yahoo provider.

Phase 6.1 test suite verifying:
- Successful retrieval and normalization of company data and periodic statements.
- Strict preservation of None for missing or NaN values (no zero fabrication).
- Distinguishing HTTP 429 rate limit errors from generic connection failures.
- Handling unknown/invalid tickers, empty responses, and malformed data.
- Factory behavior and configuration defaults.
- Domain models and boundary validation (no derived metrics in Phase 6.1).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.models.financial_data import (
    CompanyFundamentals,
    CompanyProfile,
    IncomeStatementPeriod,
    ProviderRawSnapshot,
)
from app.providers.exceptions import (
    FinancialDataError,
    ProviderMalformedDataError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    TickerNotFoundError,
)
from app.providers.financial_data import (
    FinancialDataProvider,
    get_financial_data_provider,
)
from app.providers.yahoo_finance import (
    YahooFinanceProvider,
    _extract_row_value,
    _safe_float,
    _safe_int,
)


@pytest.fixture
def mock_financials_df() -> pd.DataFrame:
    """Deterministic sample income statement DataFrame mirroring yfinance structure."""
    dates = [pd.Timestamp("2023-09-30"), pd.Timestamp("2022-09-24")]
    data = {
        dates[0]: {
            "Total Revenue": 383285000000.0,
            "Operating Income": 114301000000.0,
            "Net Income": 96995000000.0,
            "Diluted EPS": 6.13,
        },
        dates[1]: {
            "Total Revenue": 394328000000.0,
            "Operating Income": 119437000000.0,
            "Net Income": 99803000000.0,
            "Diluted EPS": 6.11,
        },
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_balance_sheet_df() -> pd.DataFrame:
    """Deterministic sample balance sheet DataFrame mirroring yfinance structure."""
    dates = [pd.Timestamp("2023-09-30"), pd.Timestamp("2022-09-24")]
    data = {
        dates[0]: {
            "Total Assets": 352583000000.0,
            "Total Liabilities Net Minority Interest": 290437000000.0,
            "Total Debt": 111088000000.0,
            "Stockholders Equity": 62146000000.0,
            "Cash And Cash Equivalents": 29965000000.0,
        },
        dates[1]: {
            "Total Assets": 352755000000.0,
            "Total Liabilities Net Minority Interest": 302083000000.0,
            "Total Debt": 120069000000.0,
            "Stockholders Equity": 50672000000.0,
            "Cash And Cash Equivalents": 23646000000.0,
        },
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_cashflow_df() -> pd.DataFrame:
    """Sample cash flow statement DataFrame mirroring yfinance structure."""
    dates = [pd.Timestamp("2023-09-30"), pd.Timestamp("2022-09-24")]
    data = {
        dates[0]: {
            "Operating Cash Flow": 110543000000.0,
            "Capital Expenditure": -10959000000.0,
            "Free Cash Flow": 99584000000.0,
        },
        dates[1]: {
            "Operating Cash Flow": 122151000000.0,
            "Capital Expenditure": -10708000000.0,
            "Free Cash Flow": 111443000000.0,
        },
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_info_dict() -> dict:
    """Deterministic sample info dictionary mirroring yfinance."""
    return {
        "symbol": "AAPL",
        "longName": "Apple Inc.",
        "currency": "USD",
        "exchange": "NMS",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "country": "United States",
        "website": "https://www.apple.com",
        "longBusinessSummary": (
            "Apple Inc. designs, manufactures, and markets smartphones."
        ),
        "regularMarketPrice": 180.5,
        "marketCap": 2800000000000.0,
        "trailingPE": 29.5,
        "forwardPE": 26.2,
        "priceToBook": 45.1,
        "debtToEquity": 178.7,
        "dividendYield": 0.0055,
        "trailingEps": 6.13,
        "forwardEps": 6.90,
        "sharesOutstanding": 15500000000.0,
        "floatShares": 15400000000.0,
    }


def test_safe_float_and_int_helpers():
    """Verify numeric helper functions strictly preserve None."""
    assert _safe_float(None) is None
    assert _safe_float(float("nan")) is None
    assert _safe_float(np.nan) is None
    assert _safe_float(float("inf")) is None
    assert _safe_float("invalid") is None
    assert _safe_float(123.45) == 123.45
    assert _safe_float("456.78") == 456.78
    assert _safe_float(0.0) == 0.0  # Legitimate zero is preserved

    assert _safe_int(None) is None
    assert _safe_int(float("nan")) is None
    assert _safe_int("abc") is None
    assert _safe_int(2023.0) == 2023
    assert _safe_int("2024") == 2024


def test_extract_row_value_helper():
    """Verify DataFrame row extraction handles missing candidates and NaN safely."""
    dates = [pd.Timestamp("2023-09-30")]
    df = pd.DataFrame(
        {
            dates[0]: {
                "Total Revenue": 1000.0,
                "Net Income": np.nan,
            }
        }
    )
    assert _extract_row_value(df, dates[0], ["Total Revenue"]) == 1000.0
    assert (
        _extract_row_value(df, dates[0], ["total revenue"]) == 1000.0
    )  # Case-insensitive
    assert (
        _extract_row_value(df, dates[0], ["Operating Revenue", "Total Revenue"])
        == 1000.0
    )
    assert _extract_row_value(df, dates[0], ["Net Income"]) is None  # NaN becomes None
    assert _extract_row_value(df, dates[0], ["Nonexistent Field"]) is None
    assert _extract_row_value(None, dates[0], ["Total Revenue"]) is None


def test_yahoo_provider_success(
    mock_info_dict, mock_financials_df, mock_balance_sheet_df, mock_cashflow_df
):
    """Verify successful retrieval and normalization of company data and statements."""
    provider = YahooFinanceProvider()
    assert provider.provider_name == "yahoo"
    assert isinstance(provider, FinancialDataProvider)

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = mock_info_dict
    mock_ticker_instance.financials = mock_financials_df
    mock_ticker_instance.balance_sheet = mock_balance_sheet_df
    mock_ticker_instance.cashflow = mock_cashflow_df

    with patch("yfinance.Ticker", return_value=mock_ticker_instance):
        fundamentals = provider.get_fundamentals("aapl")

    assert isinstance(fundamentals, CompanyFundamentals)
    assert fundamentals.ticker == "AAPL"  # Normalized to uppercase
    assert fundamentals.provider == "yahoo"

    # Profile
    assert fundamentals.profile.ticker == "AAPL"
    assert fundamentals.profile.company_name == "Apple Inc."
    assert fundamentals.profile.currency == "USD"
    assert fundamentals.profile.sector == "Technology"
    assert fundamentals.profile.industry == "Consumer Electronics"
    assert fundamentals.profile.country == "United States"

    # Statements sorted chronologically (2022 before 2023)
    assert len(fundamentals.income_statements) == 2
    stmt_2022 = fundamentals.income_statements[0]
    stmt_2023 = fundamentals.income_statements[1]
    assert stmt_2022.fiscal_year == 2022
    assert stmt_2022.period_end_date == "2022-09-24"
    assert stmt_2022.total_revenue == 394328000000.0
    assert stmt_2022.net_income == 99803000000.0
    assert stmt_2022.eps == 6.11

    assert stmt_2023.fiscal_year == 2023
    assert stmt_2023.period_end_date == "2023-09-30"
    assert stmt_2023.total_revenue == 383285000000.0
    assert stmt_2023.operating_income == 114301000000.0

    # Balance Sheet
    assert len(fundamentals.balance_sheets) == 2
    bs_2023 = fundamentals.balance_sheets[1]
    assert bs_2023.total_assets == 352583000000.0
    assert bs_2023.total_debt == 111088000000.0
    assert bs_2023.total_equity == 62146000000.0
    assert bs_2023.cash_and_cash_equivalents == 29965000000.0

    # Cash Flow
    assert len(fundamentals.cash_flow_statements) == 2
    cf_2023 = fundamentals.cash_flow_statements[1]
    assert cf_2023.operating_cash_flow == 110543000000.0
    assert cf_2023.capital_expenditures == -10959000000.0
    assert cf_2023.free_cash_flow == 99584000000.0

    # Raw Snapshot
    assert fundamentals.raw_snapshot is not None
    assert fundamentals.raw_snapshot.market_cap == 2800000000000.0
    assert fundamentals.raw_snapshot.trailing_pe == 29.5
    assert fundamentals.raw_snapshot.debt_to_equity == 178.7


def test_yahoo_provider_preserves_none_for_missing_or_nan():
    """Verify missing fields and NaN values are strictly preserved as None."""
    provider = YahooFinanceProvider()

    sparse_info = {
        "regularMarketPrice": 10.0,
        "shortName": "Sparse Co",
        # Missing marketCap, trailingPE, etc.
    }

    dates = [pd.Timestamp("2023-12-31")]
    sparse_financials = pd.DataFrame(
        {
            dates[0]: {
                "Total Revenue": 500000.0,
                "Operating Income": np.nan,  # NaN value
                # Net income row completely absent
            }
        }
    )

    mock_ticker = MagicMock()
    mock_ticker.info = sparse_info
    mock_ticker.financials = sparse_financials
    mock_ticker.balance_sheet = pd.DataFrame()
    mock_ticker.cashflow = pd.DataFrame()

    with patch("yfinance.Ticker", return_value=mock_ticker):
        fundamentals = provider.get_fundamentals("SPARSE")

    assert fundamentals.profile.sector is None
    assert fundamentals.profile.currency is None

    stmt = fundamentals.income_statements[0]
    assert stmt.total_revenue == 500000.0
    assert stmt.operating_income is None  # NaN preserved as None, not 0.0
    assert stmt.net_income is None  # Missing row preserved as None, not 0.0
    assert stmt.eps is None

    assert fundamentals.raw_snapshot is not None
    assert fundamentals.raw_snapshot.market_cap is None
    assert fundamentals.raw_snapshot.trailing_pe is None
    assert fundamentals.raw_snapshot.forward_pe is None


def test_yahoo_provider_unknown_ticker_raises_not_found():
    """Verify empty info and empty statements raise TickerNotFoundError."""
    provider = YahooFinanceProvider()

    mock_ticker = MagicMock()
    mock_ticker.info = {}
    mock_ticker.financials = pd.DataFrame()
    mock_ticker.balance_sheet = pd.DataFrame()
    mock_ticker.cashflow = pd.DataFrame()

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(TickerNotFoundError) as exc_info:
            provider.get_fundamentals("INVALIDTICKER123")

    assert "INVALIDTICKER123" in str(exc_info.value)
    assert exc_info.value.provider == "yahoo"


def test_yahoo_provider_empty_ticker_raises_not_found():
    """Verify empty/non-string ticker raises TickerNotFoundError."""
    provider = YahooFinanceProvider()

    with pytest.raises(TickerNotFoundError):
        provider.get_fundamentals("")

    with pytest.raises(TickerNotFoundError):
        provider.get_fundamentals("   ")

    with pytest.raises(TickerNotFoundError):
        provider.get_fundamentals(None)  # type: ignore[arg-type]


def test_yahoo_provider_rate_limit_distinguished():
    """Verify HTTP 429 specifically raises ProviderRateLimitError."""
    provider = YahooFinanceProvider()

    mock_ticker = MagicMock()
    mock_ticker.info = MagicMock(
        side_effect=Exception("HTTP Error 429: Too Many Requests")
    )

    with patch(
        "yfinance.Ticker", side_effect=Exception("HTTP Error 429: Too Many Requests")
    ):
        with pytest.raises(ProviderRateLimitError) as exc_info:
            provider.get_fundamentals("AAPL")

    assert "rate limit" in str(exc_info.value).lower()
    assert exc_info.value.provider == "yahoo"


def test_yahoo_provider_generic_network_error_raises_unavailable():
    """Verify generic connection/timeout raises ProviderUnavailableError."""
    provider = YahooFinanceProvider()

    with patch(
        "yfinance.Ticker", side_effect=Exception("Connection timed out after 30000ms")
    ):
        with pytest.raises(ProviderUnavailableError) as exc_info:
            provider.get_fundamentals("AAPL")

    assert "failed to connect" in str(exc_info.value).lower()
    assert not isinstance(exc_info.value, ProviderRateLimitError)
    assert exc_info.value.provider == "yahoo"


def test_yahoo_provider_malformed_data_raises_malformed():
    """Verify unparseable data structures raise ProviderMalformedDataError."""
    provider = YahooFinanceProvider()

    mock_ticker = MagicMock()
    # Provide valid info so existence check passes, but make financials unparseable
    mock_ticker.info = {"regularMarketPrice": 100.0, "shortName": "Test"}

    # Simulate an object whose columns iteration raises an unhandled error
    class BrokenDF:
        @property
        def empty(self):
            return False

        @property
        def columns(self):
            raise TypeError("Corrupted DataFrame columns structure")

    mock_ticker.financials = BrokenDF()
    mock_ticker.balance_sheet = pd.DataFrame()
    mock_ticker.cashflow = pd.DataFrame()

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(ProviderMalformedDataError) as exc_info:
            provider.get_fundamentals("TEST")

    assert "malformed" in str(exc_info.value).lower()
    assert exc_info.value.provider == "yahoo"


def test_yahoo_provider_empty_statements_handled_gracefully():
    """Verify empty statement DataFrames return empty lists without crashing."""
    provider = YahooFinanceProvider()

    mock_ticker = MagicMock()
    mock_ticker.info = {
        "regularMarketPrice": 50.0,
        "shortName": "New IPO Corp",
        "currency": "USD",
    }
    mock_ticker.financials = pd.DataFrame()
    mock_ticker.balance_sheet = pd.DataFrame()
    mock_ticker.cashflow = pd.DataFrame()

    with patch("yfinance.Ticker", return_value=mock_ticker):
        fundamentals = provider.get_fundamentals("IPO")

    assert fundamentals.ticker == "IPO"
    assert fundamentals.income_statements == []
    assert fundamentals.balance_sheets == []
    assert fundamentals.cash_flow_statements == []
    assert fundamentals.profile.company_name == "New IPO Corp"


def test_factory_default_yahoo():
    """Verify get_financial_data_provider returns YahooFinanceProvider by default."""
    provider = get_financial_data_provider()
    assert isinstance(provider, YahooFinanceProvider)
    assert provider.provider_name == "yahoo"


def test_factory_custom_settings():
    """Verify factory respects custom Settings."""
    settings = Settings(FINANCIAL_DATA_PROVIDER="yahoo")
    provider = get_financial_data_provider(settings=settings)
    assert isinstance(provider, YahooFinanceProvider)


def test_factory_unsupported_provider_raises():
    """Verify configuring unsupported provider raises FinancialDataError."""
    settings = Settings(FINANCIAL_DATA_PROVIDER="bloomberg")
    with pytest.raises(FinancialDataError) as exc_info:
        get_financial_data_provider(settings=settings)

    assert "Unsupported FINANCIAL_DATA_PROVIDER" in str(exc_info.value)


def test_factory_empty_provider_raises():
    """Verify empty provider configuration raises FinancialDataError."""
    settings = Settings(FINANCIAL_DATA_PROVIDER="")
    with pytest.raises(FinancialDataError) as exc_info:
        get_financial_data_provider(settings=settings)

    assert "cannot be empty" in str(exc_info.value)


def test_domain_models_no_derived_metrics():
    """Verify Phase 6.1 models omit derived/computed analytical ratios."""
    # Ensure no analytical metrics exist on CompanyFundamentals or statements
    fund_fields = CompanyFundamentals.model_fields.keys()
    assert "roe" not in fund_fields
    assert "pe_ratio" not in fund_fields
    assert "revenue_growth" not in fund_fields
    assert "earnings_growth" not in fund_fields
    assert "fundamental_score" not in fund_fields

    income_fields = IncomeStatementPeriod.model_fields.keys()
    assert "gross_margin" not in income_fields
    assert "net_margin" not in income_fields
    assert "revenue_growth" not in income_fields

    snapshot_fields = ProviderRawSnapshot.model_fields.keys()
    # Check that snapshot fields are explicitly documented as raw reported
    assert "market_cap" in snapshot_fields
    assert "trailing_pe" in snapshot_fields


def test_domain_models_validation():
    """Verify Pydantic validation on models."""
    with pytest.raises(ValidationError):
        CompanyProfile(ticker="")

    with pytest.raises(ValidationError):
        CompanyFundamentals(
            ticker="",
            profile=CompanyProfile(ticker="AAPL"),
        )

    # Valid model construction
    profile = CompanyProfile(ticker="aapl")
    assert profile.ticker == "AAPL"  # Normalized to uppercase
