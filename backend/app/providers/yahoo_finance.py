"""Yahoo Finance Data Provider implementation for FinPilot.

Phase 6.1 implements Yahoo Finance via yfinance:
- Retrieves underlying company fundamentals, profile metadata, and statements.
- Preserves missing/unavailable fields as None (no zero fabrication or estimation).
- Strictly maintains provider boundaries (no yfinance/pandas DataFrame leak).
- Distinct typed exceptions for rate limiting, unavailable provider, and errors.
"""

import math
from typing import Any, List, Optional

import yfinance

from app.core.logging import get_logger
from app.models.financial_data import (
    BalanceSheetPeriod,
    CashFlowPeriod,
    CompanyFundamentals,
    CompanyProfile,
    IncomeStatementPeriod,
    ProviderRawSnapshot,
)
from app.providers.exceptions import (
    ProviderMalformedDataError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    TickerNotFoundError,
)
from app.providers.financial_data import FinancialDataProvider

logger = get_logger("app.providers.yahoo_finance")


def _safe_float(val: Any) -> Optional[float]:
    """Safely convert a value to float, preserving missing data as None.

    Never defaults to 0.0. Returns None on NaN, Inf, empty, or parsing failures.
    """
    if val is None:
        return None
    try:
        # Avoid treating collections or DataFrames as single floats
        if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
            if len(val) == 0:
                return None
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Safely convert a value to int, preserving missing data as None."""
    f = _safe_float(val)
    if f is None:
        return None
    try:
        return int(f)
    except (ValueError, TypeError, OverflowError):
        return None


def _extract_row_value(df: Any, col: Any, candidates: List[str]) -> Optional[float]:
    """Extract and parse a numeric line item from a statement DataFrame.

    Searches df.index case-insensitively across candidate row names.
    Returns None if row is not found or value is NaN/empty.
    """
    if df is None:
        return None
    try:
        if not hasattr(df, "index"):
            return None

        # Build lookup from lowercase string to original index label
        index_map = {str(idx).strip().lower(): idx for idx in df.index}

        for candidate in candidates:
            cand_lower = candidate.strip().lower()
            if cand_lower in index_map:
                actual_idx = index_map[cand_lower]
                val = df.loc[actual_idx, col]
                return _safe_float(val)
        return None
    except Exception:
        return None


def _is_empty_df(df: Any) -> bool:
    """Return True if DataFrame is None, empty, or lacks columns."""
    if df is None:
        return True
    try:
        if hasattr(df, "empty") and df.empty:
            return True
        if hasattr(df, "columns") and len(df.columns) == 0:
            return True
        return False
    except Exception:
        return False


class YahooFinanceProvider(FinancialDataProvider):
    """Financial data provider implementation backed by Yahoo Finance (yfinance)."""

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return "yahoo"

    def get_fundamentals(self, ticker: str) -> CompanyFundamentals:
        """Retrieve and normalize company fundamentals from Yahoo Finance.

        Args:
            ticker: Stock symbol (e.g. 'AAPL', 'MSFT').

        Returns:
            CompanyFundamentals: Strongly typed normalized domain model.

        Raises:
            TickerNotFoundError: If ticker is invalid or returns no data.
            ProviderUnavailableError: If Yahoo Finance is unreachable or times out.
            ProviderRateLimitError: If Yahoo Finance returns HTTP 429 rate limit.
            ProviderMalformedDataError: If provider returns unparseable structure.
            FinancialDataError: For unexpected operational failures.
        """
        if not ticker or not isinstance(ticker, str) or not ticker.strip():
            raise TickerNotFoundError(
                message="Ticker symbol cannot be empty or non-string.",
                provider=self.provider_name,
            )

        clean_ticker = ticker.strip().upper()
        logger.info("Fetching fundamentals for ticker: %s", clean_ticker)

        try:
            yf_ticker = yfinance.Ticker(clean_ticker)
            info = yf_ticker.info
            financials = yf_ticker.financials
            balance_sheet = yf_ticker.balance_sheet
            cashflow = yf_ticker.cashflow
        except Exception as exc:
            err_msg = str(exc)
            err_lower = err_msg.lower()
            if "429" in err_msg or "too many requests" in err_lower:
                logger.error(
                    "Rate limit error from Yahoo Finance for ticker %s: %s",
                    clean_ticker,
                    err_msg,
                )
                raise ProviderRateLimitError(
                    message=f"Rate limit exceeded for ticker '{clean_ticker}'.",
                    provider=self.provider_name,
                ) from exc

            logger.error(
                "Network/connection failure from Yahoo Finance for ticker %s: %s",
                clean_ticker,
                err_msg,
            )
            raise ProviderUnavailableError(
                message=(
                    f"Failed to connect to Yahoo Finance for ticker '{clean_ticker}'."
                ),
                provider=self.provider_name,
            ) from exc

        try:
            # Validate response content: verify that the ticker exists and has data
            has_valid_info = (
                bool(info)
                and isinstance(info, dict)
                and any(
                    info.get(k) is not None
                    for k in (
                        "regularMarketPrice",
                        "currentPrice",
                        "shortName",
                        "longName",
                        "sector",
                        "totalRevenue",
                        "marketCap",
                        "enterpriseValue",
                    )
                )
            )
            has_statement_data = not (
                _is_empty_df(financials)
                and _is_empty_df(balance_sheet)
                and _is_empty_df(cashflow)
            )

            # Check for empty or invalid symbol responses from Yahoo Finance
            if not has_valid_info and not has_statement_data:
                logger.warning("No fundamental data found for ticker: %s", clean_ticker)
                raise TickerNotFoundError(
                    message=(
                        f"Ticker '{clean_ticker}' not found or no financial data "
                        "available from Yahoo Finance."
                    ),
                    provider=self.provider_name,
                )

            profile = self._normalize_profile(clean_ticker, info)
            income_stmts = self._normalize_income_statements(financials)
            balance_sheets = self._normalize_balance_sheets(balance_sheet)
            cash_flows = self._normalize_cash_flow(cashflow)
            raw_snapshot = self._normalize_raw_snapshot(info)

            return CompanyFundamentals(
                ticker=clean_ticker,
                profile=profile,
                income_statements=income_stmts,
                balance_sheets=balance_sheets,
                cash_flow_statements=cash_flows,
                raw_snapshot=raw_snapshot,
                provider=self.provider_name,
            )
        except (TickerNotFoundError, ProviderRateLimitError, ProviderUnavailableError):
            raise
        except Exception as exc:
            logger.error(
                "Malformed or unparseable data from Yahoo Finance for ticker %s: %s",
                clean_ticker,
                str(exc),
            )
            raise ProviderMalformedDataError(
                message=(
                    f"Malformed financial data returned for ticker '{clean_ticker}': "
                    f"{exc}"
                ),
                provider=self.provider_name,
            ) from exc

    def _normalize_profile(self, ticker: str, info: Optional[dict]) -> CompanyProfile:
        """Extract and normalize CompanyProfile metadata from info dict."""
        safe_info = info if isinstance(info, dict) else {}
        company_name = safe_info.get("longName") or safe_info.get("shortName")

        return CompanyProfile(
            ticker=ticker,
            company_name=str(company_name).strip() if company_name else None,
            currency=safe_info.get("currency"),
            exchange=safe_info.get("exchange"),
            sector=safe_info.get("sector"),
            industry=safe_info.get("industry"),
            country=safe_info.get("country"),
            website=safe_info.get("website"),
            business_summary=safe_info.get("longBusinessSummary"),
        )

    def _normalize_income_statements(self, df: Any) -> List[IncomeStatementPeriod]:
        """Normalize periodic income statements from DataFrame."""
        if _is_empty_df(df):
            return []

        periods: List[IncomeStatementPeriod] = []
        for col in df.columns:
            period_date = self._format_period_date(col)
            fiscal_year = self._format_fiscal_year(col)

            total_revenue = _extract_row_value(
                df,
                col,
                ["Total Revenue", "Operating Revenue", "TotalRevenue", "Revenue"],
            )
            operating_income = _extract_row_value(
                df, col, ["Operating Income", "OperatingIncome", "EBIT"]
            )
            net_income = _extract_row_value(
                df,
                col,
                [
                    "Net Income",
                    "Net Income Common Stockholders",
                    "NetIncome",
                    "Net Income To Common",
                ],
            )
            eps = _extract_row_value(
                df, col, ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"]
            )

            periods.append(
                IncomeStatementPeriod(
                    fiscal_year=fiscal_year,
                    period_end_date=period_date,
                    total_revenue=total_revenue,
                    operating_income=operating_income,
                    net_income=net_income,
                    eps=eps,
                )
            )

        # Sort chronologically (oldest to newest)
        return sorted(periods, key=lambda p: p.period_end_date or "")

    def _normalize_balance_sheets(self, df: Any) -> List[BalanceSheetPeriod]:
        """Normalize periodic balance sheets from DataFrame."""
        if _is_empty_df(df):
            return []

        periods: List[BalanceSheetPeriod] = []
        for col in df.columns:
            period_date = self._format_period_date(col)
            fiscal_year = self._format_fiscal_year(col)

            total_assets = _extract_row_value(df, col, ["Total Assets", "TotalAssets"])
            total_liabilities = _extract_row_value(
                df,
                col,
                [
                    "Total Liabilities Net Minority Interest",
                    "Total Liabilities",
                    "TotalLiabilitiesNetMinorityInterest",
                ],
            )
            total_debt = _extract_row_value(df, col, ["Total Debt", "TotalDebt"])
            total_equity = _extract_row_value(
                df,
                col,
                [
                    "Stockholders Equity",
                    "Common Stock Equity",
                    "Total Stockholder Equity",
                    "StockholdersEquity",
                ],
            )
            cash = _extract_row_value(
                df,
                col,
                [
                    "Cash And Cash Equivalents",
                    "Cash Cash Equivalents And Short Term Investments",
                    "CashAndCashEquivalents",
                ],
            )

            periods.append(
                BalanceSheetPeriod(
                    fiscal_year=fiscal_year,
                    period_end_date=period_date,
                    total_assets=total_assets,
                    total_liabilities=total_liabilities,
                    total_debt=total_debt,
                    total_equity=total_equity,
                    cash_and_cash_equivalents=cash,
                )
            )

        return sorted(periods, key=lambda p: p.period_end_date or "")

    def _normalize_cash_flow(self, df: Any) -> List[CashFlowPeriod]:
        """Normalize periodic cash flow statements from DataFrame."""
        if _is_empty_df(df):
            return []

        periods: List[CashFlowPeriod] = []
        for col in df.columns:
            period_date = self._format_period_date(col)
            fiscal_year = self._format_fiscal_year(col)

            operating_cf = _extract_row_value(
                df,
                col,
                [
                    "Operating Cash Flow",
                    "Cash Flow From Continuing Operating Activities",
                    "OperatingCashFlow",
                ],
            )
            capex = _extract_row_value(
                df,
                col,
                [
                    "Capital Expenditure",
                    "Capital Expenditures",
                    "CapitalExpenditure",
                ],
            )
            free_cf = _extract_row_value(df, col, ["Free Cash Flow", "FreeCashFlow"])

            periods.append(
                CashFlowPeriod(
                    fiscal_year=fiscal_year,
                    period_end_date=period_date,
                    operating_cash_flow=operating_cf,
                    capital_expenditures=capex,
                    free_cash_flow=free_cf,
                )
            )

        return sorted(periods, key=lambda p: p.period_end_date or "")

    def _normalize_raw_snapshot(
        self, info: Optional[dict]
    ) -> Optional[ProviderRawSnapshot]:
        """Extract purely provider-reported snapshot figures without derivations."""
        if not isinstance(info, dict):
            return None

        return ProviderRawSnapshot(
            market_cap=_safe_float(info.get("marketCap")),
            trailing_pe=_safe_float(info.get("trailingPE")),
            forward_pe=_safe_float(info.get("forwardPE")),
            price_to_book=_safe_float(info.get("priceToBook")),
            debt_to_equity=_safe_float(info.get("debtToEquity")),
            dividend_yield=_safe_float(info.get("dividendYield")),
            trailing_eps=_safe_float(info.get("trailingEps")),
            forward_eps=_safe_float(info.get("forwardEps")),
            shares_outstanding=_safe_float(info.get("sharesOutstanding")),
            float_shares=_safe_float(info.get("floatShares")),
        )

    @staticmethod
    def _format_period_date(col: Any) -> Optional[str]:
        """Format column header date to ISO YYYY-MM-DD string."""
        if col is None:
            return None
        if hasattr(col, "strftime"):
            return col.strftime("%Y-%m-%d")
        col_str = str(col).strip()
        if len(col_str) >= 10:
            return col_str[:10]
        return col_str or None

    @staticmethod
    def _format_fiscal_year(col: Any) -> Optional[int]:
        """Format column header date to fiscal year int."""
        if col is None:
            return None
        if hasattr(col, "year"):
            return int(col.year)
        col_str = str(col).strip()
        if len(col_str) >= 4 and col_str[:4].isdigit():
            return int(col_str[:4])
        return None
