"""Normalized domain models for financial statement and company fundamentals.

Phase 6.1 defines the strongly typed data models capturing raw/underlying company
financial information from external data providers.

Boundary Rules:
- All financial metrics represent provider-reported data and underlying statement items.
- No derived ratios, growth calculations, profitability metrics, or fundamental scoring
  are implemented here (reserved for Phase 6.2).
- Missing/unavailable values are strictly preserved as None; values are never defaulted
  to 0.0 or fabricated.
"""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CompanyProfile(BaseModel):
    """Company profile and security identification metadata.

    Attributes:
        ticker: Uppercase ticker symbol (e.g. 'AAPL').
        company_name: Full registered company name.
        currency: Reporting currency ISO code (e.g. 'USD').
        exchange: Exchange code/name (e.g. 'NASDAQ').
        sector: Economic sector classification.
        industry: Industry group classification.
        country: Domicile country name or ISO code.
        website: Official corporate website URL.
        business_summary: Summary of company operations.
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(..., description="Normalized ticker symbol.")
    company_name: Optional[str] = Field(
        default=None, description="Registered company name."
    )
    currency: Optional[str] = Field(
        default=None, description="Reporting currency (e.g. USD)."
    )
    exchange: Optional[str] = Field(default=None, description="Listing exchange code.")
    sector: Optional[str] = Field(default=None, description="Sector classification.")
    industry: Optional[str] = Field(
        default=None, description="Industry classification."
    )
    country: Optional[str] = Field(default=None, description="Country of domicile.")
    website: Optional[str] = Field(default=None, description="Company website URL.")
    business_summary: Optional[str] = Field(
        default=None, description="Summary of business operations."
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate and normalize ticker symbol."""
        if not v or not v.strip():
            raise ValueError("Ticker symbol cannot be empty or whitespace.")
        return v.strip().upper()


class IncomeStatementPeriod(BaseModel):
    """Periodic income statement line items reported by data provider.

    Attributes:
        fiscal_year: Fiscal year for the period (e.g. 2023).
        period_end_date: Period ending date in ISO format (YYYY-MM-DD).
        total_revenue: Total revenue/turnover for the period.
        operating_income: Operating income / EBIT.
        net_income: Net income attributable to common shareholders.
        eps: Provider-reported earnings per share (basic or diluted).
    """

    model_config = ConfigDict(extra="ignore")

    fiscal_year: Optional[int] = Field(
        default=None, description="Fiscal year of reporting period."
    )
    period_end_date: Optional[str] = Field(
        default=None, description="Period end date (YYYY-MM-DD)."
    )
    total_revenue: Optional[float] = Field(
        default=None, description="Total revenue reported for period."
    )
    operating_income: Optional[float] = Field(
        default=None, description="Operating income reported for period."
    )
    net_income: Optional[float] = Field(
        default=None, description="Net income reported for period."
    )
    eps: Optional[float] = Field(
        default=None, description="Raw provider-reported EPS for period."
    )


class BalanceSheetPeriod(BaseModel):
    """Periodic balance sheet line items reported by data provider.

    Attributes:
        fiscal_year: Fiscal year for the period (e.g. 2023).
        period_end_date: Period ending date in ISO format (YYYY-MM-DD).
        total_assets: Total assets at period end.
        total_liabilities: Total liabilities at period end.
        total_debt: Total debt (short-term + long-term) at period end.
        total_equity: Total stockholders' equity at period end.
        cash_and_cash_equivalents: Cash and cash equivalents at period end.
    """

    model_config = ConfigDict(extra="ignore")

    fiscal_year: Optional[int] = Field(
        default=None, description="Fiscal year of reporting period."
    )
    period_end_date: Optional[str] = Field(
        default=None, description="Period end date (YYYY-MM-DD)."
    )
    total_assets: Optional[float] = Field(
        default=None, description="Total assets at period end."
    )
    total_liabilities: Optional[float] = Field(
        default=None, description="Total liabilities at period end."
    )
    total_debt: Optional[float] = Field(
        default=None, description="Total debt at period end."
    )
    total_equity: Optional[float] = Field(
        default=None, description="Total stockholders' equity at period end."
    )
    cash_and_cash_equivalents: Optional[float] = Field(
        default=None, description="Cash and equivalents at period end."
    )


class CashFlowPeriod(BaseModel):
    """Periodic cash flow statement line items reported by data provider.

    Attributes:
        fiscal_year: Fiscal year for the period (e.g. 2023).
        period_end_date: Period ending date in ISO format (YYYY-MM-DD).
        operating_cash_flow: Net cash from operating activities.
        capital_expenditures: Capital expenditure outflows.
        free_cash_flow: Provider-reported free cash flow (if reported).
    """

    model_config = ConfigDict(extra="ignore")

    fiscal_year: Optional[int] = Field(
        default=None, description="Fiscal year of reporting period."
    )
    period_end_date: Optional[str] = Field(
        default=None, description="Period end date (YYYY-MM-DD)."
    )
    operating_cash_flow: Optional[float] = Field(
        default=None, description="Net cash provided by operating activities."
    )
    capital_expenditures: Optional[float] = Field(
        default=None, description="Capital expenditures for period."
    )
    free_cash_flow: Optional[float] = Field(
        default=None, description="Provider-reported free cash flow."
    )


class ProviderRawSnapshot(BaseModel):
    """Raw snapshot market and valuation fields as directly reported by provider.

    Note:
        These represent raw external data points preserved directly from the provider.
        No client-side ratio derivations, normalization, or calculations are performed
        in Phase 6.1.
    """

    model_config = ConfigDict(extra="ignore")

    market_cap: Optional[float] = Field(
        default=None, description="Reported total market capitalization."
    )
    trailing_pe: Optional[float] = Field(
        default=None, description="Reported trailing price-to-earnings ratio."
    )
    forward_pe: Optional[float] = Field(
        default=None, description="Reported forward price-to-earnings ratio."
    )
    price_to_book: Optional[float] = Field(
        default=None, description="Reported price-to-book ratio."
    )
    debt_to_equity: Optional[float] = Field(
        default=None, description="Reported debt-to-equity ratio."
    )
    dividend_yield: Optional[float] = Field(
        default=None, description="Reported dividend yield."
    )
    trailing_eps: Optional[float] = Field(
        default=None, description="Reported trailing twelve month EPS."
    )
    forward_eps: Optional[float] = Field(
        default=None, description="Reported forward EPS."
    )
    shares_outstanding: Optional[float] = Field(
        default=None, description="Reported number of shares outstanding."
    )
    float_shares: Optional[float] = Field(
        default=None, description="Reported float shares."
    )


class CompanyFundamentals(BaseModel):
    """Complete normalized financial fundamentals container for a security.

    Attributes:
        ticker: Uppercase ticker symbol.
        profile: CompanyProfile metadata.
        income_statements: Periodic income statements (sorted chronologically).
        balance_sheets: Periodic balance sheets (sorted chronologically).
        cash_flow_statements: Periodic cash flow statements (sorted chronologically).
        raw_snapshot: Raw provider-reported market snapshot.
        provider: Provider identifier (e.g. 'yahoo').
        retrieved_at: UTC timestamp when data was fetched and normalized.
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(..., description="Security ticker symbol.")
    profile: CompanyProfile = Field(
        ..., description="Company profile and classification metadata."
    )
    income_statements: List[IncomeStatementPeriod] = Field(
        default_factory=list,
        description="Historical periodic income statements.",
    )
    balance_sheets: List[BalanceSheetPeriod] = Field(
        default_factory=list,
        description="Historical periodic balance sheets.",
    )
    cash_flow_statements: List[CashFlowPeriod] = Field(
        default_factory=list,
        description="Historical periodic cash flow statements.",
    )
    raw_snapshot: Optional[ProviderRawSnapshot] = Field(
        default=None,
        description="Raw provider-reported market snapshot.",
    )
    provider: str = Field(
        default="yahoo",
        description="Source financial data provider identifier.",
    )
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of data retrieval.",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate and normalize ticker symbol."""
        if not v or not v.strip():
            raise ValueError("Ticker symbol cannot be empty or whitespace.")
        return v.strip().upper()
