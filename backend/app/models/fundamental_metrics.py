"""Normalized domain models for calculated fundamental metrics and valuation indicators.

Phase 6.2 defines strongly typed models representing derived and provider-reported
fundamental metrics across growth, profitability, leverage, cash flow, and valuation.

Boundary Rules:
- Preserves full numeric precision without premature rounding.
- Explicit provenance tracking for valuation metrics ('provider' vs. 'derived').
- Dedicated zero-crossing flags for non-standard earnings/EPS transitions.
- Strictly keeps missing or undefined data as None (no zero fabrication).
"""

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

MetricSource = Literal["provider", "derived"]


class RevenueHistoryItem(BaseModel):
    """Historical annual revenue data point for trend analysis."""

    model_config = ConfigDict(extra="ignore")

    fiscal_year: Optional[int] = Field(
        default=None, description="Fiscal reporting year."
    )
    period_end_date: Optional[str] = Field(
        default=None, description="Period end date in ISO format (YYYY-MM-DD)."
    )
    total_revenue: Optional[float] = Field(
        default=None, description="Total revenue reported for period."
    )
    yoy_growth: Optional[float] = Field(
        default=None, description="Year-over-year revenue growth rate."
    )


class GrowthMetrics(BaseModel):
    """Annual growth metrics with explicit zero-crossing semantics."""

    model_config = ConfigDict(extra="ignore")

    revenue_growth_yoy: Optional[float] = Field(
        default=None, description="Latest YoY revenue growth rate."
    )
    net_income_growth_yoy: Optional[float] = Field(
        default=None, description="Latest YoY net income growth rate."
    )
    eps_growth_yoy: Optional[float] = Field(
        default=None, description="Latest YoY EPS growth rate."
    )
    revenue_cagr_3yr: Optional[float] = Field(
        default=None, description="3-year compound annual growth rate of revenue."
    )

    # Net Income Zero-Crossing Semantics
    earnings_turnaround: bool = Field(
        default=False,
        description="True if net income transitioned from loss (<0) to profit (>0).",
    )
    earnings_deficit_turnaround: bool = Field(
        default=False,
        description="True if net income transitioned from profit (>0) to loss (<0).",
    )
    both_periods_deficit: bool = Field(
        default=False,
        description="True if both current and prior periods suffered net losses (<0).",
    )

    # EPS Zero-Crossing Semantics
    eps_turnaround: bool = Field(
        default=False,
        description="True if EPS transitioned from loss (<0) to profit (>0).",
    )
    eps_deficit_turnaround: bool = Field(
        default=False,
        description="True if EPS transitioned from profit (>0) to loss (<0).",
    )
    both_periods_eps_deficit: bool = Field(
        default=False,
        description="True if both current and prior periods had negative EPS (<0).",
    )


class ProfitabilityMetrics(BaseModel):
    """Profitability margins and return on capital metrics."""

    model_config = ConfigDict(extra="ignore")

    operating_margin: Optional[float] = Field(
        default=None, description="Operating income divided by total revenue."
    )
    net_profit_margin: Optional[float] = Field(
        default=None, description="Net income divided by total revenue."
    )
    roe: Optional[float] = Field(
        default=None,
        description="Return on equity (net income / total equity).",
    )
    roa: Optional[float] = Field(
        default=None,
        description="Return on assets (net income divided by total assets).",
    )
    negative_equity: bool = Field(
        default=False,
        description="True if total stockholders' equity is zero or negative.",
    )


class LeverageMetrics(BaseModel):
    """Capital structure, debt position, and solvency metrics."""

    model_config = ConfigDict(extra="ignore")

    total_debt: Optional[float] = Field(
        default=None, description="Total reported debt (short-term + long-term)."
    )
    total_equity: Optional[float] = Field(
        default=None, description="Total stockholders' equity."
    )
    cash_and_equivalents: Optional[float] = Field(
        default=None, description="Cash and cash equivalents."
    )
    net_debt: Optional[float] = Field(
        default=None,
        description="Net debt position (total debt minus cash and equivalents).",
    )
    debt_to_equity: Optional[float] = Field(
        default=None, description="Total debt divided by total stockholders' equity."
    )
    is_net_cash_positive: Optional[bool] = Field(
        default=None,
        description="True if cash and equivalents exceed total debt (net debt < 0).",
    )
    negative_equity: bool = Field(
        default=False,
        description="True if stockholders' equity is zero or negative.",
    )


class CashFlowMetrics(BaseModel):
    """Cash generation, capital expenditures, and conversion metrics."""

    model_config = ConfigDict(extra="ignore")

    operating_cash_flow: Optional[float] = Field(
        default=None, description="Net cash flow from operating activities."
    )
    capital_expenditures: Optional[float] = Field(
        default=None,
        description="Capital expenditures (standardized magnitude).",
    )
    free_cash_flow: Optional[float] = Field(
        default=None,
        description="Free cash flow (OCF minus normalized CapEx, or reported).",
    )
    fcf_conversion_ratio: Optional[float] = Field(
        default=None,
        description="Free cash flow divided by net income.",
    )


class ValuationMetrics(BaseModel):
    """Market valuation multiples with explicit provenance tracking."""

    model_config = ConfigDict(extra="ignore")

    market_cap: Optional[float] = Field(
        default=None, description="Total market capitalization."
    )
    pe_ratio: Optional[float] = Field(
        default=None, description="Price to earnings multiple."
    )
    pe_ratio_source: Optional[MetricSource] = Field(
        default=None,
        description="Provenance of P/E: 'provider' if reported, 'derived' if computed.",
    )
    forward_pe: Optional[float] = Field(
        default=None, description="Provider-reported forward P/E ratio."
    )
    price_to_book: Optional[float] = Field(
        default=None, description="Price to book value multiple."
    )
    price_to_book_source: Optional[MetricSource] = Field(
        default=None,
        description="Provenance of P/B: 'provider' if reported, 'derived' if computed.",
    )
    dividend_yield: Optional[float] = Field(
        default=None, description="Provider-reported dividend yield."
    )
    trailing_eps: Optional[float] = Field(
        default=None,
        description="Provider-reported trailing twelve month EPS.",
    )


class FundamentalMetrics(BaseModel):
    """Container for calculated and normalized fundamental metrics for a security."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(..., description="Normalized uppercase ticker symbol.")
    company_name: Optional[str] = Field(
        default=None, description="Registered company name."
    )
    currency: Optional[str] = Field(
        default=None, description="Reporting currency (e.g. USD)."
    )
    period_end_date: Optional[str] = Field(
        default=None,
        description="Period end date of the evaluated annual period (YYYY-MM-DD).",
    )
    fiscal_year: Optional[int] = Field(
        default=None, description="Fiscal year of evaluated period."
    )
    calculated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when metrics were computed.",
    )

    growth: GrowthMetrics = Field(
        default_factory=GrowthMetrics, description="Annual growth and trend metrics."
    )
    profitability: ProfitabilityMetrics = Field(
        default_factory=ProfitabilityMetrics,
        description="Profit margins and return metrics.",
    )
    leverage: LeverageMetrics = Field(
        default_factory=LeverageMetrics,
        description="Debt position and capital structure metrics.",
    )
    cash_flow: CashFlowMetrics = Field(
        default_factory=CashFlowMetrics,
        description="Cash generation and free cash flow metrics.",
    )
    valuation: ValuationMetrics = Field(
        default_factory=ValuationMetrics,
        description="Valuation ratios with provenance indicators.",
    )
    revenue_history: List[RevenueHistoryItem] = Field(
        default_factory=list,
        description="Chronological annual revenue history.",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate and uppercase ticker symbol."""
        if not v or not v.strip():
            raise ValueError("Ticker symbol cannot be empty or whitespace.")
        return v.strip().upper()
