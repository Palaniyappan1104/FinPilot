"""Deterministic fundamental metrics calculation engine for FinPilot.

Phase 6.2 implements pure-Python financial metric calculations:
- Consumes CompanyFundamentals from Phase 6.1 as read-only input.
- Computes growth, profitability, leverage, cash flow, and valuation metrics.
- Preserves full float precision without premature rounding.
- Explicit valuation metric provenance ('provider' vs. 'derived').
- Rigorous zero-crossing and negative earnings/EPS flags.
- CapEx sign normalization for Free Cash Flow: FCF = OCF - |CapEx|.
- Strictly independent of LangGraph, LLMs, and external networks.
"""

import math
from typing import List, Optional, Tuple

from app.core.logging import get_logger
from app.models.financial_data import (
    BalanceSheetPeriod,
    CashFlowPeriod,
    CompanyFundamentals,
    IncomeStatementPeriod,
)
from app.models.fundamental_metrics import (
    CashFlowMetrics,
    FundamentalMetrics,
    GrowthMetrics,
    LeverageMetrics,
    MetricSource,
    ProfitabilityMetrics,
    RevenueHistoryItem,
    ValuationMetrics,
)

logger = get_logger("app.services.fundamental_metrics")


def safe_divide(
    numerator: Optional[float], denominator: Optional[float]
) -> Optional[float]:
    """Safely divide two numbers, returning None on zero division or missing data.

    Preserves full floating-point precision without premature rounding.
    """
    if numerator is None or denominator is None:
        return None
    try:
        num = float(numerator)
        denom = float(denominator)
        if (
            denom == 0.0
            or math.isnan(num)
            or math.isnan(denom)
            or math.isinf(num)
            or math.isinf(denom)
        ):
            return None
        res = num / denom
        if math.isnan(res) or math.isinf(res):
            return None
        return res
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def calculate_growth_with_zero_crossing(
    current: Optional[float], prior: Optional[float]
) -> Tuple[Optional[float], bool, bool, bool]:
    """Calculate YoY growth rate and classify earnings zero-crossing transitions.

    Returns:
        (growth_rate, turnaround, deficit_turnaround, both_deficit)
    """
    if current is None or prior is None:
        return None, False, False, False

    try:
        curr = float(current)
        prev = float(prior)
    except (ValueError, TypeError):
        return None, False, False, False

    if math.isnan(curr) or math.isnan(prev) or math.isinf(curr) or math.isinf(prev):
        return None, False, False, False

    if prev == 0.0:
        return None, False, False, False

    # Zero-crossing classifications
    turnaround = prev < 0.0 and curr > 0.0
    deficit_turnaround = prev > 0.0 and curr < 0.0
    both_deficit = prev < 0.0 and curr < 0.0

    growth = (curr - prev) / abs(prev)
    if math.isnan(growth) or math.isinf(growth):
        return None, turnaround, deficit_turnaround, both_deficit

    return growth, turnaround, deficit_turnaround, both_deficit


def calculate_cagr(
    end_val: Optional[float], start_val: Optional[float], periods: int
) -> Optional[float]:
    """Calculate Compound Annual Growth Rate across given number of periods.

    Requires positive baseline and positive end values.
    """
    if end_val is None or start_val is None or periods <= 0:
        return None
    try:
        end_f = float(end_val)
        start_f = float(start_val)
        if (
            start_f <= 0.0
            or end_f <= 0.0
            or math.isinf(start_f)
            or math.isinf(end_f)
            or math.isnan(start_f)
            or math.isnan(end_f)
        ):
            return None
        cagr = (end_f / start_f) ** (1.0 / periods) - 1.0
        if math.isnan(cagr) or math.isinf(cagr):
            return None
        return cagr
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def _statement_sort_key(stmt: IncomeStatementPeriod) -> Tuple[str, int]:
    """Chronological sort key prioritizing period_end_date with fiscal_year fallback."""
    date_str = stmt.period_end_date or ""
    year_val = stmt.fiscal_year if stmt.fiscal_year is not None else 0
    if not date_str and year_val:
        date_str = f"{year_val:04d}-12-31"
    return (date_str, year_val)


def _extract_year(stmt: IncomeStatementPeriod) -> Optional[int]:
    """Extract fiscal year or calendar year from period_end_date."""
    if stmt.fiscal_year is not None:
        return stmt.fiscal_year
    if stmt.period_end_date and len(stmt.period_end_date) >= 4:
        try:
            return int(stmt.period_end_date[:4])
        except (ValueError, TypeError):
            pass
    return None


def _find_aligned_balance_sheet(
    balance_sheets: List[BalanceSheetPeriod],
    target_date: Optional[str],
    target_year: Optional[int],
) -> Optional[BalanceSheetPeriod]:
    """Find balance sheet matching period_end_date or fiscal_year."""
    if not balance_sheets:
        return None

    if target_date:
        for bs in balance_sheets:
            if bs.period_end_date == target_date:
                return bs

    if target_year is not None:
        for bs in balance_sheets:
            if bs.fiscal_year == target_year:
                return bs

    # Fallback to the latest available balance sheet if date unaligned
    sorted_bs = sorted(
        balance_sheets,
        key=lambda b: (
            b.period_end_date
            or (f"{b.fiscal_year:04d}-12-31" if b.fiscal_year else ""),
            b.fiscal_year or 0,
        ),
    )
    return sorted_bs[-1]


def _find_aligned_cash_flow(
    cash_flow_statements: List[CashFlowPeriod],
    target_date: Optional[str],
    target_year: Optional[int],
) -> Optional[CashFlowPeriod]:
    """Find cash flow statement matching period_end_date or fiscal_year."""
    if not cash_flow_statements:
        return None

    if target_date:
        for cf in cash_flow_statements:
            if cf.period_end_date == target_date:
                return cf

    if target_year is not None:
        for cf in cash_flow_statements:
            if cf.fiscal_year == target_year:
                return cf

    # Fallback to the latest available cash flow statement
    sorted_cf = sorted(
        cash_flow_statements,
        key=lambda c: (
            c.period_end_date
            or (f"{c.fiscal_year:04d}-12-31" if c.fiscal_year else ""),
            c.fiscal_year or 0,
        ),
    )
    return sorted_cf[-1]


def calculate_fundamental_metrics(
    fundamentals: CompanyFundamentals,
) -> FundamentalMetrics:
    """Compute standardized fundamental metrics from normalized company data.

    Pure deterministic Python execution without LLM dependencies.
    Does not mutate the incoming CompanyFundamentals object.

    Args:
        fundamentals: Normalized CompanyFundamentals from Phase 6.1.

    Returns:
        FundamentalMetrics: Calculated growth, profitability, leverage, cash flow,
        and valuation metrics with provenance indicators.
    """
    ticker = fundamentals.ticker
    profile = fundamentals.profile
    raw_income = fundamentals.income_statements or []
    # Create chronologically ascending representation without mutating input
    income_stmts: List[IncomeStatementPeriod] = sorted(
        raw_income, key=_statement_sort_key
    )
    balance_sheets = fundamentals.balance_sheets or []
    cash_flows = fundamentals.cash_flow_statements or []
    snapshot = fundamentals.raw_snapshot

    # Identify the primary (latest) annual period from sorted income statements
    latest_income: Optional[IncomeStatementPeriod] = (
        income_stmts[-1] if income_stmts else None
    )
    prior_income: Optional[IncomeStatementPeriod] = None
    if latest_income is not None:
        latest_year = _extract_year(latest_income)
        if latest_year is not None:
            # Search for the annual observation exactly 1 fiscal year before
            for s in reversed(income_stmts[:-1]):
                if _extract_year(s) == latest_year - 1:
                    prior_income = s
                    break
        elif len(income_stmts) >= 2:
            prior_income = income_stmts[-2]

    baseline_date = latest_income.period_end_date if latest_income else None
    baseline_year = latest_income.fiscal_year if latest_income else None

    # Align balance sheet and cash flow to the latest annual period
    latest_bs = _find_aligned_balance_sheet(
        balance_sheets, baseline_date, baseline_year
    )
    latest_cf = _find_aligned_cash_flow(cash_flows, baseline_date, baseline_year)

    # -------------------------------------------------------------------------
    # 1. Revenue History & Growth Metrics (6.2.1 & 6.2.8)
    # -------------------------------------------------------------------------
    revenue_history: List[RevenueHistoryItem] = []
    for i, stmt in enumerate(income_stmts):
        yoy: Optional[float] = None
        prev_stmt = income_stmts[i - 1] if i > 0 else None
        if (
            prev_stmt is not None
            and prev_stmt.total_revenue is not None
            and stmt.total_revenue is not None
        ):
            stmt_y = _extract_year(stmt)
            prev_y = _extract_year(prev_stmt)
            if stmt_y is not None and prev_y is not None and (stmt_y - prev_y != 1):
                yoy = None
            else:
                prev_rev = prev_stmt.total_revenue
                if prev_rev is not None and prev_rev > 0:
                    yoy = safe_divide(stmt.total_revenue - prev_rev, prev_rev)

        revenue_history.append(
            RevenueHistoryItem(
                fiscal_year=stmt.fiscal_year,
                period_end_date=stmt.period_end_date,
                total_revenue=stmt.total_revenue,
                yoy_growth=yoy,
            )
        )

    # YoY Revenue Growth
    curr_rev = latest_income.total_revenue if latest_income else None
    prev_rev = prior_income.total_revenue if prior_income else None
    revenue_growth_yoy: Optional[float] = None
    if curr_rev is not None and prev_rev is not None and prev_rev > 0:
        revenue_growth_yoy = safe_divide(curr_rev - prev_rev, prev_rev)

    # 3-Year CAGR (requires at least 4 annual periods spanning exactly 3 years)
    revenue_cagr_3yr: Optional[float] = None
    if len(income_stmts) >= 4 and latest_income is not None:
        latest_year = _extract_year(latest_income)
        baseline_cagr_stmt: Optional[IncomeStatementPeriod] = None
        if latest_year is not None:
            target_baseline_year = latest_year - 3
            for s in income_stmts:
                if _extract_year(s) == target_baseline_year:
                    baseline_cagr_stmt = s
                    break

        if baseline_cagr_stmt is not None:
            base_year = _extract_year(baseline_cagr_stmt)
            if (
                latest_year is not None
                and base_year is not None
                and (latest_year - base_year == 3)
            ):
                rev_t = latest_income.total_revenue
                rev_t_3 = baseline_cagr_stmt.total_revenue
                revenue_cagr_3yr = calculate_cagr(rev_t, rev_t_3, periods=3)

    # -------------------------------------------------------------------------
    # 2. Net Income & Zero-Crossing Growth (6.2.2)
    # -------------------------------------------------------------------------
    curr_ni = latest_income.net_income if latest_income else None
    prev_ni = prior_income.net_income if prior_income else None
    (
        ni_growth,
        ni_turnaround,
        ni_deficit_turnaround,
        both_ni_deficit,
    ) = calculate_growth_with_zero_crossing(curr_ni, prev_ni)

    # -------------------------------------------------------------------------
    # 3. EPS & Zero-Crossing Growth (6.2.3)
    # -------------------------------------------------------------------------
    curr_eps = latest_income.eps if latest_income else None
    prev_eps = prior_income.eps if prior_income else None
    (
        eps_growth,
        eps_turnaround,
        eps_deficit_turnaround,
        both_eps_deficit,
    ) = calculate_growth_with_zero_crossing(curr_eps, prev_eps)

    growth = GrowthMetrics(
        revenue_growth_yoy=revenue_growth_yoy,
        net_income_growth_yoy=ni_growth,
        eps_growth_yoy=eps_growth,
        revenue_cagr_3yr=revenue_cagr_3yr,
        earnings_turnaround=ni_turnaround,
        earnings_deficit_turnaround=ni_deficit_turnaround,
        both_periods_deficit=both_ni_deficit,
        eps_turnaround=eps_turnaround,
        eps_deficit_turnaround=eps_deficit_turnaround,
        both_periods_eps_deficit=both_eps_deficit,
    )

    # -------------------------------------------------------------------------
    # 4. Profitability & Returns (6.2.5)
    # -------------------------------------------------------------------------
    curr_op_inc = latest_income.operating_income if latest_income else None
    curr_equity = latest_bs.total_equity if latest_bs else None
    curr_assets = latest_bs.total_assets if latest_bs else None

    op_margin: Optional[float] = None
    net_margin: Optional[float] = None
    if curr_rev is not None and curr_rev > 0:
        op_margin = safe_divide(curr_op_inc, curr_rev)
        net_margin = safe_divide(curr_ni, curr_rev)

    # Return on Equity & Negative Equity Flag
    roe: Optional[float] = None
    negative_equity = False
    if curr_equity is not None:
        if curr_equity <= 0:
            negative_equity = True
            roe = None
        else:
            roe = safe_divide(curr_ni, curr_equity)

    roa: Optional[float] = None
    if curr_assets is not None and curr_assets > 0:
        roa = safe_divide(curr_ni, curr_assets)

    profitability = ProfitabilityMetrics(
        operating_margin=op_margin,
        net_profit_margin=net_margin,
        roe=roe,
        roa=roa,
        negative_equity=negative_equity,
    )

    # -------------------------------------------------------------------------
    # 5. Leverage & Capital Structure (6.2.6)
    # -------------------------------------------------------------------------
    curr_debt = latest_bs.total_debt if latest_bs else None
    curr_cash = latest_bs.cash_and_cash_equivalents if latest_bs else None

    net_debt: Optional[float] = None
    is_net_cash: Optional[bool] = None
    if curr_debt is not None and curr_cash is not None:
        net_debt = curr_debt - curr_cash
        is_net_cash = net_debt < 0

    debt_to_equity: Optional[float] = None
    if curr_equity is not None and curr_equity > 0:
        debt_to_equity = safe_divide(curr_debt, curr_equity)
    elif curr_equity is not None and curr_equity <= 0:
        debt_to_equity = None

    leverage = LeverageMetrics(
        total_debt=curr_debt,
        total_equity=curr_equity,
        cash_and_equivalents=curr_cash,
        net_debt=net_debt,
        debt_to_equity=debt_to_equity,
        is_net_cash_positive=is_net_cash,
        negative_equity=negative_equity,
    )

    # -------------------------------------------------------------------------
    # 6. Cash Flow & Free Cash Flow (6.2.7)
    # -------------------------------------------------------------------------
    ocf = latest_cf.operating_cash_flow if latest_cf else None
    capex = latest_cf.capital_expenditures if latest_cf else None
    reported_fcf = latest_cf.free_cash_flow if latest_cf else None

    # Free Cash Flow Normalization
    free_cash_flow: Optional[float] = None
    if reported_fcf is not None:
        # Preserve provider reported FCF if available
        free_cash_flow = reported_fcf
    elif ocf is not None and capex is not None:
        # FCF = OCF - |CapEx| to normalize negative/positive outflow representation
        free_cash_flow = ocf - abs(capex)

    fcf_conversion: Optional[float] = None
    if free_cash_flow is not None and curr_ni is not None and curr_ni > 0:
        fcf_conversion = safe_divide(free_cash_flow, curr_ni)

    cash_flow = CashFlowMetrics(
        operating_cash_flow=ocf,
        capital_expenditures=abs(capex) if capex is not None else None,
        free_cash_flow=free_cash_flow,
        fcf_conversion_ratio=fcf_conversion,
    )

    # -------------------------------------------------------------------------
    # 7. Valuation Multiples & Provenance Tracking (6.2.4)
    # -------------------------------------------------------------------------
    market_cap = snapshot.market_cap if snapshot else None
    forward_pe = snapshot.forward_pe if snapshot else None
    div_yield = snapshot.dividend_yield if snapshot else None
    trailing_eps = snapshot.trailing_eps if snapshot else None

    # P/E Ratio Provenance
    pe_ratio: Optional[float] = None
    pe_source: Optional[MetricSource] = None
    if snapshot and snapshot.trailing_pe is not None:
        pe_ratio = snapshot.trailing_pe
        pe_source = "provider"
    elif market_cap is not None and curr_ni is not None and curr_ni > 0:
        derived_pe = safe_divide(market_cap, curr_ni)
        if derived_pe is not None:
            pe_ratio = derived_pe
            pe_source = "derived"

    # Price to Book Provenance
    price_to_book: Optional[float] = None
    pb_source: Optional[MetricSource] = None
    if snapshot and snapshot.price_to_book is not None:
        price_to_book = snapshot.price_to_book
        pb_source = "provider"
    elif market_cap is not None and curr_equity is not None and curr_equity > 0:
        derived_pb = safe_divide(market_cap, curr_equity)
        if derived_pb is not None:
            price_to_book = derived_pb
            pb_source = "derived"

    valuation = ValuationMetrics(
        market_cap=market_cap,
        pe_ratio=pe_ratio,
        pe_ratio_source=pe_source,
        forward_pe=forward_pe,
        price_to_book=price_to_book,
        price_to_book_source=pb_source,
        dividend_yield=div_yield,
        trailing_eps=trailing_eps,
    )

    return FundamentalMetrics(
        ticker=ticker,
        company_name=profile.company_name,
        currency=profile.currency,
        period_end_date=baseline_date,
        fiscal_year=baseline_year,
        growth=growth,
        profitability=profitability,
        leverage=leverage,
        cash_flow=cash_flow,
        valuation=valuation,
        revenue_history=revenue_history,
    )
