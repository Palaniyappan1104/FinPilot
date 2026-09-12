"""Unit tests for Phase 6.2 Fundamental Metrics calculation engine.

Verifies:
- Complete metric calculations across multi-year data.
- Full IEEE-754 precision preservation without premature rounding.
- Explicit valuation metric provenance ('provider' vs. 'derived').
- CapEx sign normalization for Free Cash Flow (|CapEx|).
- Zero-crossing semantics for net income and EPS.
- Negative equity detection and ROE protection.
- Zero-division safety and missing data handling.
- 3-Year CAGR calculation across sufficient and insufficient periods.
- Immutability of input CompanyFundamentals object.
"""

import math

import pytest

from app.models.financial_data import (
    BalanceSheetPeriod,
    CashFlowPeriod,
    CompanyFundamentals,
    CompanyProfile,
    IncomeStatementPeriod,
    ProviderRawSnapshot,
)
from app.models.fundamental_metrics import FundamentalMetrics
from app.services.fundamental_metrics import (
    calculate_cagr,
    calculate_fundamental_metrics,
    calculate_growth_with_zero_crossing,
    safe_divide,
)


@pytest.fixture
def sample_company_fundamentals() -> CompanyFundamentals:
    """Multi-year company fundamentals dataset mirroring Apple Inc."""
    profile = CompanyProfile(
        ticker="AAPL",
        company_name="Apple Inc.",
        currency="USD",
        sector="Technology",
        industry="Consumer Electronics",
    )

    income_stmts = [
        IncomeStatementPeriod(
            fiscal_year=2020,
            period_end_date="2020-09-26",
            total_revenue=274515000000.0,
            operating_income=66288000000.0,
            net_income=57411000000.0,
            eps=3.28,
        ),
        IncomeStatementPeriod(
            fiscal_year=2021,
            period_end_date="2021-09-25",
            total_revenue=365817000000.0,
            operating_income=108949000000.0,
            net_income=94680000000.0,
            eps=5.61,
        ),
        IncomeStatementPeriod(
            fiscal_year=2022,
            period_end_date="2022-09-24",
            total_revenue=394328000000.0,
            operating_income=119437000000.0,
            net_income=99803000000.0,
            eps=6.11,
        ),
        IncomeStatementPeriod(
            fiscal_year=2023,
            period_end_date="2023-09-30",
            total_revenue=383285000000.0,
            operating_income=114301000000.0,
            net_income=96995000000.0,
            eps=6.13,
        ),
    ]

    balance_sheets = [
        BalanceSheetPeriod(
            fiscal_year=2022,
            period_end_date="2022-09-24",
            total_assets=352755000000.0,
            total_liabilities=302083000000.0,
            total_debt=120069000000.0,
            total_equity=50672000000.0,
            cash_and_cash_equivalents=23646000000.0,
        ),
        BalanceSheetPeriod(
            fiscal_year=2023,
            period_end_date="2023-09-30",
            total_assets=352583000000.0,
            total_liabilities=290437000000.0,
            total_debt=111088000000.0,
            total_equity=62146000000.0,
            cash_and_cash_equivalents=29965000000.0,
        ),
    ]

    cash_flows = [
        CashFlowPeriod(
            fiscal_year=2022,
            period_end_date="2022-09-24",
            operating_cash_flow=122151000000.0,
            capital_expenditures=-10708000000.0,
            free_cash_flow=111443000000.0,
        ),
        CashFlowPeriod(
            fiscal_year=2023,
            period_end_date="2023-09-30",
            operating_cash_flow=110543000000.0,
            capital_expenditures=-10959000000.0,
            free_cash_flow=99584000000.0,
        ),
    ]

    raw_snapshot = ProviderRawSnapshot(
        market_cap=2800000000000.0,
        trailing_pe=28.87,
        forward_pe=25.50,
        price_to_book=45.05,
        debt_to_equity=1.7875,
        dividend_yield=0.0055,
        trailing_eps=6.13,
        forward_eps=6.90,
        shares_outstanding=15500000000.0,
    )

    return CompanyFundamentals(
        ticker="AAPL",
        profile=profile,
        income_statements=income_stmts,
        balance_sheets=balance_sheets,
        cash_flow_statements=cash_flows,
        raw_snapshot=raw_snapshot,
        provider="yahoo",
    )


def test_safe_divide_and_cagr_helpers():
    """Verify safe division and CAGR helpers handle edge cases correctly."""
    assert safe_divide(10.0, 2.0) == 5.0
    assert safe_divide(None, 2.0) is None
    assert safe_divide(10.0, None) is None
    assert safe_divide(10.0, 0.0) is None
    assert safe_divide(0.0, 10.0) == 0.0

    # CAGR
    assert calculate_cagr(1331.0, 1000.0, periods=3) == pytest.approx(0.1, rel=1e-5)
    assert calculate_cagr(None, 1000.0, periods=3) is None
    assert calculate_cagr(1000.0, 0.0, periods=3) is None
    assert calculate_cagr(-1000.0, 1000.0, periods=3) is None
    assert calculate_cagr(1000.0, 1000.0, periods=0) is None


def test_normal_multi_year_calculations(sample_company_fundamentals):
    """Verify comprehensive metrics calculation for healthy multi-year company."""
    metrics = calculate_fundamental_metrics(sample_company_fundamentals)
    assert isinstance(metrics, FundamentalMetrics)
    assert metrics.ticker == "AAPL"
    assert metrics.period_end_date == "2023-09-30"
    assert metrics.fiscal_year == 2023

    # Growth
    expected_rev_growth = (383285000000.0 - 394328000000.0) / 394328000000.0
    assert metrics.growth.revenue_growth_yoy == pytest.approx(
        expected_rev_growth, rel=1e-6
    )

    expected_ni_growth = (96995000000.0 - 99803000000.0) / 99803000000.0
    assert metrics.growth.net_income_growth_yoy == pytest.approx(
        expected_ni_growth, rel=1e-6
    )

    expected_eps_growth = (6.13 - 6.11) / 6.11
    assert metrics.growth.eps_growth_yoy == pytest.approx(expected_eps_growth, rel=1e-6)

    # 3-Year CAGR (2020 to 2023)
    expected_cagr = (383285000000.0 / 274515000000.0) ** (1.0 / 3.0) - 1.0
    assert metrics.growth.revenue_cagr_3yr == pytest.approx(expected_cagr, rel=1e-6)

    # Profitability
    expected_op_margin = 114301000000.0 / 383285000000.0
    assert metrics.profitability.operating_margin == pytest.approx(
        expected_op_margin, rel=1e-6
    )

    expected_net_margin = 96995000000.0 / 383285000000.0
    assert metrics.profitability.net_profit_margin == pytest.approx(
        expected_net_margin, rel=1e-6
    )

    expected_roe = 96995000000.0 / 62146000000.0
    assert metrics.profitability.roe == pytest.approx(expected_roe, rel=1e-6)
    assert not metrics.profitability.negative_equity

    # Leverage
    assert metrics.leverage.total_debt == 111088000000.0
    assert metrics.leverage.cash_and_equivalents == 29965000000.0
    assert metrics.leverage.net_debt == 111088000000.0 - 29965000000.0
    assert metrics.leverage.is_net_cash_positive is False
    assert metrics.leverage.debt_to_equity == pytest.approx(
        111088000000.0 / 62146000000.0, rel=1e-6
    )

    # Cash Flow
    assert metrics.cash_flow.operating_cash_flow == 110543000000.0
    assert metrics.cash_flow.free_cash_flow == 99584000000.0
    assert metrics.cash_flow.fcf_conversion_ratio == pytest.approx(
        99584000000.0 / 96995000000.0, rel=1e-6
    )

    # Valuation Provenance (Provider)
    assert metrics.valuation.pe_ratio == 28.87
    assert metrics.valuation.pe_ratio_source == "provider"
    assert metrics.valuation.price_to_book == 45.05
    assert metrics.valuation.price_to_book_source == "provider"
    assert metrics.valuation.trailing_eps == 6.13

    # Revenue History
    assert len(metrics.revenue_history) == 4
    assert metrics.revenue_history[0].fiscal_year == 2020
    assert metrics.revenue_history[0].yoy_growth is None
    assert metrics.revenue_history[1].yoy_growth is not None


def test_full_precision_maintained():
    """Verify calculations preserve full IEEE-754 precision without rounding."""
    fundamentals = CompanyFundamentals(
        ticker="PREC",
        profile=CompanyProfile(ticker="PREC"),
        income_statements=[
            IncomeStatementPeriod(
                fiscal_year=2022,
                period_end_date="2022-12-31",
                total_revenue=100.0,
                net_income=1.0,
            ),
            IncomeStatementPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                total_revenue=103.0,
                net_income=1.0,
            ),
        ],
        balance_sheets=[
            BalanceSheetPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                total_equity=3.0,
            )
        ],
    )
    metrics = calculate_fundamental_metrics(fundamentals)

    # (103 - 100) / 100 = 0.03
    assert metrics.growth.revenue_growth_yoy == 0.03
    # 1.0 / 3.0 is 0.3333333333333333, must NOT be truncated to 0.3333
    assert metrics.profitability.roe == (1.0 / 3.0)
    assert str(metrics.profitability.roe).startswith("0.33333333333333")


def test_valuation_provenance_provider_vs_derived():
    """Verify provenance for provider-reported vs derived valuation multiples."""
    # Case A: Provider snapshot has trailing_pe and price_to_book
    fund_with_provider = CompanyFundamentals(
        ticker="PROV",
        profile=CompanyProfile(ticker="PROV"),
        raw_snapshot=ProviderRawSnapshot(
            trailing_pe=15.5,
            price_to_book=2.4,
            market_cap=1000000.0,
        ),
    )
    res_a = calculate_fundamental_metrics(fund_with_provider)
    assert res_a.valuation.pe_ratio == 15.5
    assert res_a.valuation.pe_ratio_source == "provider"
    assert res_a.valuation.price_to_book == 2.4
    assert res_a.valuation.price_to_book_source == "provider"

    # Case B: Provider snapshot lacks P/E and P/B, derived from inputs
    fund_derived = CompanyFundamentals(
        ticker="DERIV",
        profile=CompanyProfile(ticker="DERIV"),
        income_statements=[
            IncomeStatementPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                net_income=50000.0,
            )
        ],
        balance_sheets=[
            BalanceSheetPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                total_equity=200000.0,
            )
        ],
        raw_snapshot=ProviderRawSnapshot(
            trailing_pe=None,
            price_to_book=None,
            market_cap=1000000.0,
        ),
    )
    res_b = calculate_fundamental_metrics(fund_derived)
    assert res_b.valuation.pe_ratio == 1000000.0 / 50000.0  # 20.0
    assert res_b.valuation.pe_ratio_source == "derived"
    assert res_b.valuation.price_to_book == 1000000.0 / 200000.0  # 5.0
    assert res_b.valuation.price_to_book_source == "derived"

    # Case C: Neither provider nor inputs available
    fund_none = CompanyFundamentals(
        ticker="NONE",
        profile=CompanyProfile(ticker="NONE"),
        raw_snapshot=None,
    )
    res_c = calculate_fundamental_metrics(fund_none)
    assert res_c.valuation.pe_ratio is None
    assert res_c.valuation.pe_ratio_source is None
    assert res_c.valuation.price_to_book is None
    assert res_c.valuation.price_to_book_source is None


def test_capex_sign_conventions_negative_and_positive():
    """Verify CapEx normalization: OCF - |CapEx| handles -20 and +20 identically."""
    # Sub-case 1: Negative CapEx convention (-20)
    fund_neg = CompanyFundamentals(
        ticker="TEST",
        profile=CompanyProfile(ticker="TEST"),
        cash_flow_statements=[
            CashFlowPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                operating_cash_flow=100.0,
                capital_expenditures=-20.0,
                free_cash_flow=None,
            )
        ],
    )
    res_neg = calculate_fundamental_metrics(fund_neg)
    assert res_neg.cash_flow.free_cash_flow == 80.0
    assert res_neg.cash_flow.capital_expenditures == 20.0

    # Sub-case 2: Positive CapEx convention (+20)
    fund_pos = CompanyFundamentals(
        ticker="TEST",
        profile=CompanyProfile(ticker="TEST"),
        cash_flow_statements=[
            CashFlowPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                operating_cash_flow=100.0,
                capital_expenditures=20.0,
                free_cash_flow=None,
            )
        ],
    )
    res_pos = calculate_fundamental_metrics(fund_pos)
    assert res_pos.cash_flow.free_cash_flow == 80.0
    assert res_pos.cash_flow.capital_expenditures == 20.0


def test_earnings_zero_crossing_semantics():
    """Verify zero-crossing classifications and flags for net income."""
    # 1. Turnaround: Loss to Profit
    rate, turnaround, def_turnaround, both_def = calculate_growth_with_zero_crossing(
        current=50.0, prior=-100.0
    )
    assert turnaround is True
    assert def_turnaround is False
    assert both_def is False
    # (50 - (-100)) / 100 = 1.5 (+150%)
    assert rate == 1.5

    # 2. Deficit Turnaround: Profit to Loss
    rate, turnaround, def_turnaround, both_def = calculate_growth_with_zero_crossing(
        current=-50.0, prior=100.0
    )
    assert turnaround is False
    assert def_turnaround is True
    assert both_def is False
    # (-50 - 100) / 100 = -1.5 (-150%)
    assert rate == -1.5

    # 3. Dual Deficit: Loss to Loss
    rate, turnaround, def_turnaround, both_def = calculate_growth_with_zero_crossing(
        current=-40.0, prior=-100.0
    )
    assert turnaround is False
    assert def_turnaround is False
    assert both_def is True
    # (-40 - (-100)) / 100 = 0.60 (Loss narrowed by 60%)
    assert rate == 0.6

    # 4. Normal Profit Growth: Profit to Profit
    rate, turnaround, def_turnaround, both_def = calculate_growth_with_zero_crossing(
        current=120.0, prior=100.0
    )
    assert turnaround is False
    assert def_turnaround is False
    assert both_def is False
    assert rate == 0.20


def test_eps_zero_crossing_semantics():
    """Verify zero-crossing classifications for EPS transitions."""
    fund = CompanyFundamentals(
        ticker="EPSZ",
        profile=CompanyProfile(ticker="EPSZ"),
        income_statements=[
            IncomeStatementPeriod(
                fiscal_year=2022,
                period_end_date="2022-12-31",
                eps=-1.50,
            ),
            IncomeStatementPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                eps=0.75,
            ),
        ],
    )
    res = calculate_fundamental_metrics(fund)
    assert res.growth.eps_turnaround is True
    assert res.growth.eps_deficit_turnaround is False
    assert res.growth.both_periods_eps_deficit is False
    assert res.growth.eps_growth_yoy == (0.75 - (-1.50)) / 1.50


def test_negative_equity_handling():
    """Verify negative equity prevents false positive ROE and flags distress."""
    fund = CompanyFundamentals(
        ticker="DIST",
        profile=CompanyProfile(ticker="DIST"),
        income_statements=[
            IncomeStatementPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                net_income=-1000000.0,
            )
        ],
        balance_sheets=[
            BalanceSheetPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                total_debt=5000000.0,
                total_equity=-2000000.0,  # Negative equity
            )
        ],
    )
    res = calculate_fundamental_metrics(fund)
    # Does not produce false positive -1M / -2M = +50%
    assert res.profitability.roe is None
    assert res.profitability.negative_equity is True
    assert res.leverage.debt_to_equity is None
    assert res.leverage.negative_equity is True


def test_single_period_company_growth_none():
    """Verify single-period history computes snapshot ratios while growth stays None."""
    fund = CompanyFundamentals(
        ticker="IPO",
        profile=CompanyProfile(ticker="IPO"),
        income_statements=[
            IncomeStatementPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                total_revenue=1000000.0,
                operating_income=200000.0,
                net_income=150000.0,
                eps=1.50,
            )
        ],
    )
    res = calculate_fundamental_metrics(fund)
    assert res.growth.revenue_growth_yoy is None
    assert res.growth.net_income_growth_yoy is None
    assert res.growth.eps_growth_yoy is None
    assert res.growth.revenue_cagr_3yr is None
    assert res.profitability.operating_margin == 0.20
    assert res.profitability.net_profit_margin == 0.15


def test_cagr_calculation_4_periods_vs_insufficient():
    """Verify 3-year CAGR requires at least 4 periods."""
    fund_3_periods = CompanyFundamentals(
        ticker="CAGR",
        profile=CompanyProfile(ticker="CAGR"),
        income_statements=[
            IncomeStatementPeriod(fiscal_year=2021, total_revenue=100.0),
            IncomeStatementPeriod(fiscal_year=2022, total_revenue=110.0),
            IncomeStatementPeriod(fiscal_year=2023, total_revenue=120.0),
        ],
    )
    res_3 = calculate_fundamental_metrics(fund_3_periods)
    assert res_3.growth.revenue_cagr_3yr is None

    fund_4_periods = CompanyFundamentals(
        ticker="CAGR",
        profile=CompanyProfile(ticker="CAGR"),
        income_statements=[
            IncomeStatementPeriod(fiscal_year=2020, total_revenue=100.0),
            IncomeStatementPeriod(fiscal_year=2021, total_revenue=110.0),
            IncomeStatementPeriod(fiscal_year=2022, total_revenue=121.0),
            IncomeStatementPeriod(fiscal_year=2023, total_revenue=133.1),
        ],
    )
    res_4 = calculate_fundamental_metrics(fund_4_periods)
    assert res_4.growth.revenue_cagr_3yr == pytest.approx(0.1, rel=1e-5)


def test_empty_statements_handled_gracefully():
    """Verify completely empty statements return structured metrics without crashing."""
    fund_empty = CompanyFundamentals(
        ticker="EMPTY",
        profile=CompanyProfile(ticker="EMPTY"),
        income_statements=[],
        balance_sheets=[],
        cash_flow_statements=[],
        raw_snapshot=None,
    )
    res = calculate_fundamental_metrics(fund_empty)
    assert res.ticker == "EMPTY"
    assert res.growth.revenue_growth_yoy is None
    assert res.profitability.operating_margin is None
    assert res.leverage.total_debt is None
    assert res.cash_flow.operating_cash_flow is None
    assert res.valuation.pe_ratio is None
    assert res.revenue_history == []


def test_zero_division_safety():
    """Verify zero division in inputs does not raise exceptions."""
    fund_zero = CompanyFundamentals(
        ticker="ZERO",
        profile=CompanyProfile(ticker="ZERO"),
        income_statements=[
            IncomeStatementPeriod(
                fiscal_year=2022,
                period_end_date="2022-12-31",
                total_revenue=0.0,
                net_income=0.0,
            ),
            IncomeStatementPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                total_revenue=0.0,
                operating_income=0.0,
                net_income=0.0,
            ),
        ],
        balance_sheets=[
            BalanceSheetPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                total_debt=0.0,
                total_equity=0.0,
            )
        ],
        cash_flow_statements=[
            CashFlowPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                operating_cash_flow=0.0,
                capital_expenditures=0.0,
            )
        ],
    )
    res = calculate_fundamental_metrics(fund_zero)
    assert res.growth.revenue_growth_yoy is None
    assert res.profitability.operating_margin is None
    assert res.profitability.net_profit_margin is None
    assert res.profitability.roe is None
    assert res.leverage.debt_to_equity is None
    assert res.cash_flow.fcf_conversion_ratio is None


def test_input_immutability(sample_company_fundamentals):
    """Verify calculator treats input CompanyFundamentals as strictly read-only."""
    # Record snapshots of statements and ticker
    orig_order = [s.fiscal_year for s in sample_company_fundamentals.income_statements]
    income_len = len(sample_company_fundamentals.income_statements)
    orig_rev = sample_company_fundamentals.income_statements[0].total_revenue

    _ = calculate_fundamental_metrics(sample_company_fundamentals)

    assert len(sample_company_fundamentals.income_statements) == income_len
    assert sample_company_fundamentals.income_statements[0].total_revenue == orig_rev
    assert [
        s.fiscal_year for s in sample_company_fundamentals.income_statements
    ] == orig_order


def test_trailing_eps_preservation():
    """Verify provider trailing_eps survives into ValuationMetrics exactly."""
    fund = CompanyFundamentals(
        ticker="TEPS",
        profile=CompanyProfile(ticker="TEPS"),
        raw_snapshot=ProviderRawSnapshot(
            trailing_eps=6.1287654321987,
        ),
    )
    res = calculate_fundamental_metrics(fund)
    assert res.valuation.trailing_eps == 6.1287654321987


def test_provider_fcf_precedence_over_derived():
    """Verify provider-reported FCF has priority over derived CapEx formula."""
    fund = CompanyFundamentals(
        ticker="PFCF",
        profile=CompanyProfile(ticker="PFCF"),
        cash_flow_statements=[
            CashFlowPeriod(
                fiscal_year=2023,
                period_end_date="2023-12-31",
                operating_cash_flow=100.0,
                capital_expenditures=-20.0,
                free_cash_flow=95.0,  # Provider reported (differs from 100 - 20 = 80)
            )
        ],
    )
    res = calculate_fundamental_metrics(fund)
    assert res.cash_flow.free_cash_flow == 95.0
    assert res.cash_flow.operating_cash_flow == 100.0
    assert res.cash_flow.capital_expenditures == 20.0


def test_out_of_order_income_statements_sorted_chronologically():
    """Verify income statements provided in reverse or shuffled order are sorted."""
    stmt_2020 = IncomeStatementPeriod(
        fiscal_year=2020,
        period_end_date="2020-12-31",
        total_revenue=100.0,
        net_income=10.0,
    )
    stmt_2021 = IncomeStatementPeriod(
        fiscal_year=2021,
        period_end_date="2021-12-31",
        total_revenue=110.0,
        net_income=12.0,
    )
    stmt_2022 = IncomeStatementPeriod(
        fiscal_year=2022,
        period_end_date="2022-12-31",
        total_revenue=120.0,
        net_income=15.0,
    )
    stmt_2023 = IncomeStatementPeriod(
        fiscal_year=2023,
        period_end_date="2023-12-31",
        total_revenue=135.0,
        net_income=20.0,
    )

    # Deliberately reversed order: 2023, 2021, 2020, 2022
    shuffled = [stmt_2023, stmt_2021, stmt_2020, stmt_2022]
    fund = CompanyFundamentals(
        ticker="SHUF",
        profile=CompanyProfile(ticker="SHUF"),
        income_statements=shuffled,
    )
    res = calculate_fundamental_metrics(fund)

    # Latest should be 2023, prior should be 2022
    assert res.fiscal_year == 2023
    assert res.period_end_date == "2023-12-31"
    # YoY = (135 - 120) / 120 = 0.125
    assert res.growth.revenue_growth_yoy == pytest.approx(0.125, rel=1e-6)
    # Revenue history must be in ascending chronological order: 2020, 2021, 2022, 2023
    assert [item.fiscal_year for item in res.revenue_history] == [
        2020,
        2021,
        2022,
        2023,
    ]
    # Input list must NOT have been mutated
    assert [s.fiscal_year for s in fund.income_statements] == [
        2023,
        2021,
        2020,
        2022,
    ]


def test_cagr_gap_returns_none():
    """Verify 3-year CAGR returns None when 4 statements have a gap."""
    # 4 statements: 2019, 2021, 2022, 2023 (missing 2020, span is 4 years, not 3)
    fund = CompanyFundamentals(
        ticker="GAP",
        profile=CompanyProfile(ticker="GAP"),
        income_statements=[
            IncomeStatementPeriod(fiscal_year=2019, total_revenue=100.0),
            IncomeStatementPeriod(fiscal_year=2021, total_revenue=110.0),
            IncomeStatementPeriod(fiscal_year=2022, total_revenue=120.0),
            IncomeStatementPeriod(fiscal_year=2023, total_revenue=133.1),
        ],
    )
    res = calculate_fundamental_metrics(fund)
    assert res.growth.revenue_cagr_3yr is None


def test_cagr_valid_annual_span():
    """Verify 3-year CAGR calculates correctly for 4 consecutive annual statements."""
    fund = CompanyFundamentals(
        ticker="CAGR",
        profile=CompanyProfile(ticker="CAGR"),
        income_statements=[
            IncomeStatementPeriod(fiscal_year=2020, total_revenue=1000.0),
            IncomeStatementPeriod(fiscal_year=2021, total_revenue=1100.0),
            IncomeStatementPeriod(fiscal_year=2022, total_revenue=1200.0),
            IncomeStatementPeriod(fiscal_year=2023, total_revenue=1331.0),
        ],
    )
    res = calculate_fundamental_metrics(fund)
    # (1331 / 1000) ** (1/3) - 1 = 0.10
    assert res.growth.revenue_cagr_3yr == pytest.approx(0.10, rel=1e-6)


def test_nan_and_inf_handling():
    """Verify NaN and Inf values safely evaluate to None across safe_divide and CAGR."""
    assert safe_divide(math.nan, 10.0) is None
    assert safe_divide(10.0, math.nan) is None
    assert safe_divide(math.inf, 10.0) is None
    assert safe_divide(10.0, math.inf) is None
    assert safe_divide(-math.inf, 10.0) is None
    assert safe_divide(10.0, -math.inf) is None

    # CAGR with NaN/Inf
    assert calculate_cagr(math.nan, 100.0, periods=3) is None
    assert calculate_cagr(100.0, math.nan, periods=3) is None
    assert calculate_cagr(math.inf, 100.0, periods=3) is None
    assert calculate_cagr(100.0, math.inf, periods=3) is None
