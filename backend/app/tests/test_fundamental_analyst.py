"""Unit tests for Phase 6.3 Fundamental Analyst Agent.

All tests run 100% offline using deterministic mock providers.
ZERO real Gemini API or network calls are made.

Verifies:
- Grounded interpretation of deterministic FundamentalMetrics.
- Categorical fundamental score:
  ('favorable', 'neutral', 'unfavorable', 'insufficient_data').
- Dimension assessments with evidence-linked supporting_metrics.
- Zero-crossing turnaround interpretations (profit/loss transitions).
- Negative equity handling and ratio distortion safeguards.
- Provider vs derived valuation provenance interpretation.
- Trailing EPS preservation in narrative.
- Investor profile context invariance (metrics are never altered).
- Malformed output retry and error isolation via AgentResult.
- LangGraph node adapter contract in isolation.
"""

import json
from typing import Any, List, Optional

import pytest

from app.agents.fundamental import (
    FundamentalAnalystAgent,
    format_fundamental_prompt,
    fundamental_analyst_node,
    validate_fundamental_analysis_consistency,
)
from app.agents.fundamental_schema import (
    DimensionAssessment,
    FundamentalAnalysisOutput,
    FundamentalAnalystInput,
)
from app.agents.state import GraphState
from app.core.llm import (
    LLMError,
    LLMProvider,
    LLMResponse,
)
from app.models.financial_data import (
    BalanceSheetPeriod,
    CashFlowPeriod,
    CompanyFundamentals,
    CompanyProfile,
    IncomeStatementPeriod,
    ProviderRawSnapshot,
)
from app.models.fundamental_metrics import FundamentalMetrics
from app.services.fundamental_metrics import calculate_fundamental_metrics


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for Phase 6.3 Fundamental Analyst tests."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        fail_with: Optional[Exception] = None,
        name: str = "mock_fundamental_provider",
    ) -> None:
        self.responses = responses or []
        self.call_count = 0
        self.prompts_received: List[str] = []
        self.fail_with = fail_with
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        schema: Optional[Any] = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.prompts_received.append(prompt)

        if self.fail_with is not None:
            raise self.fail_with

        if not self.responses:
            raise RuntimeError("MockLLMProvider: No responses queued.")

        resp_idx = min(self.call_count - 1, len(self.responses) - 1)
        content = self.responses[resp_idx]

        return LLMResponse(
            content=content,
            provider=self._name,
            model="mock-model",
        )


@pytest.fixture
def apple_fundamental_metrics() -> FundamentalMetrics:
    """Pre-calculated healthy fundamental metrics mirroring Apple Inc."""
    profile = CompanyProfile(
        ticker="AAPL",
        company_name="Apple Inc.",
        currency="USD",
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
            fiscal_year=2023,
            period_end_date="2023-09-30",
            total_assets=352583000000.0,
            total_liabilities=290437000000.0,
            total_debt=111088000000.0,
            total_equity=62146000000.0,
            cash_and_cash_equivalents=29965000000.0,
        )
    ]
    cash_flows = [
        CashFlowPeriod(
            fiscal_year=2023,
            period_end_date="2023-09-30",
            operating_cash_flow=110543000000.0,
            capital_expenditures=-10959000000.0,
            free_cash_flow=99584000000.0,
        )
    ]
    snapshot = ProviderRawSnapshot(
        market_cap=2800000000000.0,
        trailing_pe=28.87,
        forward_pe=25.50,
        price_to_book=45.05,
        dividend_yield=0.0055,
        trailing_eps=6.13,
    )
    fundamentals = CompanyFundamentals(
        ticker="AAPL",
        profile=profile,
        income_statements=income_stmts,
        balance_sheets=balance_sheets,
        cash_flow_statements=cash_flows,
        raw_snapshot=snapshot,
    )
    return calculate_fundamental_metrics(fundamentals)


def _build_mock_apple_output() -> str:
    """Helper creating valid JSON output for AAPL fundamental analysis."""
    payload = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "currency": "USD",
        "financial_health": {
            "rating": "strong",
            "explanation": (
                "Solid solvency supported by substantial operating cash generation."
            ),
            "supporting_metrics": ["operating_cash_flow", "cash_and_equivalents"],
        },
        "growth_assessment": {
            "rating": "moderate",
            "explanation": (
                "Latest revenue contracted 2.8% YoY following prior expansions."
            ),
            "supporting_metrics": ["revenue_growth_yoy", "revenue_cagr_3yr"],
        },
        "profitability_assessment": {
            "rating": "strong",
            "explanation": (
                "Superior operating margin of 29.8% and exceptional ROE of 156%."
            ),
            "supporting_metrics": ["operating_margin", "net_profit_margin", "roe"],
        },
        "valuation_assessment": {
            "rating": "neutral",
            "explanation": (
                "Premium valuation at 28.87 provider-reported P/E and 45.05 P/B."
            ),
            "supporting_metrics": ["pe_ratio", "price_to_book", "trailing_eps"],
        },
        "leverage_assessment": {
            "rating": "moderate",
            "explanation": (
                "Net debt position of $81.1B with manageable debt-to-equity of 1.79."
            ),
            "supporting_metrics": ["total_debt", "net_debt", "debt_to_equity"],
        },
        "cash_flow_assessment": {
            "rating": "strong",
            "explanation": (
                "Outstanding free cash flow generation of $99.6B with 102% conversion."
            ),
            "supporting_metrics": [
                "operating_cash_flow",
                "free_cash_flow",
                "fcf_conversion_ratio",
            ],
        },
        "key_strengths": [
            "Industry-leading operating margin of 29.8%",
            "Immense annual free cash flow generation approaching $100B",
            "Healthy 3-year revenue compound annual growth rate",
        ],
        "key_weaknesses": [
            "Latest year-over-year revenue contraction of 2.8%",
            "Substantial net debt position of $81.1B",
        ],
        "notable_flags": [],
        "overall_assessment": "favorable",
        "overall_summary": (
            "Exceptional profitability and free cash flow generation "
            "offset short-term revenue softness."
        ),
        "confidence": 0.95,
    }
    return json.dumps(payload)


def test_apple_healthy_multi_year_assessment(apple_fundamental_metrics):
    """Verify comprehensive grounded analysis for healthy multi-year company."""
    mock_resp = _build_mock_apple_output()
    mock_provider = MockLLMProvider(responses=[mock_resp])
    agent = FundamentalAnalystAgent(provider=mock_provider)

    res = agent.run(apple_fundamental_metrics)
    assert res.success is True
    data: FundamentalAnalysisOutput = res.data
    assert data.ticker == "AAPL"
    assert data.overall_assessment == "favorable"
    assert data.confidence == 0.95
    assert data.profitability_assessment.rating == "strong"
    assert "roe" in data.profitability_assessment.supporting_metrics
    assert len(data.key_strengths) >= 1
    assert len(data.key_weaknesses) >= 1

    # Verify prompt received the complete deterministic payload
    prompt = mock_provider.prompts_received[0]
    assert "AAPL" in prompt
    assert "274515000000.0" in prompt
    assert "28.87" in prompt


def test_high_growth_profitable_company():
    """Verify high growth is recognized and references revenue CAGR and YoY."""
    profile = CompanyProfile(ticker="GROWTH", company_name="Fast Tech")
    income_stmts = [
        IncomeStatementPeriod(fiscal_year=2020, total_revenue=100.0, net_income=10.0),
        IncomeStatementPeriod(fiscal_year=2021, total_revenue=150.0, net_income=20.0),
        IncomeStatementPeriod(fiscal_year=2022, total_revenue=225.0, net_income=40.0),
        IncomeStatementPeriod(fiscal_year=2023, total_revenue=337.5, net_income=80.0),
    ]
    fund = CompanyFundamentals(
        ticker="GROWTH", profile=profile, income_statements=income_stmts
    )
    metrics = calculate_fundamental_metrics(fund)

    resp_payload = {
        "ticker": "GROWTH",
        "financial_health": {
            "rating": "strong",
            "explanation": "Healthy.",
            "supporting_metrics": [],
        },
        "growth_assessment": {
            "rating": "strong",
            "explanation": "Exceptional 50% YoY growth and 50% 3-year CAGR.",
            "supporting_metrics": ["revenue_growth_yoy", "revenue_cagr_3yr"],
        },
        "profitability_assessment": {
            "rating": "strong",
            "explanation": "Rising margins.",
            "supporting_metrics": [],
        },
        "valuation_assessment": {
            "rating": "neutral",
            "explanation": "Not evaluated.",
            "supporting_metrics": [],
        },
        "leverage_assessment": {
            "rating": "neutral",
            "explanation": "No debt.",
            "supporting_metrics": [],
        },
        "cash_flow_assessment": {
            "rating": "neutral",
            "explanation": "Missing.",
            "supporting_metrics": [],
        },
        "key_strengths": ["Rapid revenue expansion of 50% YoY"],
        "key_weaknesses": ["Lack of balance sheet data"],
        "notable_flags": [],
        "overall_assessment": "favorable",
        "overall_summary": "High growth business.",
        "confidence": 0.85,
    }
    mock_provider = MockLLMProvider(responses=[json.dumps(resp_payload)])
    agent = FundamentalAnalystAgent(provider=mock_provider)

    res = agent.run(metrics)
    assert res.success is True
    assert res.data.growth_assessment.rating == "strong"
    assert "revenue_growth_yoy" in res.data.growth_assessment.supporting_metrics


def test_declining_distressed_company():
    """Verify distressed fundamentals result in unfavorable overall assessment."""
    profile = CompanyProfile(ticker="DIST", company_name="Distressed Corp")
    income_stmts = [
        IncomeStatementPeriod(fiscal_year=2022, total_revenue=200.0, net_income=10.0),
        IncomeStatementPeriod(fiscal_year=2023, total_revenue=120.0, net_income=-30.0),
    ]
    fund = CompanyFundamentals(
        ticker="DIST", profile=profile, income_statements=income_stmts
    )
    metrics = calculate_fundamental_metrics(fund)

    resp_payload = {
        "ticker": "DIST",
        "financial_health": {
            "rating": "weak",
            "explanation": "Severe stress.",
            "supporting_metrics": [],
        },
        "growth_assessment": {
            "rating": "weak",
            "explanation": "Revenue fell 40%.",
            "supporting_metrics": [],
        },
        "profitability_assessment": {
            "rating": "weak",
            "explanation": "Net loss.",
            "supporting_metrics": [],
        },
        "valuation_assessment": {
            "rating": "insufficient_data",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "leverage_assessment": {
            "rating": "weak",
            "explanation": "High debt.",
            "supporting_metrics": [],
        },
        "cash_flow_assessment": {
            "rating": "weak",
            "explanation": "Burning cash.",
            "supporting_metrics": [],
        },
        "key_strengths": ["Historic market presence"],
        "key_weaknesses": ["Net loss of $30M", "Revenue decline of 40%"],
        "notable_flags": ["Earnings deficit transition"],
        "overall_assessment": "unfavorable",
        "overall_summary": "Distressed fundamentals.",
        "confidence": 0.90,
    }
    agent = FundamentalAnalystAgent(
        provider=MockLLMProvider(responses=[json.dumps(resp_payload)])
    )
    res = agent.run(metrics)
    assert res.success is True
    assert res.data.overall_assessment == "unfavorable"


def test_loss_to_profit_turnaround():
    """Verify earnings turnaround is reflected in notable flags and narrative."""
    profile = CompanyProfile(ticker="TURN")
    income_stmts = [
        IncomeStatementPeriod(
            fiscal_year=2022, total_revenue=100.0, net_income=-20.0, eps=-1.0
        ),
        IncomeStatementPeriod(
            fiscal_year=2023, total_revenue=140.0, net_income=15.0, eps=0.75
        ),
    ]
    fund = CompanyFundamentals(
        ticker="TURN", profile=profile, income_statements=income_stmts
    )
    metrics = calculate_fundamental_metrics(fund)
    assert metrics.growth.earnings_turnaround is True

    resp_payload = {
        "ticker": "TURN",
        "financial_health": {
            "rating": "moderate",
            "explanation": "Improving.",
            "supporting_metrics": [],
        },
        "growth_assessment": {
            "rating": "strong",
            "explanation": "Successful turnaround to profitability.",
            "supporting_metrics": ["earnings_turnaround", "revenue_growth_yoy"],
        },
        "profitability_assessment": {
            "rating": "moderate",
            "explanation": "Returned to positive net margin.",
            "supporting_metrics": [],
        },
        "valuation_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "leverage_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "cash_flow_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "key_strengths": ["Swing to $15M net profit from $20M loss"],
        "key_weaknesses": ["Short history of profitability"],
        "notable_flags": ["earnings_turnaround"],
        "overall_assessment": "favorable",
        "overall_summary": "Promising turnaround.",
        "confidence": 0.85,
    }
    agent = FundamentalAnalystAgent(
        provider=MockLLMProvider(responses=[json.dumps(resp_payload)])
    )
    res = agent.run(metrics)
    assert res.success is True
    assert "earnings_turnaround" in res.data.notable_flags


def test_profit_to_loss_deficit_turnaround():
    """Verify profit to loss transition is flagged."""
    profile = CompanyProfile(ticker="SLIP")
    income_stmts = [
        IncomeStatementPeriod(fiscal_year=2022, total_revenue=100.0, net_income=20.0),
        IncomeStatementPeriod(fiscal_year=2023, total_revenue=90.0, net_income=-10.0),
    ]
    fund = CompanyFundamentals(
        ticker="SLIP", profile=profile, income_statements=income_stmts
    )
    metrics = calculate_fundamental_metrics(fund)
    assert metrics.growth.earnings_deficit_turnaround is True

    resp_payload = {
        "ticker": "SLIP",
        "financial_health": {
            "rating": "weak",
            "explanation": "Deteriorating.",
            "supporting_metrics": [],
        },
        "growth_assessment": {
            "rating": "weak",
            "explanation": "Slipped into loss.",
            "supporting_metrics": ["earnings_deficit_turnaround"],
        },
        "profitability_assessment": {
            "rating": "weak",
            "explanation": "Negative margin.",
            "supporting_metrics": [],
        },
        "valuation_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "leverage_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "cash_flow_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "key_strengths": ["Historic profit in 2022"],
        "key_weaknesses": ["Slipped into net loss of $10M"],
        "notable_flags": ["earnings_deficit_turnaround"],
        "overall_assessment": "unfavorable",
        "overall_summary": "Loss transition.",
        "confidence": 0.85,
    }
    agent = FundamentalAnalystAgent(
        provider=MockLLMProvider(responses=[json.dumps(resp_payload)])
    )
    res = agent.run(metrics)
    assert res.success is True
    assert "earnings_deficit_turnaround" in res.data.notable_flags


def test_consecutive_losses_dual_deficit():
    """Verify consecutive losses prevent rating profitability as strong."""
    profile = CompanyProfile(ticker="DUAL")
    income_stmts = [
        IncomeStatementPeriod(fiscal_year=2022, total_revenue=100.0, net_income=-50.0),
        IncomeStatementPeriod(fiscal_year=2023, total_revenue=110.0, net_income=-40.0),
    ]
    fund = CompanyFundamentals(
        ticker="DUAL", profile=profile, income_statements=income_stmts
    )
    metrics = calculate_fundamental_metrics(fund)
    assert metrics.growth.both_periods_deficit is True

    # LLM hallucinates "strong" profitability despite dual deficit
    resp_payload = {
        "ticker": "DUAL",
        "financial_health": {
            "rating": "weak",
            "explanation": "Losses.",
            "supporting_metrics": [],
        },
        "growth_assessment": {
            "rating": "strong",
            "explanation": "Revenue grew.",
            "supporting_metrics": [],
        },
        "profitability_assessment": {
            "rating": "strong",
            "explanation": "False strong.",
            "supporting_metrics": [],
        },
        "valuation_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "leverage_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "cash_flow_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "key_strengths": ["Revenue up 10%"],
        "key_weaknesses": ["Consecutive net losses"],
        "notable_flags": ["both_periods_deficit"],
        "overall_assessment": "unfavorable",
        "overall_summary": "Dual losses.",
        "confidence": 0.85,
    }
    agent = FundamentalAnalystAgent(
        provider=MockLLMProvider(responses=[json.dumps(resp_payload)])
    )
    res = agent.run(metrics)
    assert res.success is True
    # Validator must correct the false "strong" ratings
    assert res.data.profitability_assessment.rating == "weak"
    assert res.data.growth_assessment.rating == "weak"


def test_negative_equity_distress():
    """Verify negative equity corrects false strong financial health rating."""
    profile = CompanyProfile(ticker="NEGEQ")
    fund = CompanyFundamentals(
        ticker="NEGEQ",
        profile=profile,
        income_statements=[
            IncomeStatementPeriod(
                fiscal_year=2023, total_revenue=100.0, net_income=-10.0
            )
        ],
        balance_sheets=[
            BalanceSheetPeriod(fiscal_year=2023, total_debt=50.0, total_equity=-20.0)
        ],
    )
    metrics = calculate_fundamental_metrics(fund)
    assert metrics.profitability.negative_equity is True

    # LLM incorrectly rates financial health as "strong"
    resp_payload = {
        "ticker": "NEGEQ",
        "financial_health": {
            "rating": "strong",
            "explanation": "False health.",
            "supporting_metrics": [],
        },
        "growth_assessment": {
            "rating": "weak",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "profitability_assessment": {
            "rating": "weak",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "valuation_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "leverage_assessment": {
            "rating": "weak",
            "explanation": "Deficit equity.",
            "supporting_metrics": [],
        },
        "cash_flow_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "key_strengths": ["Operating presence"],
        "key_weaknesses": ["Negative stockholders' equity of -$20M"],
        "notable_flags": ["negative_equity"],
        "overall_assessment": "unfavorable",
        "overall_summary": "Balance sheet deficit.",
        "confidence": 0.85,
    }
    agent = FundamentalAnalystAgent(
        provider=MockLLMProvider(responses=[json.dumps(resp_payload)])
    )
    res = agent.run(metrics)
    assert res.success is True
    # Validator must correct false strong to weak
    assert res.data.financial_health.rating == "weak"


def test_single_period_ipo_sparse_data():
    """Verify single-period history caps confidence and marks growth."""
    profile = CompanyProfile(ticker="IPO")
    fund = CompanyFundamentals(
        ticker="IPO",
        profile=profile,
        income_statements=[
            IncomeStatementPeriod(
                fiscal_year=2023, total_revenue=100.0, net_income=15.0
            )
        ],
    )
    metrics = calculate_fundamental_metrics(fund)

    resp_payload = {
        "ticker": "IPO",
        "financial_health": {
            "rating": "neutral",
            "explanation": "Limited balance sheet.",
            "supporting_metrics": [],
        },
        "growth_assessment": {
            "rating": "insufficient_data",
            "explanation": (
                "Only one annual period available; YoY growth cannot be evaluated."
            ),
            "supporting_metrics": [],
        },
        "profitability_assessment": {
            "rating": "moderate",
            "explanation": "15% net margin.",
            "supporting_metrics": ["net_profit_margin"],
        },
        "valuation_assessment": {
            "rating": "insufficient_data",
            "explanation": "No multiples.",
            "supporting_metrics": [],
        },
        "leverage_assessment": {
            "rating": "insufficient_data",
            "explanation": "No balance sheet.",
            "supporting_metrics": [],
        },
        "cash_flow_assessment": {
            "rating": "insufficient_data",
            "explanation": "No cash flow statement.",
            "supporting_metrics": [],
        },
        "key_strengths": ["Profitable initial year"],
        "key_weaknesses": ["Insufficient operating history"],
        "notable_flags": [],
        "overall_assessment": "neutral",
        "overall_summary": "Single period data.",
        "confidence": 0.95,  # Too high for single period
    }
    agent = FundamentalAnalystAgent(
        provider=MockLLMProvider(responses=[json.dumps(resp_payload)])
    )
    res = agent.run(metrics)
    assert res.success is True
    assert res.data.growth_assessment.rating == "insufficient_data"
    # Confidence capped due to sparse statements
    assert res.data.confidence <= 0.70


def test_completely_empty_metrics():
    """Verify completely empty metrics yield overall_assessment as insufficient_data."""
    fund = CompanyFundamentals(ticker="EMPTY", profile=CompanyProfile(ticker="EMPTY"))
    metrics = calculate_fundamental_metrics(fund)

    resp_payload = {
        "ticker": "EMPTY",
        "financial_health": {
            "rating": "insufficient_data",
            "explanation": "No statements.",
            "supporting_metrics": [],
        },
        "growth_assessment": {
            "rating": "insufficient_data",
            "explanation": "No data.",
            "supporting_metrics": [],
        },
        "profitability_assessment": {
            "rating": "insufficient_data",
            "explanation": "No data.",
            "supporting_metrics": [],
        },
        "valuation_assessment": {
            "rating": "insufficient_data",
            "explanation": "No data.",
            "supporting_metrics": [],
        },
        "leverage_assessment": {
            "rating": "insufficient_data",
            "explanation": "No data.",
            "supporting_metrics": [],
        },
        "cash_flow_assessment": {
            "rating": "insufficient_data",
            "explanation": "No data.",
            "supporting_metrics": [],
        },
        "key_strengths": ["None identified"],
        "key_weaknesses": ["No financial statement data available"],
        "notable_flags": ["insufficient_data"],
        "overall_assessment": "insufficient_data",
        "overall_summary": "No fundamentals available.",
        "confidence": 0.10,
    }
    agent = FundamentalAnalystAgent(
        provider=MockLLMProvider(responses=[json.dumps(resp_payload)])
    )
    res = agent.run(metrics)
    assert res.success is True
    assert res.data.overall_assessment == "insufficient_data"


def test_provider_vs_derived_pe_provenance():
    """Verify valuation narrative respects provider vs derived P/E provenance."""
    profile = CompanyProfile(ticker="PROV")
    fund = CompanyFundamentals(
        ticker="PROV",
        profile=profile,
        raw_snapshot=ProviderRawSnapshot(trailing_pe=22.5, market_cap=500000.0),
    )
    metrics = calculate_fundamental_metrics(fund)
    assert metrics.valuation.pe_ratio_source == "provider"

    resp_payload = {
        "ticker": "PROV",
        "financial_health": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "growth_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "profitability_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "valuation_assessment": {
            "rating": "moderate",
            "explanation": "Provider-reported P/E multiple of 22.5.",
            "supporting_metrics": ["pe_ratio", "pe_ratio_source"],
        },
        "leverage_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "cash_flow_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "key_strengths": ["Valuation multiple available"],
        "key_weaknesses": ["No operating data"],
        "notable_flags": [],
        "overall_assessment": "neutral",
        "overall_summary": "Valuation context.",
        "confidence": 0.50,
    }
    agent = FundamentalAnalystAgent(
        provider=MockLLMProvider(responses=[json.dumps(resp_payload)])
    )
    res = agent.run(metrics)
    assert res.success is True
    assert "pe_ratio" in res.data.valuation_assessment.supporting_metrics


def test_provider_vs_derived_fcf_precedence():
    """Verify cash flow assessment preserves reported FCF priority."""
    profile = CompanyProfile(ticker="FCF")
    fund = CompanyFundamentals(
        ticker="FCF",
        profile=profile,
        cash_flow_statements=[
            CashFlowPeriod(
                fiscal_year=2023,
                operating_cash_flow=100.0,
                capital_expenditures=-20.0,
                free_cash_flow=95.0,  # Reported priority
            )
        ],
    )
    metrics = calculate_fundamental_metrics(fund)
    assert metrics.cash_flow.free_cash_flow == 95.0

    resp_payload = {
        "ticker": "FCF",
        "financial_health": {
            "rating": "strong",
            "explanation": "Positive cash flow.",
            "supporting_metrics": [],
        },
        "growth_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "profitability_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "valuation_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "leverage_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "cash_flow_assessment": {
            "rating": "strong",
            "explanation": (
                "Provider-reported free cash flow of $95M supported by $100M OCF."
            ),
            "supporting_metrics": ["free_cash_flow", "operating_cash_flow"],
        },
        "key_strengths": ["Strong reported free cash flow of $95M"],
        "key_weaknesses": ["Single year observation"],
        "notable_flags": [],
        "overall_assessment": "favorable",
        "overall_summary": "Solid cash generation.",
        "confidence": 0.70,
    }
    agent = FundamentalAnalystAgent(
        provider=MockLLMProvider(responses=[json.dumps(resp_payload)])
    )
    res = agent.run(metrics)
    assert res.success is True
    assert "free_cash_flow" in res.data.cash_flow_assessment.supporting_metrics


def test_trailing_eps_preservation_in_narrative():
    """Verify provider trailing_eps is preserved and available to analyst."""
    profile = CompanyProfile(ticker="EPS")
    fund = CompanyFundamentals(
        ticker="EPS",
        profile=profile,
        raw_snapshot=ProviderRawSnapshot(trailing_eps=4.85),
    )
    metrics = calculate_fundamental_metrics(fund)
    assert metrics.valuation.trailing_eps == 4.85

    resp_payload = {
        "ticker": "EPS",
        "financial_health": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "growth_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "profitability_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "valuation_assessment": {
            "rating": "moderate",
            "explanation": "Trailing EPS reported at $4.85.",
            "supporting_metrics": ["trailing_eps"],
        },
        "leverage_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "cash_flow_assessment": {
            "rating": "neutral",
            "explanation": "N/A",
            "supporting_metrics": [],
        },
        "key_strengths": ["Positive trailing earnings"],
        "key_weaknesses": ["No historical trend"],
        "notable_flags": [],
        "overall_assessment": "neutral",
        "overall_summary": "EPS context.",
        "confidence": 0.50,
    }
    agent = FundamentalAnalystAgent(
        provider=MockLLMProvider(responses=[json.dumps(resp_payload)])
    )
    res = agent.run(metrics)
    assert res.success is True
    assert "trailing_eps" in res.data.valuation_assessment.supporting_metrics


def test_investor_profile_context_invariance(apple_fundamental_metrics):
    """Verify investor profile context does NOT mutate the underlying metrics."""
    orig_roe = apple_fundamental_metrics.profitability.roe
    orig_rev = apple_fundamental_metrics.growth.revenue_growth_yoy

    input_data = FundamentalAnalystInput(
        metrics=apple_fundamental_metrics,
        time_horizon="10 years",
        risk_tolerance="conservative",
    )

    prompt = format_fundamental_prompt(input_data)
    assert "Time Horizon: 10 years" in prompt
    assert "Risk Tolerance: conservative" in prompt

    # Metrics themselves must remain 100% unchanged
    assert apple_fundamental_metrics.profitability.roe == orig_roe
    assert apple_fundamental_metrics.growth.revenue_growth_yoy == orig_rev


def test_malformed_structured_output_retry_and_exhaustion(apple_fundamental_metrics):
    """Verify malformed JSON triggers retry and graceful AgentResult failure."""
    # Two invalid JSON responses to exhaust the 2 attempts in generate_structured
    invalid_json = "{ invalid_json: true, "
    mock_provider = MockLLMProvider(responses=[invalid_json, invalid_json])
    agent = FundamentalAnalystAgent(provider=mock_provider)

    res = agent.run(apple_fundamental_metrics)
    assert res.success is False
    assert "structured validation failed" in res.error.lower()
    assert mock_provider.call_count == 2


def test_llm_provider_network_failure(apple_fundamental_metrics):
    """Verify LLM provider error is caught gracefully without raising exceptions."""
    mock_provider = MockLLMProvider(
        fail_with=LLMError("Connection timeout to Gemini API")
    )
    agent = FundamentalAnalystAgent(provider=mock_provider)

    res = agent.run(apple_fundamental_metrics)
    assert res.success is False
    assert "Connection timeout" in res.error


def test_supporting_metric_validation():
    """Verify unknown or hallucinated supporting metric names are filtered out."""
    output = FundamentalAnalysisOutput(
        ticker="TEST",
        financial_health=DimensionAssessment(
            rating="strong",
            explanation="Good.",
            supporting_metrics=["operating_cash_flow", "hallucinated_magic_ratio"],
        ),
        growth_assessment=DimensionAssessment(
            rating="neutral", explanation="N/A", supporting_metrics=[]
        ),
        profitability_assessment=DimensionAssessment(
            rating="neutral", explanation="N/A", supporting_metrics=[]
        ),
        valuation_assessment=DimensionAssessment(
            rating="neutral", explanation="N/A", supporting_metrics=[]
        ),
        leverage_assessment=DimensionAssessment(
            rating="neutral", explanation="N/A", supporting_metrics=[]
        ),
        cash_flow_assessment=DimensionAssessment(
            rating="neutral", explanation="N/A", supporting_metrics=[]
        ),
        key_strengths=["Strength"],
        key_weaknesses=["Weakness"],
        overall_assessment="neutral",
        overall_summary="Summary.",
        confidence=0.8,
    )
    metrics = calculate_fundamental_metrics(
        CompanyFundamentals(ticker="TEST", profile=CompanyProfile(ticker="TEST"))
    )

    validate_fundamental_analysis_consistency(output, metrics)
    # Valid metric retained, hallucinated metric removed
    assert "operating_cash_flow" in output.financial_health.supporting_metrics
    assert "hallucinated_magic_ratio" not in output.financial_health.supporting_metrics


def test_langgraph_node_adapter_in_isolation(apple_fundamental_metrics):
    """Verify fundamental_analyst_node operates as a clean LangGraph node adapter."""
    mock_resp = _build_mock_apple_output()
    agent = FundamentalAnalystAgent(provider=MockLLMProvider(responses=[mock_resp]))

    # Case A: Metrics present in state
    state_with_metrics: GraphState = {
        "user_query": "Analyze Apple fundamentals",
        "fundamental_metrics": apple_fundamental_metrics,
        "investor_profile": {"time_horizon": "5 years", "risk_tolerance": "moderate"},
    }
    update = fundamental_analyst_node(state_with_metrics, agent=agent)
    assert "fundamental_result" in update
    f_res = update["fundamental_result"]
    assert f_res["success"] is True
    assert f_res["specialist"] == "fundamental"
    assert f_res["data"]["ticker"] == "AAPL"

    # Case B: No metrics in state -> graceful failure
    empty_state: GraphState = {"user_query": "Analyze Apple"}
    fail_update = fundamental_analyst_node(empty_state, agent=agent)
    assert "fundamental_result" in fail_update
    assert fail_update["fundamental_result"]["success"] is False
    assert "No fundamental metrics" in fail_update["fundamental_result"]["error"]
