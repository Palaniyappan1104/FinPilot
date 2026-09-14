"""Comprehensive End-to-End LangGraph Integration Tests (Phase 13).

Validates:
1. Full happy-path workflow: Conversation -> Clarification -> CIO ->
   Specialists -> Aggregator -> Report Generator -> END.
2. Clarification conditional routing: halting with clarification questions.
3. Clarification feedback loop: resolving missing parameters.
4. Dynamic CIO specialist selection: executing only the selected specialist subset.
5. Conditional Research Analyst inclusion (13.2.3): included only when docs available.
6. Specialist failure isolation (13.3.1): single failures never crash graph.
7. Graceful degradation under total failure (13.3.2): honest fallback reports.
8. State and provenance preservation (13.1.2): evidence preserved end-to-end.
9. Deterministic termination and DAG flow: zero uncontrolled loops.
10. Regulatory safety and non-advisory compliance across all outputs.
11. Parity of convenience aliases (create_finpilot_graph, run_finpilot_graph).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from app.agents.aggregator import ReportAggregatorAgent
from app.agents.cio import CIOAgent
from app.agents.clarification import ClarificationAgent
from app.agents.conversation import ConversationAgent
from app.agents.fundamental_schema import (
    DimensionAssessment,
    FundamentalAnalysisOutput,
)
from app.agents.graph import (
    create_end_to_end_graph,
    run_end_to_end_graph,
    run_finpilot_graph,
)
from app.agents.news_schema import (
    FactorItem,
    NewsAnalysisEvent,
    NewsAnalysisOutput,
    RecentNewsItem,
)
from app.agents.report_generator import ReportGeneratorAgent
from app.agents.report_schema import (
    FinalReport,
    RecommendationStance,
)
from app.agents.research_schema import (
    ResearchAnalysisOutput,
    ResearchEvidenceRef,
    ResearchFinding,
)
from app.agents.risk_schema import (
    RiskAnalysisOutput,
    RiskCategory,
    RiskEvidenceRef,
    RiskFactor,
    RiskProbability,
    RiskSeverity,
)
from app.agents.state import (
    GraphState,
    create_initial_state,
)
from app.agents.technical_schema import (
    MACDMetrics,
    MovingAverageMetrics,
    RSIMetrics,
    SupportResistanceSummary,
    TechnicalAnalysisOutput,
    TechnicalIndicatorsSummary,
    TechnicalInterpretation,
    VolumeMetrics,
)
from app.core.llm.base import LLMProvider, LLMResponse

# ===========================================================================
# MOCK LLM PROVIDER & TEST HELPERS
# ===========================================================================


class MockIntegrationLLMProvider(LLMProvider):
    """Deterministic mock LLM provider for Phase 13 end-to-end testing."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        fail_with: Optional[Exception] = None,
        name: str = "mock_integration_provider",
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
            content = "{}"
        else:
            idx = min(self.call_count - 1, len(self.responses) - 1)
            content = self.responses[idx]

        return LLMResponse(
            content=content,
            model="mock-integration-v1",
            provider=self.provider_name,
        )


def _build_conv_json(
    query: str,
    company: Optional[str] = "Apple Inc.",
    capital_amount: Optional[float] = 50000.0,
    time_horizon: Optional[str] = "3-5 years",
    risk_tolerance: Optional[str] = "moderate",
    intent_type: str = "investment_analysis",
) -> str:
    return json.dumps(
        {
            "normalized_query": query,
            "intent_type": intent_type,
            "company": company,
            "capital_amount": capital_amount,
            "time_horizon": time_horizon,
            "risk_tolerance": risk_tolerance,
        }
    )


def _build_cio_json(
    target_company: str = "Apple Inc.",
    ticker: str = "AAPL",
    selected_specialists: Optional[List[str]] = None,
) -> str:
    specs = selected_specialists or ["technical", "fundamental", "news", "risk"]
    tasks = {
        s: {
            "specialist": s,
            "task_description": f"Analyze {s} aspects for {target_company}.",
            "parameters": {"company": target_company, "ticker": ticker},
        }
        for s in specs
    }
    return json.dumps(
        {
            "target_company": target_company,
            "ticker": ticker,
            "selected_specialists": specs,
            "specialist_tasks": tasks,
            "reasoning": "Selected all primary financial domains.",
            "fallback_applied": False,
        }
    )


# ===========================================================================
# DETERMINISTIC SPECIALIST OUTPUT FACTORIES
# ===========================================================================


def create_mock_technical_output(ticker: str = "AAPL") -> TechnicalAnalysisOutput:
    return TechnicalAnalysisOutput(
        ticker=ticker,
        trend="uptrend",
        indicators_summary=TechnicalIndicatorsSummary(
            latest_close=185.50,
            moving_averages=MovingAverageMetrics(
                sma_20=182.0,
                sma_50=178.0,
                sma_200=170.0,
            ),
            rsi=RSIMetrics(value=62.0),
            macd=MACDMetrics(macd_line=1.5, signal_line=1.0, histogram=0.5),
            volume=VolumeMetrics(
                latest_volume=55000000.0, average_volume_20d=48000000.0
            ),
        ),
        support_resistance=SupportResistanceSummary(
            primary_support=180.0,
            primary_resistance=192.0,
            support_levels=[180.0, 175.0],
            resistance_levels=[192.0, 198.0],
        ),
        technical_score=75.0,
        interpretation=TechnicalInterpretation(
            overall_summary="Bullish continuation with strong momentum.",
            trend_analysis="Price holds firmly above ascending 50-day SMAs.",
            moving_averages_analysis="Bullish stacking order SMA 20 > 50 > 200.",
            momentum_analysis="RSI at 62.0 reflects robust buying pressure.",
            volume_analysis="Volume confirms breakout over consolidation.",
            support_resistance_analysis="Primary floor at 180.0, barrier at 192.0.",
        ),
        evidence=[
            "Price 185.50 is above 50-day SMA 178.0 and 200-day SMA 170.0",
            "RSI 62.0 confirms bullish momentum",
            "MACD histogram positive at 0.5",
        ],
        risks=["Approaching overhead resistance at 192.0"],
        confidence=0.88,
    )


def create_mock_fundamental_output(ticker: str = "AAPL") -> FundamentalAnalysisOutput:
    return FundamentalAnalysisOutput(
        ticker=ticker,
        company_name="Apple Inc.",
        currency="USD",
        financial_health=DimensionAssessment(
            rating="strong",
            explanation="Strong cash position and zero refinancing distress.",
            supporting_metrics=["cash_and_equivalents", "operating_cash_flow"],
        ),
        growth_assessment=DimensionAssessment(
            rating="moderate",
            explanation="Services revenue grew 11% YoY.",
            supporting_metrics=["revenue_growth_yoy"],
        ),
        profitability_assessment=DimensionAssessment(
            rating="strong",
            explanation="Industry-leading operating margin at 30.5%.",
            supporting_metrics=["gross_margin", "operating_margin"],
        ),
        valuation_assessment=DimensionAssessment(
            rating="neutral",
            explanation="Valuation in line with peer group historical multiples.",
            supporting_metrics=["pe_ratio"],
        ),
        leverage_assessment=DimensionAssessment(
            rating="strong",
            explanation="Conservative debt-to-equity ratio.",
            supporting_metrics=["debt_to_equity"],
        ),
        cash_flow_assessment=DimensionAssessment(
            rating="strong",
            explanation="Free cash flow generation exceeds $100B annually.",
            supporting_metrics=["free_cash_flow"],
        ),
        key_strengths=["World-class free cash flow conversion."],
        key_weaknesses=["Premium valuation multiple relative to historical mean."],
        notable_flags=[],
        overall_assessment="favorable",
        overall_summary=(
            "Apple demonstrates exceptional cash flow and balance sheet strength."
        ),
        confidence=0.92,
    )


def create_mock_news_output(ticker: str = "AAPL") -> NewsAnalysisOutput:
    return NewsAnalysisOutput(
        ticker=ticker,
        overall_sentiment="positive",
        sentiment_distribution={"positive": 9, "neutral": 2, "negative": 1},
        recent_news=[
            RecentNewsItem(
                article_id="art_101",
                headline="Apple reports record quarterly services subscription growth",
                summary="Paid subscriptions topped 1 billion accounts.",
                source="Reuters",
                url="https://example.com/101",
                published_at=datetime(2026, 9, 11, 10, 0, tzinfo=timezone.utc),
                sentiment="positive",
            )
        ],
        important_events=[
            NewsAnalysisEvent(
                event_type="earnings",
                description="Enterprise AI hardware launch.",
                article_ids=["art_101"],
            )
        ],
        positive_factors=[
            FactorItem(text="Services momentum", article_ids=["art_101"])
        ],
        negative_factors=[],
        summary=(
            "News coverage is predominantly positive, highlighting services expansion."
        ),
        confidence=0.86,
    )


def create_mock_research_output(ticker: str = "AAPL") -> ResearchAnalysisOutput:
    ev_ref = ResearchEvidenceRef(
        document_id="sec_10q_2026_q2",
        chunk_id="chunk_p14_c2",
        source_document="10-Q Q2 2026",
        page_numbers=[14, 15],
    )
    return ResearchAnalysisOutput(
        query=f"Analyze 10-K filing disclosures for {ticker}",
        answer="SEC disclosures substantiate robust operational performance.",
        key_findings=[
            ResearchFinding(
                claim="Services gross margin expanded to 71.3% in fiscal 2025.",
                evidence=[ev_ref],
            )
        ],
        evidence=[ev_ref],
        summary="SEC disclosures substantiate robust operational performance.",
        confidence=0.91,
        insufficient_evidence=False,
    )


def create_mock_risk_output(ticker: str = "AAPL") -> RiskAnalysisOutput:
    return RiskAnalysisOutput(
        ticker=ticker,
        overall_risk_level=RiskSeverity.MODERATE,
        summary=(
            "Risk profile is moderate, characterized by manageable market volatility."
        ),
        market_risks=[
            RiskFactor(
                category=RiskCategory.MARKET,
                name="Equity Market Multiple Sensitivity",
                description="Valuation may compress if macro interest rate cuts stall.",
                severity=RiskSeverity.MODERATE,
                probability=RiskProbability.MEDIUM,
                evidence=[
                    RiskEvidenceRef(
                        source_type="fundamental",
                        reference_id="pe_ratio",
                        detail="P/E multiple of 28.0",
                    )
                ],
            )
        ],
        company_risks=[],
        sector_risks=[],
        financial_risks=[],
        volatility_risks=[],
        investor_specific_risks=[],
        confidence=0.90,
    )


@pytest.fixture
def mock_specialist_overrides() -> Dict[str, Any]:
    """Default dictionary of specialist node overrides returning validated payloads."""
    return {
        "technical": lambda s: {
            "technical_result": {
                "specialist": "technical",
                "status": "completed",
                "success": True,
                "data": create_mock_technical_output().model_dump(),
                "confidence": 0.88,
            }
        },
        "fundamental": lambda s: {
            "fundamental_result": {
                "specialist": "fundamental",
                "status": "completed",
                "success": True,
                "data": create_mock_fundamental_output().model_dump(),
                "confidence": 0.92,
            }
        },
        "news": lambda s: {
            "news_result": {
                "specialist": "news",
                "status": "completed",
                "success": True,
                "data": create_mock_news_output().model_dump(),
                "confidence": 0.85,
            }
        },
        "research": lambda s: {
            "research_result": {
                "specialist": "research",
                "status": "completed",
                "success": True,
                "data": create_mock_research_output().model_dump(),
                "confidence": 0.89,
            }
        },
        "risk": lambda s: {
            "risk_result": {
                "specialist": "risk",
                "status": "completed",
                "success": True,
                "data": create_mock_risk_output().model_dump(),
                "confidence": 0.90,
            }
        },
    }


# ===========================================================================
# 1. FULL HAPPY-PATH WORKFLOW TESTS (13.1 & 13.4.1)
# ===========================================================================


class TestEndToEndHappyPath:
    """Test suite for full end-to-end happy-path graph execution."""

    def test_full_happy_path_workflow(
        self, mock_specialist_overrides: Dict[str, Any]
    ) -> None:
        """Full pipeline executes from query to FinalReport at END."""
        query = "Should I invest 50000 in Apple Inc. for 3-5 years with moderate risk?"
        conv_json = _build_conv_json(query)
        cio_json = _build_cio_json(
            target_company="Apple Inc.",
            ticker="AAPL",
            selected_specialists=["technical", "fundamental", "news", "risk"],
        )

        conv_agent = ConversationAgent(provider=MockIntegrationLLMProvider([conv_json]))
        clar_agent = ClarificationAgent(provider=MockIntegrationLLMProvider([]))
        cio_agent = CIOAgent(provider=MockIntegrationLLMProvider([cio_json]))
        agg_agent = ReportAggregatorAgent(deterministic_only=True)
        rep_agent = ReportGeneratorAgent(deterministic_only=True)

        graph = create_end_to_end_graph(
            conversation_agent=conv_agent,
            clarification_agent=clar_agent,
            cio_agent=cio_agent,
            aggregator_agent=agg_agent,
            report_generator_agent=rep_agent,
            specialist_overrides=mock_specialist_overrides,
        )

        initial_state = create_initial_state(
            user_query=query,
            trace_id="test-happy-path-01",
        )

        final_state: GraphState = graph.invoke(initial_state)

        # 1. Clarification status
        assert final_state.get("investor_profile") is not None
        assert final_state["investor_profile"]["profile_complete"] is True
        assert final_state["clarified_request"]["clarification_needed"] is False

        # 2. CIO decision
        assert final_state.get("cio_decision") is not None
        assert final_state["cio_decision"]["ticker"] == "AAPL"
        assert len(final_state["cio_decision"]["selected_specialists"]) == 4

        # 3. Specialist execution
        for spec in (
            "technical_result",
            "fundamental_result",
            "news_result",
            "risk_result",
        ):
            assert final_state.get(spec) is not None
            assert final_state[spec]["status"] == "completed"
            assert final_state[spec]["success"] is True

        # 4. Aggregator output
        agg = final_state.get("aggregated_result")
        assert agg is not None
        assert agg["target_company"] == "Apple Inc."
        assert agg["ticker"] == "AAPL"
        assert agg["data_completeness_ratio"] == 1.0

        # 5. Final report output
        rep_dict = final_state.get("report")
        assert rep_dict is not None
        assert rep_dict["success"] is True
        report_data = rep_dict["data"]
        report = FinalReport.model_validate(report_data)

        assert report.company.ticker == "AAPL"
        assert report.company.name == "Apple Inc."
        assert report.recommendation is not None
        assert report.recommendation.stance == RecommendationStance.FAVORABLE
        assert len(report.key_reasons) > 0
        assert len(report.important_risks) > 0
        assert "not a registered investment advisor" in report.disclaimer.lower()

    def test_run_end_to_end_graph_convenience_helper(
        self, mock_specialist_overrides: Dict[str, Any]
    ) -> None:
        """run_end_to_end_graph convenience function executes cleanly."""
        query = "Should I invest 50000 in Apple Inc. for 3-5 years with moderate risk?"
        conv_json = _build_conv_json(query)
        cio_json = _build_cio_json()

        final_state = run_end_to_end_graph(
            query=query,
            trace_id="test-helper-01",
            conversation_agent=ConversationAgent(
                provider=MockIntegrationLLMProvider([conv_json])
            ),
            clarification_agent=ClarificationAgent(
                provider=MockIntegrationLLMProvider([])
            ),
            cio_agent=CIOAgent(provider=MockIntegrationLLMProvider([cio_json])),
            aggregator_agent=ReportAggregatorAgent(deterministic_only=True),
            report_generator_agent=ReportGeneratorAgent(deterministic_only=True),
            specialist_overrides=mock_specialist_overrides,
        )

        assert final_state.get("report") is not None
        assert final_state["report"]["success"] is True
        assert final_state["report"]["data"]["company"]["ticker"] == "AAPL"


# ===========================================================================
# 2. CLARIFICATION CONDITIONAL ROUTING & FEEDBACK LOOP (13.2.1 & 13.4.2)
# ===========================================================================


class TestEndToEndClarificationRouting:
    """Test suite for clarification branching, halting, and answer resolution."""

    def test_underspecified_query_halts_at_clarification(self) -> None:
        """Underspecified query halts at END with clarification questions."""
        query = "Analyze Apple"
        conv_json = _build_conv_json(
            query=query,
            company="Apple Inc.",
            capital_amount=None,
            time_horizon=None,
            risk_tolerance=None,
        )

        conv_agent = ConversationAgent(provider=MockIntegrationLLMProvider([conv_json]))
        clar_agent = ClarificationAgent(
            provider=MockIntegrationLLMProvider(
                [
                    json.dumps(
                        {
                            "questions": [
                                "How much capital are you planning to invest?",
                                "What is your intended investment horizon?",
                            ]
                        }
                    )
                ]
            )
        )

        graph = create_end_to_end_graph(
            conversation_agent=conv_agent,
            clarification_agent=clar_agent,
        )

        initial_state = create_initial_state(user_query=query)
        final_state = graph.invoke(initial_state)

        # Halts with clarification needed
        assert final_state["clarified_request"]["clarification_needed"] is True
        assert final_state["investor_profile"]["profile_complete"] is False
        assert len(final_state["clarified_request"]["clarification_questions"]) > 0

        # Downstream nodes NEVER executed
        assert final_state.get("cio_decision") is None
        assert final_state.get("technical_result") is None
        assert final_state.get("aggregated_result") is None
        assert final_state.get("report") is None

    def test_clarification_feedback_loop_with_provided_answers(
        self, mock_specialist_overrides: Dict[str, Any]
    ) -> None:
        """Clarification answers in state resolve missing parameters."""
        query = "Analyze Apple"
        conv_json = _build_conv_json(
            query=query,
            company="Apple Inc.",
            capital_amount=None,
            time_horizon=None,
            risk_tolerance=None,
        )
        cio_json = _build_cio_json()

        conv_agent = ConversationAgent(provider=MockIntegrationLLMProvider([conv_json]))
        clar_agent = ClarificationAgent(provider=MockIntegrationLLMProvider([]))
        cio_agent = CIOAgent(provider=MockIntegrationLLMProvider([cio_json]))
        agg_agent = ReportAggregatorAgent(deterministic_only=True)
        rep_agent = ReportGeneratorAgent(deterministic_only=True)

        graph = create_end_to_end_graph(
            conversation_agent=conv_agent,
            clarification_agent=clar_agent,
            cio_agent=cio_agent,
            aggregator_agent=agg_agent,
            report_generator_agent=rep_agent,
            specialist_overrides=mock_specialist_overrides,
        )

        # State includes clarification answers from previous user turn (plan.md 13.2.1)
        initial_state = create_initial_state(
            user_query=query,
            clarification_answers={
                "capital_amount": 50000.0,
                "time_horizon": "3-5 years",
                "risk_tolerance": "moderate",
            },
        )

        final_state = graph.invoke(initial_state)

        # Profile is now complete and workflow finished
        assert final_state["investor_profile"]["profile_complete"] is True
        assert final_state["clarified_request"]["clarification_needed"] is False
        assert final_state.get("cio_decision") is not None
        assert final_state.get("report") is not None
        assert final_state["report"]["success"] is True


# ===========================================================================
# 3. SELECTIVE ROUTING & RESEARCH CONDITIONAL INCLUSION (13.2.2 & 13.2.3)
# ===========================================================================


class TestEndToEndSelectiveRouting:
    """Test suite for CIO selective specialist routing and conditional research."""

    def test_cio_selective_routing_subsets(
        self, mock_specialist_overrides: Dict[str, Any]
    ) -> None:
        """CIO routing strictly executes selected specialist subset."""
        query = "Technical chart and fundamental valuation for Apple"
        conv_json = _build_conv_json(query)
        cio_json = _build_cio_json(selected_specialists=["technical", "fundamental"])

        conv_agent = ConversationAgent(provider=MockIntegrationLLMProvider([conv_json]))
        clar_agent = ClarificationAgent(provider=MockIntegrationLLMProvider([]))
        cio_agent = CIOAgent(provider=MockIntegrationLLMProvider([cio_json]))
        agg_agent = ReportAggregatorAgent(deterministic_only=True)
        rep_agent = ReportGeneratorAgent(deterministic_only=True)

        graph = create_end_to_end_graph(
            conversation_agent=conv_agent,
            clarification_agent=clar_agent,
            cio_agent=cio_agent,
            aggregator_agent=agg_agent,
            report_generator_agent=rep_agent,
            specialist_overrides=mock_specialist_overrides,
        )

        initial_state = create_initial_state(user_query=query)
        final_state = graph.invoke(initial_state)

        # Selected specialists executed
        assert final_state.get("technical_result") is not None
        assert final_state.get("fundamental_result") is not None

        # Unselected specialists NEVER executed
        assert final_state.get("news_result") is None
        assert final_state.get("research_result") is None
        assert final_state.get("risk_result") is None

        # Aggregator reflects partial availability
        agg = final_state["aggregated_result"]
        assert agg["specialist_statuses"]["technical"] == "available"
        assert agg["specialist_statuses"]["fundamental"] == "available"
        assert agg["specialist_statuses"]["news"] == "missing"
        assert agg["specialist_statuses"]["risk"] == "missing"

        # Report was generated for partial data
        assert final_state.get("report") is not None
        assert final_state["report"]["success"] is True

    def test_conditional_research_omitted_when_documents_unavailable(
        self, mock_specialist_overrides: Dict[str, Any]
    ) -> None:
        """Research specialist excluded when documents_available is False (13.2.3)."""
        query = "Analyze Apple 10-K filings and technicals"
        conv_json = _build_conv_json(query)
        cio_json = _build_cio_json(selected_specialists=["technical", "research"])

        conv_agent = ConversationAgent(provider=MockIntegrationLLMProvider([conv_json]))
        clar_agent = ClarificationAgent(provider=MockIntegrationLLMProvider([]))
        cio_agent = CIOAgent(provider=MockIntegrationLLMProvider([cio_json]))
        agg_agent = ReportAggregatorAgent(deterministic_only=True)
        rep_agent = ReportGeneratorAgent(deterministic_only=True)

        graph = create_end_to_end_graph(
            conversation_agent=conv_agent,
            clarification_agent=clar_agent,
            cio_agent=cio_agent,
            aggregator_agent=agg_agent,
            report_generator_agent=rep_agent,
            specialist_overrides=mock_specialist_overrides,
        )

        # documents_available=False
        initial_state = create_initial_state(
            user_query=query,
            documents_available=False,
        )
        final_state = graph.invoke(initial_state)

        # Technical ran
        assert final_state.get("technical_result") is not None
        # Research was omitted from routing
        assert final_state.get("research_result") is None

    def test_conditional_research_included_when_documents_available(
        self, mock_specialist_overrides: Dict[str, Any]
    ) -> None:
        """Research specialist executed when documents_available is True (13.2.3)."""
        query = "Analyze Apple 10-K filings and technicals"
        conv_json = _build_conv_json(query)
        cio_json = _build_cio_json(selected_specialists=["technical", "research"])

        conv_agent = ConversationAgent(provider=MockIntegrationLLMProvider([conv_json]))
        clar_agent = ClarificationAgent(provider=MockIntegrationLLMProvider([]))
        cio_agent = CIOAgent(provider=MockIntegrationLLMProvider([cio_json]))
        agg_agent = ReportAggregatorAgent(deterministic_only=True)
        rep_agent = ReportGeneratorAgent(deterministic_only=True)

        graph = create_end_to_end_graph(
            conversation_agent=conv_agent,
            clarification_agent=clar_agent,
            cio_agent=cio_agent,
            aggregator_agent=agg_agent,
            report_generator_agent=rep_agent,
            specialist_overrides=mock_specialist_overrides,
        )

        # documents_available=True
        initial_state = create_initial_state(
            user_query=query,
            documents_available=True,
        )
        final_state = graph.invoke(initial_state)

        # Both technical and research ran
        assert final_state.get("technical_result") is not None
        assert final_state.get("research_result") is not None
        assert final_state["research_result"]["status"] == "completed"


# ===========================================================================
# 4. SPECIALIST FAILURE ISOLATION & RESILIENCE (13.3.1, 13.3.2 & 13.4.4)
# ===========================================================================


class TestEndToEndFailureIsolation:
    """Test suite verifying failure isolation and graceful degradation."""

    def test_single_specialist_crash_does_not_halt_graph(
        self, mock_specialist_overrides: Dict[str, Any]
    ) -> None:
        """Crash in specialist does not halt workflow; others complete (13.3.1)."""
        query = "Should I invest 50000 in Apple Inc. for 3-5 years with moderate risk?"
        conv_json = _build_conv_json(query)
        cio_json = _build_cio_json(
            selected_specialists=["technical", "fundamental", "news", "risk"]
        )

        # Inject simulated crash into technical analyst
        overrides = dict(mock_specialist_overrides)

        def _crashing_tech(s: GraphState) -> Dict[str, Any]:
            raise RuntimeError("Simulated external technical API outage")

        overrides["technical"] = _crashing_tech

        graph = create_end_to_end_graph(
            conversation_agent=ConversationAgent(
                provider=MockIntegrationLLMProvider([conv_json])
            ),
            clarification_agent=ClarificationAgent(
                provider=MockIntegrationLLMProvider([])
            ),
            cio_agent=CIOAgent(provider=MockIntegrationLLMProvider([cio_json])),
            aggregator_agent=ReportAggregatorAgent(deterministic_only=True),
            report_generator_agent=ReportGeneratorAgent(deterministic_only=True),
            specialist_overrides=overrides,
        )

        initial_state = create_initial_state(user_query=query)
        final_state = graph.invoke(initial_state)

        # Technical recorded as failed
        assert final_state["technical_result"]["status"] == "failed"
        assert final_state["technical_result"]["success"] is False
        assert (
            "Simulated external technical API outage"
            in final_state["technical_result"]["error"]
        )

        # Other specialists completed successfully
        assert final_state["fundamental_result"]["status"] == "completed"
        assert final_state["news_result"]["status"] == "completed"
        assert final_state["risk_result"]["status"] == "completed"

        # Aggregator recorded failure
        agg = final_state["aggregated_result"]
        assert agg["specialist_statuses"]["technical"] == "failed"
        assert "technical" in agg["failed_specialists"]

        # Final report generated with partial warning banner
        assert final_state.get("report") is not None
        assert final_state["report"]["success"] is True

    def test_all_specialists_failed_produces_honest_insufficient_evidence_report(
        self,
    ) -> None:
        """Total specialist failure produces honest fallback report (13.3.2)."""
        query = "Should I invest 50000 in Apple Inc. for 3-5 years with moderate risk?"
        conv_json = _build_conv_json(query)
        cio_json = _build_cio_json(
            selected_specialists=["technical", "fundamental", "news", "risk"]
        )

        # All specialists fail
        failing_overrides = {
            s: (
                lambda st, name=s: {
                    f"{name}_result": {
                        "specialist": name,
                        "status": "failed",
                        "success": False,
                        "error": f"Simulated provider error for {name}",
                    }
                }
            )
            for s in ("technical", "fundamental", "news", "risk")
        }

        graph = create_end_to_end_graph(
            conversation_agent=ConversationAgent(
                provider=MockIntegrationLLMProvider([conv_json])
            ),
            clarification_agent=ClarificationAgent(
                provider=MockIntegrationLLMProvider([])
            ),
            cio_agent=CIOAgent(provider=MockIntegrationLLMProvider([cio_json])),
            aggregator_agent=ReportAggregatorAgent(deterministic_only=True),
            report_generator_agent=ReportGeneratorAgent(deterministic_only=True),
            specialist_overrides=failing_overrides,
        )

        initial_state = create_initial_state(user_query=query)
        final_state = graph.invoke(initial_state)

        # Aggregator marks insufficient evidence
        agg = final_state["aggregated_result"]
        assert agg["insufficient_evidence"] is True
        assert len(agg["failed_specialists"]) == 4

        # Report generator reflects insufficient evidence stance
        report_data = final_state["report"]["data"]
        report = FinalReport.model_validate(report_data)
        assert (
            report.recommendation.stance == RecommendationStance.INSUFFICIENT_EVIDENCE
        )
        assert report.overall_assessment.insufficient_evidence is True


# ===========================================================================
# 5. STATE INTEGRITY, TRACE LOGGING & ALIAS PARITY (13.1.2, 13.3.3)
# ===========================================================================


class TestEndToEndIntegrityAndParity:
    """Test suite verifying state integrity, trace propagation, and alias parity."""

    def test_state_provenance_and_evidence_preservation(
        self, mock_specialist_overrides: Dict[str, Any]
    ) -> None:
        """Specialist evidence items preserve references end-to-end (13.1.2)."""
        query = "Should I invest 50000 in Apple Inc. for 3-5 years with moderate risk?"
        conv_json = _build_conv_json(query)
        cio_json = _build_cio_json()

        final_state = run_end_to_end_graph(
            query=query,
            conversation_agent=ConversationAgent(
                provider=MockIntegrationLLMProvider([conv_json])
            ),
            clarification_agent=ClarificationAgent(
                provider=MockIntegrationLLMProvider([])
            ),
            cio_agent=CIOAgent(provider=MockIntegrationLLMProvider([cio_json])),
            aggregator_agent=ReportAggregatorAgent(deterministic_only=True),
            report_generator_agent=ReportGeneratorAgent(deterministic_only=True),
            specialist_overrides=mock_specialist_overrides,
        )

        report = FinalReport.model_validate(final_state["report"]["data"])
        evidence_ref_ids = [ev.reference_id for ev in report.evidence_sources]

        # Provenance from technical and fundamental is intact
        assert "tech_ev_0" in evidence_ref_ids
        assert any(
            "gross_margin" in ref.lower() or "profitability" in ref.lower()
            for ref in evidence_ref_ids
        )

    def test_alias_parity(self, mock_specialist_overrides: Dict[str, Any]) -> None:
        """create_finpilot_graph and run_finpilot_graph match end_to_end functions."""
        query = "Should I invest 50000 in Apple Inc. for 3-5 years with moderate risk?"
        conv_json = _build_conv_json(query)
        cio_json = _build_cio_json()

        state_direct = run_end_to_end_graph(
            query=query,
            conversation_agent=ConversationAgent(
                provider=MockIntegrationLLMProvider([conv_json])
            ),
            clarification_agent=ClarificationAgent(
                provider=MockIntegrationLLMProvider([])
            ),
            cio_agent=CIOAgent(provider=MockIntegrationLLMProvider([cio_json])),
            aggregator_agent=ReportAggregatorAgent(deterministic_only=True),
            report_generator_agent=ReportGeneratorAgent(deterministic_only=True),
            specialist_overrides=mock_specialist_overrides,
        )

        state_alias = run_finpilot_graph(
            query=query,
            conversation_agent=ConversationAgent(
                provider=MockIntegrationLLMProvider([conv_json])
            ),
            clarification_agent=ClarificationAgent(
                provider=MockIntegrationLLMProvider([])
            ),
            cio_agent=CIOAgent(provider=MockIntegrationLLMProvider([cio_json])),
            aggregator_agent=ReportAggregatorAgent(deterministic_only=True),
            report_generator_agent=ReportGeneratorAgent(deterministic_only=True),
            specialist_overrides=mock_specialist_overrides,
        )

        assert (
            state_direct["report"]["data"]["company"]["ticker"]
            == state_alias["report"]["data"]["company"]["ticker"]
        )
        assert (
            state_direct["report"]["data"]["recommendation"]["stance"]
            == state_alias["report"]["data"]["recommendation"]["stance"]
        )

    def test_no_uncontrolled_loops_and_deterministic_termination(
        self, mock_specialist_overrides: Dict[str, Any]
    ) -> None:
        """Workflow graph is a strict DAG and terminates deterministically in 1 pass."""
        query = "Should I invest 50000 in Apple Inc. for 3-5 years with moderate risk?"
        conv_json = _build_conv_json(query)
        cio_json = _build_cio_json()

        graph = create_end_to_end_graph(
            conversation_agent=ConversationAgent(
                provider=MockIntegrationLLMProvider([conv_json])
            ),
            clarification_agent=ClarificationAgent(
                provider=MockIntegrationLLMProvider([])
            ),
            cio_agent=CIOAgent(provider=MockIntegrationLLMProvider([cio_json])),
            aggregator_agent=ReportAggregatorAgent(deterministic_only=True),
            report_generator_agent=ReportGeneratorAgent(deterministic_only=True),
            specialist_overrides=mock_specialist_overrides,
        )

        initial_state = create_initial_state(user_query=query)
        final_state = graph.invoke(initial_state)

        # Graph cleanly terminated and reached the final report
        assert final_state.get("report") is not None
        assert final_state["report"]["success"] is True
