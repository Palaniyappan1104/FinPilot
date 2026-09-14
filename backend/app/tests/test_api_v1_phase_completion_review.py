"""Phase 15.6 Phase Completion Review Test Suite.

Requirement:
15.6.1 Verify the entire user journey (query -> clarification -> analysis -> report)
works purely through the API.

Covers:
1. Complete Synchronous User Journey:
   - Initial chat query halting for missing constraints
   - Clarification submission resuming to report generation
   - Status polling confirming state transitions
   - Multi-format report retrieval (JSON, Markdown, Summary)
2. Complete Asynchronous / Background User Journey:
   - Chat query with background analysis triggering
   - In-flight status tracking transitioning from running to completed
   - Report retrieval matching generated report_id
3. Document-augmented research journey purely through the API
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.agents.report_schema import (
    STANDARD_DISCLAIMER,
    FinalReport,
    FundamentalReportSection,
    NewsReportSection,
    OverallAssessmentSection,
    RecommendationStance,
    ReportCapitalInfo,
    ReportCompanyInfo,
    ReportRecommendation,
    RiskReportSection,
    TechnicalReportSection,
)
from app.agents.state import GraphState
from app.api.v1.analysis import get_graph_runner
from app.api.v1.documents import get_document_storage
from app.main import create_application
from app.storage.local import LocalDocumentStorage

VALID_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n"
    b"<< /Type /Catalog /Pages 2 0 R >>\n"
    b"endobj\n"
    b"2 0 obj\n"
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
    b"endobj\n"
    b"3 0 obj\n"
    b"<< /Type /Page /Parent 2 0 R >>\n"
    b"endobj\n"
    b"xref\n"
    b"0 4\n"
    b"trailer\n"
    b"<< /Root 1 0 R >>\n"
    b"%%EOF\n"
)


def create_comprehensive_final_report(ticker: str = "AAPL") -> FinalReport:
    """Build a complete multi-section FinalReport for end-to-end API review."""
    return FinalReport(
        company=ReportCompanyInfo(
            ticker=ticker,
            name="Apple Inc.",
            currency="USD",
        ),
        recommendation=ReportRecommendation(
            stance=RecommendationStance.FAVORABLE,
            rationale=(
                "Exceptional ecosystem retention, gross margin expansion, "
                "and cash flow generation."
            ),
            profile_alignment=(
                "Matches moderate-risk long-term equity growth objective."
            ),
            monitoring_points=[
                "Services segment gross margins",
                "Greater China market revenue trajectory",
                "Regulatory scrutiny regarding App Store policies",
            ],
        ),
        capital=ReportCapitalInfo(
            amount=50000.0,
            currency="USD",
            formatted="$50,000",
        ),
        technical=TechnicalReportSection(
            summary=(
                "Constructive technical structure with price holding above key "
                "moving averages."
            ),
            trend="bullish",
            support_levels=[175.0, 170.0],
            resistance_levels=[195.0, 200.0],
        ),
        fundamental=FundamentalReportSection(
            summary=(
                "Solid revenue expansion, resilient operating margins, and "
                "healthy free cash flow."
            ),
            financial_health=(
                "Outstanding liquidity and investment grade balance sheet"
            ),
            valuation="Fairly valued relative to historical multiples",
        ),
        news=NewsReportSection(
            summary=(
                "Constructive news coverage centered on AI developments and "
                "developer adoption."
            ),
            overall_sentiment="positive",
            sentiment_score=0.65,
        ),
        risk=RiskReportSection(
            summary=(
                "Risks are centered on regulatory antitrust actions and "
                "consumer cyclicality."
            ),
            overall_risk_level="moderate",
            top_risk_factors=[
                "Hardware cycle elongation",
                "Antitrust litigation",
            ],
        ),
        overall_assessment=OverallAssessmentSection(
            data_completeness_ratio=1.0,
            specialist_consensus="favorable",
            synthesis=(
                "High-conviction investment idea with stable dividend and "
                "strong share repurchase capacity."
            ),
        ),
        disclaimer=STANDARD_DISCLAIMER,
    )


@pytest.fixture
def test_app(tmp_path: Path):
    """Create fresh FastAPI test application instance with isolated storage."""
    app = create_application()
    test_storage = LocalDocumentStorage(base_directory=tmp_path)
    app.dependency_overrides[get_document_storage] = lambda: test_storage
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(test_app) -> TestClient:
    """FastAPI TestClient bound to test app."""
    return TestClient(test_app)


# ===========================================================================
# 15.6.1 PHASE COMPLETION REVIEW: USER JOURNEY VERIFICATION
# ===========================================================================


class TestPhase156CompletionReviewUserJourney:
    """Requirement 15.6.1: Verify entire user journey purely through the API."""

    def test_sync_full_journey_query_clarification_analysis_report(
        self, test_app, client: TestClient
    ) -> None:
        """Verify synchronous journey: query -> clarification -> analysis -> report.

        Executed strictly through API endpoints with zero direct LangGraph calls:
        1. POST /api/v1/documents/upload -> upload supporting filing
        2. POST /api/v1/chat -> underspecified goal triggers clarification
        3. GET  /api/v1/analysis/{id}/status -> check clarification status
        4. POST /api/v1/clarification -> submit missing fields & generate report
        5. GET  /api/v1/analysis/{id}/status -> check completed status
        6. GET  /api/v1/reports/{id} -> retrieve JSON report
        7. GET  /api/v1/reports/{id}?format=markdown -> retrieve Markdown report
        8. GET  /api/v1/reports/{id}?format=summary -> retrieve Executive Summary
        """
        report_fixture = create_comprehensive_final_report("AAPL")

        def _mock_graph_runner(**kwargs: Any) -> GraphState:
            query = kwargs.get("query", "")
            answers = kwargs.get("clarification_answers")

            # Stage 1: Underspecified query triggers clarification needed
            if query == "Should I buy Apple stock?" and not answers:
                return {
                    "user_query": query,
                    "ticker": "AAPL",
                    "clarified_request": {
                        "clarification_needed": True,
                        "clarification_questions": [
                            "What is your target investment horizon?",
                            "What is your risk tolerance level?",
                            "How much capital do you plan to allocate?",
                        ],
                    },
                    "investor_profile": {"ticker": "AAPL"},
                    "trace_id": kwargs.get("trace_id", "trace-sync-review-1"),
                }

            # Stage 2: Clarification provided -> complete analysis report
            return {
                "user_query": query,
                "ticker": kwargs.get("ticker", "AAPL"),
                "clarified_request": {
                    "clarification_needed": False,
                    "clarification_questions": [],
                },
                "investor_profile": {
                    "ticker": "AAPL",
                    "target_company": "Apple Inc.",
                    "risk_tolerance": "moderate",
                    "time_horizon": "long-term",
                    "capital_amount": 50000.0,
                },
                "report": {
                    "success": True,
                    "data": report_fixture.model_dump(),
                },
                "trace_id": kwargs.get("trace_id", "trace-sync-review-2"),
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_graph_runner

        # 1. Document upload
        upload_resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("AAPL_10K.pdf", VALID_PDF_BYTES, "application/pdf")},
            data={"ticker": "AAPL", "document_type": "annual_report"},
        )
        assert upload_resp.status_code == status.HTTP_201_CREATED
        upload_body = upload_resp.json()
        assert upload_body["ticker"] == "AAPL"
        assert upload_body["upload_status"] == "stored"

        # 2. Conversational query (underspecified)
        chat_resp = client.post(
            "/api/v1/chat",
            json={
                "query": "Should I buy Apple stock?",
                "trace_id": "trace-review-sync-chat",
            },
        )
        assert chat_resp.status_code == status.HTTP_200_OK
        chat_body = chat_resp.json()
        assert chat_body["status"] == "clarification_needed"
        assert chat_body["clarification_needed"] is True
        assert len(chat_body["clarification_questions"]) == 3
        assert chat_body["report"] is None
        analysis_id = chat_body["analysis_id"]

        # 3. Status polling during clarification
        status_resp_1 = client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert status_resp_1.status_code == status.HTTP_200_OK
        assert status_resp_1.json()["status"] == "clarification_needed"
        assert status_resp_1.json()["report_id"] is None

        # 4. Clarification submission
        clarify_resp = client.post(
            "/api/v1/clarification",
            json={
                "analysis_id": analysis_id,
                "ticker": "AAPL",
                "clarification_answers": {
                    "time_horizon": "long-term",
                    "risk_tolerance": "moderate",
                    "capital_amount": 50000.0,
                },
                "trace_id": "trace-review-sync-clarify",
            },
        )
        assert clarify_resp.status_code == status.HTTP_200_OK
        clarify_body = clarify_resp.json()
        assert clarify_body["status"] == "completed"
        assert clarify_body["clarification_needed"] is False
        report_id = clarify_body["report_id"]
        assert report_id is not None
        assert clarify_body["report"]["company"]["ticker"] == "AAPL"

        # 5. Status polling after completion
        status_resp_2 = client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert status_resp_2.status_code == status.HTTP_200_OK
        assert status_resp_2.json()["status"] == "completed"
        assert status_resp_2.json()["report_id"] == report_id
        assert status_resp_2.json()["completed_at"] is not None

        # 6. Report retrieval: JSON
        rep_json = client.get(f"/api/v1/reports/{report_id}")
        assert rep_json.status_code == status.HTTP_200_OK
        assert rep_json.json()["format"] == "json"
        assert rep_json.json()["report"]["company"]["ticker"] == "AAPL"
        assert rep_json.json()["report"]["company"]["name"] == "Apple Inc."
        assert rep_json.json()["report"]["recommendation"]["stance"] == "favorable"
        assert rep_json.json()["report"]["capital"]["amount"] == 50000.0

        # 7. Report retrieval: Markdown
        rep_md = client.get(f"/api/v1/reports/{report_id}?format=markdown")
        assert rep_md.status_code == status.HTTP_200_OK
        assert rep_md.json()["format"] == "markdown"
        assert "Investment Research Report" in rep_md.json()["rendered_content"]
        assert "Apple Inc." in rep_md.json()["rendered_content"]
        assert "AAPL" in rep_md.json()["rendered_content"]

        # 8. Report retrieval: Summary
        rep_sum = client.get(f"/api/v1/reports/{report_id}?format=summary")
        assert rep_sum.status_code == status.HTTP_200_OK
        assert rep_sum.json()["format"] == "summary"
        assert "INVESTMENT RESEARCH REPORT" in rep_sum.json()["rendered_content"]
        assert "AAPL" in rep_sum.json()["rendered_content"]

    def test_async_full_journey_with_background_status_polling(
        self, test_app, client: TestClient
    ) -> None:
        """Verify async background execution journey with status lifecycle polling."""
        report_fixture = create_comprehensive_final_report("MSFT")

        def _mock_graph_runner(**kwargs: Any) -> GraphState:
            return {
                "ticker": kwargs.get("ticker", "MSFT"),
                "clarified_request": {
                    "clarification_needed": False,
                    "clarification_questions": [],
                },
                "investor_profile": {
                    "ticker": "MSFT",
                    "capital_amount": 100000.0,
                    "risk_tolerance": "aggressive",
                },
                "report": {
                    "success": True,
                    "data": report_fixture.model_dump(),
                },
                "trace_id": "trace-async-review-1",
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_graph_runner

        # 1. Trigger background analysis
        post_resp = client.post(
            "/api/v1/analysis?background=true",
            json={
                "ticker": "MSFT",
                "investor_profile": {
                    "risk_tolerance": "aggressive",
                    "capital_amount": 100000.0,
                    "time_horizon": "5 years",
                },
            },
        )
        assert post_resp.status_code == status.HTTP_200_OK
        init_body = post_resp.json()
        assert init_body["status"] == "running"
        analysis_id = init_body["analysis_id"]

        # 2. Poll status (after background task completion)
        status_resp = client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert status_resp.status_code == status.HTTP_200_OK
        status_body = status_resp.json()
        assert status_body["status"] == "completed"
        assert status_body["ticker"] == "MSFT"
        report_id = status_body["report_id"]
        assert report_id is not None

        # 3. Retrieve report
        report_resp = client.get(f"/api/v1/reports/{report_id}?format=json")
        assert report_resp.status_code == status.HTTP_200_OK
        assert report_resp.json()["report"]["company"]["ticker"] == "MSFT"

    def test_research_document_query_journey(
        self, test_app, client: TestClient
    ) -> None:
        """Verify research querying against uploaded filings purely through the API."""
        report_fixture = create_comprehensive_final_report("NVDA")

        def _mock_graph_runner(**kwargs: Any) -> GraphState:
            return {
                "ticker": "NVDA",
                "user_query": kwargs.get("query", ""),
                "documents_available": True,
                "clarified_request": {
                    "clarification_needed": False,
                    "clarification_questions": [],
                },
                "report": {
                    "success": True,
                    "data": report_fixture.model_dump(),
                },
                "trace_id": "trace-research-review-1",
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_graph_runner

        # 1. Upload filing
        upload_resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("NVDA_10Q.pdf", VALID_PDF_BYTES, "application/pdf")},
            data={"ticker": "NVDA", "document_type": "company_report"},
        )
        assert upload_resp.status_code == status.HTTP_201_CREATED

        # 2. Query document research via API
        query_resp = client.post(
            "/api/v1/research/query",
            json={
                "ticker": "NVDA",
                "query": (
                    "What are the latest revenue trends in compute & networking?"
                ),
                "documents_available": True,
            },
        )
        assert query_resp.status_code == status.HTTP_200_OK
        query_body = query_resp.json()
        assert query_body["status"] == "completed"
        assert query_body["report_id"] is not None
        assert query_body["report"]["company"]["ticker"] == "NVDA"

        # 3. Verify report retrieved from document query
        report_id = query_body["report_id"]
        rep_resp = client.get(f"/api/v1/reports/{report_id}?format=json")
        assert rep_resp.status_code == status.HTTP_200_OK
        assert rep_resp.json()["report"]["company"]["ticker"] == "NVDA"
