"""Comprehensive tests for Phase 15.1 FastAPI Backend Integration (API Foundation).

Validates:
1. POST /api/v1/analysis: Happy path company analysis returning FinalReport.
2. POST /api/v1/chat: Clarification halting path when queries are underspecified.
3. POST /api/v1/clarification: Resumption path submitting answers and report.
4. POST /api/v1/research/query: Research query against documents.
5. GET  /api/v1/analysis/{analysis_id}/status: Status polling and 404 on unknown ID.
6. GET  /api/v1/reports/{report_id}: Report retrieval in JSON, Markdown, and Summary.
7. Unknown report ID -> 404.
8. Request validation errors -> 422 with sanitized details.
9. FinalReport schema structure preservation.
10. Trace ID propagation across request and response.
11. Safe internal error handling on graph crashes without leaking tracebacks.
12. Health check endpoint regression (/health and /api/v1/health).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.agents.report_schema import (
    STANDARD_DISCLAIMER,
    FinalReport,
    OverallAssessmentSection,
    RecommendationStance,
    ReportCompanyInfo,
    ReportRecommendation,
)
from app.agents.state import GraphState
from app.api.v1.analysis import get_graph_runner
from app.main import create_application

# ===========================================================================
# TEST FIXTURES & DETERMINISTIC MOCKS
# ===========================================================================


def create_deterministic_test_report(ticker: str = "AAPL") -> FinalReport:
    """Build a valid FinalReport fixture fulfilling the schema contract."""
    return FinalReport(
        company=ReportCompanyInfo(
            ticker=ticker,
            company_name="Apple Inc.",
            currency="USD",
        ),
        recommendation=ReportRecommendation(
            stance=RecommendationStance.FAVORABLE,
            rationale="Robust ecosystem fundamentals and durable operating margin.",
            profile_alignment="Well-aligned with moderate risk long-term horizon.",
            monitoring_points=[
                "Upcoming quarterly disclosures",
                "Gross margin stability",
            ],
        ),
        overall_assessment=OverallAssessmentSection(
            data_completeness_ratio=1.0,
            specialist_consensus="favorable",
            synthesis=(
                "Strong operational performance and solid balance sheet liquidity."
            ),
        ),
        disclaimer=STANDARD_DISCLAIMER,
    )


@pytest.fixture
def test_app():
    """Create fresh FastAPI test application instance."""
    app = create_application()
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(test_app) -> TestClient:
    """FastAPI TestClient bound to the test app."""
    return TestClient(test_app)


# ===========================================================================
# 1. ANALYSIS HAPPY PATH (15.1.3)
# ===========================================================================


class TestCompanyAnalysisAPI:
    """Tests for POST /api/v1/analysis endpoint."""

    def test_company_analysis_happy_path(self, test_app, client: TestClient) -> None:
        """Valid analysis request runs graph and returns structured FinalReport."""
        expected_report = create_deterministic_test_report("AAPL")

        def _mock_runner(**kwargs: Any) -> GraphState:
            return {
                "user_query": kwargs.get("query", ""),
                "ticker": kwargs.get("ticker", "AAPL"),
                "clarified_request": {
                    "clarification_needed": False,
                    "clarification_questions": [],
                },
                "investor_profile": kwargs.get("investor_profile")
                or {"ticker": "AAPL"},
                "report": {
                    "success": True,
                    "data": expected_report.model_dump(),
                },
                "trace_id": kwargs.get("trace_id", "trace-test-123"),
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_runner

        payload = {
            "ticker": "AAPL",
            "target_company": "Apple Inc.",
            "query": "Comprehensive investment analysis for Apple Inc.",
            "investor_profile": {
                "investment_goal": "growth",
                "time_horizon": "3-5 years",
                "capital_amount": 50000.0,
                "risk_tolerance": "moderate",
            },
            "documents_available": False,
            "trace_id": "custom-trace-999",
        }

        response = client.post("/api/v1/analysis", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["status"] == "completed"
        assert data["clarification_needed"] is False
        assert data["trace_id"] == "custom-trace-999"
        assert uuid.UUID(data["analysis_id"])  # Valid UUID
        assert data["report_id"] is not None
        assert uuid.UUID(data["report_id"])  # Valid UUID

        # Validate structured report preservation
        report_data = data["report"]
        assert report_data is not None
        assert report_data["company"]["ticker"] == "AAPL"
        assert report_data["recommendation"]["stance"] == "favorable"
        assert "disclaimer" in report_data


# ===========================================================================
# 2. CHAT / QUERY CLARIFICATION HALTING (15.1.2)
# ===========================================================================


class TestChatQueryAPI:
    """Tests for POST /api/v1/chat endpoint."""

    def test_chat_query_underspecified_halts_with_clarification(
        self, test_app, client: TestClient
    ) -> None:
        """Underspecified query returns clarification_needed=True and questions list."""

        def _mock_clarification_runner(**kwargs: Any) -> GraphState:
            return {
                "user_query": kwargs.get("query", ""),
                "clarified_request": {
                    "clarification_needed": True,
                    "clarification_questions": [
                        "What is your investment time horizon?",
                        "How much capital do you plan to invest?",
                    ],
                },
                "investor_profile": {
                    "target_company": "Tesla Inc.",
                    "ticker": "TSLA",
                    "profile_complete": False,
                },
                "report": None,
                "trace_id": kwargs.get("trace_id", "trace-chat-456"),
            }

        test_app.dependency_overrides[get_graph_runner] = (
            lambda: _mock_clarification_runner
        )

        payload = {
            "query": "Should I invest in Tesla?",
            "documents_available": False,
        }

        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["status"] == "clarification_needed"
        assert data["clarification_needed"] is True
        assert len(data["clarification_questions"]) == 2
        assert "investment time horizon" in data["clarification_questions"][0]
        assert data["report"] is None
        assert data["report_id"] is None


# ===========================================================================
# 3. CLARIFICATION SUBMISSION & RESUMPTION (15.1.4)
# ===========================================================================


class TestClarificationSubmitAPI:
    """Tests for POST /api/v1/clarification endpoint."""

    def test_submit_clarification_resumes_to_report(
        self, test_app, client: TestClient
    ) -> None:
        """Clarification submission supplies answers and completes final report."""
        expected_report = create_deterministic_test_report("TSLA")

        def _mock_resumed_runner(**kwargs: Any) -> GraphState:
            # Confirm answers were passed through
            assert kwargs.get("clarification_answers") == {
                "time_horizon": "5 years",
                "capital_amount": 25000.0,
            }
            return {
                "user_query": kwargs.get("query", ""),
                "ticker": "TSLA",
                "clarified_request": {
                    "clarification_needed": False,
                    "clarification_questions": [],
                },
                "investor_profile": {
                    "target_company": "Tesla Inc.",
                    "ticker": "TSLA",
                    "time_horizon": "5 years",
                    "capital_amount": 25000.0,
                    "profile_complete": True,
                },
                "report": {
                    "success": True,
                    "data": expected_report.model_dump(),
                },
                "trace_id": kwargs.get("trace_id", "trace-clar-789"),
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_resumed_runner

        payload = {
            "ticker": "TSLA",
            "clarification_answers": {
                "time_horizon": "5 years",
                "capital_amount": 25000.0,
            },
        }

        response = client.post("/api/v1/clarification", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["status"] == "completed"
        assert data["clarification_needed"] is False
        assert data["report"] is not None
        assert data["report_id"] is not None


# ===========================================================================
# 4. RESEARCH QUERY (15.1.6)
# ===========================================================================


class TestResearchQueryAPI:
    """Tests for POST /api/v1/research/query endpoint."""

    def test_research_query_executes_with_documents(
        self, test_app, client: TestClient
    ) -> None:
        """Research query sets documents_available=True and runs workflow."""

        def _mock_research_runner(**kwargs: Any) -> GraphState:
            assert kwargs.get("documents_available") is True
            return {
                "user_query": kwargs.get("query", ""),
                "ticker": kwargs.get("ticker", "AAPL"),
                "clarified_request": {
                    "clarification_needed": False,
                    "clarification_questions": [],
                },
                "report": {
                    "success": True,
                    "data": create_deterministic_test_report("AAPL").model_dump(),
                },
                "trace_id": kwargs.get("trace_id", "trace-res-101"),
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_research_runner

        payload = {
            "query": "What are Apple's gross margin disclosures in the 10-K?",
            "ticker": "AAPL",
            "documents_available": True,
        }

        response = client.post("/api/v1/research/query", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["status"] == "completed"
        assert data["report"] is not None


# ===========================================================================
# 5. STATUS POLLING & REPORT RETRIEVAL (15.1.7 & 15.1.8)
# ===========================================================================


class TestStatusAndReportRetrievalAPI:
    """Tests for GET status and report retrieval endpoints."""

    def test_status_polling_and_report_retrieval_lifecycle(
        self, test_app, client: TestClient
    ) -> None:
        """Running analysis registers IDs retrievable by status and report endpoints."""
        expected_report = create_deterministic_test_report("NVDA")

        def _mock_runner(**kwargs: Any) -> GraphState:
            return {
                "user_query": kwargs.get("query", ""),
                "ticker": "NVDA",
                "clarified_request": {
                    "clarification_needed": False,
                    "clarification_questions": [],
                },
                "report": {
                    "success": True,
                    "data": expected_report.model_dump(),
                },
                "trace_id": "trace-lifecycle-123",
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_runner

        # 1. Trigger analysis
        post_resp = client.post("/api/v1/analysis", json={"ticker": "NVDA"})
        assert post_resp.status_code == status.HTTP_200_OK
        analysis_id = post_resp.json()["analysis_id"]
        report_id = post_resp.json()["report_id"]

        # 2. Poll status (15.1.7)
        status_resp = client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert status_resp.status_code == status.HTTP_200_OK
        status_data = status_resp.json()
        assert status_data["analysis_id"] == analysis_id
        assert status_data["status"] == "completed"
        assert status_data["ticker"] == "NVDA"
        assert status_data["report_id"] == report_id

        # 3. Retrieve report in JSON format (15.1.8)
        rep_json_resp = client.get(f"/api/v1/reports/{report_id}?format=json")
        assert rep_json_resp.status_code == status.HTTP_200_OK
        rep_json_data = rep_json_resp.json()
        assert rep_json_data["format"] == "json"
        assert rep_json_data["report"]["company"]["ticker"] == "NVDA"

        # 4. Retrieve report in Markdown format (15.1.8)
        rep_md_resp = client.get(f"/api/v1/reports/{report_id}?format=markdown")
        assert rep_md_resp.status_code == status.HTTP_200_OK
        rep_md_data = rep_md_resp.json()
        assert rep_md_data["format"] == "markdown"
        assert "NVDA" in rep_md_data["rendered_content"]
        assert "Investment Research Report" in rep_md_data["rendered_content"]

        # 5. Retrieve report in Executive Summary format (15.1.8)
        rep_sum_resp = client.get(f"/api/v1/reports/{report_id}?format=summary")
        assert rep_sum_resp.status_code == status.HTTP_200_OK
        rep_sum_data = rep_sum_resp.json()
        assert rep_sum_data["format"] == "summary"
        assert "INVESTMENT RESEARCH REPORT" in rep_sum_data["rendered_content"]
        assert "NVDA" in rep_sum_data["rendered_content"]

    def test_unknown_analysis_id_returns_404(self, client: TestClient) -> None:
        """Nonexistent analysis ID returns HTTP 404."""
        unknown_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/analysis/{unknown_id}/status")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert (
            f"Analysis '{unknown_id}' not found" in response.json()["error"]["message"]
        )

    def test_unknown_report_id_returns_404(self, client: TestClient) -> None:
        """Nonexistent report ID returns HTTP 404."""
        unknown_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/reports/{unknown_id}")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert f"Report '{unknown_id}' not found" in response.json()["error"]["message"]


# ===========================================================================
# 6. VALIDATION & ERROR HANDLING (HTTP 422 & 500)
# ===========================================================================


class TestValidationAndErrorHandlingAPI:
    """Tests for input validation and safe error responses."""

    def test_invalid_capital_amount_returns_422(self, client: TestClient) -> None:
        """Negative or zero capital amount returns HTTP 422 Unprocessable Entity."""
        payload = {
            "ticker": "AAPL",
            "investor_profile": {
                "capital_amount": -500.0,  # Invalid: must be gt 0
            },
        }
        response = client.post("/api/v1/analysis", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_empty_query_in_chat_returns_422(self, client: TestClient) -> None:
        """Empty query string returns HTTP 422."""
        response = client.post("/api/v1/chat", json={"query": ""})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_empty_clarification_answers_returns_422(self, client: TestClient) -> None:
        """Empty clarification answers dictionary returns HTTP 422."""
        response = client.post(
            "/api/v1/clarification",
            json={"clarification_answers": {}},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_unhandled_runner_crash_returns_safe_500(
        self, test_app, client: TestClient
    ) -> None:
        """Unhandled runner exception returns safe HTTP 500 without traceback."""

        def _crashing_runner(**kwargs: Any) -> GraphState:
            raise RuntimeError("Database connection string: secret_db_pw_123")

        test_app.dependency_overrides[get_graph_runner] = lambda: _crashing_runner

        response = client.post(
            "/api/v1/analysis",
            json={"ticker": "AAPL"},
        )
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        # Ensure secret / raw traceback is NOT in the user-facing message
        assert "secret_db_pw_123" not in str(data)
        assert "Failed to execute company analysis" in data["error"]["message"]


# ===========================================================================
# 7. REGRESSION: PRE-EXISTING HEALTH CHECKS
# ===========================================================================


class TestPreExistingHealthEndpointsRegression:
    """Verify pre-existing endpoints continue functioning as expected."""

    def test_health_check_top_level(self, client: TestClient) -> None:
        """Top-level /health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "healthy"

    def test_health_check_v1(self, client: TestClient) -> None:
        """Versioned /api/v1/health endpoint returns healthy status."""
        response = client.get("/api/v1/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "healthy"
