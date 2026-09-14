"""Comprehensive tests for Phase 15.3 Async Execution and Status Tracking.

Validates:
1. 15.3.1 Support background task execution alongside synchronous execution:
   - Default synchronous path returns completed report directly.
   - Asynchronous path (background=true) returns immediate 'running' status.
2. 15.3.2 Status tracking for in-progress analyses:
   - Polling status while task is running shows status='running' and completed_at=None.
   - Polling status after completion shows status='completed', report_id, completed_at.
   - Background clarification halt correctly transitions to 'clarification_needed'.
   - Background failure correctly transitions to 'failed' with safe error message.
   - Completed report is retrievable via GET /reports/{report_id}.
3. Background execution across endpoints:
   - POST /api/v1/analysis?background=true
   - POST /api/v1/chat?background=true
   - POST /api/v1/clarification?background=true
   - POST /api/v1/research/query?background=true
"""

from __future__ import annotations

import threading
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


def create_deterministic_test_report(ticker: str = "AAPL") -> FinalReport:
    """Build a deterministic FinalReport fixture."""
    return FinalReport(
        company=ReportCompanyInfo(
            ticker=ticker,
            company_name="Apple Inc.",
            currency="USD",
        ),
        recommendation=ReportRecommendation(
            stance=RecommendationStance.FAVORABLE,
            rationale="Robust operating margins and consistent capital returns.",
            profile_alignment="Fits moderate long-term profile.",
            monitoring_points=["Services segment growth", "Supply chain stability"],
        ),
        overall_assessment=OverallAssessmentSection(
            data_completeness_ratio=1.0,
            specialist_consensus="favorable",
            synthesis="Solid operational fundamentals.",
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
# 1. ASYNC BACKGROUND EXECUTION LIFECYCLE (15.3.1, 15.3.2)
# ===========================================================================


class TestAsyncExecutionLifecycle:
    """Tests for asynchronous background analysis execution and status polling."""

    def test_company_analysis_background_lifecycle(
        self, test_app, client: TestClient
    ) -> None:
        """Background analysis returns 'running', completes, and report is saved."""
        expected_report = create_deterministic_test_report("GOOGL")

        def _mock_runner(**kwargs: Any) -> GraphState:
            return {
                "user_query": kwargs.get("query", ""),
                "ticker": "GOOGL",
                "clarified_request": {"clarification_needed": False},
                "report": {
                    "success": True,
                    "data": expected_report.model_dump(),
                },
                "trace_id": "trace-async-1",
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_runner

        # 1. Trigger background execution
        response = client.post(
            "/api/v1/analysis?background=true", json={"ticker": "GOOGL"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "running"
        assert data["report"] is None
        assert data["report_id"] is None
        analysis_id = data["analysis_id"]

        # 2. Poll status (after background task finished in TestClient)
        status_resp = client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert status_resp.status_code == status.HTTP_200_OK
        status_data = status_resp.json()
        assert status_data["status"] == "completed"
        assert status_data["ticker"] == "GOOGL"
        assert status_data["report_id"] is not None
        assert status_data["completed_at"] is not None
        report_id = status_data["report_id"]

        # 3. Retrieve completed report by ID
        report_resp = client.get(f"/api/v1/reports/{report_id}?format=json")
        assert report_resp.status_code == status.HTTP_200_OK
        assert report_resp.json()["report"]["company"]["ticker"] == "GOOGL"

    def test_in_progress_status_tracking_while_executing(
        self, test_app, client: TestClient
    ) -> None:
        """Verify status shows 'running' while background execution is in-flight."""
        from app.api.v1.analysis import _RECENT_ANALYSES, _REGISTRY_LOCK

        resume_event = threading.Event()
        started_event = threading.Event()
        expected_report = create_deterministic_test_report("TSLA")

        def _blocking_runner(**kwargs: Any) -> GraphState:
            started_event.set()
            resume_event.wait(timeout=3.0)
            return {
                "user_query": kwargs.get("query", ""),
                "ticker": "TSLA",
                "clarified_request": {"clarification_needed": False},
                "report": {
                    "success": True,
                    "data": expected_report.model_dump(),
                },
                "trace_id": "trace-blocking-1",
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _blocking_runner

        thread = threading.Thread(
            target=lambda: client.post(
                "/api/v1/analysis?background=true", json={"ticker": "TSLA"}
            )
        )
        thread.start()

        try:
            assert started_event.wait(timeout=2.0), "Runner did not start in time"
            # Locate the in-progress analysis in the registry
            with _REGISTRY_LOCK:
                running_entries = [
                    k
                    for k, v in _RECENT_ANALYSES.items()
                    if v.get("ticker") == "TSLA" and v.get("status") == "running"
                ]
            assert len(running_entries) >= 1
            analysis_id = running_entries[0]

            # Poll status while still in-flight
            mid_status = client.get(f"/api/v1/analysis/{analysis_id}/status")
            assert mid_status.status_code == status.HTTP_200_OK
            mid_data = mid_status.json()
            assert mid_data["status"] == "running"
            assert mid_data["completed_at"] is None
            assert mid_data["report_id"] is None
            assert mid_data["progress_stage"] == "running"
        finally:
            # Resume the runner and wait for thread to finish
            resume_event.set()
            thread.join(timeout=3.0)

        # Poll status after completion
        final_status = client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert final_status.status_code == status.HTTP_200_OK
        final_data = final_status.json()
        assert final_data["status"] == "completed"
        assert final_data["completed_at"] is not None
        assert final_data["report_id"] is not None
        assert final_data["progress_stage"] == "completed"

    def test_background_clarification_halt_lifecycle(
        self, test_app, client: TestClient
    ) -> None:
        """Background chat halting transitions to clarification_needed."""

        def _clarification_runner(**kwargs: Any) -> GraphState:
            return {
                "user_query": kwargs.get("query", ""),
                "clarified_request": {
                    "clarification_needed": True,
                    "clarification_questions": [
                        "What is your investment time horizon?"
                    ],
                },
                "trace_id": "trace-clarify-async",
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _clarification_runner

        response = client.post(
            "/api/v1/chat?background=true", json={"query": "Evaluate growth stock"}
        )
        assert response.status_code == status.HTTP_200_OK
        analysis_id = response.json()["analysis_id"]

        status_resp = client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert status_resp.status_code == status.HTTP_200_OK
        status_data = status_resp.json()
        assert status_data["status"] == "clarification_needed"
        assert len(status_data["clarification_questions"]) == 1
        assert "time horizon" in status_data["clarification_questions"][0]

    def test_background_execution_safe_failure_handling(
        self, test_app, client: TestClient
    ) -> None:
        """Background error updates status to 'failed' without leaking secrets."""

        def _crashing_runner(**kwargs: Any) -> GraphState:
            raise RuntimeError("Database password leaked: secret_pass_xyz")

        test_app.dependency_overrides[get_graph_runner] = lambda: _crashing_runner

        response = client.post(
            "/api/v1/analysis?background=true", json={"ticker": "AAPL"}
        )
        assert response.status_code == status.HTTP_200_OK
        analysis_id = response.json()["analysis_id"]

        status_resp = client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert status_resp.status_code == status.HTTP_200_OK
        status_data = status_resp.json()
        assert status_data["status"] == "failed"
        assert "secret_pass_xyz" not in str(status_data)
        assert "Failed to execute company analysis" in status_data["error"]

    def test_background_clarification_resumption(
        self, test_app, client: TestClient
    ) -> None:
        """Submitting clarification with background=true completes asynchronously."""
        expected_report = create_deterministic_test_report("MSFT")

        def _resume_runner(**kwargs: Any) -> GraphState:
            return {
                "user_query": kwargs.get("query", ""),
                "ticker": "MSFT",
                "clarified_request": {"clarification_needed": False},
                "report": {
                    "success": True,
                    "data": expected_report.model_dump(),
                },
                "trace_id": "trace-resume-async",
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _resume_runner

        response = client.post(
            "/api/v1/clarification?background=true",
            json={
                "clarification_answers": {"time_horizon": "5 years"},
                "ticker": "MSFT",
            },
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "running"
        analysis_id = response.json()["analysis_id"]

        status_resp = client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert status_resp.status_code == status.HTTP_200_OK
        assert status_resp.json()["status"] == "completed"
        assert status_resp.json()["report_id"] is not None

    def test_background_research_query(self, test_app, client: TestClient) -> None:
        """Research query with background=true executes and completes asynchronously."""
        expected_report = create_deterministic_test_report("NVDA")

        def _research_runner(**kwargs: Any) -> GraphState:
            return {
                "user_query": kwargs.get("query", ""),
                "ticker": "NVDA",
                "clarified_request": {"clarification_needed": False},
                "report": {
                    "success": True,
                    "data": expected_report.model_dump(),
                },
                "trace_id": "trace-research-async",
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _research_runner

        response = client.post(
            "/api/v1/research/query?background=true",
            json={"query": "What are the latest gross margin disclosures?"},
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "running"
        analysis_id = response.json()["analysis_id"]

        status_resp = client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert status_resp.status_code == status.HTTP_200_OK
        assert status_resp.json()["status"] == "completed"

    def test_sync_execution_default_backward_compatible(
        self, test_app, client: TestClient
    ) -> None:
        """Omitting background runs synchronously and returns report immediately."""
        expected_report = create_deterministic_test_report("AMZN")

        def _mock_runner(**kwargs: Any) -> GraphState:
            return {
                "user_query": kwargs.get("query", ""),
                "ticker": "AMZN",
                "clarified_request": {"clarification_needed": False},
                "report": {
                    "success": True,
                    "data": expected_report.model_dump(),
                },
                "trace_id": "trace-sync-default",
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_runner

        response = client.post("/api/v1/analysis", json={"ticker": "AMZN"})
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "completed"
        assert data["report"] is not None
        assert data["report_id"] is not None
