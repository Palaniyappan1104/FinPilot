"""Phase 15.5 FastAPI Backend Testing Suite.

Implements:
- 15.5.1 API tests for every endpoint (happy path)
- 15.5.2 API tests for validation/error cases
- 15.5.3 Integration test: full user journey via API only (no direct graph calls)
"""

from __future__ import annotations

import uuid
from pathlib import Path
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


def create_mock_final_report(ticker: str = "AAPL") -> FinalReport:
    """Build a deterministic FinalReport instance fulfilling the schema."""
    return FinalReport(
        company=ReportCompanyInfo(
            ticker=ticker,
            company_name="Apple Inc.",
            currency="USD",
        ),
        recommendation=ReportRecommendation(
            stance=RecommendationStance.FAVORABLE,
            rationale="Robust financial resilience and durable operating margins.",
            profile_alignment="Aligned with moderate-risk growth portfolio.",
            monitoring_points=[
                "Quarterly services revenue growth",
                "Supply chain margins",
            ],
        ),
        overall_assessment=OverallAssessmentSection(
            data_completeness_ratio=1.0,
            specialist_consensus="favorable",
            synthesis="Consistent operational cash flows and strong balance sheet.",
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
# 15.5.1 API TESTS FOR EVERY ENDPOINT (HAPPY PATH)
# ===========================================================================


class TestPhase155AllEndpointsHappyPath:
    """Requirement 15.5.1: Happy-path API tests for every defined endpoint."""

    def test_health_root_happy_path(self, client: TestClient) -> None:
        """GET /health returns 200 with service health metadata."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "environment" in data

    def test_health_v1_happy_path(self, client: TestClient) -> None:
        """GET /api/v1/health returns 200 with service health metadata."""
        response = client.get("/api/v1/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "environment" in data

    def test_document_upload_happy_path(self, client: TestClient) -> None:
        """POST /api/v1/documents/upload uploads research PDF document."""
        files = {"file": ("apple_10k.pdf", VALID_PDF_BYTES, "application/pdf")}
        data = {"ticker": "AAPL", "document_type": "annual_report"}
        response = client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == status.HTTP_201_CREATED
        body = response.json()
        assert body["ticker"] == "AAPL"
        assert body["document_type"] == "annual_report"
        assert "document_id" in body
        assert body["original_filename"].endswith(".pdf")

    def test_chat_query_happy_path(self, test_app, client: TestClient) -> None:
        """POST /api/v1/chat runs chat query returning AnalysisExecutionResponse."""
        expected_report = create_mock_final_report("AAPL")

        def _mock_runner(**kwargs: Any) -> GraphState:
            return {
                "user_query": kwargs.get("query", ""),
                "ticker": "AAPL",
                "clarified_request": {
                    "clarification_needed": False,
                    "clarification_questions": [],
                },
                "report": {"success": True, "data": expected_report.model_dump()},
                "trace_id": kwargs.get("trace_id", "trace-chat-happy"),
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_runner

        response = client.post(
            "/api/v1/chat",
            json={"query": "Analyze Apple fundamentals for moderate risk investor."},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "completed"
        assert data["clarification_needed"] is False
        assert data["report_id"] is not None
        assert data["report"]["company"]["ticker"] == "AAPL"

    def test_company_analysis_sync_happy_path(
        self, test_app, client: TestClient
    ) -> None:
        """POST /api/v1/analysis (sync) executes workflow and returns report."""
        expected_report = create_mock_final_report("MSFT")

        def _mock_runner(**kwargs: Any) -> GraphState:
            return {
                "ticker": kwargs.get("ticker", "MSFT"),
                "clarified_request": {
                    "clarification_needed": False,
                    "clarification_questions": [],
                },
                "report": {"success": True, "data": expected_report.model_dump()},
                "trace_id": kwargs.get("trace_id", "trace-sync-happy"),
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_runner

        response = client.post(
            "/api/v1/analysis",
            json={
                "ticker": "MSFT",
                "investor_profile": {
                    "risk_tolerance": "aggressive",
                    "capital_amount": 50000.0,
                    "time_horizon": "long",
                },
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "completed"
        assert data["report_id"] is not None
        assert data["report"]["company"]["ticker"] == "MSFT"

    def test_company_analysis_async_happy_path(
        self, test_app, client: TestClient
    ) -> None:
        """POST /api/v1/analysis (async) schedules background run."""
        expected_report = create_mock_final_report("GOOGL")

        def _mock_runner(**kwargs: Any) -> GraphState:
            return {
                "ticker": "GOOGL",
                "clarified_request": {
                    "clarification_needed": False,
                    "clarification_questions": [],
                },
                "report": {"success": True, "data": expected_report.model_dump()},
                "trace_id": "trace-async-happy",
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_runner

        response = client.post(
            "/api/v1/analysis?background=true",
            json={"ticker": "GOOGL"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "running"
        assert data["analysis_id"] is not None

    def test_clarification_submit_happy_path(
        self, test_app, client: TestClient
    ) -> None:
        """POST /api/v1/clarification resumes halted session with answers."""
        expected_report = create_mock_final_report("AMZN")

        def _mock_runner(**kwargs: Any) -> GraphState:
            return {
                "ticker": "AMZN",
                "clarified_request": {
                    "clarification_needed": False,
                    "clarification_questions": [],
                },
                "report": {"success": True, "data": expected_report.model_dump()},
                "trace_id": "trace-clarification-happy",
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_runner

        response = client.post(
            "/api/v1/clarification",
            json={
                "ticker": "AMZN",
                "clarification_answers": {
                    "risk_tolerance": "conservative",
                    "time_horizon": "medium",
                    "capital_amount": 10000.0,
                },
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "completed"
        assert data["clarification_needed"] is False
        assert data["report_id"] is not None

    def test_research_query_happy_path(self, test_app, client: TestClient) -> None:
        """POST /api/v1/research/query runs research against document context."""
        expected_report = create_mock_final_report("NVDA")

        def _mock_runner(**kwargs: Any) -> GraphState:
            return {
                "ticker": "NVDA",
                "user_query": kwargs.get("query", ""),
                "documents_available": True,
                "clarified_request": {
                    "clarification_needed": False,
                    "clarification_questions": [],
                },
                "report": {"success": True, "data": expected_report.model_dump()},
                "trace_id": "trace-research-happy",
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_runner

        response = client.post(
            "/api/v1/research/query",
            json={
                "query": "What are NVIDIA's primary datacenter growth drivers?",
                "ticker": "NVDA",
                "documents_available": True,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "completed"
        assert data["report_id"] is not None

    def test_analysis_status_polling_happy_path(
        self, test_app, client: TestClient
    ) -> None:
        """GET /api/v1/analysis/{analysis_id}/status returns session status."""
        expected_report = create_mock_final_report("AAPL")

        def _mock_runner(**kwargs: Any) -> GraphState:
            return {
                "ticker": "AAPL",
                "clarified_request": {"clarification_needed": False},
                "report": {"success": True, "data": expected_report.model_dump()},
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_runner

        init_resp = client.post("/api/v1/analysis", json={"ticker": "AAPL"})
        analysis_id = init_resp.json()["analysis_id"]

        status_resp = client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert status_resp.status_code == status.HTTP_200_OK
        status_data = status_resp.json()
        assert status_data["analysis_id"] == analysis_id
        assert status_data["status"] == "completed"
        assert status_data["report_id"] is not None

    def test_report_retrieval_all_formats_happy_path(
        self, test_app, client: TestClient
    ) -> None:
        """GET /api/v1/reports/{report_id} retrieves JSON, Markdown, and Summary."""
        expected_report = create_mock_final_report("AAPL")

        def _mock_runner(**kwargs: Any) -> GraphState:
            return {
                "ticker": "AAPL",
                "clarified_request": {"clarification_needed": False},
                "report": {"success": True, "data": expected_report.model_dump()},
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_runner

        init_resp = client.post("/api/v1/analysis", json={"ticker": "AAPL"})
        report_id = init_resp.json()["report_id"]

        # 1. JSON format (default)
        json_resp = client.get(f"/api/v1/reports/{report_id}")
        assert json_resp.status_code == status.HTTP_200_OK
        json_data = json_resp.json()
        assert json_data["format"] == "json"
        assert json_data["report"]["company"]["ticker"] == "AAPL"

        # 2. Markdown format
        md_resp = client.get(f"/api/v1/reports/{report_id}?format=markdown")
        assert md_resp.status_code == status.HTTP_200_OK
        md_data = md_resp.json()
        assert md_data["format"] == "markdown"
        assert "Investment Research Report" in md_data["rendered_content"]
        assert "AAPL" in md_data["rendered_content"]

        # 3. Summary format
        summary_resp = client.get(f"/api/v1/reports/{report_id}?format=summary")
        assert summary_resp.status_code == status.HTTP_200_OK
        summary_data = summary_resp.json()
        assert summary_data["format"] == "summary"
        assert "INVESTMENT RESEARCH REPORT" in summary_data["rendered_content"]
        assert "AAPL" in summary_data["rendered_content"]


# ===========================================================================
# 15.5.2 API TESTS FOR VALIDATION & ERROR CASES
# ===========================================================================


class TestPhase155ValidationAndErrorCases:
    """Requirement 15.5.2: Validation and error cases across endpoints."""

    def test_validation_error_missing_body_payloads(self, client: TestClient) -> None:
        """Missing required JSON bodies return 422 with validation errors."""
        # Chat missing body
        resp = client.post("/api/v1/chat", json={})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert resp.json()["error"]["code"] == "VALIDATION_ERROR"

        # Analysis missing ticker
        resp = client.post("/api/v1/analysis", json={})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Clarification missing clarification_answers
        resp = client.post("/api/v1/clarification", json={"ticker": "AAPL"})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Research query missing query
        resp = client.post("/api/v1/research/query", json={"ticker": "AAPL"})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Document upload missing file and ticker
        resp = client.post("/api/v1/documents/upload", data={})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_validation_error_invalid_field_constraints(
        self, test_app, client: TestClient
    ) -> None:
        """Invalid field constraints (regex, numbers, lengths) return 422."""

        def _dummy_runner(**kwargs: Any) -> GraphState:
            return {"ticker": "AAPL"}

        test_app.dependency_overrides[get_graph_runner] = lambda: _dummy_runner

        # Invalid ticker format
        resp = client.post("/api/v1/analysis", json={"ticker": "INVALID_LONG_TICKER"})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Whitespace-only query
        resp = client.post("/api/v1/chat", json={"query": "    \n\t  "})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Empty clarification answers
        resp = client.post(
            "/api/v1/clarification",
            json={"ticker": "AAPL", "clarification_answers": {}},
        )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Negative capital amount
        resp = client.post(
            "/api/v1/analysis",
            json={
                "ticker": "AAPL",
                "investor_profile": {"capital_amount": -500.0},
            },
        )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_document_upload_unsupported_extension_returns_400(
        self, client: TestClient
    ) -> None:
        """Uploading non-PDF unsupported file returns 400."""
        files = {
            "file": (
                "malicious.exe",
                b"binary content",
                "application/octet-stream",
            )
        }
        data = {"ticker": "AAPL", "document_type": "annual_report"}
        resp = client.post("/api/v1/documents/upload", files=files, data=data)
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

    def test_malformed_path_parameters_return_400(self, client: TestClient) -> None:
        """Malformed UUID path parameters return 400 with HTTP_ERROR envelope."""
        resp = client.get("/api/v1/analysis/not-a-valid-uuid/status")
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert resp.json()["error"]["code"] == "HTTP_ERROR"

        resp = client.get("/api/v1/reports/not-a-valid-uuid")
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert resp.json()["error"]["code"] == "HTTP_ERROR"

    def test_not_found_resources_return_404(self, client: TestClient) -> None:
        """Nonexistent but valid UUIDs return 404 with HTTP_ERROR envelope."""
        random_id = str(uuid.uuid4())
        resp = client.get(f"/api/v1/analysis/{random_id}/status")
        assert resp.status_code == status.HTTP_404_NOT_FOUND
        assert resp.json()["error"]["code"] == "HTTP_ERROR"

        resp = client.get(f"/api/v1/reports/{random_id}")
        assert resp.status_code == status.HTTP_404_NOT_FOUND
        assert resp.json()["error"]["code"] == "HTTP_ERROR"

    def test_invalid_report_format_query_param_returns_422(
        self, client: TestClient
    ) -> None:
        """Invalid format parameter on report retrieval returns 422."""
        valid_id = str(uuid.uuid4())
        resp = client.get(f"/api/v1/reports/{valid_id}?format=unsupported_format")
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_safe_500_error_envelope_masks_tracebacks(
        self, test_app, client: TestClient
    ) -> None:
        """Unexpected internal crashes return sanitized 500 error envelope."""

        def _crashing_runner(**kwargs: Any) -> GraphState:
            raise RuntimeError("Database connection string: secret_db_pw_123")

        test_app.dependency_overrides[get_graph_runner] = lambda: _crashing_runner

        resp = client.post(
            "/api/v1/analysis",
            json={"ticker": "AAPL", "trace_id": "test-trace-safe-500"},
        )
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = resp.json()
        assert "secret_db_pw_123" not in str(data)
        assert "Failed to execute company analysis" in data["error"]["message"]
        assert data["error"]["code"] == "HTTP_ERROR"


# ===========================================================================
# 15.5.3 INTEGRATION TEST: FULL USER JOURNEY VIA API ONLY
# ===========================================================================


class TestPhase155FullUserJourneyIntegration:
    """Requirement 15.5.3: Full user journey via API ONLY (no direct graph calls).

    Simulates realistic user journey:
    1. Upload 10-K document (POST /api/v1/documents/upload)
    2. Conversational query requiring clarification (POST /api/v1/chat)
    3. Poll analysis status -> clarification_needed (GET /api/v1/analysis/{id}/status)
    4. Submit clarification answers (POST /api/v1/clarification)
    5. Poll analysis status -> completed (GET /api/v1/analysis/{id}/status)
    6. Retrieve report as JSON (GET /api/v1/reports/{id}?format=json)
    7. Retrieve report as Markdown (GET /api/v1/reports/{id}?format=markdown)
    8. Retrieve report as Executive Summary (GET /api/v1/reports/{id}?format=summary)
    9. Document research query (POST /api/v1/research/query)
    """

    def test_complete_user_journey_api_only(self, test_app, client: TestClient) -> None:
        """Execute user journey purely via API calls with 0 direct graph calls."""
        expected_report = create_mock_final_report("AAPL")

        def _simulated_graph_runner(**kwargs: Any) -> GraphState:
            """Simulates LangGraph orchestrator behavior across workflow stages."""
            query = kwargs.get("query", "")
            clarification_answers = kwargs.get("clarification_answers")

            # Stage 1: Underspecified chat query triggers clarification
            if query == "I want to invest in Apple" and not clarification_answers:
                return {
                    "user_query": query,
                    "ticker": "AAPL",
                    "clarified_request": {
                        "clarification_needed": True,
                        "clarification_questions": [
                            "What is your investment time horizon?",
                            "What is your risk tolerance level?",
                            "What is your intended capital allocation?",
                        ],
                    },
                    "investor_profile": {"ticker": "AAPL"},
                    "trace_id": kwargs.get("trace_id", "trace-journey-step2"),
                }

            # Stage 2: Clarification provided or complete analysis
            return {
                "user_query": query,
                "ticker": kwargs.get("ticker", "AAPL"),
                "clarified_request": {
                    "clarification_needed": False,
                    "clarification_questions": [],
                },
                "investor_profile": {
                    "ticker": "AAPL",
                    "risk_tolerance": "moderate",
                    "time_horizon": "long",
                    "capital_amount": 25000.0,
                },
                "report": {
                    "success": True,
                    "data": expected_report.model_dump(),
                },
                "trace_id": kwargs.get("trace_id", "trace-journey-success"),
            }

        test_app.dependency_overrides[get_graph_runner] = (
            lambda: _simulated_graph_runner
        )

        # -------------------------------------------------------------------
        # Step 1: Upload research filing (POST /api/v1/documents/upload)
        # -------------------------------------------------------------------
        upload_resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("AAPL_10K_2023.pdf", VALID_PDF_BYTES, "application/pdf")},
            data={"ticker": "AAPL", "document_type": "annual_report"},
        )
        assert upload_resp.status_code == status.HTTP_201_CREATED
        upload_data = upload_resp.json()
        assert upload_data["ticker"] == "AAPL"
        assert upload_data["document_type"] == "annual_report"
        document_id = upload_data["document_id"]
        assert document_id is not None

        # -------------------------------------------------------------------
        # Step 2: Chat query requiring clarification (POST /api/v1/chat)
        # -------------------------------------------------------------------
        chat_resp = client.post(
            "/api/v1/chat",
            json={
                "query": "I want to invest in Apple",
                "trace_id": "journey-trace-001",
            },
        )
        assert chat_resp.status_code == status.HTTP_200_OK
        chat_data = chat_resp.json()
        assert chat_data["status"] == "clarification_needed"
        assert chat_data["clarification_needed"] is True
        assert len(chat_data["clarification_questions"]) == 3
        assert chat_data["report"] is None
        analysis_id = chat_data["analysis_id"]
        assert analysis_id is not None

        # -------------------------------------------------------------------
        # Step 3: Poll status during clarification (GET /api/v1/analysis/{id}/status)
        # -------------------------------------------------------------------
        status_step3 = client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert status_step3.status_code == status.HTTP_200_OK
        assert status_step3.json()["status"] == "clarification_needed"
        assert len(status_step3.json()["clarification_questions"]) == 3
        assert status_step3.json()["report_id"] is None

        # -------------------------------------------------------------------
        # Step 4: Submit clarification answers (POST /api/v1/clarification)
        # -------------------------------------------------------------------
        clarification_resp = client.post(
            "/api/v1/clarification",
            json={
                "analysis_id": analysis_id,
                "ticker": "AAPL",
                "clarification_answers": {
                    "time_horizon": "long",
                    "risk_tolerance": "moderate",
                    "capital_amount": 25000.0,
                },
                "trace_id": "journey-trace-002",
            },
        )
        assert clarification_resp.status_code == status.HTTP_200_OK
        clarification_data = clarification_resp.json()
        assert clarification_data["status"] == "completed"
        assert clarification_data["clarification_needed"] is False
        report_id = clarification_data["report_id"]
        assert report_id is not None
        assert clarification_data["report"]["company"]["ticker"] == "AAPL"

        # -------------------------------------------------------------------
        # Step 5: Poll status after completion (GET /api/v1/analysis/{id}/status)
        # -------------------------------------------------------------------
        status_step5 = client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert status_step5.status_code == status.HTTP_200_OK
        assert status_step5.json()["status"] == "completed"
        assert status_step5.json()["report_id"] == report_id
        assert status_step5.json()["completed_at"] is not None

        # -------------------------------------------------------------------
        # Step 6: Retrieve report in JSON format (GET /api/v1/reports/{id})
        # -------------------------------------------------------------------
        report_json_resp = client.get(f"/api/v1/reports/{report_id}")
        assert report_json_resp.status_code == status.HTTP_200_OK
        report_json_data = report_json_resp.json()
        assert report_json_data["format"] == "json"
        assert report_json_data["report"]["company"]["ticker"] == "AAPL"
        assert report_json_data["report"]["recommendation"]["stance"] == "favorable"

        # -------------------------------------------------------------------
        # Step 7: Retrieve report in Markdown (GET /api/v1/reports/{id}?format=markdown)
        # -------------------------------------------------------------------
        report_md_resp = client.get(f"/api/v1/reports/{report_id}?format=markdown")
        assert report_md_resp.status_code == status.HTTP_200_OK
        report_md_data = report_md_resp.json()
        assert report_md_data["format"] == "markdown"
        assert "Investment Research Report" in report_md_data["rendered_content"]
        assert "AAPL" in report_md_data["rendered_content"]

        # -------------------------------------------------------------------
        # Step 8: Retrieve report in Summary (GET /api/v1/reports/{id}?format=summary)
        # -------------------------------------------------------------------
        report_summary_resp = client.get(f"/api/v1/reports/{report_id}?format=summary")
        assert report_summary_resp.status_code == status.HTTP_200_OK
        report_summary_data = report_summary_resp.json()
        assert report_summary_data["format"] == "summary"
        assert "INVESTMENT RESEARCH REPORT" in report_summary_data["rendered_content"]
        assert "AAPL" in report_summary_data["rendered_content"]

        # -------------------------------------------------------------------
        # Step 9: Query research document vault (POST /api/v1/research/query)
        # -------------------------------------------------------------------
        research_resp = client.post(
            "/api/v1/research/query",
            json={
                "query": "Summarize Apple's latest gross margin commentary.",
                "ticker": "AAPL",
                "documents_available": True,
                "trace_id": "journey-trace-003",
            },
        )
        assert research_resp.status_code == status.HTTP_200_OK
        research_data = research_resp.json()
        assert research_data["status"] == "completed"
        assert research_data["report_id"] is not None
        assert research_data["report"]["company"]["ticker"] == "AAPL"
