"""Phase 17 Frontend-Backend UI Contracts Integration Tests.

Validates that exact JSON request payloads produced by the React frontend HttpApiService
are fully accepted by FastAPI Pydantic v2 schemas with zero validation errors (422),
and that FastAPI responses match the frontend adapter expectations for:
- 17.1 API Client Layer contracts
- 17.2 Full UI Request/Response lifecycle (Chat -> Analysis -> Clarification -> Report)
- 17.3 Document Upload and Research Query payloads
- 17.4 Error envelope compliance
"""

from __future__ import annotations

import uuid
from typing import Any
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.agents.report_schema import (
    FinalReport,
    OverallAssessmentSection,
    RecommendationStance,
    ReportCompanyInfo,
    ReportRecommendation,
    STANDARD_DISCLAIMER,
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


def _build_dummy_final_report(ticker: str = "AAPL") -> FinalReport:
    """Build a deterministic FinalReport fulfilling Phase 15/17 contracts."""
    return FinalReport(
        company=ReportCompanyInfo(
            ticker=ticker,
            company_name="Apple Inc.",
            currency="USD",
        ),
        recommendation=ReportRecommendation(
            stance=RecommendationStance.FAVORABLE,
            rationale="Robust hardware revenue, sticky service ecosystem, disciplined capital allocation.",
        ),
        overall_assessment=OverallAssessmentSection(
            synthesis="Consensus indicates favorable risk-adjusted returns over long horizons.",
            data_completeness_ratio=1.0,
        ),
        disclaimer=STANDARD_DISCLAIMER,
    )


@pytest.fixture
def client_and_storage(tmp_path: Any) -> tuple[TestClient, LocalDocumentStorage]:
    app = create_application()
    storage = LocalDocumentStorage(base_directory=str(tmp_path / "documents"))
    app.dependency_overrides[get_document_storage] = lambda: storage

    def mock_runner(**kwargs: Any) -> dict[str, Any]:
        ticker = kwargs.get("ticker") or "AAPL"
        clarifications = kwargs.get("clarification_answers") or {}
        if not clarifications and kwargs.get("query") == "needs_clarification":
            return {
                "user_query": kwargs.get("query", ""),
                "ticker": ticker,
                "clarified_request": {
                    "clarification_needed": True,
                    "clarification_questions": ["What is your investment time horizon?"],
                },
                "report": None,
                "trace_id": kwargs.get("trace_id", str(uuid.uuid4())),
            }

        report = _build_dummy_final_report(ticker=ticker)
        return {
            "user_query": kwargs.get("query", ""),
            "ticker": ticker,
            "clarified_request": {
                "clarification_needed": False,
                "clarification_questions": [],
            },
            "report": {"success": True, "data": report.model_dump()},
            "trace_id": kwargs.get("trace_id", str(uuid.uuid4())),
        }

    app.dependency_overrides[get_graph_runner] = lambda: mock_runner

    with TestClient(app) as test_client:
        yield test_client, storage

    app.dependency_overrides.clear()


class TestPhase17UIContracts:
    """Validate that exact payload structures sent by frontend HttpApiService are accepted."""

    def test_frontend_chat_query_payload_accepted(
        self, client_and_storage: tuple[TestClient, LocalDocumentStorage]
    ) -> None:
        client, _ = client_and_storage

        # Exact payload structure from frontend HttpApiService.sendChatMessage
        payload = {
            "query": "Should I invest in Apple for high growth?",
            "investor_profile": {
                "ticker": "AAPL",
                "investment_goal": "Capital Growth",
                "time_horizon": "3-5 years",
                "capital_amount": 50000.0,
                "risk_tolerance": "moderate",
            },
            "documents_available": False,
            "trace_id": None,
        }

        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "analysis_id" in data
        assert "trace_id" in data
        assert "status" in data
        assert "clarification_needed" in data
        assert data["clarification_needed"] is False

    def test_frontend_company_analysis_background_payload_accepted(
        self, client_and_storage: tuple[TestClient, LocalDocumentStorage]
    ) -> None:
        client, _ = client_and_storage

        # Exact payload from frontend HttpApiService.startAnalysis
        payload = {
            "ticker": "NVDA",
            "target_company": "NVIDIA Corporation",
            "query": "Analyze investment feasibility for NVDA",
            "investor_profile": {
                "ticker": "NVDA",
                "investment_goal": "Wealth Generation",
                "time_horizon": "5+ years",
                "capital_amount": 100000.0,
                "risk_tolerance": "aggressive",
            },
            "documents_available": False,
        }

        response = client.post("/api/v1/analysis?background=true", json=payload)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "running"
        assert data["analysis_id"] is not None

        # Verify status polling endpoint receives valid response for this analysis_id
        poll_resp = client.get(f"/api/v1/analysis/{data['analysis_id']}/status")
        assert poll_resp.status_code == status.HTTP_200_OK
        poll_data = poll_resp.json()
        assert poll_data["analysis_id"] == data["analysis_id"]
        assert poll_data["status"] in ["running", "completed", "queued"]

    def test_frontend_clarification_submit_payload_accepted(
        self, client_and_storage: tuple[TestClient, LocalDocumentStorage]
    ) -> None:
        client, _ = client_and_storage

        # Exact payload from frontend HttpApiService.submitClarification
        payload = {
            "clarification_answers": {
                "time_horizon": "3-5 years",
                "risk_tolerance": "moderate",
                "capital_amount": 25000,
            },
            "analysis_id": "test-analysis-clarification-01",
            "ticker": "AAPL",
            "query": "Proceed with analysis using provided clarifications",
            "documents_available": False,
        }

        response = client.post("/api/v1/clarification?background=false", json=payload)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["analysis_id"] == "test-analysis-clarification-01"
        assert data["clarification_needed"] is False
        assert data["report_id"] is not None

        # Retrieve report using report_id
        rep_resp = client.get(f"/api/v1/reports/{data['report_id']}?format=json")
        assert rep_resp.status_code == status.HTTP_200_OK
        rep_data = rep_resp.json()
        assert rep_data["report_id"] == data["report_id"]
        assert rep_data["format"] == "json"
        assert rep_data["report"]["company"]["ticker"] == "AAPL"

    def test_frontend_research_query_payload_accepted(
        self, client_and_storage: tuple[TestClient, LocalDocumentStorage]
    ) -> None:
        client, _ = client_and_storage

        # Exact payload from frontend HttpApiService.queryResearchDocuments
        payload = {
            "query": "What are management comments on AI gross margin expansion?",
            "ticker": "AAPL",
            "documents_available": True,
        }

        response = client.post("/api/v1/research/query", json=payload)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "completed"
        assert data["report"] is not None

    def test_frontend_document_upload_multipart_accepted(
        self, client_and_storage: tuple[TestClient, LocalDocumentStorage]
    ) -> None:
        client, _ = client_and_storage

        # Exact multipart form payload from frontend HttpApiService.uploadDocument
        files = {
            "file": ("AAPL_10K.pdf", VALID_PDF_BYTES, "application/pdf"),
        }
        data = {
            "ticker": "AAPL",
            "document_type": "annual_report",  # Mapped from '10-K' via mapFrontendDocTypeToBackend
        }

        response = client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == status.HTTP_201_CREATED
        upload_data = response.json()
        assert upload_data["ticker"] == "AAPL"
        assert upload_data["document_type"] == "annual_report"
        assert upload_data["original_filename"] == "AAPL_10K.pdf"
        assert upload_data["file_size_bytes"] == len(VALID_PDF_BYTES)
        assert upload_data["upload_status"] == "stored"

    def test_frontend_error_envelope_structure(
        self, client_and_storage: tuple[TestClient, LocalDocumentStorage]
    ) -> None:
        client, _ = client_and_storage

        # Non-existent report (valid UUID) -> should return structured APIErrorEnvelope with HTTP_ERROR
        valid_uuid = str(uuid.uuid4())
        response = client.get(f"/api/v1/reports/{valid_uuid}?format=json")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        err_json = response.json()
        assert "error" in err_json
        assert err_json["error"]["code"] == "HTTP_ERROR"
        assert "not found" in err_json["error"]["message"]

        # Validation error (missing required query parameter in chat) -> VALIDATION_ERROR
        bad_response = client.post("/api/v1/chat", json={})
        assert bad_response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        bad_err = bad_response.json()
        assert "error" in bad_err
        assert bad_err["error"]["code"] == "VALIDATION_ERROR"
        assert "details" in bad_err["error"]
