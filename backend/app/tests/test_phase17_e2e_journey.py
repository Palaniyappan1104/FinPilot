"""Phase 17.4.2 Automated End-to-End Journey Test.

Tests the full end-to-end integration across the real HTTP/API boundary:
Frontend Request Contracts
-> FastAPI /api/v1 Endpoints
-> Graph Runner (deterministic test runner)
-> Asynchronous Analysis Execution
-> Status Polling
-> Report Retrieval & Validation
-> Frontend Adapter Consumption Compatibility.
"""

from __future__ import annotations

import time
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


def _build_deterministic_report(ticker: str = "AAPL") -> FinalReport:
    """Construct a full FinalReport instance fulfilling Phase 12/15 contracts."""
    return FinalReport(
        company=ReportCompanyInfo(
            ticker=ticker,
            name=f"{ticker} Inc.",
            currency="USD",
            sector="Technology",
        ),
        recommendation=ReportRecommendation(
            stance=RecommendationStance.FAVORABLE,
            rationale=f"Multi-agent evaluation of {ticker} shows strong fundamentals and disciplined capital allocation.",
        ),
        overall_assessment=OverallAssessmentSection(
            synthesis=f"Holistic cross-specialist consensus for {ticker} indicates strong operational resilience.",
            data_completeness_ratio=1.0,
        ),
        important_risks=[
            "Regulatory antitrust scrutiny regarding ecosystem services.",
            "Supply chain dependencies on concentrated manufacturing partners.",
        ],
        key_reasons=[
            "Robust recurring service revenues expanding above 30% gross margin.",
            "Consistent free cash flow generation exceeding $100B annually.",
        ],
        disclaimer=STANDARD_DISCLAIMER,
    )


@pytest.fixture
def e2e_environment(tmp_path: Any):
    """Provides isolated FastAPI test client with deterministic graph runner."""
    app = create_application()
    storage = LocalDocumentStorage(base_directory=str(tmp_path / "vault"))
    app.dependency_overrides[get_document_storage] = lambda: storage

    def deterministic_runner(**kwargs: Any) -> dict[str, Any]:
        ticker = kwargs.get("ticker") or "AAPL"
        clarifications = kwargs.get("clarification_answers") or {}
        query = kwargs.get("query", "")

        # Clarification trigger condition
        if not clarifications and ("clarify" in query.lower() or "ambiguous" in query.lower()):
            return {
                "user_query": query,
                "ticker": ticker,
                "clarified_request": {
                    "clarification_needed": True,
                    "clarification_questions": [
                        "What is your target investment time horizon?",
                        "What is your risk tolerance level (conservative, moderate, aggressive)?",
                    ],
                },
                "report": None,
                "trace_id": kwargs.get("trace_id", str(uuid.uuid4())),
            }

        report = _build_deterministic_report(ticker=ticker)
        return {
            "user_query": query,
            "ticker": ticker,
            "clarified_request": {
                "clarification_needed": False,
                "clarification_questions": [],
            },
            "report": {"success": True, "data": report.model_dump()},
            "trace_id": kwargs.get("trace_id", str(uuid.uuid4())),
        }

    app.dependency_overrides[get_graph_runner] = lambda: deterministic_runner

    with TestClient(app) as client:
        yield client, storage

    app.dependency_overrides.clear()


class TestPhase17EndToEndJourney:
    """Automated E2E test exercising the complete user journey through HTTP API."""

    def test_complete_frontend_to_report_e2e_journey(self, e2e_environment: Any) -> None:
        client, _ = e2e_environment

        # =====================================================================
        # Step 1: Frontend sends initial query via POST /api/v1/chat
        # Expecting clarification needed state
        # =====================================================================
        chat_payload = {
            "query": "Should I invest in Apple? Please clarify my risk profile first.",
            "investor_profile": {
                "ticker": "AAPL",
                "investment_goal": "Capital Growth",
            },
            "documents_available": False,
        }

        chat_resp = client.post("/api/v1/chat", json=chat_payload)
        assert chat_resp.status_code == status.HTTP_200_OK
        chat_data = chat_resp.json()

        assert chat_data["clarification_needed"] is True
        assert len(chat_data["clarification_questions"]) == 2
        assert "time horizon" in chat_data["clarification_questions"][0]
        analysis_id = chat_data["analysis_id"]
        assert analysis_id is not None

        # =====================================================================
        # Step 2: Frontend responds with clarification answers via
        # POST /api/v1/clarification?background=true
        # =====================================================================
        clarification_payload = {
            "clarification_answers": {
                "time_horizon": "3-5 years",
                "risk_tolerance": "moderate",
                "capital_amount": 50000,
            },
            "analysis_id": analysis_id,
            "ticker": "AAPL",
            "query": "Proceed with analysis using provided clarifications",
            "documents_available": False,
        }

        clarify_resp = client.post(
            "/api/v1/clarification?background=true", json=clarification_payload
        )
        assert clarify_resp.status_code == status.HTTP_200_OK
        clarify_data = clarify_resp.json()
        assert clarify_data["analysis_id"] == analysis_id
        assert clarify_data["status"] == "running"

        # =====================================================================
        # Step 3: Frontend polls GET /api/v1/analysis/{analysis_id}/status
        # Observe status completion and retrieve report_id
        # =====================================================================
        report_id = None
        for _ in range(20):
            poll_resp = client.get(f"/api/v1/analysis/{analysis_id}/status")
            assert poll_resp.status_code == status.HTTP_200_OK
            poll_data = poll_resp.json()
            assert poll_data["analysis_id"] == analysis_id

            if poll_data["status"] == "completed":
                report_id = poll_data.get("report_id")
                break
            time.sleep(0.05)

        assert report_id is not None, "Analysis should complete and yield a valid report_id"

        # =====================================================================
        # Step 4: Frontend retrieves final report via GET /api/v1/reports/{report_id}
        # =====================================================================
        report_resp = client.get(f"/api/v1/reports/{report_id}?format=json")
        assert report_resp.status_code == status.HTTP_200_OK
        rep_envelope = report_resp.json()

        assert rep_envelope["report_id"] == report_id
        assert rep_envelope["format"] == "json"
        report_body = rep_envelope["report"]
        assert report_body is not None

        # =====================================================================
        # Step 5: Verify report structure matches frontend normalizer expectations
        # =====================================================================
        assert report_body["company"]["ticker"] == "AAPL"
        assert report_body["company"]["name"] == "AAPL Inc."
        assert report_body["recommendation"]["stance"] == "favorable"
        assert len(report_body["key_reasons"]) >= 1
        assert len(report_body["important_risks"]) >= 1
        assert "FinPilot provides automated financial research" in report_body["disclaimer"]
        assert report_body["overall_assessment"]["data_completeness_ratio"] == 1.0

    def test_document_vault_and_grounded_research_e2e_journey(
        self, e2e_environment: Any
    ) -> None:
        client, _ = e2e_environment

        # 1. Upload research PDF
        files = {"file": ("NVDA_10K.pdf", VALID_PDF_BYTES, "application/pdf")}
        data = {"ticker": "NVDA", "document_type": "annual_report"}

        upload_resp = client.post("/api/v1/documents/upload", files=files, data=data)
        assert upload_resp.status_code == status.HTTP_201_CREATED
        upload_json = upload_resp.json()
        assert upload_json["ticker"] == "NVDA"
        assert upload_json["upload_status"] == "stored"

        # 2. Query research against uploaded document
        query_payload = {
            "query": "What are the disclosures regarding data center revenue growth?",
            "ticker": "NVDA",
            "documents_available": True,
        }
        res_resp = client.post("/api/v1/research/query", json=query_payload)
        assert res_resp.status_code == status.HTTP_200_OK
        res_json = res_resp.json()
        assert res_json["status"] == "completed"
        assert res_json["report"] is not None
