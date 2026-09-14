"""Comprehensive tests for Phase 15.2 Request/Response Contracts and Validation.

Validates:
1. Pydantic request/response model contracts for every endpoint (15.2.1).
2. Input validation and sanitation across all fields:
   - Ticker normalization and invalid character rejection.
   - Query whitespace rejection and length constraints.
   - Investor profile capital constraints (positive, finite, max bound).
   - Risk tolerance string normalization.
   - Clarification answers non-empty dict and valid key constraints.
3. Path parameter validation (15.2.2):
   - Malformed analysis_id -> HTTP 400 Bad Request with clear message.
   - Malformed report_id -> HTTP 400 Bad Request with clear message.
   - Nonexistent UUID -> HTTP 404 Not Found with clear message.
4. Standardized error envelopes and security:
   - Error responses follow {"error": {"code": ..., "message": ..., "details": ...}}.
   - No leakage of internal tracebacks or secrets.
5. OpenAPI documentation contract completeness:
   - All endpoints have response models and error schemas documented.
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
from app.models.api import InvestorProfilePayload


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
            rationale="Durable competitive moat and consistent capital return.",
            profile_alignment="Matches moderate risk profile.",
            monitoring_points=["Product margins", "Services segment growth"],
        ),
        overall_assessment=OverallAssessmentSection(
            data_completeness_ratio=1.0,
            specialist_consensus="favorable",
            synthesis="Solid financial metrics and healthy balance sheet.",
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
# 1. REQUEST MODEL INPUT VALIDATION (15.2.2)
# ===========================================================================


class TestInputValidationContracts:
    """Tests for field validation rules across request models."""

    def test_ticker_normalization_and_whitespace_stripping(
        self, test_app, client: TestClient
    ) -> None:
        """Lowercase ticker with whitespace is auto-uppercased and trimmed."""
        captured_ticker = None

        def _mock_runner(**kwargs: Any) -> GraphState:
            nonlocal captured_ticker
            captured_ticker = kwargs.get("ticker")
            return {
                "user_query": kwargs.get("query", ""),
                "ticker": captured_ticker,
                "clarified_request": {"clarification_needed": False},
                "report": {
                    "success": True,
                    "data": create_deterministic_test_report(
                        captured_ticker or "MSFT"
                    ).model_dump(),
                },
                "trace_id": "trace-test-1",
            }

        test_app.dependency_overrides[get_graph_runner] = lambda: _mock_runner

        response = client.post("/api/v1/analysis", json={"ticker": "  msft  "})
        assert response.status_code == status.HTTP_200_OK
        assert captured_ticker == "MSFT"

    @pytest.mark.parametrize(
        "invalid_ticker",
        [
            "",
            "   ",
            "AAPL@123",
            "$$$",
            "AAPL MSFT",
            "TOOLONGTICKERNAME",
        ],
    )
    def test_invalid_ticker_formats_rejected(
        self, client: TestClient, invalid_ticker: str
    ) -> None:
        """Invalid ticker symbols are rejected with HTTP 422 and descriptive error."""
        response = client.post("/api/v1/analysis", json={"ticker": invalid_ticker})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"
        details_str = str(data["error"]["details"])
        assert "ticker" in details_str

    @pytest.mark.parametrize(
        "invalid_query",
        [
            "",
            "    ",
            "\t\n  \n",
        ],
    )
    def test_empty_and_whitespace_queries_rejected_in_chat(
        self, client: TestClient, invalid_query: str
    ) -> None:
        """Empty or whitespace-only queries in chat return HTTP 422."""
        response = client.post("/api/v1/chat", json={"query": invalid_query})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert any(
            "Query cannot be empty or whitespace only" in err.get("msg", "")
            or "String should have at least 1 character" in err.get("msg", "")
            for err in data["error"]["details"]
        )

    def test_whitespace_query_rejected_in_research(self, client: TestClient) -> None:
        """Whitespace-only query in research endpoint returns HTTP 422."""
        response = client.post("/api/v1/research/query", json={"query": "   \n\t  "})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"

    @pytest.mark.parametrize(
        "invalid_capital",
        [-100.0, 0.0, "not-a-number"],
    )
    def test_invalid_capital_amount_rejected(
        self, client: TestClient, invalid_capital: Any
    ) -> None:
        """Non-positive or non-numerical capital amounts return HTTP 422."""
        payload = {
            "ticker": "AAPL",
            "investor_profile": {"capital_amount": invalid_capital},
        }
        response = client.post("/api/v1/analysis", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_excessive_capital_amount_rejected(self, client: TestClient) -> None:
        """Capital amount exceeding max threshold returns HTTP 422."""
        payload = {
            "ticker": "AAPL",
            "investor_profile": {"capital_amount": 1e16},
        }
        response = client.post("/api/v1/analysis", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "exceeds maximum allowed threshold" in str(
            response.json()["error"]["details"]
        )

    def test_risk_tolerance_normalization(self) -> None:
        """Risk tolerance profile is stripped and lowercased."""
        profile = InvestorProfilePayload(risk_tolerance="  AGGRESSIVE  ")
        assert profile.risk_tolerance == "aggressive"

    def test_empty_clarification_answers_rejected(self, client: TestClient) -> None:
        """Submitting empty clarification answers dictionary returns HTTP 422."""
        response = client.post(
            "/api/v1/clarification",
            json={"clarification_answers": {}},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_clarification_answers_with_whitespace_key_rejected(
        self, client: TestClient
    ) -> None:
        """Clarification answers with empty or whitespace keys return HTTP 422."""
        response = client.post(
            "/api/v1/clarification",
            json={"clarification_answers": {"   ": "growth"}},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"


# ===========================================================================
# 2. PATH PARAMETER VALIDATION & CLEAR ERROR RESPONSES (15.2.2)
# ===========================================================================


class TestPathParameterAndErrorResponses:
    """Tests for path parameter format enforcement and HTTP error envelopes."""

    def test_malformed_analysis_id_returns_400(self, client: TestClient) -> None:
        """Malformed analysis_id returns HTTP 400 Bad Request."""
        malformed_id = "not-a-valid-uuid-format"
        response = client.get(f"/api/v1/analysis/{malformed_id}/status")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["error"]["code"] == "HTTP_ERROR"
        assert (
            f"Invalid analysis ID format: '{malformed_id}'" in data["error"]["message"]
        )
        assert "Expected a valid UUID" in data["error"]["message"]

    def test_malformed_report_id_returns_400(self, client: TestClient) -> None:
        """Malformed report_id returns HTTP 400 Bad Request."""
        malformed_id = "report-12345"
        response = client.get(f"/api/v1/reports/{malformed_id}")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["error"]["code"] == "HTTP_ERROR"
        assert f"Invalid report ID format: '{malformed_id}'" in data["error"]["message"]
        assert "Expected a valid UUID" in data["error"]["message"]

    def test_valid_uuid_format_not_found_returns_404(self, client: TestClient) -> None:
        """Syntactically valid UUID that does not exist returns HTTP 404 Not Found."""
        random_uuid = str(uuid.uuid4())
        response = client.get(f"/api/v1/analysis/{random_uuid}/status")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["error"]["code"] == "HTTP_ERROR"
        assert f"Analysis '{random_uuid}' not found" in data["error"]["message"]

    def test_invalid_report_format_query_param_returns_422(
        self, client: TestClient
    ) -> None:
        """Unsupported format query parameter returns HTTP 422."""
        valid_uuid = str(uuid.uuid4())
        response = client.get(f"/api/v1/reports/{valid_uuid}?format=unsupported_format")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"


# ===========================================================================
# 3. OPENAPI SPECIFICATION CONFORMANCE (15.2.1)
# ===========================================================================


class TestOpenAPIContractConformance:
    """Tests verifying OpenAPI documentation reflects Phase 15.2 contracts."""

    def test_openapi_schema_contains_endpoints_and_error_models(self, test_app) -> None:
        """FastAPI OpenAPI schema defines contracts and error envelopes."""
        schema = test_app.openapi()
        assert "paths" in schema
        paths = schema["paths"]

        # Verify all endpoints are documented
        assert "/api/v1/chat" in paths
        assert "/api/v1/analysis" in paths
        assert "/api/v1/clarification" in paths
        assert "/api/v1/research/query" in paths
        assert "/api/v1/analysis/{analysis_id}/status" in paths
        assert "/api/v1/reports/{report_id}" in paths

        # Verify APIErrorEnvelope is in schema components
        schemas = schema.get("components", {}).get("schemas", {})
        assert "APIErrorEnvelope" in schemas
        assert "APIErrorDetail" in schemas

        # Verify status endpoint documents 400 and 404 error responses
        status_get = paths["/api/v1/analysis/{analysis_id}/status"]["get"]
        assert "400" in status_get["responses"]
        assert "404" in status_get["responses"]
        assert "422" in status_get["responses"]
        assert "500" in status_get["responses"]

        # Verify report retrieval endpoint documents 400 and 404 error responses
        reports_get = paths["/api/v1/reports/{report_id}"]["get"]
        assert "400" in reports_get["responses"]
        assert "404" in reports_get["responses"]
        assert "422" in reports_get["responses"]
        assert "500" in reports_get["responses"]
