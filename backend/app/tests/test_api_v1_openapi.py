"""Phase 15.4 OpenAPI Documentation Test Suite.

Verifies:
- GET /openapi.json returns valid OpenAPI 3.1.x schema with accurate metadata
- GET /docs returns 200 with Swagger UI
- GET /redoc returns 200 with ReDoc UI
- All routes have operation IDs, summaries, descriptions, and tags
- Key models and error envelopes are registered in components.schemas
"""

from fastapi.testclient import TestClient

from app.main import TAGS_METADATA, app

client = TestClient(app)


class TestOpenAPISpecification:
    """Tests verifying the OpenAPI 3 schema generated at /openapi.json."""

    def test_openapi_json_status_and_metadata(self) -> None:
        """Verify /openapi.json returns 200 and top-level info is accurate."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()

        assert "openapi" in data
        assert data["openapi"].startswith("3.")

        info = data["info"]
        assert info["title"] == "FinPilot"
        assert info["version"] == "0.1.0"
        assert "FinPilot" in info["description"]
        assert "multi-agent" in info["description"]

    def test_openapi_tags_metadata(self) -> None:
        """Verify openapi tags contain configured documentation tags."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()

        tags = data.get("tags", [])
        tag_names = {t["name"] for t in tags}
        expected_tag_names = {t["name"] for t in TAGS_METADATA}
        assert expected_tag_names.issubset(tag_names)

        for tag in tags:
            assert tag["name"]
            assert tag.get("description")

    def test_docs_ui_available(self) -> None:
        """Verify Swagger UI is accessible at /docs."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "swagger-ui" in response.text.lower() or "openapi.json" in response.text

    def test_redoc_ui_available(self) -> None:
        """Verify ReDoc UI is accessible at /redoc."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "redoc" in response.text.lower() or "openapi.json" in response.text


class TestOpenAPIRoutesAndOperations:
    """Tests verifying all endpoints are properly documented in OpenAPI."""

    EXPECTED_ENDPOINTS = {
        ("/health", "get"): {
            "operation_id": "getHealthRoot",
            "tag": "health",
        },
        ("/api/v1/health", "get"): {
            "operation_id": "getHealthV1",
            "tag": "health",
        },
        ("/api/v1/documents/upload", "post"): {
            "operation_id": "uploadResearchDocument",
            "tag": "documents",
        },
        ("/api/v1/chat", "post"): {
            "operation_id": "submitChatQuery",
            "tag": "analysis",
        },
        ("/api/v1/analysis", "post"): {
            "operation_id": "triggerCompanyAnalysis",
            "tag": "analysis",
        },
        ("/api/v1/clarification", "post"): {
            "operation_id": "submitClarificationAnswers",
            "tag": "analysis",
        },
        ("/api/v1/research/query", "post"): {
            "operation_id": "queryDocumentResearch",
            "tag": "analysis",
        },
        ("/api/v1/analysis/{analysis_id}/status", "get"): {
            "operation_id": "getAnalysisStatus",
            "tag": "analysis",
        },
        ("/api/v1/reports/{report_id}", "get"): {
            "operation_id": "getCompletedReport",
            "tag": "reports",
        },
    }

    def test_all_routes_present_and_documented(self) -> None:
        """Verify all planned API endpoints exist with documentation metadata."""
        response = client.get("/openapi.json")
        data = response.json()
        paths = data["paths"]

        operation_ids = set()

        for (path, method), expected in self.EXPECTED_ENDPOINTS.items():
            assert path in paths, f"Path {path} missing from OpenAPI spec"
            assert method in paths[path], f"Method {method} missing for {path}"

            endpoint_def = paths[path][method]
            m_upper = method.upper()
            assert endpoint_def.get("summary"), f"Missing summary for {m_upper} {path}"
            assert endpoint_def.get(
                "description"
            ), f"Missing description for {m_upper} {path}"

            op_id = endpoint_def.get("operationId")
            assert op_id == expected["operation_id"], (
                f"Expected operationId {expected['operation_id']} "
                f"for {m_upper} {path}, got {op_id}"
            )
            assert op_id not in operation_ids, f"Duplicate operationId: {op_id}"
            operation_ids.add(op_id)

            tags = endpoint_def.get("tags", [])
            assert (
                expected["tag"] in tags
            ), f"Expected tag {expected['tag']} in {tags} for {path}"


class TestOpenAPIComponentSchemas:
    """Tests verifying component schemas in the OpenAPI specification."""

    EXPECTED_SCHEMAS = [
        "APIErrorEnvelope",
        "APIErrorDetail",
        "AnalysisExecutionResponse",
        "AnalysisStatusResponse",
        "ReportRetrievalResponse",
        "ChatQueryRequest",
        "CompanyAnalysisRequest",
        "ClarificationSubmitRequest",
        "ResearchQueryRequest",
        "DocumentUploadResponse",
        "HealthResponse",
        "FinalReport",
    ]

    def test_core_schemas_present(self) -> None:
        """Verify all essential API request, response, and domain models are present."""
        response = client.get("/openapi.json")
        data = response.json()
        schemas = data.get("components", {}).get("schemas", {})

        for schema_name in self.EXPECTED_SCHEMAS:
            assert (
                schema_name in schemas
            ), f"Schema '{schema_name}' missing from components.schemas"

    def test_analysis_execution_response_schema_properties(self) -> None:
        """Verify AnalysisExecutionResponse schema has all documented properties."""
        response = client.get("/openapi.json")
        schemas = response.json().get("components", {}).get("schemas", {})
        schema = schemas["AnalysisExecutionResponse"]

        props = schema.get("properties", {})
        expected_props = [
            "analysis_id",
            "trace_id",
            "status",
            "clarification_needed",
            "clarification_questions",
            "report_id",
            "report",
            "error",
        ]
        for prop in expected_props:
            assert (
                prop in props
            ), f"Property '{prop}' missing in AnalysisExecutionResponse schema"

    def test_analysis_status_response_schema_properties(self) -> None:
        """Verify AnalysisStatusResponse schema has all documented properties."""
        response = client.get("/openapi.json")
        schemas = response.json().get("components", {}).get("schemas", {})
        schema = schemas["AnalysisStatusResponse"]

        props = schema.get("properties", {})
        expected_props = [
            "analysis_id",
            "trace_id",
            "status",
            "ticker",
            "clarification_questions",
            "report_id",
            "created_at",
            "completed_at",
            "error",
            "progress_stage",
        ]
        for prop in expected_props:
            assert (
                prop in props
            ), f"Property '{prop}' missing in AnalysisStatusResponse schema"

    def test_error_envelope_documented_on_routes(self) -> None:
        """Verify error responses reference APIErrorEnvelope on reports route."""
        response = client.get("/openapi.json")
        paths = response.json()["paths"]

        reports_route_responses = paths["/api/v1/reports/{report_id}"]["get"][
            "responses"
        ]
        for code in ["400", "404", "422", "500"]:
            assert (
                code in reports_route_responses
            ), f"Status {code} not documented in responses"
            schema_ref = (
                reports_route_responses[code]
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
                .get("$ref", "")
            )
            assert "APIErrorEnvelope" in schema_ref
