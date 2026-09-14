"""FastAPI routes for FinPilot End-to-End Analysis and Reports (Phase 15.1-15.3).

Implements Phase 15.1 API endpoints, 15.2 contracts, and 15.3 async execution:
- POST /api/v1/chat: Conversational query endpoint (15.1.2, 15.2.1, 15.3.1, 15.3.2)
- POST /api/v1/analysis: Company analysis endpoint (15.1.3, 15.2.1, 15.3.1, 15.3.2)
- POST /api/v1/clarification: Clarification answer submission (15.1.4, 15.2.1, 15.3.2)
- POST /api/v1/research/query: Research query against documents (15.1.6, 15.3.1, 15.3.2)
- GET  /api/v1/analysis/{analysis_id}/status: Analysis status polling (15.1.7, 15.3.2)
- GET  /api/v1/reports/{report_id}: Report retrieval by ID (15.1.8, 15.2.1)
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    status,
)

from app.agents.graph import run_end_to_end_graph
from app.agents.report_formatter import (
    format_report_markdown,
    format_report_text,
)
from app.agents.report_schema import FinalReport
from app.agents.state import GraphState
from app.core.logging import get_logger
from app.models.api import (
    AnalysisExecutionResponse,
    AnalysisStatusResponse,
    APIErrorEnvelope,
    ChatQueryRequest,
    ClarificationSubmitRequest,
    CompanyAnalysisRequest,
    ReportRetrievalResponse,
    ResearchQueryRequest,
)

logger = get_logger("app.api.v1.analysis")

router = APIRouter()

# ---------------------------------------------------------------------------
# Standard OpenAPI Response Schemas for Error States (15.2.1, 15.2.2)
# ---------------------------------------------------------------------------
ERROR_400_RESPONSE = {
    "model": APIErrorEnvelope,
    "description": "Malformed request parameters or invalid identifier format.",
}
ERROR_404_RESPONSE = {
    "model": APIErrorEnvelope,
    "description": "Requested resource not found.",
}
ERROR_422_RESPONSE = {
    "model": APIErrorEnvelope,
    "description": "Validation error in request parameters or body payload.",
}
ERROR_500_RESPONSE = {
    "model": APIErrorEnvelope,
    "description": "Unexpected internal server execution failure.",
}

# ---------------------------------------------------------------------------
# Ephemeral API-Layer Status & Report Registry (Phase 15.1 & 15.3 requirement)
# Bounded to recent executions within the running process (no fake DB layer).
# ---------------------------------------------------------------------------
_RECENT_ANALYSES: Dict[str, Dict[str, Any]] = {}
_RECENT_REPORTS: Dict[str, FinalReport] = {}
_REGISTRY_LOCK = threading.Lock()
_MAX_STORED_ENTRIES = 200


def _validate_uuid_param(param_value: str, param_name: str) -> str:
    """Validate that a path parameter string conforms to a valid UUID format.

    Args:
        param_value: Raw string from path parameter.
        param_name: Label used in user-facing error message.

    Returns:
        str: Normalized canonical UUID string.

    Raises:
        HTTPException: 400 Bad Request if param_value is not a valid UUID.
    """
    try:
        parsed = uuid.UUID(param_value)
        return str(parsed)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid {param_name} format: '{param_value}'. "
                "Expected a valid UUID."
            ),
        )


def _register_running_analysis(
    analysis_id: str,
    trace_id: str,
    ticker: Optional[str] = None,
    progress_stage: str = "running",
) -> None:
    """Register an analysis session in 'running' state for async status tracking.

    Args:
        analysis_id: Unique analysis session identifier.
        trace_id: Correlated trace identifier.
        ticker: Optional associated stock ticker symbol.
        progress_stage: Initial progress descriptor (default 'running').
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    with _REGISTRY_LOCK:
        _RECENT_ANALYSES[analysis_id] = {
            "analysis_id": analysis_id,
            "trace_id": trace_id,
            "status": "running",
            "ticker": ticker,
            "clarification_questions": [],
            "report_id": None,
            "created_at": now_iso,
            "completed_at": None,
            "error": None,
            "progress_stage": progress_stage,
        }
        if len(_RECENT_ANALYSES) > _MAX_STORED_ENTRIES:
            oldest_key = next(iter(_RECENT_ANALYSES))
            del _RECENT_ANALYSES[oldest_key]


def _store_analysis_record(
    analysis_id: str,
    trace_id: str,
    execution_status: str,
    ticker: Optional[str],
    questions: List[str],
    report: Optional[FinalReport],
    error: Optional[str] = None,
    progress_stage: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Record or update an analysis lifecycle state in the ephemeral API registry."""
    now_iso = datetime.now(timezone.utc).isoformat()
    report_id: Optional[str] = None

    with _REGISTRY_LOCK:
        existing = _RECENT_ANALYSES.get(analysis_id)
        created_at = existing["created_at"] if existing else now_iso

        if report is not None:
            report_id = str(uuid.uuid4())
            _RECENT_REPORTS[report_id] = report

        resolved_stage = progress_stage or (
            "completed" if execution_status == "completed" else execution_status
        )

        _RECENT_ANALYSES[analysis_id] = {
            "analysis_id": analysis_id,
            "trace_id": trace_id,
            "status": execution_status,
            "ticker": ticker,
            "clarification_questions": questions,
            "report_id": report_id,
            "created_at": created_at,
            "completed_at": now_iso,
            "error": error,
            "progress_stage": resolved_stage,
        }

        # Keep memory footprint bounded
        if len(_RECENT_ANALYSES) > _MAX_STORED_ENTRIES:
            oldest_key = next(iter(_RECENT_ANALYSES))
            del _RECENT_ANALYSES[oldest_key]
        if len(_RECENT_REPORTS) > _MAX_STORED_ENTRIES:
            oldest_rep = next(iter(_RECENT_REPORTS))
            del _RECENT_REPORTS[oldest_rep]

    return analysis_id, report_id


def get_graph_runner() -> Callable[..., GraphState]:
    """Dependency provider returning the Phase 13 end-to-end graph runner."""
    return run_end_to_end_graph


def _extract_response_from_state(
    final_state: GraphState,
    analysis_id: str,
    trace_id: str,
    fallback_ticker: Optional[str] = None,
) -> AnalysisExecutionResponse:
    """Transform resulting GraphState into a clean AnalysisExecutionResponse."""
    clarified_req = final_state.get("clarified_request") or {}
    clarification_needed = bool(clarified_req.get("clarification_needed", False))
    clarification_questions: List[str] = clarified_req.get(
        "clarification_questions", []
    )
    inv_profile = final_state.get("investor_profile")
    report_dict = (final_state.get("report") or {}).get("data")
    ticker = (
        final_state.get("ticker")
        or (inv_profile.get("ticker") if inv_profile else None)
        or fallback_ticker
    )

    final_report: Optional[FinalReport] = None
    if report_dict:
        try:
            final_report = FinalReport.model_validate(report_dict)
        except Exception as err:
            logger.warning(
                "Failed to validate report_dict against FinalReport: %s", err
            )

    # Determine execution status
    if clarification_needed:
        execution_status = "clarification_needed"
    elif final_report is not None:
        execution_status = "completed"
    elif final_state.get("error"):
        execution_status = "failed"
    else:
        execution_status = "completed"

    # Store in API registry for status polling and report retrieval
    _, report_id = _store_analysis_record(
        analysis_id=analysis_id,
        trace_id=trace_id,
        execution_status=execution_status,
        ticker=ticker,
        questions=clarification_questions,
        report=final_report,
        error=final_state.get("error"),
    )

    return AnalysisExecutionResponse(
        analysis_id=analysis_id,
        trace_id=trace_id,
        status=execution_status,
        clarification_needed=clarification_needed,
        clarification_questions=clarification_questions,
        investor_profile=dict(inv_profile) if inv_profile else None,
        report=final_report,
        report_id=report_id,
        error=final_state.get("error"),
    )


# ---------------------------------------------------------------------------
# 15.1.2 Chat / Query Endpoint (Async enabled via 15.3)
# ---------------------------------------------------------------------------
@router.post(
    "/chat",
    response_model=AnalysisExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit user query or chat message",
    description=(
        "Processes a natural language query through Conversation and "
        "Clarification agents. Halts with clarification questions if missing "
        "constraints, or proceeds through the full pipeline to a completed report. "
        "Supports asynchronous execution via background=true."
    ),
    responses={
        422: ERROR_422_RESPONSE,
        500: ERROR_500_RESPONSE,
    },
    operation_id="submitChatQuery",
)
async def chat_query(
    request: ChatQueryRequest,
    background_tasks: BackgroundTasks,
    background: bool = Query(
        default=False,
        description=(
            "If true, execute analysis asynchronously in a background task "
            "and immediately return status 'running'. Poll GET /analysis/{id}/status "
            "for completion."
        ),
    ),
    runner: Callable[..., GraphState] = Depends(get_graph_runner),
) -> AnalysisExecutionResponse:
    """Execute conversational query through the end-to-end graph."""
    analysis_id = str(uuid.uuid4())
    trace_id = request.trace_id or f"trace-{uuid.uuid4().hex[:12]}"

    profile_dict = (
        request.investor_profile.model_dump(exclude_none=True)
        if request.investor_profile
        else None
    )

    if background:
        _register_running_analysis(analysis_id, trace_id, progress_stage="running")

        def _run_bg_chat() -> None:
            try:
                final_state = runner(
                    query=request.query,
                    investor_profile=profile_dict,
                    documents_available=request.documents_available,
                    trace_id=trace_id,
                )
                _extract_response_from_state(final_state, analysis_id, trace_id)
            except Exception as exc:
                logger.exception(
                    "Background chat execution failed for %s: %s", analysis_id, exc
                )
                _store_analysis_record(
                    analysis_id=analysis_id,
                    trace_id=trace_id,
                    execution_status="failed",
                    ticker=None,
                    questions=[],
                    report=None,
                    error="Failed to process analysis query. Please try again.",
                    progress_stage="failed",
                )

        background_tasks.add_task(_run_bg_chat)

        return AnalysisExecutionResponse(
            analysis_id=analysis_id,
            trace_id=trace_id,
            status="running",
            clarification_needed=False,
            clarification_questions=[],
            investor_profile=profile_dict,
            report=None,
            report_id=None,
            error=None,
        )

    try:
        final_state = runner(
            query=request.query,
            investor_profile=profile_dict,
            documents_available=request.documents_available,
            trace_id=trace_id,
        )
    except Exception as exc:
        logger.exception("Error executing chat query through graph: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process analysis query. Please try again.",
        ) from exc

    return _extract_response_from_state(final_state, analysis_id, trace_id)


# ---------------------------------------------------------------------------
# 15.1.3 Company Analysis Endpoint (Async enabled via 15.3)
# ---------------------------------------------------------------------------
@router.post(
    "/analysis",
    response_model=AnalysisExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger full company analysis",
    description=(
        "Triggers the complete FinPilot multi-agent workflow for a target equity "
        "ticker and investor profile. Returns a structured FinalReport or "
        "clarification questions. Supports asynchronous background execution "
        "via background=true."
    ),
    responses={
        422: ERROR_422_RESPONSE,
        500: ERROR_500_RESPONSE,
    },
    operation_id="triggerCompanyAnalysis",
)
async def company_analysis(
    request: CompanyAnalysisRequest,
    background_tasks: BackgroundTasks,
    background: bool = Query(
        default=False,
        description=(
            "If true, execute analysis asynchronously in a background task "
            "and immediately return status 'running'. Poll GET /analysis/{id}/status "
            "for progress."
        ),
    ),
    runner: Callable[..., GraphState] = Depends(get_graph_runner),
) -> AnalysisExecutionResponse:
    """Execute company investment analysis through the end-to-end graph."""
    analysis_id = str(uuid.uuid4())
    trace_id = request.trace_id or f"trace-{uuid.uuid4().hex[:12]}"
    clean_ticker = request.ticker

    profile_dict = (
        request.investor_profile.model_dump(exclude_none=True)
        if request.investor_profile
        else {}
    )
    profile_dict.setdefault("ticker", clean_ticker)
    if request.target_company:
        profile_dict.setdefault("target_company", request.target_company)

    query = request.query or f"Analyze investment feasibility for {clean_ticker}"

    if background:
        _register_running_analysis(
            analysis_id, trace_id, ticker=clean_ticker, progress_stage="running"
        )

        def _run_bg_analysis() -> None:
            try:
                final_state = runner(
                    query=query,
                    investor_profile=profile_dict,
                    documents_available=request.documents_available,
                    clarification_answers=request.clarification_answers,
                    ticker=clean_ticker,
                    target_company=request.target_company,
                    trace_id=trace_id,
                )
                _extract_response_from_state(
                    final_state,
                    analysis_id,
                    trace_id,
                    fallback_ticker=clean_ticker,
                )
            except Exception as exc:
                logger.exception(
                    "Background analysis execution failed for %s: %s",
                    analysis_id,
                    exc,
                )
                _store_analysis_record(
                    analysis_id=analysis_id,
                    trace_id=trace_id,
                    execution_status="failed",
                    ticker=clean_ticker,
                    questions=[],
                    report=None,
                    error="Failed to execute company analysis. Please try again.",
                    progress_stage="failed",
                )

        background_tasks.add_task(_run_bg_analysis)

        return AnalysisExecutionResponse(
            analysis_id=analysis_id,
            trace_id=trace_id,
            status="running",
            clarification_needed=False,
            clarification_questions=[],
            investor_profile=profile_dict,
            report=None,
            report_id=None,
            error=None,
        )

    try:
        final_state = runner(
            query=query,
            investor_profile=profile_dict,
            documents_available=request.documents_available,
            clarification_answers=request.clarification_answers,
            ticker=clean_ticker,
            target_company=request.target_company,
            trace_id=trace_id,
        )
    except Exception as exc:
        logger.exception("Error executing company analysis: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute company analysis. Please try again.",
        ) from exc

    return _extract_response_from_state(
        final_state, analysis_id, trace_id, fallback_ticker=clean_ticker
    )


# ---------------------------------------------------------------------------
# 15.1.4 Clarification Submission Endpoint (Async enabled via 15.3)
# ---------------------------------------------------------------------------
@router.post(
    "/clarification",
    response_model=AnalysisExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit answers to clarification questions",
    description=(
        "Submits answers to missing investor profile fields, resolving the "
        "clarification halt and continuing graph execution to full report completion. "
        "Supports asynchronous background execution via background=true."
    ),
    responses={
        422: ERROR_422_RESPONSE,
        500: ERROR_500_RESPONSE,
    },
    operation_id="submitClarificationAnswers",
)
async def submit_clarification(
    request: ClarificationSubmitRequest,
    background_tasks: BackgroundTasks,
    background: bool = Query(
        default=False,
        description=(
            "If true, resume analysis asynchronously in a background task "
            "and immediately return status 'running'."
        ),
    ),
    runner: Callable[..., GraphState] = Depends(get_graph_runner),
) -> AnalysisExecutionResponse:
    """Submit clarification answers and resume pipeline execution."""
    analysis_id = request.analysis_id or str(uuid.uuid4())
    trace_id = request.trace_id or f"trace-{uuid.uuid4().hex[:12]}"

    profile_dict = (
        request.investor_profile.model_dump(exclude_none=True)
        if request.investor_profile
        else None
    )

    query = request.query or "Proceed with analysis using provided clarifications"

    if background:
        _register_running_analysis(
            analysis_id, trace_id, ticker=request.ticker, progress_stage="running"
        )

        def _run_bg_clarification() -> None:
            try:
                final_state = runner(
                    query=query,
                    investor_profile=profile_dict,
                    documents_available=request.documents_available,
                    clarification_answers=request.clarification_answers,
                    ticker=request.ticker,
                    trace_id=trace_id,
                )
                _extract_response_from_state(
                    final_state,
                    analysis_id,
                    trace_id,
                    fallback_ticker=request.ticker,
                )
            except Exception as exc:
                logger.exception(
                    "Background clarification execution failed for %s: %s",
                    analysis_id,
                    exc,
                )
                _store_analysis_record(
                    analysis_id=analysis_id,
                    trace_id=trace_id,
                    execution_status="failed",
                    ticker=request.ticker,
                    questions=[],
                    report=None,
                    error="Failed to submit clarification answers. Please try again.",
                    progress_stage="failed",
                )

        background_tasks.add_task(_run_bg_clarification)

        return AnalysisExecutionResponse(
            analysis_id=analysis_id,
            trace_id=trace_id,
            status="running",
            clarification_needed=False,
            clarification_questions=[],
            investor_profile=profile_dict,
            report=None,
            report_id=None,
            error=None,
        )

    try:
        final_state = runner(
            query=query,
            investor_profile=profile_dict,
            documents_available=request.documents_available,
            clarification_answers=request.clarification_answers,
            ticker=request.ticker,
            trace_id=trace_id,
        )
    except Exception as exc:
        logger.exception("Error executing clarification resumption: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit clarification answers. Please try again.",
        ) from exc

    return _extract_response_from_state(
        final_state, analysis_id, trace_id, fallback_ticker=request.ticker
    )


# ---------------------------------------------------------------------------
# 15.1.6 Research Query Endpoint (Async enabled via 15.3)
# ---------------------------------------------------------------------------
@router.post(
    "/research/query",
    response_model=AnalysisExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask research questions against uploaded documents",
    description=(
        "Submits a research query targeted at document filings and corporate "
        "disclosures, routing through the Research Analyst specialist. "
        "Supports asynchronous background execution via background=true."
    ),
    responses={
        422: ERROR_422_RESPONSE,
        500: ERROR_500_RESPONSE,
    },
    operation_id="queryDocumentResearch",
)
async def research_query(
    request: ResearchQueryRequest,
    background_tasks: BackgroundTasks,
    background: bool = Query(
        default=False,
        description=(
            "If true, execute research query asynchronously in a background task "
            "and immediately return status 'running'."
        ),
    ),
    runner: Callable[..., GraphState] = Depends(get_graph_runner),
) -> AnalysisExecutionResponse:
    """Execute research document query through the end-to-end graph."""
    analysis_id = str(uuid.uuid4())
    trace_id = request.trace_id or f"trace-{uuid.uuid4().hex[:12]}"
    clean_ticker = request.ticker

    if background:
        _register_running_analysis(
            analysis_id, trace_id, ticker=clean_ticker, progress_stage="running"
        )

        def _run_bg_research() -> None:
            try:
                final_state = runner(
                    query=request.query,
                    ticker=clean_ticker,
                    documents_available=request.documents_available,
                    trace_id=trace_id,
                )
                _extract_response_from_state(
                    final_state,
                    analysis_id,
                    trace_id,
                    fallback_ticker=clean_ticker,
                )
            except Exception as exc:
                logger.exception(
                    "Background research execution failed for %s: %s",
                    analysis_id,
                    exc,
                )
                _store_analysis_record(
                    analysis_id=analysis_id,
                    trace_id=trace_id,
                    execution_status="failed",
                    ticker=clean_ticker,
                    questions=[],
                    report=None,
                    error="Failed to execute research query. Please try again.",
                    progress_stage="failed",
                )

        background_tasks.add_task(_run_bg_research)

        return AnalysisExecutionResponse(
            analysis_id=analysis_id,
            trace_id=trace_id,
            status="running",
            clarification_needed=False,
            clarification_questions=[],
            investor_profile=None,
            report=None,
            report_id=None,
            error=None,
        )

    try:
        final_state = runner(
            query=request.query,
            ticker=clean_ticker,
            documents_available=request.documents_available,
            trace_id=trace_id,
        )
    except Exception as exc:
        logger.exception("Error executing research query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute research query. Please try again.",
        ) from exc

    return _extract_response_from_state(
        final_state, analysis_id, trace_id, fallback_ticker=clean_ticker
    )


# ---------------------------------------------------------------------------
# 15.1.7 Analysis Status Polling Endpoint (Phase 15.1 & 15.3.2)
# ---------------------------------------------------------------------------
@router.get(
    "/analysis/{analysis_id}/status",
    response_model=AnalysisStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Poll analysis status",
    description=(
        "Polls the lifecycle execution status of an active, running, "
        "or completed analysis."
    ),
    responses={
        400: ERROR_400_RESPONSE,
        404: ERROR_404_RESPONSE,
        422: ERROR_422_RESPONSE,
        500: ERROR_500_RESPONSE,
    },
    operation_id="getAnalysisStatus",
)
async def get_analysis_status(
    analysis_id: str,
) -> AnalysisStatusResponse:
    """Retrieve status for a specified analysis session."""
    valid_id = _validate_uuid_param(analysis_id, "analysis ID")

    with _REGISTRY_LOCK:
        record = _RECENT_ANALYSES.get(valid_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{valid_id}' not found.",
        )

    return AnalysisStatusResponse(
        analysis_id=record["analysis_id"],
        trace_id=record["trace_id"],
        status=record["status"],
        ticker=record.get("ticker"),
        clarification_questions=record.get("clarification_questions", []),
        report_id=record.get("report_id"),
        created_at=record["created_at"],
        completed_at=record.get("completed_at"),
        error=record.get("error"),
        progress_stage=record.get("progress_stage"),
    )


# ---------------------------------------------------------------------------
# 15.1.8 Report Retrieval Endpoint
# ---------------------------------------------------------------------------
@router.get(
    "/reports/{report_id}",
    response_model=ReportRetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve completed investment report",
    description=(
        "Retrieves a completed report by ID. Supports 'format=json' (default), "
        "'format=markdown', and 'format=summary'."
    ),
    tags=["reports"],
    responses={
        400: ERROR_400_RESPONSE,
        404: ERROR_404_RESPONSE,
        422: ERROR_422_RESPONSE,
        500: ERROR_500_RESPONSE,
    },
    operation_id="getCompletedReport",
)
async def get_report(
    report_id: str,
    format: Literal["json", "markdown", "summary"] = Query(
        "json", description="Target output rendering format"
    ),
) -> ReportRetrievalResponse:
    """Retrieve a completed report by ID in the requested format."""
    valid_id = _validate_uuid_param(report_id, "report ID")

    with _REGISTRY_LOCK:
        report = _RECENT_REPORTS.get(valid_id)

    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{valid_id}' not found.",
        )

    if format == "json":
        return ReportRetrievalResponse(
            report_id=valid_id,
            format="json",
            report=report,
            rendered_content=None,
        )
    elif format == "markdown":
        rendered = format_report_markdown(report)
        return ReportRetrievalResponse(
            report_id=valid_id,
            format="markdown",
            report=None,
            rendered_content=rendered,
        )
    else:  # summary
        rendered = format_report_text(report)
        return ReportRetrievalResponse(
            report_id=valid_id,
            format="summary",
            report=None,
            rendered_content=rendered,
        )
