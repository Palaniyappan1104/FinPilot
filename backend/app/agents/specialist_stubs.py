"""Specialist stub nodes and execution isolation wrappers for Phase 5.3.

These stub nodes act strictly as orchestration placeholders within LangGraph
during Phase 5. They:
- Identify which specialist they represent (using SpecialistName).
- Extract parameters from GraphState (e.g. from cio_decision or investor_profile).
- Emulate execution isolation and timeout boundaries.
- Return deterministic state updates without calling LLMs or external APIs.
- Are cleanly isolated from future real specialist implementations (Phases 6-10).
"""

import concurrent.futures
from typing import Any, Callable, Dict, Optional

from app.agents.base import AgentResult
from app.agents.cio_schema import SpecialistName
from app.agents.state import GraphState
from app.core.logging import get_logger

logger = get_logger("app.agents.specialist_stubs")

# Mapping of SpecialistName enum to the corresponding GraphState result key
SPECIALIST_STATE_KEY_MAP: Dict[SpecialistName, str] = {
    SpecialistName.TECHNICAL: "technical_result",
    SpecialistName.FUNDAMENTAL: "fundamental_result",
    SpecialistName.NEWS: "news_result",
    SpecialistName.RESEARCH: "research_result",
    SpecialistName.RISK: "risk_result",
}

# Shared bounded thread pool for specialist node timeout isolation.
# Using a shared executor avoids recreating and shutting down an executor per-call.
# Crucially, this ensures that when a timeout occurs, future.result(timeout=...)
# immediately unblocks the caller without waiting for the underlying timed-out
# worker thread.
_SPECIALIST_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=10,
    thread_name_prefix="finpilot-specialist",
)


def run_with_timeout_and_isolation(
    fn: Callable[[GraphState], Dict[str, Any]],
    state: GraphState,
    specialist_name: SpecialistName,
    timeout_seconds: Optional[float] = 30.0,
) -> Dict[str, Any]:
    """Execute a specialist node function with robust error and timeout isolation.

    Ensures that if a specialist node raises an unhandled exception or times out,
    the failure is trapped at the branch level, converted into an AgentResult failure,
    and written to the specialist's result field in state without crashing the graph.

    Note on Python Thread Limitation:
    In CPython, an already-running thread executing blocking operations or CPU-bound
    loops cannot be forcibly killed without terminating the process. Therefore, upon
    timeout, the specialist node immediately returns a timeout failure result to the
    graph workflow, while the underlying background worker thread runs to its natural
    cooperative conclusion.

    Args:
        fn: Specialist node callable accepting GraphState and returning state update.
        state: Incoming GraphState.
        specialist_name: Identifier of the specialist being executed.
        timeout_seconds: Maximum allowed execution duration in seconds.
            None for no timeout.

    Returns:
        Dict[str, Any]: State update containing the specialist result.
    """
    state_key = SPECIALIST_STATE_KEY_MAP[specialist_name]

    if timeout_seconds is not None and timeout_seconds > 0:
        try:
            future = _SPECIALIST_EXECUTOR.submit(fn, state)
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            logger.error(
                "Specialist %s timed out after %.2f seconds.",
                specialist_name.value,
                timeout_seconds,
            )
            failure_result = AgentResult.create_failure(
                error=(
                    f"Specialist '{specialist_name.value}' timed out after "
                    f"{timeout_seconds} seconds."
                ),
            )
            return {
                state_key: {
                    "specialist": specialist_name.value,
                    "status": "timeout",
                    "success": False,
                    "error": failure_result.error,
                    "is_stub": True,
                }
            }
        except Exception as exc:
            logger.error(
                "Specialist %s execution failed with exception: %s",
                specialist_name.value,
                exc,
                exc_info=True,
            )
            failure_result = AgentResult.create_failure(
                error=f"Specialist '{specialist_name.value}' failed: {exc}",
            )
            return {
                state_key: {
                    "specialist": specialist_name.value,
                    "status": "failed",
                    "success": False,
                    "error": failure_result.error,
                    "is_stub": True,
                }
            }
    else:
        try:
            return fn(state)
        except Exception as exc:
            logger.error(
                "Specialist %s execution failed with exception: %s",
                specialist_name.value,
                exc,
                exc_info=True,
            )
            failure_result = AgentResult.create_failure(
                error=f"Specialist '{specialist_name.value}' failed: {exc}",
            )
            return {
                state_key: {
                    "specialist": specialist_name.value,
                    "status": "failed",
                    "success": False,
                    "error": failure_result.error,
                    "is_stub": True,
                }
            }


def create_specialist_stub_node(
    specialist_name: SpecialistName,
    timeout_seconds: Optional[float] = 30.0,
    custom_handler: Optional[Callable[[GraphState], Dict[str, Any]]] = None,
) -> Callable[[GraphState], Dict[str, Any]]:
    """Factory creating an isolated orchestration stub node for a specialist.

    Args:
        specialist_name: The SpecialistName represented by this stub node.
        timeout_seconds: Optional per-agent timeout in seconds.
        custom_handler: Optional handler replacing default stub behavior.

    Returns:
        Callable[[GraphState], Dict[str, Any]]: A LangGraph-compatible node function.
    """
    state_key = SPECIALIST_STATE_KEY_MAP[specialist_name]

    def _default_stub(state: GraphState) -> Dict[str, Any]:
        cio_dec = state.get("cio_decision") or {}
        tasks = cio_dec.get("specialist_tasks") or {}
        task_info = tasks.get(specialist_name.value) or {}

        company = (
            task_info.get("parameters", {}).get("company")
            or cio_dec.get("target_company")
            or (state.get("investor_profile") or {}).get("target_company")
            or "Unknown"
        )
        ticker = (
            task_info.get("parameters", {}).get("ticker")
            or cio_dec.get("ticker")
            or (state.get("investor_profile") or {}).get("ticker")
        )

        res = AgentResult.create_success(
            data={
                "specialist": specialist_name.value,
                "status": "completed",
                "is_stub": True,
                "company": company,
                "ticker": ticker,
                "task_description": task_info.get("task_description", ""),
                "parameters": task_info.get("parameters", {}),
            },
            confidence=1.0,
        )

        return {
            state_key: {
                "specialist": specialist_name.value,
                "status": "completed",
                "success": True,
                "is_stub": True,
                "data": res.data,
                "confidence": res.confidence,
            }
        }

    handler = custom_handler or _default_stub

    def _node(state: GraphState) -> Dict[str, Any]:
        return run_with_timeout_and_isolation(
            fn=handler,
            state=state,
            specialist_name=specialist_name,
            timeout_seconds=timeout_seconds,
        )

    _node.__name__ = f"{specialist_name.value}_node"
    return _node
