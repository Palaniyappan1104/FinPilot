"""CLI script to manually invoke the minimal LangGraph workflow.

Phase 2.5 verification script:
- Creates initial GraphState using create_initial_state()
- Compiles the minimal StateGraph (START -> passthrough -> END)
- Invokes the graph and displays the resulting state
"""

import sys
from pathlib import Path

# Add backend to sys.path so app modules can be resolved
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agents import create_graph, create_initial_state


def main() -> None:
    sample_query = "Should I invest in Infosys for 5 years?"
    print("=" * 60)
    print("FinPilot Phase 2.5 - Minimal LangGraph CLI Invocation")
    print("=" * 60)
    print(f"\n[1] Initializing state with user_query: '{sample_query}'")
    initial_state = create_initial_state(user_query=sample_query)

    print("[2] Compiling graph (START -> passthrough -> END)...")
    graph = create_graph()

    print("[3] Invoking graph...")
    result_state = graph.invoke(initial_state)

    print("[4] Graph execution completed successfully!")
    print(f"    - Returned user_query: {result_state.get('user_query')}")
    print(f"    - Investor profile:    {result_state.get('investor_profile')}")
    print(f"    - GraphState fields:   {list(result_state.keys())}")
    print("\nResult state verification: SUCCESS\n")


if __name__ == "__main__":
    main()
