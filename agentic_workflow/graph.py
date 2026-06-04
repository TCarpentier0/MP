# agentic_workflow/graph.py
from langgraph.graph import StateGraph, END
from agentic_workflow.state import PipelineState
from agentic_workflow.agents.agent1_rule_analyser import run_agent_1
from agentic_workflow.agents.agent2_rag_retriever import run_agent_2
from agentic_workflow.agents.agent3_code_generator import run_agent_3
from agentic_workflow.agents.agent4_syntax_validator import run_agent_4
from agentic_workflow.agents.agent5_semantic_judge import run_agent_5
from agentic_workflow.agents.agent6_report_generator import run_agent_6
from workflow_config import MAX_SYNTAX_ITERATIONS, MAX_SEMANTIC_ITERATIONS


# --- ROUTING FUNCTIONS ---
def route_after_agent_4(state: PipelineState) -> str:
    """Route after syntax validation."""
    if state.pipeline_failed:
        return "error"
    if state.current_agent == "agent_5":
        return "agent_5"
    if state.syntax_iterations >= MAX_SYNTAX_ITERATIONS:
        return "error"
    return "agent_3"


def route_after_agent_5(state: PipelineState) -> str:
    """Route after semantic validation."""
    if state.pipeline_failed:
        return "error"
    if state.current_agent == "agent_6":
        return "agent_6"
    if state.semantic_iterations >= MAX_SEMANTIC_ITERATIONS:
        return "agent_6"
    return "agent_3"


def route_after_agent_3(state: PipelineState) -> str:
    """Route after code generation — always to Agent 4."""
    if state.pipeline_failed:
        return "error"
    return "agent_4"


def error_node(state: PipelineState) -> PipelineState:
    """Terminal error node."""
    print("\n[PIPELINE] Error state reached.")
    print(f"[PIPELINE] Error log: {state.error_log}")
    return state


# --- BUILD GRAPH ---
def build_graph():
    """Build and compile the LangGraph pipeline."""

    graph = StateGraph(PipelineState)

    # --- Add nodes ---
    graph.add_node("agent_1", run_agent_1)
    graph.add_node("agent_2", run_agent_2)
    graph.add_node("agent_3", run_agent_3)
    graph.add_node("agent_4", run_agent_4)
    graph.add_node("agent_5", run_agent_5)
    graph.add_node("agent_6", run_agent_6)
    graph.add_node("error", error_node)

    # --- Set entry point ---
    graph.set_entry_point("agent_1")

    # --- Add edges ---
    # Linear edges
    graph.add_edge("agent_1", "agent_2")
    graph.add_edge("agent_2", "agent_3")

    # Conditional edges
    graph.add_conditional_edges(
        "agent_3",
        route_after_agent_3,
        {
            "agent_4": "agent_4",
            "error": "error"
        }
    )

    graph.add_conditional_edges(
        "agent_4",
        route_after_agent_4,
        {
            "agent_5": "agent_5",
            "agent_3": "agent_3",
            "error": "error"
        }
    )

    graph.add_conditional_edges(
        "agent_5",
        route_after_agent_5,
        {
            "agent_6": "agent_6",
            "agent_3": "agent_3",
            "error": "error"
        }
    )

    # Terminal edges
    graph.add_edge("agent_6", END)
    graph.add_edge("error", END)

    return graph.compile()


# --- CONVENIENCE FUNCTION ---
def run_compliance_check(
    building_rule: str,
    ifc_file_path: str,
    api_key: str,
) -> PipelineState:
    """
    Run the full compliance checking pipeline.

    Args:
        building_rule: The building regulation rule in natural language
        ifc_file_path: Path to the IFC file to check
        api_key: OpenRouter API key

    Returns:
        Final PipelineState after pipeline completion
    """
    pipeline = build_graph()

    initial_state = PipelineState(
        building_rule_text=building_rule,
        ifc_file_path=ifc_file_path,
        api_key=api_key
    )

    print("\n" + "="*60)
    print("IFC COMPLIANCE CHECKING PIPELINE")
    print("="*60)
    print(f"Rule:      {building_rule}")
    print(f"IFC File:  {ifc_file_path}")
    print("="*60)

    final_state = pipeline.invoke(initial_state)

    if not final_state.get('pipeline_failed', False):
        report = final_state.get('report', {})
        summary = report.get('summary', {})
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Syntax iterations:   {final_state.get('syntax_iterations', 0)}")
        print(f"Semantic iterations: {final_state.get('semantic_iterations', 0)}")
        print(f"Total elements:      {summary.get('total_elements', 'N/A')}")
        print(f"Compliant:           {summary.get('compliant', 'N/A')}")
        print(f"Non-compliant:       {summary.get('non_compliant', 'N/A')}")
        print("="*60)

    return final_state

