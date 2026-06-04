# agentic_workflow/agents/agent4_syntax_validator.py
import ast
import importlib.util
from typing import Optional
from agentic_workflow.state import PipelineState
from workflow_config import MAX_SYNTAX_ITERATIONS


def _check_syntax_and_imports(code: str) -> Optional[str]:
    """Check for syntax errors and missing top-level module imports."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"{type(e).__name__}: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split('.')[0]
                if importlib.util.find_spec(top) is None:
                    return f"ModuleNotFoundError: No module named '{alias.name}'"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split('.')[0]
                if importlib.util.find_spec(top) is None:
                    return f"ModuleNotFoundError: No module named '{node.module}'"

    return None


def _check_structure(code: str) -> Optional[str]:
    """
    Check that the script contains the required structural elements.
    These are deterministic AST/string checks that catch whole classes of
    structural errors before paying for Agent 5's judge runs.
    """
    # Must access sys.argv to receive the IFC file path
    if "sys.argv" not in code:
        return (
            "StructureError: script does not access sys.argv — "
            "it will not receive the IFC file path as a command-line argument. "
            "Add: ifc_path = sys.argv[1]"
        )

    # Must call .by_type() to query any IFC elements
    if ".by_type(" not in code:
        return (
            "StructureError: script does not call model.by_type() — "
            "no IFC elements will be queried. "
            "Add at least one call such as: model.by_type('IfcDoor')"
        )

    # Must produce JSON output
    if "json.dumps(" not in code:
        return (
            "StructureError: script does not call json.dumps() — "
            "no JSON output will be produced. "
            "The final result must be printed as: print(json.dumps(result))"
        )

    if "print(" not in code:
        return (
            "StructureError: script has no print() call — "
            "the JSON result will not be written to stdout."
        )

    return None


def _check_code(code: str) -> Optional[str]:
    """Run all checks: syntax, imports, then structural requirements."""
    error = _check_syntax_and_imports(code)
    if error:
        return error
    return _check_structure(code)


def run_agent_4(state: PipelineState) -> PipelineState:
    """
    Agent 4: Syntax Validator
    Input:  state.generated_code
    Output: state.syntax_error (None if no error, error message if error)
            state.syntax_iterations (incremented per attempt)
            state.current_agent ('agent_5' if ok, 'agent_3' if error and iterations remain,
                                 'pipeline_failed' if max iterations reached)
    """
    print(f"\n[Agent 4] Syntax validation — attempt {state.syntax_iterations + 1} "
          f"of {MAX_SYNTAX_ITERATIONS}...")

    error = _check_code(state.generated_code)

    if error:
        new_iterations = state.syntax_iterations + 1
        print(f"[Agent 4] Error detected:\n{error}")

        if new_iterations >= MAX_SYNTAX_ITERATIONS:
            print(f"[Agent 4] Maximum iterations ({MAX_SYNTAX_ITERATIONS}) reached. "
                  f"Pipeline failed.")
            return state.model_copy(update={
                "syntax_error": error,
                "syntax_iterations": new_iterations,
                "pipeline_failed": True,
                "error_log": state.error_log + [
                    f"Agent 4: max iterations reached. Last error: {error}"
                ],
                "current_agent": "error"
            })

        print(f"[Agent 4] Returning to Agent 3 for correction "
              f"(iteration {new_iterations}/{MAX_SYNTAX_ITERATIONS})...")
        return state.model_copy(update={
            "syntax_error": error,
            "syntax_iterations": new_iterations,
            "current_agent": "agent_3"
        })

    print("[Agent 4] No errors detected. Advancing to Agent 5.")
    return state.model_copy(update={
        "syntax_error": None,
        "current_agent": "agent_5"
    })