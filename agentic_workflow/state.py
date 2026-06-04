from typing import List, Optional
from pydantic import BaseModel, Field


class BuildingRule(BaseModel):
    """Structured representation of a building rule — output of Agent 1."""
    primary_element: str = Field(
        description="Primary building element in natural language (e.g. 'door', 'stair', 'window')"
    )
    related_elements: List[str] = Field(
        default=[],
        description="Related building elements in natural language (e.g. ['handrail', 'threshold'])"
    )
    properties_to_check: List[str] = Field(
        default=[],
        description="Properties to verify in natural language (e.g. ['height', 'width', 'fire rating'])"
    )
    constraints: List[str] = Field(
        description="Logical constraints in plain English (e.g. ['height >= 2.1m', 'count >= 2'])"
    )
    scope_conditions: List[str] = Field(
        default=[],
        description="Conditions that limit the scope of the check (e.g. ['public building', 'more than 2 floors'])"
    )
    search_queries_bsdd: List[str] = Field(
        description="Property-focused search queries for bSDD retrieval — focus on element definitions, property names, measurements, classifications"
    )
    search_queries_express: List[str] = Field(
        description="Schema-focused search queries for EXPRESS retrieval — focus on element types, relationships, hierarchy, connections between elements"
    )
    rule_category: List[str] = Field(
        description="Ordered list of rule categories from outermost to innermost layer: "
                    "'conditional' (scope) → 'relational' (structure) → "
                    "'value', 'mathematical', or 'information_availability' (measurement). "
                    "Example: ['conditional', 'relational', 'value']"
    )


class PipelineState(BaseModel):
    """
    Central state object flowing through the full LangGraph pipeline.
    Each agent reads from this state and returns an updated copy.
    """

    # --- INPUT ---
    building_rule_text: str
    ifc_file_path: str
    api_key: str

    # --- AGENT 1 ---
    parsed_rule: Optional[BuildingRule] = None

    # --- AGENT 2 ---
    rag_context: Optional[str] = None
    retrieved_chunks_bsdd: List[str] = Field(default=[])
    retrieved_chunks_express: List[str] = Field(default=[])

    # --- AGENT 3 / 4 ---
    generated_code: Optional[str] = None
    initial_code: Optional[str] = None  # first generated code, never overwritten
    syntax_error: Optional[str] = None
    syntax_iterations: int = 0
    correction_history: List[str] = Field(default=[])

    # --- AGENT 5 ---
    execution_output: Optional[str] = None
    semantic_feedback: Optional[str] = None
    semantic_iterations: int = 0
    semantic_judgement: Optional[List[dict]] = None

    # --- AGENT 6 ---
    report: Optional[dict] = None
    report_path: Optional[str] = None

    # --- TIMING ---
    start_time: Optional[float] = None
    execution_time_seconds: Optional[float] = None

    # --- PIPELINE CONTROL ---
    current_agent: str = "agent_1"
    pipeline_failed: bool = False
    error_log: List[str] = Field(default=[])