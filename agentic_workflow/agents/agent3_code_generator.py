# agentic_workflow/agents/agent3_code_generator.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from agentic_workflow.state import PipelineState
from workflow_config import (
    OPENROUTER_KEY, OPENROUTER_BASE, MODEL_GENERATOR,
    MAX_SYNTAX_ITERATIONS, MAX_SEMANTIC_ITERATIONS
)

# --- CATEGORY-SPECIFIC HINTS ---
CATEGORY_HINTS = {
    "value": """
VALUE RULE GUIDANCE:

STEP 1 — Unit conversion (always required before numeric comparison):
  PREFIX_MAP = {"MILLI": 0.001, "CENTI": 0.01, "DECI": 0.1, "KILO": 1000.0, None: 1.0}
  def get_length_unit_factor(model):
      for assignment in model.by_type("IfcUnitAssignment"):
          for unit in assignment.Units:
              if hasattr(unit, "UnitType") and unit.UnitType == "LENGTHUNIT":
                  return PREFIX_MAP.get(getattr(unit, "Prefix", None), 1.0)
      return 1.0  # fallback: metres
  For area units: check UnitType == "AREAUNIT" (factor is prefix^2).
  For dimensionless ratios (e.g. percentages): no unit conversion needed.

STEP 2 — Property lookup:
  Use the IFCOPENSHELL ACCESS PATTERN from the RAG domain knowledge as the primary guide
  for how to access each property — follow the Pset names, property names, and traversal
  path it describes exactly.
  If the RAG context marks the access pattern as NOT FOUND IN CHUNKS, fall back to:
  1. Direct IFC attribute on the element
  2. Psets via IsDefinedBy → RelatingPropertyDefinition → HasProperties
  3. Element type via IsTypedBy → RelatingType → HasPropertySets → HasProperties

STEP 3 — Comparison:
  Apply the unit factor from STEP 1 to the raw value before comparing.
  Use the exact operator from the rule (>= not >, <= not <).
  If the property is not found (None from all sources): status = "FAIL" with a descriptive
  fail_reason — never silently skip a missing value.

COMMON MISTAKES TO AVOID:
- Do NOT compare raw values without applying the unit factor first
- Do NOT use > instead of >= (or < instead of <=) — use the exact operator from the rule
- Do NOT override the RAG access pattern with a generic fallback when RAG provides one
""",

    "information_availability": """
INFORMATION AVAILABILITY GUIDANCE:

CRITICAL: collect values from ALL sources before evaluating. Never stop at the first match.

STEP 1 — Collect values:
  Gather all non-null, non-empty values from every source:
  - Direct IFC attribute: use the attribute name from the RAG domain knowledge
  - All Psets via IsDefinedBy: search EVERY Pset — do NOT filter by Pset name, only filter
    by property name using the names from the RAG context
  - Element type via IsTypedBy: search all property sets on the type object
  Exclude both None and empty string from the collected list.

STEP 2 — Evaluate the collected set:
  If no values were collected: status = "FAIL" with a descriptive fail_reason.
  Otherwise deduplicate: round all numeric values to 4 decimal places before comparing
  (avoids floating-point false conflicts such as 2.0 vs 2.0000000001); for non-numeric
  values, strip whitespace before comparing.
  If all deduplicated values are identical: status = "PASS".
  If multiple distinct values remain: status = "FAIL" with fail_reason listing the conflicts.

COMMON MISTAKES TO AVOID:
- Do NOT stop at the first match — always collect from ALL sources before evaluating
- Do NOT apply any numeric comparison (>=, <=) — existence and uniqueness are the only criteria
- None AND empty string must both be excluded from the collected list
- Property names to search for must come from the RAG context
""",

    "relational": """
RELATIONAL RULE GUIDANCE:

Use the RELEVANT RELATIONSHIPS section from the RAG domain knowledge to determine which
IFC relationship entity to navigate and the IFCOPENSHELL ACCESS PATTERN for traversal direction.

PATTERN 1 — IfcRelAggregates (parts of a composite, e.g. stair → railing):
  Navigate via the IsDecomposedBy inverse attribute on the primary element.
  Search two levels deep if needed — e.g. stair → flight → railing — do not stop at one level.

PATTERN 2 — IfcRelContainedInSpatialStructure (elements in the same space):
  Iterate all relationships of this type globally, check if the primary element appears in
  RelatedElements, then collect sibling elements of the target type from the same relationship.

Reporting rules:
- total_elements = number of PRIMARY elements checked (e.g. number of stairs)
- GlobalId in output = GlobalId of the PRIMARY element, not the related element
- FAIL reason must state which related elements are missing or wrongly positioned

COMMON MISTAKES TO AVOID:
- Do NOT count related elements as separate entries in total_elements
- Do NOT report the GlobalId of the related (missing) element — always the primary
- Do NOT stop at one level of aggregation — search two levels deep when navigating composites
""",

    "mathematical": """
MATHEMATICAL RULE GUIDANCE:

STEP 1 — Group elements by spatial unit:
  Use IfcRelContainedInSpatialStructure to group target elements per space. Iterate all
  relationships of this type, filter RelatingStructure to IfcSpace only, and collect
  target elements per space into a dictionary keyed by the space GlobalId.

STEP 2 — Retrieve quantities:
  Use the Qto name, quantity name, and value attribute from the RAG domain knowledge
  (RELEVANT PSETS AND PROPERTIES section). Note: Qto quantities use the `Quantities`
  attribute (not `HasProperties`), and values are stored in typed attributes such as
  AreaValue, LengthValue, VolumeValue, or CountValue — not NominalValue.wrappedValue.
  If the RAG context marks the Qto as NOT FOUND IN CHUNKS, apply this traversal:
  IsDefinedBy → RelatingPropertyDefinition → Quantities → typed value attribute.

STEP 3 — Calculate and compare per spatial unit:
  Perform the complete calculation for each space before applying the threshold.
  Treat a missing quantity on an element as 0 when summing (do not skip the element).
  If the denominator (e.g. floor area) is None or zero: mark the space as FAIL with
  an explanation of which value is missing. Otherwise round the result to 4 decimal
  places before comparing, and use the exact operator from the rule.

Reporting rules:
- total_elements = number of spatial units (rooms, floors) checked, NOT number of elements
- Report both the calculated value AND the threshold in the output fields
- If any required input value is None: mark the spatial unit as FAIL with explanation

COMMON MISTAKES TO AVOID:
- Do NOT aggregate at the wrong level (e.g. per building instead of per room)
- Do NOT use HasProperties or NominalValue.wrappedValue for Qto quantities — use Quantities
- Do NOT divide before checking for None or zero in the denominator
""",

    "conditional": """
CONDITIONAL RULE GUIDANCE:

First determine which scope pattern applies to the rule.

PATTERN A — BUILDING-LEVEL scope (one global condition for the entire building):
  Use when: the rule says "if the building has X, then all elements must..."
  Examples: "if building has more than 2 floors", "if building is a public space"

  - Evaluate the scope condition ONCE before the main element loop, not inside it.
  - For building-level property conditions: follow the IFCOPENSHELL ACCESS PATTERN from
    the RAG context, querying via IfcBuilding → IsDefinedBy → Pset → property.
  - If scope NOT met: report ALL primary elements as PASS with null fail_reason,
    print the JSON result, and exit with code 0 (not 1).
  - If scope IS met: apply the main constraint to all primary elements normally.

PATTERN B — ELEMENT-LEVEL scope (condition is a property of each individual element):
  Use when: the rule says "each element WITH property X must..."

  - Check the scope condition per element inside the loop, before applying the main constraint.
  - Apply unit conversion to the scope property value before comparing (use the same
    IfcUnitAssignment approach as for the main constraint unit conversion).
  - If the scope property is missing (None) or the element does not meet the scope condition:
    exclude it from the report entirely (skip with continue) — do NOT include it as PASS or FAIL.

COMMON MISTAKES TO AVOID:
- PATTERN A: do NOT omit elements when building-level scope is not met — report all as PASS
- PATTERN A: do NOT use sys.exit(1) for the scope-not-met case — use sys.exit(0)
- PATTERN B: do NOT include out-of-scope elements in the report — skip them with continue
- Do NOT mix patterns: building-level scope → global check before loop; element-level scope → per-element check inside loop
"""
}

# --- GENERAL DATA QUALITY INSTRUCTION ---
# --- DATA QUALITY HINT PARTS (assembled per category) ---

_DQ_PROPERTY_LOOKUP = """- PROPERTY LOOKUP ORDER (never skip a step before declaring a property missing):
  1. Direct IFC attribute on the element (hasattr / getattr)
  2. All Psets via IsDefinedBy → RelatingPropertyDefinition → HasProperties
  3. Element type via IsTypedBy → RelatingType → HasPropertySets → HasProperties"""

_DQ_MULTIPLE_VALUES = """- MULTIPLE VALUES from different sources: round all to 4 decimal places before comparing.
  If all rounded values are identical → treat as one value and proceed normally.
  If rounded values differ → FAIL with fail_reason explaining the conflict.
  Example: [799.999999999999, 800.0, 799.9999999999998] all round to 800.0 → single value."""

_DQ_UNIT_CONVERSION = """- UNIT VERIFICATION: always call get_length_unit_factor(model) before any numeric comparison.
  Do NOT assume the project uses metres."""


def _build_data_quality_hint(categories: list) -> str:
    """Assemble only the data quality rules relevant for the active categories."""
    cats = set(categories)
    parts = ["DATA QUALITY HANDLING:", _DQ_PROPERTY_LOOKUP]

    if cats & {"value", "information_availability", "mathematical"}:
        parts.append(_DQ_MULTIPLE_VALUES)

    if cats & {"value", "mathematical"}:
        parts.append(_DQ_UNIT_CONVERSION)

    return "\n".join(parts)

SYSTEM_PROMPT = '''You are an expert Python developer specialising in IFC-based building 
code compliance checking using the ifcopenshell library.

Your task is to generate a complete, executable Python script that checks whether all 
relevant elements in an IFC file comply with a given building regulation rule.

STRICT OUTPUT REQUIREMENTS:
- Output ONLY raw Python code — no markdown, no code fences, no explanation
- The script must be self-contained and executable without modification
- The script receives the IFC file path as a command-line argument: sys.argv[1]
- The script must produce a single JSON object printed to stdout
- Do not print anything else to stdout — only the final JSON object

REQUIRED JSON OUTPUT STRUCTURE:
{
  "rule": "<the building rule text>",
  "ifc_file": "<filename only, not full path>",
  "timestamp": "<ISO 8601 timestamp>",
  "summary": {
    "total_elements": <int>,
    "compliant": <int>,
    "non_compliant": <int>
  },
  "elements": [
    {
      "global_id": "<IfcGloballyUniqueId>",
      "name": "<element name or Unnamed>",
      "status": "<PASS or FAIL>",
      "value": <measured value as float or null if missing>,
      "threshold": <threshold value as float or null if not applicable>,
      "unit": "<unit string e.g. m mm m2 or null>",
      "fail_reason": "<precise explanation of why element fails or null if PASS>"
    }
  ]
}

RULES FOR ELEMENT FIELDS:
- status must be exactly "PASS" or "FAIL" — never "UNKNOWN", "UNVERIFIABLE", "Cannot verify",
  "N/A", or any other value. If a property cannot be found or verified, status MUST be "FAIL".
- fail_reason must be null for PASS elements
- fail_reason must be a non-null string for FAIL elements
- value must be null if the property could not be determined
- total_elements must equal compliant plus non_compliant

{data_quality_hint}

{category_hint}'''

USER_PROMPT = """Generate a Python compliance checking script for the following:

BUILDING RULE: {rule_text}

RULE CATEGORY: {rule_category}

STRUCTURED RULE ANALYSIS:
- Primary element: {primary_element}
- Related elements: {related_elements}
- Properties to check: {properties}
- Constraints: {constraints}
- Scope conditions: {scope_conditions}

IFC DOMAIN KNOWLEDGE (from RAG):
{rag_context}

Remember: output ONLY the raw Python code, nothing else."""

CORRECTION_PROMPT = """The Python compliance checking script you generated previously
contains a syntax error. Your task is to fix the error and return a corrected script.

BUILDING RULE: {rule_text}
RULE CATEGORY: {rule_category}

STRUCTURED RULE ANALYSIS:
- Primary element: {primary_element}
- Related elements: {related_elements}
- Properties to check: {properties}
- Constraints: {constraints}
- Scope conditions: {scope_conditions}

IFC DOMAIN KNOWLEDGE (from RAG):
{rag_context}

PREVIOUS FAILED ATTEMPTS — do NOT repeat these approaches:
{correction_history}

PREVIOUS CODE WITH ERROR:
{previous_code}

ERROR MESSAGE:
{error_message}

Fix the error and return the complete corrected Python script.
Remember: output ONLY the raw Python code, nothing else."""

SEMANTIC_CORRECTION_PROMPT = """The compliance checking script you generated has been
evaluated by a semantic judge and found to be incorrect. Your task is to fix the
script based on the specific feedback provided.

BUILDING RULE: {rule_text}
RULE CATEGORY: {rule_category}

STRUCTURED RULE ANALYSIS:
- Primary element: {primary_element}
- Related elements: {related_elements}
- Properties to check: {properties}
- Constraints: {constraints}
- Scope conditions: {scope_conditions}

IFC DOMAIN KNOWLEDGE (from RAG):
{rag_context}

PREVIOUS FAILED ATTEMPTS — do NOT repeat these approaches:
{correction_history}

PREVIOUS CODE:
{previous_code}

SEMANTIC JUDGE FEEDBACK:
{semantic_feedback}

Fix all identified issues and return the complete corrected Python script.
Remember: output ONLY the raw Python code, nothing else."""


def _clean_code(raw: str) -> str:
    """Strip markdown code fences if the model includes them despite instructions."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines[-1].strip() == "```" else lines
        raw = "\n".join(lines)
    return raw.strip()


def _build_prompt(system_content: str, human_template: str) -> ChatPromptTemplate:
    """
    Build a ChatPromptTemplate using SystemMessage to prevent LangChain
    from parsing the system prompt for template variables.
    """
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        HumanMessagePromptTemplate.from_template(human_template)
    ])


def run_agent_3(state: PipelineState) -> PipelineState:
    """
    Agent 3: Code Generator
    First run:   generates code from rule + RAG context + category hint
    Subsequent:  corrects code based on syntax or semantic feedback
    """
    is_syntax_correction = state.syntax_error is not None
    is_semantic_correction = state.semantic_feedback is not None

    if is_semantic_correction:
        print(f"\n[Agent 3] Correcting code based on Agent 5 semantic feedback "
              f"(semantic iteration {state.semantic_iterations}/{MAX_SEMANTIC_ITERATIONS})...")
    elif is_syntax_correction:
        print(f"\n[Agent 3] Correcting code based on Agent 4 syntax feedback "
              f"(syntax iteration {state.syntax_iterations}/{MAX_SYNTAX_ITERATIONS})...")
    else:
        print("\n[Agent 3] Generating compliance checking code...")

    total_iterations = state.syntax_iterations + state.semantic_iterations
    temperature = 0.1

    llm = ChatOpenAI(
        model_name=MODEL_GENERATOR,
        openai_api_key=OPENROUTER_KEY,
        openai_api_base=OPENROUTER_BASE,
        temperature=temperature
    )

    parsed = state.parsed_rule
    rule_categories = parsed.rule_category if parsed.rule_category else ["value"]
    if isinstance(rule_categories, str):
        rule_categories = [rule_categories]

    if len(rule_categories) == 1:
        category_hint = CATEGORY_HINTS.get(rule_categories[0], CATEGORY_HINTS["value"])
    else:
        ordered_hints = [CATEGORY_HINTS.get(cat, "") for cat in rule_categories]
        category_hint = (
            f"This rule spans multiple categories: {', '.join(rule_categories)}.\n"
            f"Structure your code in exactly this layer order:\n"
            f"  1. {rule_categories[0].upper()} — outermost layer (first check in code)\n"
            + "".join(
                f"  {i+2}. {cat.upper()} — layer {i+2}\n"
                for i, cat in enumerate(rule_categories[1:])
            )
            + "\nGuidance per layer:\n\n"
            + "\n".join(ordered_hints)
        )

    # Build system prompt with category and data quality hints
    system_prompt_filled = SYSTEM_PROMPT.replace(
        "{data_quality_hint}", _build_data_quality_hint(rule_categories)
    ).replace(
        "{category_hint}", category_hint
    )

    rule_category_str = ", ".join(rule_categories)
    common_vars = {
        "rule_text": state.building_rule_text,
        "rule_category": rule_category_str,
        "primary_element": parsed.primary_element,
        "related_elements": ", ".join(parsed.related_elements) or "none",
        "properties": ", ".join(parsed.properties_to_check) or "none",
        "constraints": ", ".join(parsed.constraints),
        "scope_conditions": ", ".join(parsed.scope_conditions) or "none"
    }

    # --- Build correction history ---
    # Record the current failure so future correction runs know what was already tried
    updated_history = list(state.correction_history)
    if is_syntax_correction:
        updated_history.append(
            f"Syntax attempt {state.syntax_iterations}: {state.syntax_error}"
        )
    elif is_semantic_correction:
        # Truncate long feedback to keep token usage manageable
        feedback_summary = (state.semantic_feedback or "")[:300]
        updated_history.append(
            f"Semantic attempt {state.semantic_iterations}: {feedback_summary}"
        )

    formatted_history = (
        "\n".join(f"  {i+1}. {entry}" for i, entry in enumerate(updated_history))
        if updated_history else "None — this is the first correction attempt."
    )

    try:
        if is_semantic_correction:
            template = SEMANTIC_CORRECTION_PROMPT
            prompt = _build_prompt(system_prompt_filled, template)
            chain = prompt | llm | StrOutputParser()
            invoke_vars = {
                **common_vars,
                "previous_code": state.generated_code,
                "semantic_feedback": state.semantic_feedback,
                "rag_context": state.rag_context or "Not available",
                "correction_history": formatted_history,
            }

        elif is_syntax_correction:
            template = CORRECTION_PROMPT
            prompt = _build_prompt(system_prompt_filled, template)
            chain = prompt | llm | StrOutputParser()
            invoke_vars = {
                **common_vars,
                "previous_code": state.generated_code,
                "error_message": state.syntax_error,
                "rag_context": state.rag_context or "Not available",
                "correction_history": formatted_history,
            }

        else:
            prompt = _build_prompt(system_prompt_filled, USER_PROMPT)
            chain = prompt | llm | StrOutputParser()
            invoke_vars = {**common_vars, "rag_context": state.rag_context}

        raw_code = chain.invoke(invoke_vars)
        clean_code = _clean_code(raw_code)

        print("[Agent 3] Code generated successfully.")
        print(f"[Agent 3] Code length: {len(clean_code.splitlines())} lines")

        is_first_generation = not is_syntax_correction and not is_semantic_correction
        return state.model_copy(update={
            "generated_code": clean_code,
            "initial_code": clean_code if is_first_generation else state.initial_code,
            "syntax_error": None,
            "syntax_iterations": state.syntax_iterations if is_syntax_correction else 0,
            "semantic_feedback": None,
            "correction_history": updated_history,
            "current_agent": "agent_4"
        })

    except Exception as e:
        print(f"[Agent 3] ERROR: {e}")
        return state.model_copy(update={
            "error_log": state.error_log + [f"Agent 3 failed: {str(e)}"],
            "pipeline_failed": True,
            "current_agent": "error"
        })