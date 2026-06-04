# agentic_workflow/agents/agent1_rule_analyser.py
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agentic_workflow.state import PipelineState, BuildingRule
from workflow_config import OPENROUTER_KEY, OPENROUTER_BASE, MODEL_GENERATOR

SYSTEM_PROMPT = """You are a building regulation analyst specialising in IFC-based 
compliance checking. Your task is to parse a building regulation rule and extract 
its key components for downstream IFC-based compliance checking.

You must identify:
1. The PRIMARY building element that needs to be checked (in natural language)
2. Any RELATED elements connected to the primary element
3. The PROPERTIES that need to be verified
4. The CONSTRAINTS expressed as precise logical conditions
5. Any SCOPE CONDITIONS that limit which elements are in scope
6. Two sets of SEARCH QUERIES for RAG retrieval, in English:
   - bSDD queries: property-focused (element definitions, property names, measurements, classifications)
   - EXPRESS queries: schema-focused (element types, hierarchy, relationships, connections between elements)
7. The RULE CATEGORY — classify the rule as an ordered list of applicable categories.
   A rule can span multiple categories. Analyse it layer by layer.

AVAILABLE CATEGORIES:
- conditional: the rule applies only under a specific scope condition
  (building type, number of floors, occupancy class, etc.)
- relational: the check requires navigating FROM one element type TO another
- mathematical: the check requires a calculation combining values from multiple elements
- value: the check compares a single numeric property against a fixed threshold
- information_availability: the check verifies whether a property exists and is specified

CLASSIFICATION PROCEDURE:
Analyse the rule layer by layer and reason step by step before giving the final list.

STEP 1 — SCOPE LAYER:
Does the rule apply only under a specific condition at building or floor level?
→ YES: add "conditional" as the first entry
→ NO: continue to step 2

STEP 2 — STRUCTURAL LAYER:
Does applying the main check require navigating FROM one element type TO another?
→ YES: add "relational"
→ NO: continue to step 3

STEP 3 — MEASUREMENT LAYER:
What kind of check is performed on the target element(s)?
→ Check whether a property EXISTS (not its value): add "information_availability"
→ Check a NUMERIC property against a fixed threshold: add "value"
→ Count the number of directly related elements and compare to a threshold: add "value"
→ Check a CALCULATED value combining quantities across multiple spatial units: add "mathematical"

The result is an ordered list reflecting the code structure from outermost to innermost.

Critical rules:
- Use natural language for elements and properties — do NOT use IFC class names
- Express constraints precisely: 'height >= 2.1m' not 'must be tall enough'
- Generate a separate search query for each distinct concept in the rule
- Always respond in English regardless of the input language
- Always express numeric constraints in SI base units — convert before writing:
  * lengths  : mm → m  (÷1000),  cm → m  (÷100),  km → m  (×1000)
  * areas    : mm² → m² (÷1e6),  cm² → m² (÷1e4)
  * volumes  : mm³ → m³ (÷1e9), cm³ → m³ (÷1e6),  dm³ → m³ (÷1000)
  * examples : '2100mm' → '2.1m',  '15cm' → '0.15m',  '5000cm²' → '0.5m²'
  * ratios and percentages: leave as-is (dimensionless)

SEARCH QUERY GENERATION RULES:
Generate two separate lists of queries — one per database:

bSDD QUERIES (property-focused):
- Focus on: element definitions, property names, measurements, classifications
- One query per distinct element or property in the rule

EXPRESS QUERIES (schema-focused):
- Focus on: element type structure, hierarchy, connections between elements
- One query per distinct structural aspect in the rule
- For RELATIONAL rules: always include at least one query about how the two element
  types are connected, using this pattern:
  "how is [element A] connected to [element B] relationship"
- For MATHEMATICAL rules: always include a query about spatial containment:
  "how to get [element] per [spatial unit] spatial containment"

{format_instructions}"""

USER_PROMPT = """Parse the following building regulation rule and extract its components.

RULE: {rule_text}

Return a JSON object with the extracted components."""


MEASUREMENT_CATEGORIES = {"value", "mathematical", "information_availability"}

# --- UNIT NORMALISATION ---

_UNIT_CONVERSIONS = {
    # Length → metres
    "mm": ("m",  1e-3),
    "cm": ("m",  1e-2),
    "dm": ("m",  1e-1),
    "km": ("m",  1e3),
    # Area → square metres (both ascii and unicode superscripts)
    "mm2": ("m2", 1e-6),  "mm²": ("m2", 1e-6),
    "cm2": ("m2", 1e-4),  "cm²": ("m2", 1e-4),
    "dm2": ("m2", 1e-2),  "dm²": ("m2", 1e-2),
    # Volume → cubic metres
    "mm3": ("m3", 1e-9),  "mm³": ("m3", 1e-9),
    "cm3": ("m3", 1e-6),  "cm³": ("m3", 1e-6),
    "dm3": ("m3", 1e-3),  "dm³": ("m3", 1e-3),
    "litre":  ("m3", 1e-3),
    "litres": ("m3", 1e-3),
}

# Match a number followed by a unit — longer units listed first to avoid partial matches
_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(mm²|cm²|dm²|mm2|cm2|dm2|mm³|cm³|dm³|mm3|cm3|dm3|litres|litre|km|mm|cm|dm)",
    re.IGNORECASE,
)


def _normalise_constraints(parsed: BuildingRule) -> tuple:
    """
    Convert non-SI units in constraint strings to SI base units.
    Returns (corrected BuildingRule, list of normalisation messages).
    """
    normalisations = []
    new_constraints = []

    for constraint in parsed.constraints:
        def _replace(match):
            raw_value = match.group(1).replace(",", ".")
            unit_key  = match.group(2).lower()
            if unit_key not in _UNIT_CONVERSIONS:
                return match.group(0)
            si_unit, factor = _UNIT_CONVERSIONS[unit_key]
            si_value = float(raw_value) * factor
            # Format: drop unnecessary decimals
            formatted = f"{si_value:.6g}"
            return f"{formatted}{si_unit}"

        new_constraint = _UNIT_RE.sub(_replace, constraint)
        if new_constraint != constraint:
            normalisations.append(f"'{constraint}' → '{new_constraint}'")
        new_constraints.append(new_constraint)

    if normalisations:
        parsed = parsed.model_copy(update={"constraints": new_constraints})

    return parsed, normalisations


def _validate_and_correct_categories(parsed: BuildingRule) -> tuple:
    """
    Cross-validate the LLM-assigned categories against the extracted fields.
    Auto-corrects missing categories and wrong ordering.
    Returns (corrected BuildingRule, list of correction messages).
    """
    categories = list(parsed.rule_category)
    corrections = []

    has_scope    = len(parsed.scope_conditions) > 0
    has_related  = len(parsed.related_elements) > 0
    has_numeric  = any(
        any(op in c for op in [">=", "<=", ">", "<", "=="])
        for c in parsed.constraints
    )
    has_measurement = bool(set(categories) & MEASUREMENT_CATEGORIES)

    # --- Missing category detection ---

    # scope_conditions extracted but "conditional" absent
    if has_scope and "conditional" not in categories:
        categories.insert(0, "conditional")
        corrections.append(
            "Added 'conditional': scope_conditions were extracted but category was missing"
        )

    # related_elements extracted but "relational" absent
    if has_related and "relational" not in categories:
        # Insert after "conditional" (if present), before first measurement category
        insert_pos = 1 if "conditional" in categories else 0
        for i, cat in enumerate(categories):
            if cat in MEASUREMENT_CATEGORIES:
                insert_pos = i
                break
        categories.insert(insert_pos, "relational")
        corrections.append(
            "Added 'relational': related_elements were extracted but category was missing"
        )

    # No measurement category at all
    if not bool(set(categories) & MEASUREMENT_CATEGORIES):
        if has_numeric:
            categories.append("value")
            corrections.append(
                "Added 'value': numeric constraints detected but no measurement category assigned"
            )
        else:
            categories.append("value")
            corrections.append(
                "Added 'value' as fallback: no measurement category assigned"
            )

    # --- Ordering correction ---

    # "conditional" must always be first
    if "conditional" in categories and categories[0] != "conditional":
        categories.remove("conditional")
        categories.insert(0, "conditional")
        corrections.append(
            "Moved 'conditional' to first position: scope layer must be outermost"
        )

    # "relational" must come before any measurement category
    if "relational" in categories:
        rel_idx = categories.index("relational")
        for mcat in ["value", "mathematical", "information_availability"]:
            if mcat in categories:
                mcat_idx = categories.index(mcat)
                if rel_idx > mcat_idx:
                    categories.remove("relational")
                    categories.insert(mcat_idx, "relational")
                    corrections.append(
                        f"Moved 'relational' before '{mcat}': "
                        f"structural layer must precede measurement layer"
                    )
                break

    if corrections:
        parsed = parsed.model_copy(update={"rule_category": categories})

    return parsed, corrections


def run_agent_1(state: PipelineState) -> PipelineState:
    """
    Agent 1: Rule Analyser
    Input:  state.building_rule_text
    Output: state.parsed_rule (BuildingRule object including rule_category)
    """
    print("\n[Agent 1] Analysing building rule...")

    parser = JsonOutputParser(pydantic_object=BuildingRule)

    llm = ChatOpenAI(
        model_name=MODEL_GENERATOR,
        openai_api_key=OPENROUTER_KEY,
        openai_api_base=OPENROUTER_BASE,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT)
    ])

    chain = (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | parser
    )

    try:
        result = chain.invoke({"rule_text": state.building_rule_text})
        parsed = BuildingRule(**result)

        print(f"[Agent 1] Primary element:  {parsed.primary_element}")
        print(f"[Agent 1] Related elements: {parsed.related_elements}")
        print(f"[Agent 1] Properties:       {parsed.properties_to_check}")
        print(f"[Agent 1] Constraints:      {parsed.constraints}")
        print(f"[Agent 1] Scope conditions: {parsed.scope_conditions}")
        print(f"[Agent 1] bSDD queries:     {parsed.search_queries_bsdd}")
        print(f"[Agent 1] EXPRESS queries:  {parsed.search_queries_express}")
        print(f"[Agent 1] Rule category:    {parsed.rule_category}")

        # --- Programmatic cross-validation ---
        parsed, corrections = _validate_and_correct_categories(parsed)
        if corrections:
            print(f"[Agent 1] Category corrections applied:")
            for c in corrections:
                print(f"  → {c}")
            print(f"[Agent 1] Corrected category: {parsed.rule_category}")

        # --- Unit normalisation ---
        parsed, normalisations = _normalise_constraints(parsed)
        if normalisations:
            print(f"[Agent 1] Unit normalisations applied:")
            for n in normalisations:
                print(f"  → {n}")

        return state.model_copy(update={
            "parsed_rule": parsed,
            "current_agent": "agent_2"
        })

    except Exception as e:
        print(f"[Agent 1] ERROR: {e}")
        return state.model_copy(update={
            "error_log": state.error_log + [f"Agent 1 failed: {str(e)}"],
            "pipeline_failed": True,
            "current_agent": "error"
        })