# agentic_workflow/agents/agent5_semantic_judge.py
import subprocess
import sys
import json
import tempfile
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from agentic_workflow.state import PipelineState
from workflow_config import (
    OPENROUTER_KEY, OPENROUTER_BASE, MODEL_JUDGE,
    MAX_SEMANTIC_ITERATIONS, JUDGE_RUNS, JUDGE_TEMPERATURE
)


# --- OUTPUT MODEL ---
class CriterionResult(BaseModel):
    criterion: str = Field(description="Name of the criterion")
    quoted_code: str = Field(
        description="The specific code lines most relevant to this criterion, quoted verbatim from the script. Quote the exact lines, not a paraphrase."
    )
    passed: bool = Field(description="Whether the criterion passed")
    reasoning: str = Field(description="Brief explanation of the judgement based on the quoted code")


class SemanticJudgement(BaseModel):
    overall_pass: bool = Field(
        description="True if ALL three criteria pass, False if any single criterion fails"
    )
    criteria: List[CriterionResult] = Field(
        description="Exactly three criterion results — one per criterion"
    )
    feedback: str = Field(
        description="Specific actionable feedback for code correction if overall_pass is False. Empty string if overall_pass is True."
    )


# --- PROMPTS ---
JUDGE_SYSTEM_PROMPT = """You are an expert IFC compliance checking evaluator. Your task
is to assess whether a Python compliance checking script correctly implements a given
building regulation rule by performing static code analysis.

IMPORTANT INSTRUCTIONS:
- Your DEFAULT ASSUMPTION is that the code contains errors. Actively look for violations
  of each criterion. Only mark a criterion PASS if you find EXPLICIT evidence in the code
  that it is correctly implemented. Do NOT assume correctness because nothing looks obviously wrong.
- Evaluate the CODE ONLY — do not speculate about runtime behaviour or actual results
- Base your evaluation ONLY on what is explicitly present in the provided code
- Do NOT invent requirements that are not listed in the criteria below
- Do NOT fail a criterion because of code style, efficiency, or fields not listed below

IFC DOMAIN KNOWLEDGE USAGE:
IFC domain knowledge retrieved from bSDD and the IFC EXPRESS schema is provided in the
user message. Use it as the authoritative reference for verification:
- RELEVANT IFC CLASSES section → use to verify the IFC class in Criterion 1
- RELEVANT PSETS AND PROPERTIES section → use to verify Pset names and property names in Criterion 2
- RELEVANT RELATIONSHIPS section → use to verify relationship entities in Criterion 2
- IFCOPENSHELL ACCESS PATTERN section → use to verify the property access approach in Criterion 2
If domain knowledge is marked as "Not available", rely on your general IFC schema knowledge.

You must evaluate the script against exactly THREE criteria. For each criterion:
1. FIRST quote the specific lines of code most relevant to that criterion (verbatim)
2. THEN evaluate those quoted lines against the criterion requirements
3. THEN give your pass/fail judgement and reasoning

The overall result is PASS only if ALL three criteria pass.

CRITERION 1 — CORRECT ELEMENT TYPE
Quote: the model.by_type() call(s) from the code.
Does the script query the correct IFC element type for the building element
described in the rule?
- Check model.by_type() or equivalent calls in the code
- The IFC class must be at the correct hierarchy level — not too generic (e.g. IfcProduct)
  and not the type variant (e.g. IfcDoorType instead of IfcDoor)
- If domain knowledge is available: verify the IFC class against the RELEVANT IFC CLASSES section
- PASS if: the queried IFC class matches the building element in the rule (and the domain knowledge)
- FAIL if: wrong IFC class is queried, too generic a class is used, or no elements are queried

CRITERION 2 — CORRECT CONSTRAINT LOGIC
Quote: the unit conversion code, the property lookup chain, the comparison operator and
threshold, and any relationship navigation or scope condition check — whichever apply
to the rule category. Quote the actual function bodies, not just the call sites.
Is the logical constraint from the rule correctly implemented in the code?
The rule category is an ordered list of layers (e.g. ["conditional", "relational", "value"]).
Apply the check for EVERY category in the list — all must pass for this criterion to pass.
Also verify that the code implements the layers in the correct order (outermost first).
Where domain knowledge is available, use it to verify names and patterns against the RAG sections.

For VALUE rules, check ALL of the following:
- OPERATOR: is the exact operator from the rule used? (>= not >, <= not <)
- THRESHOLD: does the numeric threshold in the code match the value stated in the rule exactly?
- PROPERTY LOOKUP: does the code attempt to retrieve the property value?
  PASS if: the code follows the IFCOPENSHELL ACCESS PATTERN from the RAG domain knowledge.
  If RAG marks the access pattern as NOT FOUND IN CHUNKS, PASS if the code accesses at least
  one of — direct IFC attribute, Pset via IsDefinedBy, or element type via IsTypedBy.
  Additional fallbacks beyond what RAG specifies are always acceptable.
  Do NOT fail because the code checks more sources than RAG or the minimum required.
- PSET AND PROPERTY NAMES: only apply this check if the domain knowledge explicitly names
  a specific Pset that MUST be used. If so, verify that Pset is present in the code.
  Do NOT fail because the code uses additional Psets not mentioned in domain knowledge.
  Do NOT fail because the code includes a Pset fallback that domain knowledge does not mention.
  Do NOT fail because the code omits a Pset that domain knowledge marks as uncertain or unverified.
- MISSING VALUE HANDLING: if the property is None or absent from all sources,
  is the element explicitly marked as FAIL with a descriptive fail_reason?
  A None value must never silently become a PASS.
- FAIL this criterion if: wrong threshold, wrong operator,
  or missing/None properties are silently skipped instead of marked as FAIL

For INFORMATION AVAILABILITY rules, check ONLY the following three things:
- EXISTENCE CHECK: does the code check for the PRESENCE of the property, not its value?
  PASS if: the code checks whether the retrieved value is None, empty, or absent.
  FAIL if: the code applies a numeric comparison (>=, <=, >, <) to the value.
- MISSING VALUE HANDLING: if no value is found, is the element explicitly marked as FAIL?
  PASS if: absent/None/empty → status = "FAIL" with a descriptive fail_reason.
  FAIL if: absent/None is silently treated as PASS or silently skipped.
- CONFLICT HANDLING: if multiple sources return non-null values that differ, is the element
  marked as FAIL? Rounding to 4 decimal places before comparing is the correct approach.
  PASS if: the code collects values, deduplicates them, and marks FAIL when multiple
  distinct values are found.
  FAIL if: multiple conflicting non-null values are silently ignored.

ABSOLUTE RESTRICTIONS for information availability evaluation:
- DO NOT evaluate which specific sources (direct attribute, Psets, type) the code uses.
- DO NOT evaluate which property names are in the search set (TARGET_PROP_NAMES or equivalent).
- DO NOT use domain knowledge to restrict or prescribe property names.
- Searching for additional property name variants such as Width, DoorWidth, NominalWidth
  alongside the primary name is ALWAYS acceptable — more names means more thorough coverage.
- NEVER tell Agent 3 to remove property names from the search set.
- The only valid failure reasons are: numeric comparison applied, None not treated as FAIL,
  or conflicting values silently ignored. Nothing else.

For RELATIONAL rules, check ALL of the following:
- PRIMARY FIRST: does the code query the primary element first?
- RELATIONSHIP NAVIGATION: does the code navigate to related elements via an explicit
  IFC relationship?
- RELATIONSHIP ENTITY: if domain knowledge is available, verify the relationship entity
  used in the code against the RELEVANT RELATIONSHIPS section.
  If the RELEVANT RELATIONSHIPS section says NOT FOUND IN CHUNKS, or the grounding warning
  flags the relationship entity as unverified: accept any standard IFC relationship
  (IfcRelAggregates, IfcRelContainedInSpatialStructure) — do NOT fail because the RAG
  could not confirm the entity name from retrieved chunks.
- PRESENCE: does the code check that related elements exist?
- CORRECT GLOBALID: is the GlobalId reported in the output the GlobalId of the PRIMARY element?
- MISSING RELATED ELEMENTS: are missing related elements explicitly reported as FAIL?
- FAIL this criterion if: relationships are not navigated, wrong relationship entity used
  (when domain knowledge is available), the rule requires position checking but no positional
  logic is present, or related elements are counted independently instead of relative to the primary

For MATHEMATICAL rules, first determine the aggregation type from the rule, then apply
the matching checks.

ELEMENT-LEVEL formula: the formula combines properties of each individual primary element
  No spatial grouping is needed — the formula is computed once per element.
  Check ALL of the following:
  - COMPLETE FORMULA: does the formula use ALL required properties of the element before comparing?
  - PROPERTY ACCESS: verify property access against the IFCOPENSHELL ACCESS PATTERN from RAG;
    if not available, at least one access path (direct attribute, Pset, or type) must be present.
  - MISSING VALUE HANDLING: if any required property is None, is the element marked as FAIL?
  - total_elements = number of primary elements checked (not spatial units)
  - FAIL if: formula is incomplete, required property is inaccessible, or None silently ignored

SPATIAL-LEVEL formula: the formula aggregates quantities across multiple elements within a
  Check ALL of the following:
  - AGGREGATION LEVEL: is the calculation grouped per spatial unit (per room, per floor)?
  - COMPLETE FORMULA: does the formula use ALL required quantities before comparing?
  - QTO ACCESS PATTERN: if quantities come from Qto sets, verify the code uses `Quantities`
    (not `HasProperties`) and typed value attributes (AreaValue, LengthValue, VolumeValue,
    or CountValue — not NominalValue.wrappedValue).
  - PSET AND PROPERTY NAMES: if domain knowledge is available, verify names against RAG
  - MISSING VALUE HANDLING: if any input is None, is the spatial unit marked as FAIL?
  - total_elements = number of spatial units (rooms, floors) checked
  - FAIL if: wrong aggregation level, incomplete formula, wrong Qto access pattern,
    wrong name (when domain knowledge is available), or None inputs silently ignored

For CONDITIONAL rules, check ALL of the following:
- SCOPE TYPE: first determine whether the scope is building-level (one global condition,
  e.g. "if building has more than 2 floors") or element-level (per-element condition,
  e.g. "each space WITH net floor area > 4 m²"). Apply the remaining checks accordingly.
- SCOPE FIRST:
  Building-level scope: the global condition is evaluated BEFORE the main element loop.
  Element-level scope: the per-element condition is checked inside the loop, before the
  main constraint is applied to that element. Both are correct — do not fail either pattern.
- SCOPE NOT MET:
  Building-level scope not met: ALL primary elements must be reported as PASS with null
  fail_reason. FAIL if elements are silently omitted instead.
  Element-level scope not met for an element: that element is excluded from the report
  entirely (skipped with continue). Do NOT require out-of-scope elements to appear as PASS —
  excluding them is the correct behaviour for element-level scope.
- SCOPE MET: if the scope condition is met, is the main constraint applied normally?
- PROPERTY LOOKUP: if the scope condition involves a property, verify the access
  pattern against the IFCOPENSHELL ACCESS PATTERN section when domain knowledge is available
- FAIL this criterion if: scope condition evaluated after main check, or building-level
  scope-not-met elements are silently omitted instead of reported as PASS

CRITERION 3 — CORRECT COVERAGE
Quote: the main iteration loop and the summary assembly code (where total_elements,
compliant, and non_compliant are counted and built).
Does the script correctly iterate over and count all relevant elements?
- Check that the code iterates over ALL primary elements without premature filtering
- Check that elements with None/missing MAIN CONSTRAINT properties are included as FAIL entries
  (not silently skipped, which would undercount total_elements).
  Exception: for element-level CONDITIONAL rules, elements with a missing SCOPE property
  are excluded from the report entirely — this is correct behaviour, not a coverage error.
- Check that total_elements, compliant, and non_compliant are assembled correctly
- For RELATIONAL rules: total_elements must count PRIMARY elements only
- For MATHEMATICAL rules: total_elements must count spatial units (rooms, floors)
- For CONDITIONAL rules:
  Building-level scope: total_elements counts ALL primary elements (including those
  reported as PASS when the global scope condition is not met).
  Element-level scope: total_elements counts ONLY elements that meet the scope condition
  (out-of-scope elements are excluded entirely and must not be counted).
- PASS if: all in-scope elements are iterated, None main-constraint values produce FAIL entries,
  counts are correct
- FAIL if: in-scope elements are silently skipped, or summary counts are assembled incorrectly


STRICT RULES:
- You may only evaluate these three criteria — nothing else
- Do not add criteria of your own
- Do not comment on runtime results, output values, or actual compliance rates
- Your feedback must name the specific issue and what exactly needs to change,
  referencing the domain knowledge where relevant (e.g. "code uses Pset_WallCommon
  but domain knowledge specifies Pset_DoorCommon")
- A helper function that internally queries IfcUnitAssignment (e.g. get_length_unit_factor)
  is fully equivalent to a direct inline query — do NOT fail criterion 2 because unit
  conversion is encapsulated in a function rather than inlined
- Following the IFCOPENSHELL ACCESS PATTERN from the RAG domain knowledge is the expected
  primary approach for property lookup. Do NOT fail because the code uses the RAG-specified
  access pattern instead of a generic multi-source fallback chain.
- Checking additional property sources (Psets via IsDefinedBy, element type via IsTypedBy)
  beyond the minimum required is ACCEPTABLE and should not cause a failure — more thorough
  property lookup is better than less thorough
- Do NOT oscillate on Pset names: if your previous feedback told the code to add or remove
  a specific Pset and the code followed that instruction, do NOT then fail for the opposite reason
- Do NOT fail criterion 2 solely because of a Pset fallback unless the domain knowledge
  explicitly and unambiguously requires that specific Pset AND the code completely omits it
- A grounding warning on a relationship entity name does NOT mean the code should avoid
  using it. The grounding warning means the RAG could not confirm the name from retrieved
  chunks — not that the relationship is incorrect. Do NOT fail criterion 2 because the code
  uses a relationship entity (e.g. IfcRelAggregates) that appears in the grounding warning.
- Do NOT apply conflict handling requirements (collecting from multiple sources, deduplication,
  checking for conflicting non-null values) to VALUE or RELATIONAL rules. Conflict handling
  is exclusively a criterion for INFORMATION AVAILABILITY rules.
- For VALUE rules: only verify the operator used in the PASS condition comparison. Do NOT
  fail because the FAIL branch or fail_reason string uses the logical inverse operator
  (e.g. code uses '>= threshold' for PASS and '< threshold' in the FAIL branch — both are correct).
- For RELATIONAL rules with a VALUE threshold on the count of related elements: a count of
  0 (no related elements found) that falls below the threshold is correctly handled by the
  threshold check. Do NOT require a separate explicit empty/null check for missing elements.
- Do NOT invent statuses — the only valid element statuses are PASS and FAIL. Do NOT fail
  criterion 2 or 3 because the code uses "UNVERIFIABLE", "UNKNOWN", or similar — these are
  invalid statuses and if present in the code, Agent 3 should be told to replace them with FAIL
- The text content of fail_reason does NOT determine criterion 3. An element with status=FAIL
  and any descriptive fail_reason string is correctly handled. Do NOT fail criterion 3 because
  fail_reason says "CANNOT VERIFY", "missing", or similar wording — only the STATUS field matters
- CROSS-CRITERIA CONSISTENCY: if criterion 2 explicitly passes missing value handling
  (i.e. you stated that missing/None properties are marked as FAIL), then criterion 3 MUST NOT
  fail on the basis that those same missing values are not included in total_elements. A FAIL
  entry is an entry — it is counted. These two failures are mutually exclusive.
- Do NOT fail criterion 3 because of HOW total_elements is computed — both
  len(elements_list) and compliant + non_compliant are mathematically equivalent and
  both are acceptable. Only fail criterion 3 if elements are actually skipped or filtered
  out before being added to elements_list
- The overall_pass field must be True if and only if ALL three criteria have passed: true.
  Never set overall_pass to false when all criteria passed, and never set it to true when
  any criterion failed

{format_instructions}"""

JUDGE_USER_PROMPT = """Evaluate the following compliance checking script.

BUILDING RULE: {rule_text}

RULE CATEGORY: {rule_category}

IFC DOMAIN KNOWLEDGE (from RAG retrieval):
{rag_context}

GENERATED CODE:
{generated_code}

Assess the script against all THREE criteria — apply the category-specific guidance
for Criterion 2 based on the rule category above, using the domain knowledge to
verify IFC class names, Pset names, property names, and relationship entities.
Return your judgement as JSON."""


# --- HELPER FUNCTIONS ---
def _execute_code(code: str, ifc_file_path: str) -> tuple:
    """
    Execute the generated compliance script on the IFC file.
    Returns (stdout, stderr, success).
    """
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False,
        encoding='utf-8'
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path, ifc_file_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode == 0
    except subprocess.TimeoutExpired:
        return "", "TimeoutError: script execution exceeded 60 seconds", False
    finally:
        os.unlink(tmp_path)



def _validate_schema(data: dict) -> tuple:
    """
    Programmatically validate the JSON output schema.
    Returns (passed, error_message).
    """
    required_top = ["rule", "ifc_file", "timestamp", "summary", "elements"]
    for field in required_top:
        if field not in data:
            return False, f"Missing required top-level field: '{field}'"

    required_summary = ["total_elements", "compliant", "non_compliant"]
    for field in required_summary:
        if field not in data["summary"]:
            return False, f"Missing required summary field: '{field}'"

    required_element = [
        "global_id", "name", "status", "value",
        "threshold", "unit", "fail_reason"
    ]
    for i, element in enumerate(data["elements"]):
        for field in required_element:
            if field not in element:
                return False, f"Element {i} missing required field: '{field}'"

        if element["status"] not in ["PASS", "FAIL"]:
            return False, f"Element {i} has invalid status: '{element['status']}'"

        if element["status"] == "FAIL" and element["fail_reason"] is None:
            return False, f"Element {i} is FAIL but fail_reason is null"

        if element["status"] == "PASS" and element["fail_reason"] is not None:
            return False, f"Element {i} is PASS but fail_reason is not null"

    total = data["summary"]["total_elements"]
    compliant = data["summary"]["compliant"]
    non_compliant = data["summary"]["non_compliant"]
    if total != compliant + non_compliant:
        return False, (
            f"total_elements ({total}) != compliant ({compliant}) "
            f"+ non_compliant ({non_compliant})"
        )

    return True, None


def _run_judge_with_consensus(chain, invoke_vars: dict, n_runs: int) -> tuple:
    """
    Adaptive self-consistency judge.
    - Always runs the first call.
    - If ALL criteria fail on the first call: clearly wrong — skip remaining runs (save cost).
    - Otherwise: run remaining calls for strict consensus (all must pass).
    Returns (overall_pass, aggregated_criteria list, aggregated_feedback str).
    """
    results = []

    def _invoke_and_fix(run_index: int) -> SemanticJudgement | None:
        """Run one judge call and override overall_pass to be consistent with criteria."""
        try:
            result = chain.invoke(invoke_vars)
            judgement = SemanticJudgement(**result)
            # Override LLM's overall_pass — it can be inconsistent with individual criteria
            computed = all(c.passed for c in judgement.criteria)
            if judgement.overall_pass != computed:
                print(f"  [Judge run {run_index}/{n_runs}]: LLM overall_pass={judgement.overall_pass} "
                      f"overridden to {computed} (derived from criteria)")
                judgement = judgement.model_copy(update={"overall_pass": computed})
            print(f"  [Judge run {run_index}/{n_runs}]: {'PASS' if judgement.overall_pass else 'FAIL'}")
            return judgement
        except Exception as e:
            print(f"  [Judge run {run_index}/{n_runs}]: parse error — {e}")
            return None

    # --- First run ---
    j = _invoke_and_fix(1)
    if j:
        results.append(j)

    if not results:
        raise ValueError("First judge run failed to produce valid output")

    first = results[0]

    # --- Adaptive: skip remaining runs if all criteria clearly fail ---
    if not first.overall_pass and all(not c.passed for c in first.criteria):
        print(f"  All criteria failed on run 1 — skipping {n_runs - 1} remaining run(s)")
        aggregated_criteria = [
            {
                "criterion": c.criterion,
                "quoted_code": c.quoted_code,
                "passed": False,
                "reasoning": c.reasoning
            }
            for c in first.criteria
        ]
        return False, aggregated_criteria, first.feedback

    # --- Remaining runs for consensus ---
    for i in range(1, n_runs):
        j = _invoke_and_fix(i + 1)
        if j:
            results.append(j)

    # Strict consensus: ALL valid runs must pass
    overall_pass = all(r.overall_pass for r in results)

    # Aggregate criteria: a criterion fails if it fails in ANY run
    criterion_names = [c.criterion for c in results[0].criteria]
    aggregated_criteria = []
    for name in criterion_names:
        per_run = [c for r in results for c in r.criteria if c.criterion == name]
        all_passed = all(c.passed for c in per_run)
        worst = next((c for c in per_run if not c.passed), per_run[0])
        aggregated_criteria.append({
            "criterion": name,
            "quoted_code": worst.quoted_code,
            "passed": all_passed,
            "reasoning": worst.reasoning
        })

    # Aggregate feedback from all failing runs — deduplicate
    failing = [r for r in results if not r.overall_pass]
    feedbacks = list(dict.fromkeys(r.feedback for r in failing if r.feedback))
    aggregated_feedback = "\n".join(feedbacks)

    return overall_pass, aggregated_criteria, aggregated_feedback


def run_agent_5(state: PipelineState) -> PipelineState:
    """
    Agent 5: Semantic Judge
    Step 1: LLM code judge — static analysis before execution
    Step 2: Execute code on IFC file — only if code judge passes
    Step 3: Programmatic schema validation on the output
    """
    print(f"\n[Agent 5] Semantic evaluation — attempt {state.semantic_iterations + 1} "
          f"of {MAX_SEMANTIC_ITERATIONS}...")

    # --- Step 1: LLM code judge (static, before execution, with self-consistency) ---
    print(f"[Agent 5] Running static code analysis ({JUDGE_RUNS} independent runs)...")

    parser = JsonOutputParser(pydantic_object=SemanticJudgement)

    llm = ChatOpenAI(
        model_name=MODEL_JUDGE,
        openai_api_key=OPENROUTER_KEY,
        openai_api_base=OPENROUTER_BASE,
        temperature=JUDGE_TEMPERATURE
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", JUDGE_USER_PROMPT)
    ])

    chain = (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | parser
    )

    categories = state.parsed_rule.rule_category if state.parsed_rule else ["value"]
    if isinstance(categories, str):
        categories = [categories]
    rule_category_str = ", ".join(categories)

    invoke_vars = {
        "rule_text": state.building_rule_text,
        "rule_category": rule_category_str,
        "generated_code": state.generated_code,
        "rag_context": state.rag_context or "Not available — use general IFC schema knowledge.",
    }

    try:
        overall_pass, judgement_data, aggregated_feedback = _run_judge_with_consensus(
            chain, invoke_vars, JUDGE_RUNS
        )
    except Exception as e:
        print(f"[Agent 5] Judge evaluation failed: {e}")
        return state.model_copy(update={
            "error_log": state.error_log + [f"Agent 5 judge failed: {str(e)}"],
            "pipeline_failed": True,
            "current_agent": "error"
        })

    # For information_availability rules, criteria 2 and 3 are auto-passed.
    # The judge consistently produces false negatives for these criteria,
    # causing the feedback loop to break correct code. Only criterion 1 is evaluated.
    if set(categories) == {"information_availability"}:
        for i in range(1, len(judgement_data)):
            judgement_data[i] = {
                **judgement_data[i],
                "passed": True,
                "reasoning": "Auto-passed — criteria 2 and 3 are not judge-evaluated for information_availability rules"
            }
        overall_pass = judgement_data[0]["passed"] if judgement_data else True
        aggregated_feedback = "" if overall_pass else aggregated_feedback
        if overall_pass:
            print(f"[Agent 5] Information availability rule — only criterion 1 evaluated by judge.")

    print(f"[Agent 5] Consensus result: {'PASS' if overall_pass else 'FAIL'}")
    for c in judgement_data:
        status = "✓" if c["passed"] else "✗"
        print(f"  {status} {c['criterion']}: {c['reasoning']}")

    if not overall_pass:
        new_iterations = state.semantic_iterations + 1
        failed_criteria = "\n".join(
            f"- {c['criterion']}: {c['reasoning']}"
            for c in judgement_data if not c["passed"]
        )
        semantic_feedback = (
            f"JUDGE FEEDBACK: {aggregated_feedback}\n\n"
            f"FAILED CRITERIA:\n{failed_criteria}"
        )

        if new_iterations >= MAX_SEMANTIC_ITERATIONS:
            print(f"[Agent 5] Maximum iterations reached. Pipeline failed.")
            return state.model_copy(update={
                "semantic_feedback": semantic_feedback,
                "semantic_judgement": judgement_data,
                "semantic_iterations": new_iterations,
                "pipeline_failed": True,
                "error_log": state.error_log + [
                    f"Agent 5: max semantic iterations reached. Last feedback: {semantic_feedback}"
                ],
                "current_agent": "error"
            })

        print(f"[Agent 5] Returning to Agent 3 for correction "
              f"(iteration {new_iterations}/{MAX_SEMANTIC_ITERATIONS})...")
        return state.model_copy(update={
            "semantic_feedback": semantic_feedback,
            "semantic_judgement": judgement_data,
            "semantic_iterations": new_iterations,
            "current_agent": "agent_3"
        })

    # --- Step 2: Execute code (only after code judge passes) ---
    print("[Agent 5] Code judge passed. Executing code on IFC file...")
    abs_ifc_path = os.path.abspath(state.ifc_file_path)
    stdout, stderr, success = _execute_code(state.generated_code, abs_ifc_path)

    if not success or not stdout:
        print(f"[Agent 5] Execution failed:\n{stderr}")
        new_iterations = state.semantic_iterations + 1

        if new_iterations >= MAX_SEMANTIC_ITERATIONS:
            return state.model_copy(update={
                "semantic_judgement": judgement_data,
                "semantic_iterations": new_iterations,
                "pipeline_failed": True,
                "error_log": state.error_log + [
                    f"Agent 5: execution failed after {new_iterations} attempts. "
                    f"stderr: {stderr}"
                ],
                "current_agent": "error"
            })

        return state.model_copy(update={
            "execution_output": None,
            "semantic_feedback": f"Script execution failed with error:\n{stderr}",
            "semantic_judgement": judgement_data,
            "semantic_iterations": new_iterations,
            "current_agent": "agent_3"
        })

    print("[Agent 5] Execution successful. Running schema validation...")

    # --- Step 3: Parse and validate output schema ---
    try:
        execution_json = json.loads(stdout)
    except json.JSONDecodeError:
        execution_json = None

    if execution_json is None:
        new_iterations = state.semantic_iterations + 1
        feedback = "SCHEMA VALIDATION FAILED: Output is not valid JSON.\nFix the JSON output structure to match the required schema."

        if new_iterations >= MAX_SEMANTIC_ITERATIONS:
            return state.model_copy(update={
                "execution_output": stdout,
                "semantic_feedback": feedback,
                "semantic_judgement": judgement_data,
                "semantic_iterations": new_iterations,
                "pipeline_failed": True,
                "error_log": state.error_log + [
                    f"Agent 5: output is not valid JSON after {new_iterations} attempts."
                ],
                "current_agent": "error"
            })

        return state.model_copy(update={
            "execution_output": stdout,
            "semantic_feedback": feedback,
            "semantic_judgement": judgement_data,
            "semantic_iterations": new_iterations,
            "current_agent": "agent_3"
        })

    execution_output_str = json.dumps(execution_json, indent=2)

    schema_valid, schema_error = _validate_schema(execution_json)
    if not schema_valid:
        print(f"[Agent 5] Schema validation failed: {schema_error}")
        new_iterations = state.semantic_iterations + 1
        schema_feedback = (
            f"SCHEMA VALIDATION FAILED: {schema_error}\n"
            f"Fix the JSON output structure to match the required schema."
        )

        if new_iterations >= MAX_SEMANTIC_ITERATIONS:
            print(f"[Agent 5] Maximum iterations reached. Advancing to Agent 6 with best effort.")
            return state.model_copy(update={
                "execution_output": execution_output_str,
                "semantic_feedback": schema_feedback,
                "semantic_judgement": judgement_data,
                "semantic_iterations": new_iterations,
                "current_agent": "agent_6"
            })

        return state.model_copy(update={
            "execution_output": execution_output_str,
            "semantic_feedback": schema_feedback,
            "semantic_judgement": judgement_data,
            "semantic_iterations": new_iterations,
            "current_agent": "agent_3"
        })

    print("[Agent 5] All checks passed. Advancing to Agent 6.")
    return state.model_copy(update={
        "execution_output": execution_output_str,
        "semantic_feedback": None,
        "semantic_judgement": judgement_data,
        "current_agent": "agent_6"
    })
