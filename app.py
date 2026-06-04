# app.py
import streamlit as st
import sys
import os
import time
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from agentic_workflow.state import PipelineState
from agentic_workflow.graph import build_graph
from workflow_config import OPENROUTER_KEY, MAX_SYNTAX_ITERATIONS, MAX_SEMANTIC_ITERATIONS

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="IFC Compliance Checker",
    page_icon="🏗️",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .agent-box {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .agent-running {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        color: #856404;
    }
    .agent-done {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    .agent-failed {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


def render_agent_status(agent_status: dict, placeholder):
    """Render all agent statuses in the placeholder."""
    html = ""
    for name, info in agent_status.items():
        css_class = {
            "running": "agent-running",
            "done": "agent-done",
            "failed": "agent-failed"
        }.get(info["status"], "agent-running")

        icon = {
            "running": "⏳",
            "done": "✅",
            "failed": "❌"
        }.get(info["status"], "⏳")

        detail_text = f" — {info['detail']}" if info["detail"] else ""
        html += (
            f'<div class="agent-box {css_class}">'
            f'{icon} <b>{name}</b>{detail_text}</div>'
        )

    placeholder.markdown(html, unsafe_allow_html=True)


def run_pipeline_with_progress(rule: str, ifc_path: str, placeholder):
    """Run the pipeline via LangGraph and update the UI with real-time progress."""

    pipeline = build_graph()

    initial_state = PipelineState(
        building_rule_text=rule,
        ifc_file_path=ifc_path,
        api_key=OPENROUTER_KEY,
        start_time=time.time()
    )

    agent_status = {}

    def update(name: str, status: str, detail: str = ""):
        agent_status[name] = {"status": status, "detail": detail}
        render_agent_status(agent_status, placeholder)

    LABELS = {
        "agent_1": "Agent 1 — Rule Analyser",
        "agent_2": "Agent 2 — RAG Retriever",
        "agent_3": "Agent 3 — Code Generator",
        "agent_4": "Agent 4 — Syntax Validator",
        "agent_5": "Agent 5 — Semantic Judge",
        "agent_6": "Agent 6 — Report Generator",
    }

    update(LABELS["agent_1"], "running")
    cumulative: dict = {}

    for event in pipeline.stream(initial_state, stream_mode="updates"):
        node_name = next(iter(event))
        cumulative.update(event[node_name])

        errors = cumulative.get("error_log") or []
        error_detail = errors[-1] if errors else ""

        # error node: mark the last running agent as failed
        if node_name not in LABELS:
            last_running = next(
                (k for k, v in reversed(list(agent_status.items()))
                 if v["status"] == "running"),
                None
            )
            if last_running:
                update(last_running, "failed", error_detail)
            break

        label = LABELS[node_name]

        if cumulative.get("pipeline_failed"):
            update(label, "failed", error_detail)
            break

        syntax_iter = cumulative.get("syntax_iterations", 0)
        sem_iter = cumulative.get("semantic_iterations", 0)
        next_agent = cumulative.get("current_agent", "")

        if node_name == "agent_1":
            parsed = cumulative.get("parsed_rule") or {}
            primary = (
                parsed.primary_element if hasattr(parsed, "primary_element")
                else parsed.get("primary_element", "")
            )
            update(label, "done", f"Primary element: {primary}")
            update(LABELS["agent_2"], "running")

        elif node_name == "agent_2":
            update(label, "done", "RAG context synthesised")
            update(LABELS["agent_3"], "running", "Generating code...")

        elif node_name == "agent_3":
            # Always followed by agent_4; status resolved when agent_4 reports
            update(LABELS["agent_4"], "running", "Validating...")

        elif node_name == "agent_4":
            if next_agent == "agent_5":
                agent3_detail = (
                    "Semantically corrected" if sem_iter > 0
                    else f"{syntax_iter} correction(s)"
                )
                update(LABELS["agent_3"], "done", agent3_detail)
                update(label, "done", "No syntax errors")
                update(LABELS["agent_5"], "running",
                       f"Evaluating (attempt {sem_iter + 1}/{MAX_SEMANTIC_ITERATIONS})")
            else:
                update(LABELS["agent_3"], "running",
                       f"Correcting syntax error ({syntax_iter + 1}/{MAX_SYNTAX_ITERATIONS})")

        elif node_name == "agent_5":
            if next_agent == "agent_6":
                update(label, "done", f"Passed after {sem_iter} correction(s)")
                update(LABELS["agent_6"], "running", "Generating PDF...")
            elif sem_iter >= MAX_SEMANTIC_ITERATIONS:
                update(label, "done", "Max iterations reached — best effort")
                update(LABELS["agent_6"], "running", "Generating PDF...")
            else:
                update(LABELS["agent_3"], "running",
                       f"Semantic correction {sem_iter}/{MAX_SEMANTIC_ITERATIONS}")

        elif node_name == "agent_6":
            update(label, "done", "PDF report generated")

    if cumulative.get("pipeline_failed") or not cumulative.get("report"):
        return None

    base = initial_state.model_dump()
    base.update(cumulative)
    try:
        return PipelineState(**base)
    except Exception:
        return None


# --- MAIN UI ---
st.markdown('<div class="main-header">🏗️ IFC Compliance Checker</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated building code validation using '
    'RAG-enhanced LLM agents</div>',
    unsafe_allow_html=True
)

# --- INPUT SECTION ---
building_rule = st.text_area(
    "Building Regulation Rule",
    placeholder="e.g. Every door in the building must have a minimum height of 2.1 meters.",
    height=100
)

uploaded_file = st.file_uploader(
    "Upload IFC File",
    type=["ifc"],
    help="Upload the IFC file to check for compliance"
)

run_button = st.button(
    "▶ Run Compliance Check",
    type="primary",
    disabled=not (building_rule and uploaded_file)
)

# --- PIPELINE EXECUTION ---
if run_button and building_rule and uploaded_file:

    with tempfile.NamedTemporaryFile(
        suffix=".ifc",
        delete=False,
        mode='wb'
    ) as tmp_ifc:
        tmp_ifc.write(uploaded_file.read())
        tmp_ifc_path = tmp_ifc.name

    st.divider()
    st.subheader("Pipeline Progress")

    # Single placeholder — updated in place
    progress_placeholder = st.empty()

    try:
        final_state = run_pipeline_with_progress(
            rule=building_rule,
            ifc_path=tmp_ifc_path,
            placeholder=progress_placeholder
        )

        if final_state and final_state.report:
            st.divider()
            st.subheader("Results")

            report = final_state.report
            summary = report.get('summary', {})
            total = summary.get('total_elements', 0)
            compliant = summary.get('compliant', 0)
            non_compliant = summary.get('non_compliant', 0)
            compliance_rate = (compliant / total * 100) if total > 0 else 0

            # --- Metrics ---
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Elements", total)
            with m2:
                st.metric("Compliant", compliant)
            with m3:
                st.metric("Non-compliant", non_compliant)
            with m4:
                st.metric("Compliance Rate", f"{compliance_rate:.1f}%")

            # --- Overall status ---
            if non_compliant == 0:
                st.success("✅ All elements comply with the rule.")
            else:
                st.error(
                    f"❌ {non_compliant} element(s) do not comply with the rule."
                )

            # --- Non-compliant elements ---
            if non_compliant > 0:
                st.subheader(f"Non-compliant Elements ({non_compliant})")
                for element in report.get('elements', []):
                    if element.get('status') == 'FAIL':
                        with st.expander(
                            f"❌ {element.get('name', 'Unnamed')}",
                            expanded=True
                        ):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.write(
                                    f"**GlobalId:** `{element.get('global_id', 'N/A')}`"
                                )
                                st.write(
                                    f"**Measured value:** "
                                    f"{element.get('value', 'N/A')} "
                                    f"{element.get('unit', '')}"
                                )
                            with c2:
                                st.write(
                                    f"**Required threshold:** "
                                    f"{element.get('threshold', 'N/A')} "
                                    f"{element.get('unit', '')}"
                                )
                                st.write(
                                    f"**Reason:** {element.get('fail_reason', 'N/A')}"
                                )

            # --- Download PDF ---
            st.divider()
            if final_state.report_path and Path(final_state.report_path).exists():
                pdf_path = Path(final_state.report_path)
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_file.read(),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        type="primary"
                    )

            # --- Pipeline stats ---
            with st.expander("Pipeline Statistics"):
                st.write(f"**Syntax iterations:** {final_state.syntax_iterations}")
                st.write(f"**Semantic iterations:** {final_state.semantic_iterations}")
                if final_state.execution_time_seconds is not None:
                    t = final_state.execution_time_seconds
                    if t < 60:
                        st.write(f"**Execution time:** {t:.1f}s")
                    else:
                        mins, secs = divmod(t, 60)
                        st.write(f"**Execution time:** {int(mins)}m {secs:.1f}s")

            # --- Pipeline details ---
            st.divider()
            st.subheader("Pipeline Details")

            # Agent 1
            with st.expander("Agent 1 — Rule Analyser"):
                p = final_state.parsed_rule
                if p:
                    st.write(f"**Rule category:** {p.rule_category}")
                    st.write(f"**Primary element:** {p.primary_element}")
                    if p.related_elements:
                        st.write(f"**Related elements:** {', '.join(p.related_elements)}")
                    if p.properties_to_check:
                        st.write(f"**Properties:** {', '.join(p.properties_to_check)}")
                    st.write(f"**Constraints:** {', '.join(p.constraints)}")
                    if p.scope_conditions:
                        st.write(f"**Scope conditions:** {', '.join(p.scope_conditions)}")
                    st.write("**bSDD queries:**")
                    for q in p.search_queries_bsdd:
                        st.write(f"- {q}")
                    st.write("**EXPRESS queries:**")
                    for q in p.search_queries_express:
                        st.write(f"- {q}")

            # Agent 2
            with st.expander("Agent 2 — RAG Retriever"):
                chunks_bsdd = final_state.retrieved_chunks_bsdd
                chunks_express = final_state.retrieved_chunks_express

                st.write(f"**bSDD chunks retrieved:** {len(chunks_bsdd)}")
                for i, chunk in enumerate(chunks_bsdd, 1):
                    preview = chunk.splitlines()[0][:100] if chunk else ""
                    st.write(f"{i}. {preview}")

                st.write(f"**EXPRESS chunks retrieved:** {len(chunks_express)}")
                for i, chunk in enumerate(chunks_express, 1):
                    preview = chunk.splitlines()[0][:100] if chunk else ""
                    st.write(f"{i}. {preview}")

                if final_state.rag_context:
                    st.write("**Synthesised context:**")
                    st.text(final_state.rag_context)

            # Agent 3
            with st.expander("Agent 3 — Code Generator"):
                if final_state.generated_code:
                    st.code(final_state.generated_code, language="python")

            # Tool 4
            with st.expander("Tool 4 — Syntax Validator"):
                st.write(f"**Iterations:** {final_state.syntax_iterations}")
                if final_state.syntax_error:
                    st.error(final_state.syntax_error)
                else:
                    st.success("No syntax errors detected")

            # Agent 5
            with st.expander("Agent 5 — Semantic Judge"):
                st.write(f"**Iterations:** {final_state.semantic_iterations}")
                if final_state.semantic_judgement:
                    for c in final_state.semantic_judgement:
                        icon = "✅" if c["passed"] else "❌"
                        st.write(f"{icon} **{c['criterion']}**")
                        st.write(f"  {c['reasoning']}")
                if final_state.semantic_feedback:
                    st.warning(f"**Last feedback:** {final_state.semantic_feedback}")
                if final_state.execution_output:
                    st.write("**Execution output:**")
                    st.json(final_state.execution_output)

            # Tool 6
            with st.expander("Tool 6 — Report Generator"):
                if final_state.report:
                    s = final_state.report.get("summary", {})
                    st.write(f"**Total elements:** {s.get('total_elements', 0)}")
                    st.write(f"**Compliant:** {s.get('compliant', 0)}")
                    st.write(f"**Non-compliant:** {s.get('non_compliant', 0)}")
                if final_state.report_path:
                    st.write(f"**PDF path:** {final_state.report_path}")

    finally:
        os.unlink(tmp_ifc_path)