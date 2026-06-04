# agentic_workflow/agents/agent6_report_generator.py
import json
import os
import time
from datetime import datetime
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from agentic_workflow.state import PipelineState
from workflow_config import OUTPUT_DIR


def _build_pdf(report_data: dict, output_path: str):
    """Build the PDF compliance report using ReportLab."""

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()

    # --- Custom styles ---
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=6,
        textColor=colors.HexColor('#1a1a2e')
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=13,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor('#1a1a2e')
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4
    )
    fail_style = ParagraphStyle(
        'FailStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#c0392b')
    )

    story = []

    # --- Header ---
    story.append(Paragraph("IFC Compliance Report", title_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor('#1a1a2e')))
    story.append(Spacer(1, 0.3 * cm))

    # --- Metadata ---
    story.append(Paragraph(
        f"<b>Rule:</b> {report_data.get('rule', 'N/A')}", normal_style))
    story.append(Paragraph(
        f"<b>IFC File:</b> {report_data.get('ifc_file', 'N/A')}", normal_style))
    story.append(Paragraph(
        f"<b>Generated:</b> {report_data.get('timestamp', 'N/A')}", normal_style))
    if report_data.get('execution_time_formatted'):
        story.append(Paragraph(
            f"<b>Execution time:</b> {report_data['execution_time_formatted']}",
            normal_style))
    story.append(Spacer(1, 0.4 * cm))

    # --- Summary ---
    story.append(Paragraph("Summary", heading_style))
    summary = report_data.get('summary', {})
    total = summary.get('total_elements', 0)
    compliant = summary.get('compliant', 0)
    non_compliant = summary.get('non_compliant', 0)

    compliance_rate = (compliant / total * 100) if total > 0 else 0
    status_color = colors.HexColor('#27ae60') if non_compliant == 0 \
        else colors.HexColor('#c0392b')

    summary_data = [
        ['Metric', 'Value'],
        ['Total elements checked', str(total)],
        ['Compliant elements', str(compliant)],
        ['Non-compliant elements', str(non_compliant)],
        ['Compliance rate', f"{compliance_rate:.1f}%"],
        ['Overall status',
         'COMPLIANT' if non_compliant == 0 else 'NON-COMPLIANT'],
    ]

    summary_table = Table(summary_data, colWidths=[10 * cm, 7 * cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#f8f9fa'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('TEXTCOLOR', (1, 5), (1, 5), status_color),
        ('FONTNAME', (1, 5), (1, 5), 'Helvetica-Bold'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.4 * cm))

    # --- Non-compliant elements ---
    non_compliant_elements = [
        e for e in report_data.get('elements', [])
        if e.get('status') == 'FAIL'
    ]

    if not non_compliant_elements:
        story.append(Paragraph("Non-compliant Elements", heading_style))
        story.append(Paragraph(
            "All elements comply with the rule. No violations found.",
            normal_style
        ))
    else:
        story.append(Paragraph(
            f"Non-compliant Elements ({len(non_compliant_elements)})",
            heading_style
        ))

        for element in non_compliant_elements:
            # Element header
            name = element.get('name', 'Unnamed')
            global_id = element.get('global_id', 'N/A')
            story.append(Paragraph(
                f"<b>{name}</b>", normal_style
            ))

            # Element details table
            value = element.get('value')
            threshold = element.get('threshold')
            unit = element.get('unit', '')
            fail_reason = element.get('fail_reason', 'No reason provided')

            value_str = f"{value} {unit}" if value is not None else "N/A"
            threshold_str = f"{threshold} {unit}" if threshold is not None else "N/A"

            reason_style = ParagraphStyle(
                'ReasonStyle',
                parent=styles['Normal'],
                fontSize=9,
                leading=12,
                wordWrap='CJK'
            )

            element_data = [
                ['GlobalId', Paragraph(global_id, reason_style)],
                ['Measured value', Paragraph(value_str, reason_style)],
                ['Required threshold', Paragraph(threshold_str, reason_style)],
                ['Reason', Paragraph(fail_reason or '', reason_style)],
            ]

            element_table = Table(element_data, colWidths=[5 * cm, 12 * cm])
            element_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ROWBACKGROUNDS', (0, 0), (-1, -1),
                 [colors.HexColor('#fff5f5'), colors.HexColor('#fff9f9')]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#f5c6cb')),
                ('PADDING', (0, 0), (-1, -1), 5),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1a1a2e')),
            ]))
            story.append(element_table)
            story.append(Spacer(1, 0.3 * cm))

    # --- Footer ---
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor('#dee2e6')))
    story.append(Paragraph(
        f"Generated by IFC Compliance Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle('Footer', parent=styles['Normal'],
                       fontSize=8, textColor=colors.grey,
                       alignment=TA_CENTER)
    ))

    doc.build(story)


def run_agent_6(state: PipelineState) -> PipelineState:
    """
    Agent 6: Report Generator
    Input:  state.execution_output (JSON compliance report)
    Output: state.report (dict), PDF file saved to output directory
    """
    print("\n[Agent 6] Generating compliance report...")

    try:
        report_data = json.loads(state.execution_output)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"[Agent 6] ERROR: Could not parse execution output: {e}")
        return state.model_copy(update={
            "error_log": state.error_log + [f"Agent 6 failed: {str(e)}"],
            "pipeline_failed": True,
            "current_agent": "error"
        })

    # --- Calculate execution time ---
    execution_time = time.time() - state.start_time if state.start_time else None
    if execution_time is not None:
        mins, secs = divmod(execution_time, 60)
        execution_time_str = (
            f"{int(mins)}m {secs:.1f}s" if mins >= 1 else f"{execution_time:.1f}s"
        )
        report_data["execution_time_seconds"] = round(execution_time, 1)
        report_data["execution_time_formatted"] = execution_time_str
        print(f"[Agent 6] Execution time: {execution_time_str}")

    # --- Generate output filename ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ifc_name = Path(state.ifc_file_path).stem
    output_filename = f"compliance_report_{ifc_name}_{timestamp}.pdf"
    output_path = str(OUTPUT_DIR / output_filename)

    try:
        _build_pdf(report_data, output_path)
        print(f"[Agent 6] PDF report saved to: {output_path}")

        return state.model_copy(update={
            "report": report_data,
            "report_path": output_path,
            "execution_time_seconds": execution_time,
            "current_agent": "complete"
        })

    except Exception as e:
        print(f"[Agent 6] ERROR generating PDF: {e}")
        return state.model_copy(update={
            "error_log": state.error_log + [f"Agent 6 PDF generation failed: {str(e)}"],
            "pipeline_failed": True,
            "current_agent": "error"
        })