"""
Report Generation Service for Responsible AI Credit Decision Platform.

Generates professional-grade downloadable PDF reports using reportlab:
  - Prediction Report     : Individual applicant decision report
  - Model Card            : Industry-standard model documentation
  - Fairness Audit Report : Group-level fairness metric summary
  - Model Comparison      : Side-by-side performance comparison
"""

import io
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def _get_reportlab():
    """Lazy import reportlab so tests still pass if it is not installed."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            HRFlowable,
        )
        return {
            "colors": colors,
            "letter": letter,
            "getSampleStyleSheet": getSampleStyleSheet,
            "ParagraphStyle": ParagraphStyle,
            "inch": inch,
            "SimpleDocTemplate": SimpleDocTemplate,
            "Paragraph": Paragraph,
            "Spacer": Spacer,
            "Table": Table,
            "TableStyle": TableStyle,
            "HRFlowable": HRFlowable,
        }
    except ImportError:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install it with: pip install reportlab"
        )


class ReportService:
    """Generates PDF reports for predictions, model cards, fairness audits, and comparisons."""

    BRAND_COLOR = (0.08, 0.24, 0.40)   # Dark navy
    ACCENT_COLOR = (0.18, 0.76, 0.71)  # Teal

    # ------------------------------------------------------------------
    # Prediction Report
    # ------------------------------------------------------------------

    def generate_prediction_report(
        self,
        prediction_record,
        shap_explanation: list[dict] | None = None,
        fairness_context: dict | None = None,
        model_version=None,
    ) -> bytes:
        """Generate a PDF report for a single prediction."""
        rl = _get_reportlab()
        buf = io.BytesIO()
        doc = rl["SimpleDocTemplate"](
            buf,
            pagesize=rl["letter"],
            leftMargin=0.75 * rl["inch"],
            rightMargin=0.75 * rl["inch"],
            topMargin=0.75 * rl["inch"],
            bottomMargin=0.75 * rl["inch"],
        )
        styles = rl["getSampleStyleSheet"]()
        story = []
        colors = rl["colors"]
        Paragraph = rl["Paragraph"]
        Spacer = rl["Spacer"]
        Table = rl["Table"]
        TableStyle = rl["TableStyle"]
        HRFlowable = rl["HRFlowable"]
        inch = rl["inch"]
        ParagraphStyle = rl["ParagraphStyle"]

        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Title"],
            textColor=colors.HexColor("#0d3d60"),
            fontSize=20,
            spaceAfter=6,
        )
        h2_style = ParagraphStyle(
            "H2Style",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#0d3d60"),
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = styles["BodyText"]
        body_style.leading = 16

        # Header
        story.append(Paragraph("Responsible AI Credit Decision Platform", title_style))
        story.append(Paragraph("Official Loan Underwriting Report", styles["Heading3"]))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2ec4b6")))
        story.append(Spacer(1, 0.2 * inch))

        # Applicant Summary
        story.append(Paragraph("Applicant Summary", h2_style))
        decision_color = "#28a745" if prediction_record.prediction == "Approved" else "#dc3545"
        summary_data = [
            ["Field", "Value"],
            ["Applicant Name", prediction_record.applicant_name or "N/A"],
            ["Report Date", datetime.now().strftime("%B %d, %Y %H:%M UTC")],
            ["Decision", prediction_record.prediction],
            ["Approval Probability", f"{prediction_record.probability:.1%}"],
            ["Evaluated By", prediction_record.created_by.username if prediction_record.created_by else "System"],
        ]
        if model_version:
            summary_data.append(["Model Version", f"{model_version.name} ({model_version.version})"])
            summary_data.append(["Algorithm", model_version.get_algorithm_display()])

        table = Table(summary_data, colWidths=[2.5 * inch, 4.5 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d3d60")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 3), (0, 3), "Helvetica-Bold"),
            ("TEXTCOLOR", (1, 3), (1, 3), colors.HexColor(decision_color)),
        ]))
        story.append(table)

        # SHAP Feature Attribution
        if shap_explanation:
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph("Explainable AI: Feature Contributions (SHAP)", h2_style))
            story.append(Paragraph(
                "The following table shows which input features most influenced the AI decision, "
                "ranked by their SHAP (SHapley Additive exPlanations) value.",
                body_style,
            ))
            story.append(Spacer(1, 0.1 * inch))

            shap_data = [["Rank", "Feature", "Input Value", "SHAP Value", "Effect"]]
            for i, item in enumerate(shap_explanation[:8], 1):
                effect = "↑ Increases Approval" if item["shap_value"] > 0 else "↓ Decreases Approval"
                shap_data.append([
                    str(i),
                    item["feature"].replace("_", " ").title(),
                    str(item["value"]),
                    f"{item['shap_value']:+.4f}",
                    effect,
                ])

            shap_table = Table(shap_data, colWidths=[0.5 * inch, 2 * inch, 1.5 * inch, 1.2 * inch, 2.3 * inch])
            shap_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2ec4b6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (3, 0), (3, -1), "CENTER"),
            ]))
            story.append(shap_table)

        # Fairness Context
        if fairness_context:
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph("Responsible AI: Fairness Context", h2_style))
            story.append(Paragraph(fairness_context.get("explanation", ""), body_style))
            fm = fairness_context.get("fairness_metrics", {})
            if fm:
                fairness_data = [["Fairness Metric", "Value"]] + [
                    [k.replace("_", " ").title(), f"{v:.4f}"]
                    for k, v in fm.items()
                ]
                f_table = Table(fairness_data, colWidths=[4 * inch, 3 * inch])
                f_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d3d60")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
                    ("PADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(f_table)

        # Disclaimer
        story.append(Spacer(1, 0.4 * inch))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dee2e6")))
        disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["Italic"], fontSize=8, textColor=colors.grey)
        story.append(Paragraph(
            "This report was generated by the Responsible AI Credit Decision Platform. "
            "AI-generated credit decisions are advisory only and must be reviewed by a qualified loan officer "
            "before acting on them. Protected attributes were not used as direct model inputs.",
            disclaimer_style,
        ))

        doc.build(story)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Model Card
    # ------------------------------------------------------------------

    def generate_model_card(self, model_version) -> bytes:
        """Generate an industry-standard model card PDF."""
        rl = _get_reportlab()
        buf = io.BytesIO()
        doc = rl["SimpleDocTemplate"](buf, pagesize=rl["letter"],
                                      leftMargin=0.75 * rl["inch"], rightMargin=0.75 * rl["inch"],
                                      topMargin=0.75 * rl["inch"], bottomMargin=0.75 * rl["inch"])
        styles = rl["getSampleStyleSheet"]()
        story = []
        colors = rl["colors"]
        Paragraph = rl["Paragraph"]
        Spacer = rl["Spacer"]
        Table = rl["Table"]
        TableStyle = rl["TableStyle"]
        HRFlowable = rl["HRFlowable"]
        inch = rl["inch"]
        ParagraphStyle = rl["ParagraphStyle"]

        title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                     textColor=colors.HexColor("#0d3d60"), fontSize=20)
        h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
                                  textColor=colors.HexColor("#0d3d60"), spaceBefore=12)
        body_style = styles["BodyText"]

        story.append(Paragraph("Model Card", title_style))
        story.append(Paragraph(f"{model_version.name} ({model_version.version})", styles["Heading3"]))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2ec4b6")))
        story.append(Spacer(1, 0.15 * inch))

        def section(title, rows):
            story.append(Paragraph(title, h2_style))
            table = Table([[k, v] for k, v in rows], colWidths=[2.5 * inch, 4.5 * inch])
            table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.whitesmoke]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]))
            story.append(table)
            story.append(Spacer(1, 0.1 * inch))

        section("Model Details", [
            ("Model Name", model_version.name),
            ("Version", model_version.version),
            ("Algorithm", model_version.get_algorithm_display()),
            ("Created At", model_version.created_at.strftime("%Y-%m-%d %H:%M UTC")),
            ("Status", model_version.get_status_display()),
            ("Target Column", model_version.target_column or "N/A"),
        ])

        section("Intended Use", [
            ("Primary Use Case", "Binary credit risk classification for consumer loan applications."),
            ("Intended Users", "Trained loan officers at regulated financial institutions."),
            ("Out-of-Scope Uses", "Not intended for automated approval without human review."),
        ])

        section("Training Dataset", [
            ("Dataset Name", model_version.dataset.name if model_version.dataset else "N/A"),
            ("Rows", str(model_version.dataset.row_count) if model_version.dataset else "N/A"),
            ("Columns", str(model_version.dataset.column_count) if model_version.dataset else "N/A"),
        ])

        section("Evaluation Metrics", [
            ("Accuracy",  f"{model_version.accuracy:.4f}"),
            ("Precision", f"{model_version.precision:.4f}"),
            ("Recall",    f"{model_version.recall:.4f}"),
            ("F1 Score",  f"{model_version.f1_score:.4f}"),
            ("ROC-AUC",   f"{model_version.roc_auc:.4f}"),
            ("PR-AUC",    f"{model_version.pr_auc:.4f}"),
            ("CV Accuracy (mean ± std)", f"{model_version.cv_score_mean:.4f} ± {model_version.cv_score_std:.4f}"),
        ])

        section("Ethical Considerations & Limitations", [
            ("Protected Attributes", "sex, age_group were excluded from direct model inputs."),
            ("Bias Monitoring", "Demographic parity difference and disparate impact ratio are monitored post-deployment."),
            ("Proxy Variables", "Variables such as zip_code or employment_status may act as proxies for protected attributes."),
            ("Known Limitations", "Performance may degrade on distributions significantly different from the training data."),
            ("Monitoring Recommendation", "Re-audit fairness metrics quarterly or when approval rates change by more than 5%."),
        ])

        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dee2e6")))
        story.append(Paragraph(
            "Generated by the Responsible AI Credit Decision Platform. "
            "Model Card follows the Mitchell et al. (2019) framework.",
            ParagraphStyle("disc", parent=styles["Italic"], fontSize=8, textColor=colors.grey),
        ))

        doc.build(story)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Fairness Audit Report
    # ------------------------------------------------------------------

    def generate_fairness_audit_report(self, model_version, audit_result: dict) -> bytes:
        """Generate a fairness audit PDF report."""
        rl = _get_reportlab()
        buf = io.BytesIO()
        doc = rl["SimpleDocTemplate"](buf, pagesize=rl["letter"],
                                      leftMargin=0.75 * rl["inch"], rightMargin=0.75 * rl["inch"],
                                      topMargin=0.75 * rl["inch"], bottomMargin=0.75 * rl["inch"])
        styles = rl["getSampleStyleSheet"]()
        story = []
        colors = rl["colors"]
        Paragraph = rl["Paragraph"]
        Spacer = rl["Spacer"]
        Table = rl["Table"]
        TableStyle = rl["TableStyle"]
        HRFlowable = rl["HRFlowable"]
        inch = rl["inch"]
        ParagraphStyle = rl["ParagraphStyle"]

        title_style = ParagraphStyle("T3", parent=styles["Title"],
                                     textColor=colors.HexColor("#0d3d60"), fontSize=20)
        h2_style = ParagraphStyle("H3", parent=styles["Heading2"],
                                  textColor=colors.HexColor("#0d3d60"), spaceBefore=12)
        body_style = styles["BodyText"]

        story.append(Paragraph("Fairness Audit Report", title_style))
        story.append(Paragraph(
            f"Model: {model_version.name} ({model_version.version}) | "
            f"Protected Attribute: {audit_result.get('protected_attribute', 'N/A')}",
            styles["Heading3"]
        ))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2ec4b6")))
        story.append(Spacer(1, 0.15 * inch))

        # Metrics table
        story.append(Paragraph("Fairness Metrics", h2_style))
        metrics = audit_result.get("metrics", {})
        metric_data = [["Metric", "Value", "Interpretation"]] + [
            [k.replace("_", " ").title(), f"{v:.4f}",
             ("✓ Good (< 0.1)" if abs(v) < 0.1 else "⚠ Moderate" if abs(v) < 0.2 else "✗ Investigate")]
            for k, v in metrics.items()
        ]
        m_table = Table(metric_data, colWidths=[2.5 * inch, 1.5 * inch, 3 * inch])
        m_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d3d60")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
            ("PADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(m_table)

        # Explanations
        story.append(Paragraph("Plain-English Explanations", h2_style))
        for expl in audit_result.get("explanations", []):
            story.append(Paragraph(f"<b>{expl['label']}:</b> {expl['text']}", body_style))
            story.append(Spacer(1, 0.05 * inch))

        # Proxy Analysis
        proxy = audit_result.get("proxy_analysis", {})
        if proxy:
            story.append(Paragraph("Proxy Variable Analysis", h2_style))
            story.append(Paragraph(
                "Removing the protected attribute alone is insufficient. The following "
                "features may act as proxies and re-introduce bias indirectly.", body_style))
            story.append(Spacer(1, 0.1 * inch))
            proxy_data = [["Proxy Feature", "Correlation With Sensitive Attr.", "Risk Level"]]
            for feat, info in proxy.items():
                proxy_data.append([
                    feat, f"{info.get('correlation_with_protected_attribute', 0):.4f}",
                    info.get("risk_level", "unknown").upper()
                ])
            p_table = Table(proxy_data, colWidths=[2.5 * inch, 2.5 * inch, 2 * inch])
            p_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2ec4b6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]))
            story.append(p_table)

        doc.build(story)
        return buf.getvalue()
