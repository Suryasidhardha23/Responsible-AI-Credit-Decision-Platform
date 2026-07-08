"""
Extended API views: Live prediction, bias mitigation trigger, PDF report downloads.
"""

import logging
from django.http import HttpResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from prediction.models import PredictionRecord
from prediction.services import PredictionService
from training.bias_mitigation_service import BiasMitigationService
from training.fairness_service import FairnessAuditService
from training.models import ModelVersion
from training.report_service import ReportService

logger = logging.getLogger(__name__)


def is_admin(user):
    return user.is_authenticated and user.role == "admin"


class AdminOnlyPermission(IsAuthenticated):
    def has_permission(self, request, view):
        return super().has_permission(request, view) and is_admin(request.user)


# ─────────────────────────────────────────────
# Live Prediction API
# ─────────────────────────────────────────────

class PredictAPIView(APIView):
    """POST /api/predict/ – Run a live prediction using the active model."""

    permission_classes = [IsAuthenticated]

    def post(self, request):
        active_model = PredictionService.get_active_model()
        if not active_model:
            return Response({"error": "No active model found."}, status=status.HTTP_404_NOT_FOUND)

        input_data = {k: v for k, v in request.data.items()}
        applicant_name = input_data.pop("applicant_name", "API Applicant")

        try:
            service = PredictionService()
            result = service.predict_and_explain(active_model, input_data)

            record = PredictionRecord.objects.create(
                applicant_name=applicant_name,
                prediction=result["prediction"],
                probability=result["probability"],
                created_by=request.user,
            )

            return Response(
                {
                    "record_id": record.id,
                    "applicant_name": applicant_name,
                    "prediction": result["prediction"],
                    "probability": result["probability"],
                    "confidence": result["confidence"],
                    "explanation": result["summary"],
                    "shap_explanation": result["shap_explanation"],
                    "fairness_context": result["fairness_context"],
                    "model_version": active_model.version,
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            logger.error("Prediction API failed: %s", e, exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ─────────────────────────────────────────────
# Bias Mitigation API
# ─────────────────────────────────────────────

class MitigateModelAPIView(APIView):
    """POST /api/mitigate/ – Train a fairness-aware model and store it as a new version."""

    permission_classes = [AdminOnlyPermission]

    def post(self, request):
        model_version_id = request.data.get("model_version_id")
        protected_attribute = request.data.get("protected_attribute", "sex")
        strategy = request.data.get("strategy", "reweighting")
        new_version_label = request.data.get("new_version", None)

        if not model_version_id:
            return Response({"error": "model_version_id is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            model_version = ModelVersion.objects.get(id=model_version_id)
        except ModelVersion.DoesNotExist:
            return Response({"error": "Model version not found."}, status=status.HTTP_404_NOT_FOUND)

        if not model_version.dataset:
            return Response({"error": "Model has no linked dataset."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            service = BiasMitigationService()
            result = service.mitigate(
                dataset_path=model_version.dataset.file_path,
                target_column=model_version.target_column or "target",
                protected_attribute=protected_attribute,
                algorithm=model_version.algorithm,
                strategy=strategy,
            )

            # Create a new ModelVersion entry for the mitigated model
            new_version = new_version_label or f"{model_version.version}-mitigated"
            # Avoid duplicate versions
            suffix = 0
            base = new_version
            while ModelVersion.objects.filter(version=new_version).exists():
                suffix += 1
                new_version = f"{base}-{suffix}"

            mitigated_mv = ModelVersion.objects.create(
                version=new_version,
                name=f"{model_version.name} (Mitigated – {strategy})",
                algorithm=model_version.algorithm,
                status="trained",
                created_by=request.user,
                dataset=model_version.dataset,
                target_column=model_version.target_column,
                accuracy=result["mitigated_metrics"].get("accuracy", 0.0),
                precision=result["mitigated_metrics"].get("precision", 0.0),
                recall=result["mitigated_metrics"].get("recall", 0.0),
                f1_score=result["mitigated_metrics"].get("f1_score", 0.0),
                roc_auc=result["mitigated_metrics"].get("roc_auc", 0.0),
                artifact_path=result["artifact_path"],
                notes=(
                    f"Fairness-aware model. Strategy: {strategy}. "
                    f"Protected attribute: {protected_attribute}. "
                    f"DPD: {result['mitigated_metrics'].get('demographic_parity_difference', 0):.4f}"
                ),
            )

            logger.info("Mitigated model %s created by %s", new_version, request.user.username)

            return Response(
                {
                    "message": f"Bias mitigation ({strategy}) complete.",
                    "comparison": result["comparison"],
                    "baseline_metrics": result["baseline_metrics"],
                    "mitigated_metrics": result["mitigated_metrics"],
                    "new_model_version_id": mitigated_mv.id,
                    "new_model_version": new_version,
                },
                status=status.HTTP_201_CREATED,
            )
        except Exception as e:
            logger.error("Bias mitigation failed: %s", e, exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ─────────────────────────────────────────────
# PDF Report APIs
# ─────────────────────────────────────────────

class PredictionReportPDFView(APIView):
    """GET /api/reports/prediction/<id>/pdf/ – Download prediction PDF report."""

    permission_classes = [IsAuthenticated]

    def get(self, request, record_id):
        try:
            record = PredictionRecord.objects.get(id=record_id)
        except PredictionRecord.DoesNotExist:
            return Response({"error": "Prediction record not found."}, status=status.HTTP_404_NOT_FOUND)

        # Only the creator or an admin can download
        if record.created_by != request.user and not is_admin(request.user):
            return Response({"error": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)

        try:
            model_version = record.model_version or PredictionService.get_active_model()
            service = ReportService()
            pdf_bytes = service.generate_prediction_report(
                prediction_record=record,
                shap_explanation=record.shap_explanation or [],
                fairness_context=record.fairness_context or {},
                model_version=model_version,
            )
            response = HttpResponse(pdf_bytes, content_type="application/pdf")
            response["Content-Disposition"] = (
                f'attachment; filename="prediction_report_{record.id}.pdf"'
            )
            return response
        except Exception as e:
            logger.error("PDF generation failed: %s", e, exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ModelCardPDFView(APIView):
    """GET /api/reports/model-card/<id>/pdf/ – Download model card PDF."""

    permission_classes = [IsAuthenticated]

    def get(self, request, model_version_id):
        try:
            model_version = ModelVersion.objects.get(id=model_version_id)
        except ModelVersion.DoesNotExist:
            return Response({"error": "Model not found."}, status=status.HTTP_404_NOT_FOUND)

        try:
            service = ReportService()
            pdf_bytes = service.generate_model_card(model_version)
            response = HttpResponse(pdf_bytes, content_type="application/pdf")
            response["Content-Disposition"] = (
                f'attachment; filename="model_card_{model_version.version}.pdf"'
            )
            return response
        except Exception as e:
            logger.error("Model card PDF failed: %s", e, exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FairnessAuditReportPDFView(APIView):
    """POST /api/reports/fairness-audit/<id>/pdf/ – Generate & download fairness audit PDF."""

    permission_classes = [AdminOnlyPermission]

    def post(self, request, model_version_id):
        try:
            model_version = ModelVersion.objects.get(id=model_version_id)
        except ModelVersion.DoesNotExist:
            return Response({"error": "Model not found."}, status=status.HTTP_404_NOT_FOUND)

        protected_attribute = request.data.get("protected_attribute", "sex")
        proxy_features = request.data.get("proxy_features", [])

        try:
            audit_service = FairnessAuditService()
            audit_result = audit_service.audit_model(
                model_version,
                protected_attribute=protected_attribute,
                proxy_features=proxy_features,
            )
            report_service = ReportService()
            pdf_bytes = report_service.generate_fairness_audit_report(model_version, audit_result)
            response = HttpResponse(pdf_bytes, content_type="application/pdf")
            response["Content-Disposition"] = (
                f'attachment; filename="fairness_audit_{model_version.version}.pdf"'
            )
            return response
        except Exception as e:
            logger.error("Fairness audit PDF failed: %s", e, exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
