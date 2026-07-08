"""
Training Dashboard Views – Admin-only.
Handles: model training, activation, deactivation, archiving,
         fairness auditing, and bias mitigation from the UI.
"""

import logging

from django.contrib import messages
from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import redirect, render

from dataset_manager.models import UploadedDataset
from training.bias_mitigation_service import BiasMitigationService
from training.fairness_service import FairnessAuditService
from training.models import ModelVersion
from training.services import ModelTrainingService

logger = logging.getLogger(__name__)


def is_admin(user):
    return user.is_authenticated and user.role == "admin"


@user_passes_test(is_admin)
def training_dashboard(request):
    versions = ModelVersion.objects.order_by("-created_at")
    datasets = UploadedDataset.objects.order_by("-upload_date")
    latest_model = None
    metrics_result = None
    fairness_result = None
    mitigation_result = None
    audit_model_id = None

    if request.method == "POST":
        action = request.POST.get("action")

        # ── Train ──────────────────────────────────────────────────
        if action == "train":
            dataset_id = request.POST.get("dataset_id")
            version = request.POST.get("version", "v1.0")
            model_name = request.POST.get("model_name", "Credit Model")
            algorithm = request.POST.get("algorithm", "random_forest")
            target_column = request.POST.get("target_column", "target")
            hp_tuning = "hyperparameter_tuning" in request.POST

            dataset = UploadedDataset.objects.filter(id=dataset_id).first()
            if not dataset:
                messages.error(request, "Please select a valid dataset.")
            elif ModelVersion.objects.filter(version=version).exists():
                messages.error(request, f"Model version '{version}' already exists.")
            else:
                try:
                    service = ModelTrainingService()
                    result = service.train_model(
                        dataset_path=dataset.file_path,
                        target_column=target_column,
                        algorithm=algorithm,
                        perform_hyperparameter_tuning=hp_tuning,
                    )
                    mv = ModelVersion.objects.create(
                        version=version,
                        name=model_name,
                        algorithm=algorithm,
                        status="trained",
                        created_by=request.user,
                        dataset=dataset,
                        target_column=target_column,
                        accuracy=result["metrics"]["accuracy"],
                        precision=result["metrics"]["precision"],
                        recall=result["metrics"]["recall"],
                        f1_score=result["metrics"]["f1_score"],
                        roc_auc=result["metrics"]["roc_auc"],
                        pr_auc=result["metrics"]["pr_auc"],
                        cv_score_mean=result["cv_results"]["accuracy_mean"],
                        cv_score_std=result["cv_results"]["accuracy_std"],
                        artifact_path=result["artifact_path"],
                        preprocessing_pipeline_path=result["preprocessing_pipeline_path"],
                        hyperparameters=result["hyperparameters"],
                        feature_list=result["feature_list"],
                        confusion_matrix=result["confusion_matrix"],
                        class_distribution=result["class_distribution"],
                        feature_importance=result["feature_importance"],
                        notes=f"Trained on '{dataset.name}' by {request.user.username}.",
                    )
                    latest_model = mv
                    metrics_result = result["metrics"]
                    messages.success(
                        request,
                        f"✅ Model '{version}' trained successfully! Accuracy: {result['metrics']['accuracy']:.4f}",
                    )
                    versions = ModelVersion.objects.order_by("-created_at")
                except Exception as e:
                    logger.error("Training failed: %s", e, exc_info=True)
                    messages.error(request, f"Training failed: {e}")

        # ── Activate ───────────────────────────────────────────────
        elif action == "activate":
            mv = ModelVersion.objects.filter(id=request.POST.get("model_id")).first()
            if mv:
                mv.activate()
                messages.success(request, f"🚀 Model '{mv.version}' is now the active production model.")
            return redirect("training_dashboard")

        # ── Deactivate ─────────────────────────────────────────────
        elif action == "deactivate":
            mv = ModelVersion.objects.filter(id=request.POST.get("model_id")).first()
            if mv:
                mv.deactivate()
                messages.warning(request, f"⏸ Model '{mv.version}' has been deactivated.")
            return redirect("training_dashboard")

        # ── Archive ────────────────────────────────────────────────
        elif action == "archive":
            mv = ModelVersion.objects.filter(id=request.POST.get("model_id")).first()
            if mv:
                mv.status = "archived"
                mv.is_active = False
                mv.deployment_status = "archived"
                mv.save()
                messages.info(request, f"📦 Model '{mv.version}' has been archived.")
            return redirect("training_dashboard")

        # ── Fairness Audit ─────────────────────────────────────────
        elif action == "fairness_audit":
            model_id = request.POST.get("audit_model_id")
            protected = request.POST.get("protected_attribute", "sex")
            proxies_raw = request.POST.get("proxy_features", "")
            proxies = [p.strip() for p in proxies_raw.split(",") if p.strip()]
            audit_model_id = model_id

            mv = ModelVersion.objects.filter(id=model_id).first()
            if not mv:
                messages.error(request, "Model not found.")
            elif not mv.dataset:
                messages.error(request, "Model has no linked dataset.")
            else:
                try:
                    svc = FairnessAuditService()
                    fairness_result = svc.audit_model(mv, protected_attribute=protected, proxy_features=proxies)
                    messages.success(request, f"Fairness audit complete for '{mv.version}' on attribute '{protected}'.")
                except Exception as e:
                    logger.error("Fairness audit failed: %s", e, exc_info=True)
                    messages.error(request, f"Fairness audit failed: {e}")

        # ── Bias Mitigation ────────────────────────────────────────
        elif action == "mitigate":
            model_id = request.POST.get("mitigate_model_id")
            protected = request.POST.get("mit_protected_attribute", "sex")
            strategy = request.POST.get("strategy", "reweighting")

            mv = ModelVersion.objects.filter(id=model_id).first()
            if not mv:
                messages.error(request, "Model not found.")
            elif not mv.dataset:
                messages.error(request, "Model has no linked dataset.")
            else:
                try:
                    svc = BiasMitigationService()
                    result = svc.mitigate(
                        dataset_path=mv.dataset.file_path,
                        target_column=mv.target_column or "target",
                        protected_attribute=protected,
                        algorithm=mv.algorithm,
                        strategy=strategy,
                    )
                    mitigation_result = result

                    # Persist mitigated model as a new ModelVersion
                    new_ver = f"{mv.version}-mit"
                    idx = 0
                    while ModelVersion.objects.filter(version=new_ver).exists():
                        idx += 1
                        new_ver = f"{mv.version}-mit-{idx}"

                    ModelVersion.objects.create(
                        version=new_ver,
                        name=f"{mv.name} ({strategy})",
                        algorithm=mv.algorithm,
                        status="trained",
                        created_by=request.user,
                        dataset=mv.dataset,
                        target_column=mv.target_column,
                        accuracy=result["mitigated_metrics"].get("accuracy", 0.0),
                        precision=result["mitigated_metrics"].get("precision", 0.0),
                        recall=result["mitigated_metrics"].get("recall", 0.0),
                        f1_score=result["mitigated_metrics"].get("f1_score", 0.0),
                        roc_auc=result["mitigated_metrics"].get("roc_auc", 0.0),
                        artifact_path=result["artifact_path"],
                        notes=(
                            f"Bias-mitigated ({strategy}) on '{protected}'. "
                            f"DPD: {result['mitigated_metrics'].get('demographic_parity_difference', 0):.4f}. "
                            f"Accuracy: {result['mitigated_metrics'].get('accuracy', 0):.4f}."
                        ),
                    )
                    versions = ModelVersion.objects.order_by("-created_at")
                    messages.success(
                        request,
                        f"Bias mitigation ({strategy}) complete. New model saved as '{new_ver}'.",
                    )
                except Exception as e:
                    logger.error("Bias mitigation failed: %s", e, exc_info=True)
                    messages.error(request, f"Bias mitigation failed: {e}")

    return render(request, "training/dashboard.html", {
        "title": "Model Training & Registry",
        "versions": versions,
        "datasets": datasets,
        "latest_model": latest_model,
        "metrics": metrics_result,
        "fairness_result": fairness_result,
        "mitigation_result": mitigation_result,
        "audit_model_id": audit_model_id,
    })
