import json

from django.contrib.postgres.fields import ArrayField
from django.db import models

from accounts.models import UserProfile
from dataset_manager.models import UploadedDataset


class ModelVersion(models.Model):
    """Tracks versioned ML models with comprehensive evaluation metrics."""

    ALGORITHM_CHOICES = [
        ('logistic_regression', 'Logistic Regression'),
        ('decision_tree', 'Decision Tree'),
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('xgboost', 'XGBoost'),
    ]

    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('trained', 'Trained'),
        ('evaluated', 'Evaluated'),
        ('deployed', 'Deployed'),
        ('archived', 'Archived'),
    ]

    version = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=255)
    algorithm = models.CharField(max_length=50, choices=ALGORITHM_CHOICES, default='logistic_regression')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    dataset = models.ForeignKey(UploadedDataset, on_delete=models.SET_NULL, null=True, blank=True)
    created_by = models.ForeignKey(UserProfile, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deployment_status = models.CharField(max_length=20, default='inactive')
    is_active = models.BooleanField(default=False)

    # Metrics
    accuracy = models.FloatField(default=0.0)
    precision = models.FloatField(default=0.0)
    recall = models.FloatField(default=0.0)
    f1_score = models.FloatField(default=0.0)
    roc_auc = models.FloatField(default=0.0)
    pr_auc = models.FloatField(default=0.0)
    cv_score_mean = models.FloatField(default=0.0)
    cv_score_std = models.FloatField(default=0.0)

    # Artifact storage
    artifact_path = models.CharField(max_length=500, blank=True)
    preprocessing_pipeline_path = models.CharField(max_length=500, blank=True)

    # Configuration
    hyperparameters = models.JSONField(default=dict, blank=True)
    target_column = models.CharField(max_length=100, blank=True)
    feature_list = models.JSONField(default=list, blank=True)

    # Evaluation details (JSON)
    confusion_matrix = models.JSONField(default=dict, blank=True)
    class_distribution = models.JSONField(default=dict, blank=True)
    feature_importance = models.JSONField(default=dict, blank=True)

    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['version']),
            models.Index(fields=['is_active']),
            models.Index(fields=['status']),
        ]

    def __str__(self) -> str:
        return f"{self.name} {self.version}"

    def activate(self):
        """Make this model the active production model."""
        ModelVersion.objects.filter(is_active=True).update(is_active=False)
        self.is_active = True
        self.deployment_status = 'deployed'
        self.status = 'deployed'
        self.save()

    def deactivate(self):
        """Remove this model from production."""
        self.is_active = False
        self.deployment_status = 'inactive'
        self.save()


class Hyperparameter(models.Model):
    """Stores hyperparameter configurations for model training."""

    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='param_entries')
    param_name = models.CharField(max_length=100)
    param_value = models.TextField()
    param_type = models.CharField(
        max_length=20,
        choices=[('int', 'Integer'), ('float', 'Float'), ('str', 'String'), ('bool', 'Boolean')],
    )

    class Meta:
        unique_together = ('model_version', 'param_name')

    def __str__(self) -> str:
        return f"{self.model_version.version} - {self.param_name}={self.param_value}"


class TrainingRun(models.Model):
    """Tracks individual training runs for experiment tracking."""

    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='training_runs')
    run_name = models.CharField(max_length=255)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=[('running', 'Running'), ('completed', 'Completed'), ('failed', 'Failed')],
        default='running',
    )
    error_message = models.TextField(blank=True)
    training_time_seconds = models.FloatField(null=True, blank=True)
    validation_metrics = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-started_at']

    def __str__(self) -> str:
        return f"{self.model_version.version} - {self.run_name}"


class ModelComparison(models.Model):
    """Stores comparisons between multiple model versions."""

    name = models.CharField(max_length=255)
    dataset = models.ForeignKey(UploadedDataset, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(UserProfile, on_delete=models.SET_NULL, null=True, blank=True)
    comparison_result = models.JSONField(default=dict, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self) -> str:
        return self.name


class ModelComparisonDetail(models.Model):
    """Individual model metrics within a comparison."""

    comparison = models.ForeignKey(ModelComparison, on_delete=models.CASCADE, related_name='details')
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE)
    metrics = models.JSONField(default=dict)
    rank = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        unique_together = ('comparison', 'model_version')
        ordering = ['rank']

    def __str__(self) -> str:
        return f"{self.comparison.name} - {self.model_version.version}"
