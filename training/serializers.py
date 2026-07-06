"""Serializers for training-related models."""

from rest_framework import serializers

from training.models import Hyperparameter, ModelComparison, ModelComparisonDetail, ModelVersion, TrainingRun


class HyperparameterSerializer(serializers.ModelSerializer):
    """Serializer for hyperparameters."""

    class Meta:
        model = Hyperparameter
        fields = ['id', 'param_name', 'param_value', 'param_type']


class TrainingRunSerializer(serializers.ModelSerializer):
    """Serializer for training runs."""

    class Meta:
        model = TrainingRun
        fields = ['id', 'run_name', 'started_at', 'completed_at', 'status', 'error_message', 'training_time_seconds', 'validation_metrics']


class ModelVersionSerializer(serializers.ModelSerializer):
    """Serializer for model versions."""

    param_entries = HyperparameterSerializer(many=True, read_only=True)
    training_runs = TrainingRunSerializer(many=True, read_only=True)

    class Meta:
        model = ModelVersion
        fields = [
            'id',
            'version',
            'name',
            'algorithm',
            'status',
            'created_at',
            'updated_at',
            'deployment_status',
            'is_active',
            'accuracy',
            'precision',
            'recall',
            'f1_score',
            'roc_auc',
            'pr_auc',
            'cv_score_mean',
            'cv_score_std',
            'artifact_path',
            'preprocessing_pipeline_path',
            'param_entries',
            'target_column',
            'confusion_matrix',
            'class_distribution',
            'feature_importance',
            'notes',
            'training_runs',
        ]
        read_only_fields = ['created_at', 'updated_at', 'artifact_path', 'preprocessing_pipeline_path']


class ModelVersionListSerializer(serializers.ModelSerializer):
    """Simplified serializer for model version lists."""

    class Meta:
        model = ModelVersion
        fields = ['id', 'version', 'name', 'algorithm', 'status', 'accuracy', 'f1_score', 'roc_auc', 'is_active', 'created_at']


class ModelComparisonDetailSerializer(serializers.ModelSerializer):
    """Serializer for model comparison details."""

    model_version = ModelVersionListSerializer(read_only=True)

    class Meta:
        model = ModelComparisonDetail
        fields = ['id', 'model_version', 'metrics', 'rank']


class ModelComparisonSerializer(serializers.ModelSerializer):
    """Serializer for model comparisons."""

    details = ModelComparisonDetailSerializer(many=True, read_only=True)

    class Meta:
        model = ModelComparison
        fields = ['id', 'name', 'created_at', 'created_by', 'comparison_result', 'notes', 'details']
        read_only_fields = ['created_at', 'created_by']


class TrainModelRequestSerializer(serializers.Serializer):
    """Serializer for model training requests."""

    dataset_id = serializers.IntegerField()
    version = serializers.CharField(max_length=50)
    name = serializers.CharField(max_length=255, default='Credit Model')
    algorithm = serializers.ChoiceField(
        choices=['logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting'],
        default='logistic_regression',
    )
    target_column = serializers.CharField(max_length=100, default='target')
    perform_hyperparameter_tuning = serializers.BooleanField(default=False)
    cross_validation_folds = serializers.IntegerField(default=5, min_value=2, max_value=10)


class CompareModelsRequestSerializer(serializers.Serializer):
    """Serializer for model comparison requests."""

    model_version_ids = serializers.ListField(child=serializers.IntegerField(), min_length=2)
    name = serializers.CharField(max_length=255, default='Model Comparison')
    notes = serializers.CharField(required=False, allow_blank=True)
