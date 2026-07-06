from django.contrib import admin

from .models import Hyperparameter, ModelComparison, ModelComparisonDetail, ModelVersion, TrainingRun


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ['version', 'name', 'algorithm', 'status', 'accuracy', 'is_active', 'created_at']
    list_filter = ['algorithm', 'status', 'is_active', 'created_at']
    search_fields = ['version', 'name']
    readonly_fields = ['created_at', 'updated_at', 'artifact_path', 'preprocessing_pipeline_path']

    fieldsets = (
        ('Basic Info', {'fields': ('version', 'name', 'algorithm', 'status')}),
        ('Metadata', {'fields': ('dataset', 'created_by', 'created_at', 'updated_at', 'target_column')}),
        ('Deployment', {'fields': ('deployment_status', 'is_active')}),
        ('Metrics', {'fields': ('accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'cv_score_mean', 'cv_score_std')}),
        ('Artifacts', {'fields': ('artifact_path', 'preprocessing_pipeline_path')}),
        ('Configuration', {'fields': ('hyperparameters', 'feature_list')}),
        ('Results', {'fields': ('confusion_matrix', 'class_distribution', 'feature_importance')}),
        ('Notes', {'fields': ('notes',)}),
    )


@admin.register(Hyperparameter)
class HyperparameterAdmin(admin.ModelAdmin):
    list_display = ['model_version', 'param_name', 'param_value', 'param_type']
    list_filter = ['param_type']
    search_fields = ['model_version__version', 'param_name']
    readonly_fields = ['model_version']


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ['model_version', 'run_name', 'status', 'training_time_seconds', 'started_at']
    list_filter = ['status', 'started_at']
    search_fields = ['model_version__version', 'run_name']
    readonly_fields = ['started_at', 'completed_at', 'training_time_seconds']


@admin.register(ModelComparison)
class ModelComparisonAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_by', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name']
    readonly_fields = ['created_at', 'comparison_result']


@admin.register(ModelComparisonDetail)
class ModelComparisonDetailAdmin(admin.ModelAdmin):
    list_display = ['comparison', 'model_version', 'rank']
    list_filter = ['rank']
    search_fields = ['comparison__name', 'model_version__version']
    readonly_fields = ['metrics']
