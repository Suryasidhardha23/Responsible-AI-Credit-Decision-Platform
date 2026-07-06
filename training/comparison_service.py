"""Service for comparing multiple model versions."""

import json
import logging
from typing import Dict, List

from training.models import ModelComparison, ModelComparisonDetail, ModelVersion

logger = logging.getLogger(__name__)


class ModelComparisonService:
    """Compare multiple model versions and generate comprehensive comparison reports."""

    METRIC_PRIORITIES = {
        'roc_auc': {'weight': 0.3, 'direction': 'higher'},
        'f1_score': {'weight': 0.25, 'direction': 'higher'},
        'accuracy': {'weight': 0.2, 'direction': 'higher'},
        'precision': {'weight': 0.15, 'direction': 'higher'},
        'recall': {'weight': 0.1, 'direction': 'higher'},
    }

    @staticmethod
    def compare_models(
        model_versions: List[ModelVersion],
        name: str = 'Model Comparison',
        created_by=None,
    ) -> ModelComparison:
        """Create a comprehensive comparison between multiple model versions."""

        if not model_versions:
            raise ValueError("At least one model version is required for comparison")

        logger.info(f"Comparing {len(model_versions)} models")

        comparison = ModelComparison.objects.create(
            name=name,
            created_by=created_by,
        )

        # Collect metrics for all models
        comparison_data = []
        for model in model_versions:
            metrics_dict = {
                'accuracy': model.accuracy,
                'precision': model.precision,
                'recall': model.recall,
                'f1_score': model.f1_score,
                'roc_auc': model.roc_auc,
                'pr_auc': model.pr_auc,
                'cv_score_mean': model.cv_score_mean,
            }
            comparison_data.append((model, metrics_dict))

        # Rank models based on weighted scoring
        ranked_models = ModelComparisonService._rank_models(comparison_data)

        # Create comparison details and store results
        result_data = {}
        for rank, (model, score) in enumerate(ranked_models, 1):
            metrics_dict = {
                'accuracy': model.accuracy,
                'precision': model.precision,
                'recall': model.recall,
                'f1_score': model.f1_score,
                'roc_auc': model.roc_auc,
                'pr_auc': model.pr_auc,
                'cv_score_mean': model.cv_score_mean,
                'cv_score_std': model.cv_score_std,
                'weighted_score': float(score),
                'algorithm': model.algorithm,
            }

            ModelComparisonDetail.objects.create(
                comparison=comparison,
                model_version=model,
                metrics=metrics_dict,
                rank=rank,
            )

            result_data[model.version] = metrics_dict

        comparison.comparison_result = {
            'ranked_models': [m.version for m, _ in ranked_models],
            'metrics_by_model': result_data,
            'metric_weights': ModelComparisonService.METRIC_PRIORITIES,
        }
        comparison.save()

        logger.info(f"Comparison completed. Best model: {ranked_models[0][0].version}")
        return comparison

    @staticmethod
    def _rank_models(comparison_data: List[tuple]) -> List[tuple]:
        """Rank models using weighted scoring."""
        scores = []

        for model, metrics_dict in comparison_data:
            weighted_score = 0.0

            for metric_name, metric_config in ModelComparisonService.METRIC_PRIORITIES.items():
                if metric_name in metrics_dict and metrics_dict[metric_name] is not None:
                    metric_value = metrics_dict[metric_name]
                    weight = metric_config['weight']

                    if metric_config['direction'] == 'higher':
                        weighted_score += metric_value * weight
                    else:
                        weighted_score += (1 - metric_value) * weight

            scores.append((model, weighted_score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    @staticmethod
    def get_comparison_summary(comparison: ModelComparison) -> dict:
        """Generate a human-readable summary of the comparison."""
        details = comparison.details.all().order_by('rank')

        if not details:
            return {}

        best_model = details.first()
        summary = {
            'total_models_compared': details.count(),
            'best_model': {
                'version': best_model.model_version.version,
                'algorithm': best_model.model_version.algorithm,
                'metrics': best_model.metrics,
            },
            'models': [],
        }

        for detail in details:
            summary['models'].append({
                'rank': detail.rank,
                'version': detail.model_version.version,
                'algorithm': detail.model_version.algorithm,
                'metrics': detail.metrics,
            })

        return summary

    @staticmethod
    def get_metric_differences(best_model: ModelVersion, other_model: ModelVersion) -> dict:
        """Calculate metric differences between two models."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']

        differences = {
            'models': {
                'best': {'version': best_model.version, 'algorithm': best_model.algorithm},
                'other': {'version': other_model.version, 'algorithm': other_model.algorithm},
            },
            'metric_differences': {},
        }

        for metric in metrics:
            best_val = getattr(best_model, metric, 0.0)
            other_val = getattr(other_model, metric, 0.0)
            diff = best_val - other_val

            differences['metric_differences'][metric] = {
                'best_model_value': float(best_val),
                'other_model_value': float(other_val),
                'difference': float(diff),
                'percentage_difference': float((diff / other_val * 100) if other_val != 0 else 0),
            }

        return differences
