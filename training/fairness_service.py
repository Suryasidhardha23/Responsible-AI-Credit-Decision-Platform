import logging
from typing import Any

import numpy as np
import pandas as pd

from training.models import ModelVersion

logger = logging.getLogger(__name__)


class FairnessAuditService:
    """Audit model fairness and proxy bias for protected attributes."""

    def audit_model(self, model_version: ModelVersion, protected_attribute: str, proxy_features: list[str] | None = None) -> dict[str, Any]:
        """Generate fairness metrics and plain-English explanations for a model version."""
        if not model_version.dataset:
            raise ValueError('Model is not linked to a dataset')

        dataframe = pd.read_csv(model_version.dataset.file_path)
        target_column = model_version.target_column or 'target'
        if protected_attribute not in dataframe.columns:
            raise ValueError(f'Protected attribute {protected_attribute} not found in dataset')

        proxy_features = proxy_features or []
        metrics = self._compute_metrics(dataframe, target_column, protected_attribute)
        proxy_analysis = self._analyze_proxy_bias(dataframe, protected_attribute, proxy_features)
        explanations = self._build_explanations(metrics, proxy_analysis)

        return {
            'metrics': metrics,
            'explanations': explanations,
            'proxy_analysis': proxy_analysis,
            'protected_attribute': protected_attribute,
        }

    def _compute_metrics(self, dataframe: pd.DataFrame, target_column: str, protected_attribute: str) -> dict[str, float]:
        if target_column not in dataframe.columns:
            raise ValueError('Target column not found')

        group_sizes = dataframe.groupby(protected_attribute).size()
        if group_sizes.min() == 0:
            raise ValueError('Protected attribute contains empty groups')

        positive_rate = dataframe[target_column].mean()
        return {
            'demographic_parity_difference': float(abs(positive_rate - dataframe.groupby(protected_attribute)[target_column].mean().mean())),
            'demographic_parity_ratio': float(max(0.0, min(1.0, dataframe.groupby(protected_attribute)[target_column].mean().min() / max(dataframe.groupby(protected_attribute)[target_column].mean().max(), 1e-9)))),
            'equal_opportunity_difference': float(abs(dataframe.groupby(protected_attribute)[target_column].mean().max() - dataframe.groupby(protected_attribute)[target_column].mean().min())),
            'statistical_parity_difference': float(abs(dataframe.groupby(protected_attribute)[target_column].mean().max() - dataframe.groupby(protected_attribute)[target_column].mean().min())),
            'selection_rate': float(positive_rate),
        }

    def _analyze_proxy_bias(self, dataframe: pd.DataFrame, protected_attribute: str, proxy_features: list[str]) -> dict[str, Any]:
        analysis: dict[str, Any] = {}
        for feature in proxy_features:
            if feature not in dataframe.columns:
                continue
            if dataframe[feature].dtype.kind in 'biufc':
                correlation = dataframe[feature].corr(dataframe[protected_attribute].astype('category').cat.codes)
            else:
                correlation = dataframe[feature].astype('category').cat.codes.corr(dataframe[protected_attribute].astype('category').cat.codes)
            analysis[feature] = {
                'correlation_with_protected_attribute': float(correlation) if not np.isnan(correlation) else 0.0,
                'risk_level': 'high' if abs(correlation) > 0.3 else 'medium' if abs(correlation) > 0.1 else 'low',
            }
        return analysis

    def _build_explanations(self, metrics: dict[str, float], proxy_analysis: dict[str, Any]) -> list[dict[str, str]]:
        explanations = [
            {
                'label': 'Demographic parity',
                'text': 'The model shows measurable differences across the protected attribute, which can indicate disparate treatment in approvals.',
            },
            {
                'label': 'Proxy analysis',
                'text': 'Proxy variables can indirectly encode the protected attribute even when the sensitive feature is removed from training.',
            },
        ]

        if proxy_analysis:
            explanations.append({
                'label': 'Proxy risk',
                'text': 'Observed proxy features may still carry signal related to the protected attribute and should be monitored.',
            })

        return explanations
