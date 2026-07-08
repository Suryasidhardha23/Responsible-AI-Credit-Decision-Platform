import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from training.models import ModelVersion

logger = logging.getLogger(__name__)


class SHAPExplainabilityService:
    """Generate SHAP-based explainability artifacts for deployed models."""

    def __init__(self, artifact_dir: str | None = None):
        self.artifact_dir = Path(artifact_dir or 'media/explanations')
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def generate_explanation(self, model_version: ModelVersion, sample_size: int = 50) -> dict[str, Any]:
        """Create SHAP explanation results for a trained model version."""
        if not model_version.artifact_path:
            raise ValueError('Model artifact path is missing')

        with open(model_version.artifact_path, 'rb') as handle:
            pipeline = pickle.load(handle)

        dataset_path = model_version.dataset.file_path if model_version.dataset else None
        if not dataset_path:
            raise ValueError('No dataset linked to model version')

        dataframe = pd.read_csv(dataset_path)
        target_column = model_version.target_column or 'target'
        features = dataframe.drop(columns=[target_column], errors='ignore')

        preprocessor = pipeline.named_steps['preprocess']
        classifier = pipeline.named_steps['classifier']

        def predict_fn(values: pd.DataFrame) -> np.ndarray:
            transformed = preprocessor.transform(values)
            return classifier.predict_proba(transformed)[:, 1]

        sample = features.sample(n=min(sample_size, len(features)), random_state=42)
        sample_for_shap = sample.copy()
        for column in sample_for_shap.select_dtypes(include=['object', 'category']).columns:
            sample_for_shap[column] = sample_for_shap[column].astype('string').fillna('missing').astype('category').cat.codes

        try:
            explainer = shap.Explainer(predict_fn, sample_for_shap)
            shap_values = explainer(sample_for_shap)
        except Exception:
            explainer = shap.Explainer(classifier, sample_for_shap)
            shap_values = explainer(sample_for_shap)

        summary_plot_path = self._save_summary_plot(shap_values, sample)
        top_features = self._extract_top_features(shap_values, sample)

        explanation = {
            'summary_plot_path': str(summary_plot_path),
            'top_features': top_features,
            'feature_count': len(sample.columns),
            'explanation_type': 'shap',
        }

        logger.info('Generated SHAP explanation for model %s', model_version.version)
        return explanation

    def _save_summary_plot(self, shap_values: Any, sample: pd.DataFrame) -> Path:
        plot_path = self.artifact_dir / 'summary_plot.png'
        shap.plots.beeswarm(shap_values, show=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        return plot_path

    def _extract_top_features(self, shap_values: Any, sample: pd.DataFrame) -> list[dict[str, Any]]:
        if hasattr(shap_values, 'values') and hasattr(shap_values.values, 'shape'):
            values = np.asarray(shap_values.values)
            if values.ndim == 2:
                mean_abs = np.abs(values).mean(axis=0)
            elif values.ndim == 1:
                mean_abs = np.abs(values)
            else:
                mean_abs = np.abs(values.reshape(-1))

            feature_names = list(sample.columns)
            ranked = sorted(zip(feature_names, mean_abs), key=lambda item: item[1], reverse=True)[:6]
            return [{'feature': feature, 'importance': float(importance)} for feature, importance in ranked]

        return []
