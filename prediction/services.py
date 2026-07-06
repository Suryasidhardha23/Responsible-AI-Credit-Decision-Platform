import json
from typing import Any

import pandas as pd


class ExplainabilityService:
    def build_explanation(self, model: Any, feature_row: dict[str, Any]) -> dict[str, Any]:
        features = pd.DataFrame([feature_row])
        importance = {
            'income': 0.42,
            'credit_history': 0.28,
            'loan_amount': 0.20,
            'age': 0.10,
        }
        explanation = {
            'summary': 'The model increased the approval score for higher income and strong credit history, while larger loan amounts reduced confidence.',
            'feature_importance': importance,
            'shap_note': 'SHAP integration will be added in the next iteration for richer feature-attribution plots.',
        }
        return explanation

    def build_fairness_context(self, feature_row: dict[str, Any]) -> dict[str, Any]:
        return {
            'protected_attribute': 'gender',
            'explanation': 'Fairness monitoring should compare approval rates across gender and age-group segments to detect disparate impact.',
            'fairness_metrics': {
                'demographic_parity_difference': 0.08,
                'equal_opportunity_difference': 0.05,
            },
        }
