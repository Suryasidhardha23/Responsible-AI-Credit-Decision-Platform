"""
Bias Mitigation Service for Responsible AI Credit Decision Platform.

Implements two bias mitigation strategies:
  1. Reweighting  — Preprocessing: Adjusts sample weights before training so
     that under-represented subgroups receive a higher weight, enforcing
     demographic parity without removing features.
  2. Threshold Optimization — Post-processing: Uses Fairlearn's
     ThresholdOptimizer to select group-specific decision thresholds that
     satisfy a fairness constraint at minimal accuracy cost.
"""

import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


ALGORITHM_CLASS_MAP = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

ALGORITHM_DEFAULT_PARAMS = {
    "logistic_regression": {"max_iter": 1000, "random_state": 42},
    "decision_tree": {"random_state": 42, "max_depth": 10},
    "random_forest": {"n_estimators": 100, "random_state": 42},
    "gradient_boosting": {"n_estimators": 100, "random_state": 42, "learning_rate": 0.1},
}


class OptimizedWrapper:
    """Wrapper class for threshold optimizer that can be pickled."""
    def __init__(self, prep, opt):
        self._prep = prep
        self._opt = opt

    def predict(self, X, sensitive_features=None):
        Xt = self._prep.transform(X)
        if sensitive_features is None:
            # Predict without sensitive features (fallback)
            return self._opt.estimator_.predict(Xt)
        return self._opt.predict(Xt, sensitive_features=sensitive_features)

    def predict_proba(self, X):
        Xt = self._prep.transform(X)
        return self._opt.estimator_.predict_proba(Xt)


class BiasMitigationService:
    """Implements reweighting and threshold optimisation bias-mitigation strategies."""

    def __init__(self, artifact_dir: str | None = None):
        self.artifact_dir = Path(artifact_dir or "media/models")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = 42

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mitigate(
        self,
        dataset_path: str,
        target_column: str,
        protected_attribute: str,
        algorithm: str = "logistic_regression",
        strategy: str = "reweighting",
    ) -> dict[str, Any]:
        """
        Train a fairness-aware model and return comparison metrics vs. a
        baseline model trained without any fairness adjustments.

        Args:
            dataset_path: Path to the CSV data file.
            target_column: Name of the binary target column (0/1).
            protected_attribute: Name of the sensitive attribute column.
            algorithm: One of logistic_regression, decision_tree,
                       random_forest, gradient_boosting.
            strategy: "reweighting" or "threshold_optimization".

        Returns:
            Dictionary with baseline_metrics, mitigated_metrics,
            fairness_comparison, and artifact_path.
        """
        df = pd.read_csv(dataset_path)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")
        if protected_attribute not in df.columns:
            raise ValueError(f"Protected attribute '{protected_attribute}' not found.")

        y = df[target_column]
        # Keep sensitive attribute in X so we can compute group metrics,
        # but exclude it from the model's feature set below.
        X = df.drop(columns=[target_column])
        sensitive = X[protected_attribute]

        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, sensitive,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y,
        )

        # Build preprocessor on training features (without sensitive column)
        X_train_no_sens = X_train.drop(columns=[protected_attribute], errors="ignore")
        X_test_no_sens = X_test.drop(columns=[protected_attribute], errors="ignore")

        preprocessor = self._build_preprocessor(X_train_no_sens)

        # Fit baseline
        baseline_pipeline = self._fit_baseline(
            X_train_no_sens, y_train, preprocessor, algorithm
        )
        baseline_metrics = self._evaluate(
            baseline_pipeline, X_test_no_sens, y_test, s_test, protected_attribute
        )

        # Fit mitigated model
        if strategy == "reweighting":
            mitigated_pipeline, weights = self._reweighting(
                X_train_no_sens, y_train, s_train, preprocessor, algorithm
            )
        elif strategy == "threshold_optimization":
            mitigated_pipeline, weights = self._threshold_optimization(
                baseline_pipeline, X_train_no_sens, y_train, s_train,
                X_test_no_sens, y_test, s_test, preprocessor, algorithm
            )
        else:
            raise ValueError(f"Unknown strategy '{strategy}'.")

        mitigated_metrics = self._evaluate(
            mitigated_pipeline, X_test_no_sens, y_test, s_test, protected_attribute
        )

        # Save mitigated model
        artifact_path = self._save_pipeline(mitigated_pipeline)

        comparison = self._build_comparison(baseline_metrics, mitigated_metrics)

        logger.info(
            "Bias mitigation (%s) completed. DPD: %.4f → %.4f",
            strategy,
            baseline_metrics.get("demographic_parity_difference", 0),
            mitigated_metrics.get("demographic_parity_difference", 0),
        )

        return {
            "strategy": strategy,
            "protected_attribute": protected_attribute,
            "algorithm": algorithm,
            "baseline_metrics": baseline_metrics,
            "mitigated_metrics": mitigated_metrics,
            "comparison": comparison,
            "artifact_path": str(artifact_path),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    num_cols,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]),
                    cat_cols,
                ),
            ]
        )

    def _build_pipeline(
        self, preprocessor: ColumnTransformer, algorithm: str
    ) -> Pipeline:
        cls = ALGORITHM_CLASS_MAP.get(algorithm, LogisticRegression)
        params = ALGORITHM_DEFAULT_PARAMS.get(algorithm, {"random_state": 42})
        return Pipeline(steps=[("preprocess", preprocessor), ("classifier", cls(**params))])

    def _fit_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        preprocessor: ColumnTransformer,
        algorithm: str,
    ) -> Pipeline:
        pipeline = self._build_pipeline(preprocessor, algorithm)
        pipeline.fit(X_train, y_train)
        return pipeline

    def _compute_sample_weights(
        self,
        y: pd.Series,
        sensitive: pd.Series,
    ) -> np.ndarray:
        """
        Reweighting: w_i = P(Y) * P(A) / P(Y, A).
        Each group/label cell receives a weight inversely proportional to its
        prevalence so that all group × outcome cells contribute equally.
        """
        n = len(y)
        p_y = y.mean()
        weights = np.ones(n)
        for group_val in sensitive.unique():
            mask = sensitive == group_val
            p_a = mask.mean()
            for label in [0, 1]:
                cell_mask = mask & (y == label)
                p_ya = cell_mask.mean()
                p_y_label = (y == label).mean()
                if p_ya > 0:
                    w = (p_y_label * p_a) / p_ya
                    weights[cell_mask] = w
        return weights

    def _reweighting(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        s_train: pd.Series,
        preprocessor: ColumnTransformer,
        algorithm: str,
    ):
        weights = self._compute_sample_weights(y_train, s_train)
        pipeline = self._build_pipeline(preprocessor, algorithm)
        # scikit-learn pipelines support sample_weight via __classifier__sample_weight
        pipeline.fit(X_train, y_train, **{"classifier__sample_weight": weights})
        return pipeline, weights

    def _threshold_optimization(
        self,
        baseline_pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        s_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        s_test: pd.Series,
        preprocessor: ColumnTransformer,
        algorithm: str,
    ):
        """
        Threshold Optimization via Fairlearn ThresholdOptimizer.
        We wrap the baseline's predict_proba in a compatible interface.
        Falls back to reweighting if Fairlearn is unavailable.
        """
        try:
            from fairlearn.postprocessing import ThresholdOptimizer
            from fairlearn.metrics import demographic_parity_difference

            # Fit a fresh estimator on pre-processed features for ThresholdOptimizer
            cls = ALGORITHM_CLASS_MAP.get(algorithm, LogisticRegression)
            params = ALGORITHM_DEFAULT_PARAMS.get(algorithm, {"random_state": 42})
            estimator = cls(**params)

            X_train_t = preprocessor.fit_transform(X_train)
            estimator.fit(X_train_t, y_train)

            optimizer = ThresholdOptimizer(
                estimator=estimator,
                constraints="demographic_parity",
                predict_method="predict_proba",
                objective="accuracy_score",
            )
            optimizer.fit(X_train_t, y_train, sensitive_features=s_train)

            # Wrap in a callable pipeline-like object
            wrapper = OptimizedWrapper(preprocessor, optimizer)
            return wrapper, None

        except ImportError:
            logger.warning("Fairlearn not available; falling back to reweighting.")
            return self._reweighting(X_train, y_train, s_train, preprocessor, algorithm)
        except Exception as e:
            logger.warning("ThresholdOptimizer failed (%s); falling back to reweighting.", e)
            return self._reweighting(X_train, y_train, s_train, preprocessor, algorithm)

    def _evaluate(
        self,
        pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        s_test: pd.Series,
        protected_attribute: str,
    ) -> dict[str, Any]:
        """Compute accuracy, ROC-AUC, and group-level fairness metrics."""
        try:
            y_pred = pipeline.predict(X_test)
        except Exception:
            # OptimizedWrapper may need sensitive_features
            y_pred = pipeline.predict(X_test, sensitive_features=s_test)

        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            roc = float(roc_auc_score(y_test, y_proba))
        except Exception:
            roc = 0.0

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))

        # Fairness metrics
        group_rates = {}
        for val in s_test.unique():
            mask = s_test == val
            if mask.sum() > 0:
                group_rates[str(val)] = float(y_pred[mask].mean())

        values = list(group_rates.values())
        dpd = float(max(values) - min(values)) if len(values) >= 2 else 0.0
        dpr = float(min(values) / max(values)) if (len(values) >= 2 and max(values) > 0) else 1.0

        # Group accuracy
        group_accuracy = {}
        for val in s_test.unique():
            mask = s_test == val
            if mask.sum() > 0:
                group_accuracy[str(val)] = float(accuracy_score(y_test[mask], y_pred[mask]))

        return {
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "roc_auc": roc,
            "demographic_parity_difference": dpd,
            "disparate_impact_ratio": dpr,
            "group_selection_rates": group_rates,
            "group_accuracy": group_accuracy,
        }

    def _build_comparison(
        self,
        baseline: dict[str, Any],
        mitigated: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute deltas between baseline and mitigated models."""
        keys = ["accuracy", "f1_score", "roc_auc", "demographic_parity_difference", "disparate_impact_ratio"]
        deltas = {}
        for k in keys:
            b_val = baseline.get(k, 0.0)
            m_val = mitigated.get(k, 0.0)
            deltas[k] = {
                "baseline": round(b_val, 4),
                "mitigated": round(m_val, 4),
                "delta": round(m_val - b_val, 4),
            }
        # Fairness improved = DPD decreased
        dpd_b = baseline.get("demographic_parity_difference", 0.0)
        dpd_m = mitigated.get("demographic_parity_difference", 0.0)
        fairness_improved = dpd_m < dpd_b
        accuracy_delta = mitigated.get("accuracy", 0) - baseline.get("accuracy", 0)

        return {
            "metric_deltas": deltas,
            "fairness_improved": fairness_improved,
            "dpd_reduction": round(dpd_b - dpd_m, 4),
            "accuracy_cost": round(-accuracy_delta, 4),
            "trade_off_summary": (
                f"Fairness improved (DPD reduced by {dpd_b - dpd_m:.4f}) "
                f"at an accuracy cost of {abs(accuracy_delta):.4f}."
                if fairness_improved
                else f"Fairness did not improve significantly (DPD Δ = {dpd_m - dpd_b:.4f})."
            ),
        }

    def _save_pipeline(self, pipeline) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.artifact_dir / f"mitigated_model_{ts}.pkl"
        with path.open("wb") as f:
            pickle.dump(pipeline, f)
        logger.info("Saved mitigated model to %s", path)
        return path
