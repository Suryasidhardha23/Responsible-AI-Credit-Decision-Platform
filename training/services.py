import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


class ModelTrainingService:
    """Comprehensive model training service with multiple algorithms and hyperparameter optimization."""

    ALGORITHM_CONFIGS = {
        'logistic_regression': {
            'model_class': LogisticRegression,
            'default_params': {'max_iter': 1000, 'random_state': 42, 'n_jobs': -1},
            'search_params': {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['lbfgs', 'liblinear'],
            },
        },
        'decision_tree': {
            'model_class': DecisionTreeClassifier,
            'default_params': {'random_state': 42, 'max_depth': 10},
            'search_params': {
                'classifier__max_depth': [5, 10, 15, 20],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
            },
        },
        'random_forest': {
            'model_class': RandomForestClassifier,
            'default_params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
            'search_params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 15, 20, None],
                'classifier__min_samples_split': [2, 5],
            },
        },
        'gradient_boosting': {
            'model_class': GradientBoostingClassifier,
            'default_params': {'n_estimators': 100, 'random_state': 42, 'learning_rate': 0.1},
            'search_params': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__max_depth': [3, 5, 7],
            },
        },
    }

    def __init__(self, artifact_dir: str | None = None):
        self.artifact_dir = Path(artifact_dir or 'media/models')
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = 42

    def train_model(
        self,
        dataset_path: str,
        target_column: str,
        algorithm: str = 'logistic_regression',
        hyperparameters: dict | None = None,
        perform_hyperparameter_tuning: bool = False,
        cross_validation_folds: int = 5,
    ) -> dict:
        """Train a model with comprehensive metrics and optional hyperparameter optimization."""

        logger.info(f"Starting training with {algorithm} on {dataset_path}")

        try:
            # Load and validate data
            dataframe = pd.read_csv(dataset_path)
            if target_column not in dataframe.columns:
                raise ValueError(f'Target column {target_column} not found')

            # Separate features and target
            y = dataframe[target_column]
            X = dataframe.drop(columns=[target_column])

            # Identify feature types
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_features = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

            logger.info(f"Categorical features: {categorical_features}, Numerical features: {numerical_features}")

            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        'num',
                        Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]),
                        numerical_features,
                    ),
                    (
                        'cat',
                        Pipeline([
                            ('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                        ]),
                        categorical_features,
                    ),
                ]
            )

            # Get algorithm configuration
            if algorithm not in self.ALGORITHM_CONFIGS:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            config = self.ALGORITHM_CONFIGS[algorithm]
            model_class = config['model_class']
            params = hyperparameters or config['default_params']

            # Create pipeline
            pipeline = Pipeline(steps=[('preprocess', preprocessor), ('classifier', model_class(**params))])

            # Train-test split with stratification
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )

            # Hyperparameter tuning if requested
            if perform_hyperparameter_tuning:
                logger.info(f"Performing hyperparameter tuning for {algorithm}")
                grid_search = GridSearchCV(pipeline, config['search_params'], cv=5, n_jobs=-1, verbose=1)
                grid_search.fit(X_train, y_train)
                pipeline = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
                params = grid_search.best_params_

            else:
                pipeline.fit(X_train, y_train)

            # Make predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            # Calculate comprehensive metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

            # Cross-validation
            cv_results = self._cross_validate(pipeline, X_train, y_train, cross_validation_folds)

            # Feature importance (if available)
            feature_importance = self._get_feature_importance(pipeline, X_train, categorical_features, numerical_features)

            # Save model and preprocessor
            model_path, preprocessor_path = self._save_artifacts(pipeline, preprocessor)

            # Prepare confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_dict = {
                'true_negatives': int(cm[0, 0]),
                'false_positives': int(cm[0, 1]),
                'false_negatives': int(cm[1, 0]),
                'true_positives': int(cm[1, 1]),
            }

            # Class distribution
            class_dist = {'class_0': int((y == 0).sum()), 'class_1': int((y == 1).sum())}

            result = {
                'metrics': metrics,
                'cv_results': cv_results,
                'artifact_path': str(model_path),
                'preprocessing_pipeline_path': str(preprocessor_path),
                'confusion_matrix': cm_dict,
                'class_distribution': class_dist,
                'feature_importance': feature_importance,
                'feature_list': X.columns.tolist(),
                'hyperparameters': params,
            }

            logger.info(f"Training completed successfully. Accuracy: {metrics['accuracy']:.4f}")
            return result

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

    def _calculate_metrics(self, y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        }

        if len(set(y_test)) > 1:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))

            # PR-AUC
            precision_curve, recall_curve_vals, _ = precision_recall_curve(y_test, y_pred_proba)
            metrics['pr_auc'] = float(auc(recall_curve_vals, precision_curve))
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0

        return metrics

    def _cross_validate(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray, folds: int = 5) -> dict:
        """Perform cross-validation."""
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=['accuracy', 'f1', 'roc_auc', 'precision', 'recall'], n_jobs=-1)

        return {
            'accuracy_mean': float(scores['test_accuracy'].mean()),
            'accuracy_std': float(scores['test_accuracy'].std()),
            'f1_mean': float(scores['test_f1'].mean()),
            'f1_std': float(scores['test_f1'].std()),
            'roc_auc_mean': float(scores['test_roc_auc'].mean()),
            'roc_auc_std': float(scores['test_roc_auc'].std()),
            'precision_mean': float(scores['test_precision'].mean()),
            'precision_std': float(scores['test_precision'].std()),
            'recall_mean': float(scores['test_recall'].mean()),
            'recall_std': float(scores['test_recall'].std()),
        }

    def _get_feature_importance(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        categorical_features: list,
        numerical_features: list,
    ) -> dict:
        """Extract feature importance if available."""
        feature_importance_dict = {}

        try:
            classifier = pipeline.named_steps['classifier']

            # For tree-based models
            if hasattr(classifier, 'feature_importances_'):
                preprocessor = pipeline.named_steps['preprocess']
                feature_names = []

                for name, transformer, features in preprocessor.transformers_:
                    if name == 'num':
                        feature_names.extend(features)
                    elif name == 'cat':
                        # Get one-hot encoded feature names
                        onehot = transformer.named_steps['onehot']
                        cat_features = onehot.get_feature_names_out(features)
                        feature_names.extend(cat_features)

                importances = classifier.feature_importances_
                for fname, imp in zip(feature_names, importances):
                    feature_importance_dict[fname] = float(imp)

                # Sort by importance
                feature_importance_dict = dict(
                    sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
                )
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

        return feature_importance_dict

    def _save_artifacts(self, pipeline: Pipeline, preprocessor: ColumnTransformer) -> tuple:
        """Save trained model and preprocessor."""
        model_path = self.artifact_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        preprocessor_path = self.artifact_dir / f"preprocessor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        with model_path.open('wb') as f:
            pickle.dump(pipeline, f)

        with preprocessor_path.open('wb') as f:
            pickle.dump(preprocessor, f)

        logger.info(f"Artifacts saved: {model_path}, {preprocessor_path}")
        return model_path, preprocessor_path

    @staticmethod
    def load_model(model_path: str) -> Pipeline:
        """Load a saved model."""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_preprocessor(preprocessor_path: str) -> ColumnTransformer:
        """Load a saved preprocessor."""
        with open(preprocessor_path, 'rb') as f:
            return pickle.load(f)
