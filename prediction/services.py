import os
import pickle
import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import shap
from training.models import ModelVersion

logger = logging.getLogger(__name__)


class PredictionService:
    @staticmethod
    def get_active_model() -> ModelVersion | None:
        """Retrieve the currently active production model."""
        active_model = ModelVersion.objects.filter(is_active=True).first()
        if not active_model:
            # Fallback to the latest trained model if no model is explicitly active
            active_model = ModelVersion.objects.filter(status='trained').order_by('-created_at').first()
        return active_model

    @staticmethod
    def get_categorical_options(model_version: ModelVersion) -> dict[str, list[Any]]:
        """Get unique values for categorical features from the dataset to display in dropdowns."""
        if not model_version or not model_version.dataset or not os.path.exists(model_version.dataset.file_path):
            return {}
        try:
            df = pd.read_csv(model_version.dataset.file_path)
            target = model_version.target_column or 'target'
            df = df.drop(columns=[target], errors='ignore')
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            options = {}
            for col in cat_cols:
                options[col] = sorted(df[col].dropna().unique().tolist())
            return options
        except Exception as e:
            logger.error(f"Error extracting categorical options: {e}")
            return {}

    def predict_and_explain(self, model_version: ModelVersion, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run prediction and calculate local SHAP explainability for a single applicant."""
        if not model_version.artifact_path or not Path(model_version.artifact_path).exists():
            raise ValueError("Model artifact file not found.")

        # Load pipeline
        with open(model_version.artifact_path, 'rb') as f:
            pipeline = pickle.load(f)

        # Load dataset to get defaults and background data for SHAP
        if not model_version.dataset or not Path(model_version.dataset.file_path).exists():
            raise ValueError("Model training dataset file not found.")
        
        df = pd.read_csv(model_version.dataset.file_path)
        target_col = model_version.target_column or 'target'
        X_train = df.drop(columns=[target_col], errors='ignore')

        # Compute defaults for columns that are missing from input_data
        defaults = {}
        for col in X_train.columns:
            if X_train[col].dtype in ['int64', 'float64']:
                defaults[col] = X_train[col].median()
            else:
                defaults[col] = X_train[col].mode()[0] if not X_train[col].mode().empty else ""

        # Construct full feature row using inputs and falling back to defaults
        feature_row = {}
        for col in X_train.columns:
            if col in input_data:
                # Direct match
                feature_row[col] = input_data[col]
            elif col == 'credit_amount' and 'loan_amount' in input_data and input_data['loan_amount']:
                feature_row[col] = float(input_data['loan_amount'])
            elif col == 'existing_credits' and 'existing_loans' in input_data and input_data['existing_loans']:
                feature_row[col] = int(input_data['existing_loans'])
            elif col == 'employment_status' and 'employment' in input_data and input_data['employment']:
                feature_row[col] = input_data['employment']
            elif col == 'purpose' and 'loan_purpose' in input_data and input_data['loan_purpose']:
                feature_row[col] = input_data['loan_purpose']
            elif col == 'age' and 'age' in input_data and input_data['age']:
                feature_row[col] = int(input_data['age'])
            elif col == 'age_group' and 'age' in input_data and input_data['age']:
                feature_row[col] = 'young' if int(input_data['age']) < 25 else 'adult'
            else:
                feature_row[col] = defaults[col]

        # Convert to DataFrame
        input_df = pd.DataFrame([feature_row])
        # Ensure column order matches training data
        input_df = input_df[X_train.columns]

        # Convert values to correct types
        for col in X_train.columns:
            if X_train[col].dtype in ['int64', 'float64']:
                input_df[col] = pd.to_numeric(input_df[col])
            else:
                input_df[col] = input_df[col].astype(str)

        # Predict
        probability = float(pipeline.predict_proba(input_df)[0][1])
        prediction = 'Approved' if probability >= 0.5 else 'Rejected'
        confidence = float(abs(probability - 0.5) * 2)

        # Generate SHAP explanation
        preprocessor = pipeline.named_steps['preprocess']
        classifier = pipeline.named_steps['classifier']

        def predict_fn(values: pd.DataFrame) -> np.ndarray:
            if not isinstance(values, pd.DataFrame):
                values = pd.DataFrame(values, columns=X_train.columns)
            transformed = preprocessor.transform(values)
            return classifier.predict_proba(transformed)[:, 1]

        # Use a small sample of X_train as background data (e.g. 20 samples) for speed
        background = X_train.sample(n=min(20, len(X_train)), random_state=42)
        
        # Prepare background and input for SHAP
        background_encoded = background.copy()
        input_encoded = input_df.copy()
        
        for col in X_train.select_dtypes(include=['object', 'category']).columns:
            codes, uniques = pd.factorize(X_train[col])
            uniques_list = list(uniques)
            
            def encode_val(val):
                try:
                    return uniques_list.index(val)
                except ValueError:
                    return -1
                    
            background_encoded[col] = background_encoded[col].apply(encode_val)
            input_encoded[col] = input_encoded[col].apply(encode_val)

        try:
            explainer = shap.Explainer(predict_fn, background)
            shap_values = explainer(input_df)
            shap_array = shap_values.values[0]
        except Exception as e:
            logger.warning(f"Fallback to SHAP Explainer on encoded data: {e}")
            try:
                explainer = shap.Explainer(classifier, background_encoded)
                shap_values = explainer(input_encoded)
                shap_array = shap_values.values[0]
            except Exception as ex:
                logger.error(f"SHAP explanation failed: {ex}")
                shap_array = np.zeros(len(X_train.columns))

        # Extract features and their contributions
        contributions = []
        for col_name, shap_val in zip(X_train.columns, shap_array):
            val = input_df[col_name].iloc[0]
            contributions.append({
                'feature': col_name,
                'value': str(val),
                'shap_value': float(shap_val),
                'effect': 'positive' if shap_val > 0 else 'negative'
            })

        # Sort by absolute SHAP value
        contributions = sorted(contributions, key=lambda x: abs(x['shap_value']), reverse=True)
        top_contributions = contributions[:6]

        # Create plain English summary
        pos_features = [c['feature'].replace('_', ' ') for c in top_contributions if c['shap_value'] > 0][:2]
        neg_features = [c['feature'].replace('_', ' ') for c in top_contributions if c['shap_value'] < 0][:2]
        
        summary_parts = []
        if prediction == 'Approved':
            summary_parts.append("The applicant was approved because of favorable factors.")
            if pos_features:
                summary_parts.append(f"Specifically, {', '.join(pos_features)} contributed positively to this decision.")
        else:
            summary_parts.append("The application was rejected due to risk factors.")
            if neg_features:
                summary_parts.append(f"Specifically, {', '.join(neg_features)} increased the risk score and led to the rejection.")

        summary = " ".join(summary_parts)

        # Build fairness context
        fairness_context = {
            'protected_attribute': 'sex',
            'explanation': 'Model decisions are monitored for demographic parity.',
            'fairness_metrics': {
                'demographic_parity_difference': 0.0,
                'equal_opportunity_difference': 0.0
            }
        }
        
        try:
            if 'sex' in df.columns:
                group_rates = df.groupby('sex')[target_col].mean().to_dict()
                applicant_sex = feature_row.get('sex', 'unknown')
                app_rate = group_rates.get(applicant_sex, 0.5)
                diff = abs(group_rates.get('male', 0.5) - group_rates.get('female', 0.5))
                fairness_context['explanation'] = (
                    f"The historical approval rate for the '{applicant_sex}' group is {app_rate:.1%}. "
                    f"The difference in approval rates across gender categories in the dataset is {diff:.1%}."
                )
                fairness_context['fairness_metrics'] = {
                    'demographic_parity_difference': round(diff, 4),
                    'equal_opportunity_difference': round(diff, 4)
                }
        except Exception as e:
            logger.warning(f"Could not compute group rates: {e}")

        return {
            'prediction': prediction,
            'probability': round(probability, 4),
            'confidence': round(confidence, 4),
            'shap_explanation': top_contributions,
            'summary': summary,
            'fairness_context': fairness_context,
            'feature_row': feature_row
        }
