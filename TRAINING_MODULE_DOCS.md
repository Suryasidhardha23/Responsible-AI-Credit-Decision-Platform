# Training Module Documentation

## Overview

The training module implements a production-grade ML model training and comparison system with support for multiple algorithms, hyperparameter optimization, and comprehensive evaluation metrics.

## Architecture

### Services

#### ModelTrainingService (`training/services.py`)

Handles all model training operations with the following capabilities:

- **Multi-Algorithm Support**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Hyperparameter Optimization**: GridSearchCV with algorithm-specific configurations
- **Cross-Validation**: Stratified K-fold with configurable folds
- **Preprocessing Pipeline**: Automatic feature scaling, encoding, and imputation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
- **Feature Importance**: Extracted from tree-based models

**Key Methods:**
```python
# Train a model
result = service.train_model(
    dataset_path='path/to/data.csv',
    target_column='target',
    algorithm='random_forest',
    perform_hyperparameter_tuning=True,
    cross_validation_folds=5
)

# Load saved model
model = ModelTrainingService.load_model('path/to/model.pkl')
preprocessor = ModelTrainingService.load_preprocessor('path/to/preprocessor.pkl')
```

**Returned Result Structure:**
```python
{
    'metrics': {
        'accuracy': float,
        'precision': float,
        'recall': float,
        'f1_score': float,
        'roc_auc': float,
        'pr_auc': float,
    },
    'cv_results': {
        'accuracy_mean': float,
        'accuracy_std': float,
        'f1_mean': float,
        # ... other CV metrics
    },
    'artifact_path': str,  # Path to trained model pickle
    'preprocessing_pipeline_path': str,
    'confusion_matrix': {
        'true_negatives': int,
        'false_positives': int,
        'false_negatives': int,
        'true_positives': int,
    },
    'class_distribution': {
        'class_0': int,
        'class_1': int,
    },
    'feature_importance': {
        'feature_name': float,  # Top 20 features
    },
    'feature_list': [str],  # All features used
    'hyperparameters': dict,  # Final hyperparameters used
}
```

#### ModelComparisonService (`training/comparison_service.py`)

Handles comparison and ranking of multiple model versions.

**Key Methods:**
```python
# Compare multiple models
comparison = ModelComparisonService.compare_models(
    model_versions=[model1, model2, model3],
    name='Model Comparison',
    created_by=user
)

# Get summary
summary = ModelComparisonService.get_comparison_summary(comparison)

# Compare two models
differences = ModelComparisonService.get_metric_differences(best_model, other_model)
```

**Ranking Algorithm:**
- Uses weighted scoring based on metrics
- Default weights:
  - ROC-AUC: 30%
  - F1 Score: 25%
  - Accuracy: 20%
  - Precision: 15%
  - Recall: 10%

### Models

#### ModelVersion
Main model tracking entity with the following fields:

```python
class ModelVersion(models.Model):
    # Identity
    version: CharField  # Unique version identifier (e.g., 'v1.0')
    name: CharField
    algorithm: CharField  # logistic_regression, decision_tree, random_forest, gradient_boosting
    status: CharField  # draft, trained, evaluated, deployed, archived
    
    # Relationships
    dataset: ForeignKey(UploadedDataset)
    created_by: ForeignKey(UserProfile)
    
    # Metrics
    accuracy: FloatField
    precision: FloatField
    recall: FloatField
    f1_score: FloatField
    roc_auc: FloatField
    pr_auc: FloatField
    cv_score_mean: FloatField
    cv_score_std: FloatField
    
    # Artifacts
    artifact_path: CharField  # Path to pickle file
    preprocessing_pipeline_path: CharField
    
    # Configuration
    hyperparameters: JSONField  # Dict of hyperparameters
    target_column: CharField
    feature_list: JSONField  # List of features used
    
    # Results
    confusion_matrix: JSONField
    class_distribution: JSONField
    feature_importance: JSONField
    
    # Deployment
    is_active: BooleanField  # Only one model can be active
    deployment_status: CharField  # active, inactive
    
    # Methods
    def activate()  # Make this model the active production model
    def deactivate()  # Remove from production
```

#### Supporting Models

**TrainingRun**: Track individual training runs for experiment tracking
- model_version (FK)
- run_name
- status (running, completed, failed)
- training_time_seconds
- validation_metrics (JSON)

**ModelComparison**: Store comparisons between models
- name
- dataset (FK)
- created_by (FK)
- created_at
- comparison_result (JSON)

**ModelComparisonDetail**: Individual model metrics within a comparison
- comparison (FK)
- model_version (FK)
- metrics (JSON)
- rank (integer)

## REST API Endpoints

### Training Endpoints

#### Train Model
```
POST /api/train/
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
    "dataset_id": 1,
    "version": "v1.0",
    "name": "Credit Decision Model v1",
    "algorithm": "random_forest",
    "target_column": "approval_status",
    "perform_hyperparameter_tuning": true,
    "cross_validation_folds": 5
}

Response (201):
{
    "message": "Model trained successfully",
    "model": {
        "id": 1,
        "version": "v1.0",
        "accuracy": 0.88,
        "f1_score": 0.87,
        ...
    }
}
```

#### Compare Models
```
POST /api/compare/
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
    "model_version_ids": [1, 2, 3],
    "name": "Model Comparison",
    "notes": "Optional notes about the comparison"
}

Response (201):
{
    "message": "Models compared successfully",
    "comparison": {
        "id": 1,
        "name": "Model Comparison",
        "comparison_result": {
            "ranked_models": ["v1.1", "v1.0"],
            "metrics_by_model": {...}
        },
        "details": [...]
    }
}
```

#### List Models
```
GET /api/models/
Authorization: Bearer <JWT_TOKEN>

Response (200):
{
    "count": 5,
    "next": null,
    "previous": null,
    "results": [
        {
            "id": 1,
            "version": "v1.0",
            "name": "Credit Model",
            "algorithm": "random_forest",
            "accuracy": 0.88,
            ...
        }
    ]
}
```

#### Activate Model
```
POST /api/models/<id>/activate/
Authorization: Bearer <JWT_TOKEN>

Response (200):
{
    "message": "Model v1.0 activated successfully",
    "model": {...}
}
```

#### Get Active Model
```
GET /api/active-model/
Authorization: Bearer <JWT_TOKEN>

Response (200):
{
    "active_model": {
        "id": 1,
        "version": "v1.0",
        "name": "Credit Model",
        ...
    }
}
```

## Usage Examples

### Training a Model

```python
from training.services import ModelTrainingService
from training.models import ModelVersion
from dataset_manager.models import UploadedDataset
from accounts.models import UserProfile

# Load dataset
dataset = UploadedDataset.objects.get(id=1)
user = UserProfile.objects.get(id=1)

# Train model
service = ModelTrainingService()
result = service.train_model(
    dataset_path=dataset.file_path,
    target_column='approval',
    algorithm='random_forest',
    perform_hyperparameter_tuning=True,
    cross_validation_folds=5
)

# Save to database
model = ModelVersion.objects.create(
    version='v1.0',
    name='Credit Decision Model',
    algorithm='random_forest',
    created_by=user,
    dataset=dataset,
    accuracy=result['metrics']['accuracy'],
    f1_score=result['metrics']['f1_score'],
    roc_auc=result['metrics']['roc_auc'],
    # ... other metrics
    artifact_path=result['artifact_path'],
    preprocessing_pipeline_path=result['preprocessing_pipeline_path'],
    hyperparameters=result['hyperparameters'],
)

# Activate model for production
model.activate()
```

### Comparing Models

```python
from training.comparison_service import ModelComparisonService
from training.models import ModelVersion

# Get models to compare
models = ModelVersion.objects.filter(version__in=['v1.0', 'v1.1', 'v2.0'])

# Create comparison
comparison = ModelComparisonService.compare_models(
    model_versions=list(models),
    name='Model v1 vs v2 Comparison',
    created_by=user
)

# Get results
summary = ModelComparisonService.get_comparison_summary(comparison)
print(f"Best model: {summary['best_model']['version']}")

for model_info in summary['models']:
    print(f"Rank {model_info['rank']}: {model_info['version']} ({model_info['algorithm']})")
    print(f"  Accuracy: {model_info['metrics']['accuracy']:.4f}")
```

### Making Predictions with a Model

```python
from training.services import ModelTrainingService
import pandas as pd

# Load the active model
active_model = ModelVersion.objects.get(is_active=True)
model = ModelTrainingService.load_model(active_model.artifact_path)

# Prepare data
data = pd.DataFrame({
    'age': [35],
    'income': [50000],
    'credit_history': ['good'],
})

# Make prediction
prediction = model.predict(data)[0]
probability = model.predict_proba(data)[0][1]

print(f"Prediction: {prediction}")
print(f"Approval Probability: {probability:.2%}")
```

## Algorithm Configurations

### Logistic Regression
- Default: max_iter=1000, C=1.0
- Tuning: C values [0.001, 0.01, 0.1, 1, 10], penalty and solver options

### Decision Tree
- Default: max_depth=10
- Tuning: max_depth [5, 10, 15, 20], min_samples_split, min_samples_leaf

### Random Forest
- Default: n_estimators=100
- Tuning: n_estimators [50, 100, 200], max_depth, min_samples_split

### Gradient Boosting
- Default: n_estimators=100, learning_rate=0.1
- Tuning: n_estimators, learning_rate [0.01, 0.05, 0.1], max_depth

## Tests

Run the comprehensive test suite:

```bash
# Run all training tests
pytest training/tests.py

# Run specific test class
pytest training/tests.py::ModelTrainingServiceTestCase

# Run with verbose output
pytest training/tests.py -v

# Run with coverage
pytest training/tests.py --cov=training
```

## Performance Considerations

- **Model Training**: Typically 5-30 seconds depending on algorithm and dataset size
- **Hyperparameter Tuning**: Can take 2-5 minutes with GridSearchCV
- **Cross-Validation**: Scales linearly with number of folds
- **Feature Importance**: Only computed for tree-based models, negligible overhead

## Security Notes

- Only administrators can train models or activate deployments
- Model artifacts are stored on disk; ensure proper access controls
- Feature importance may reveal sensitive business logic
- Predictions should always use the active model to maintain consistency

## Future Enhancements

1. Distributed training with Dask
2. GPU acceleration with CuML
3. Automated retraining pipelines
4. A/B testing framework
5. Model explainability with SHAP (Phase 4)
6. Fairness auditing with Fairlearn (Phase 4)
