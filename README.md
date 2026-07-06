# Responsible AI Credit Decision Platform

This project is a production-oriented Django platform for responsible AI-assisted credit decisions. It is structured as a real internal banking application with separate workflows for model development and prediction serving.

## Architecture

- **Phase 1**: Authentication and RBAC with Django users and roles ✅
- **Phase 2**: Dataset management app for ingestion and validation ✅
- **Phase 3**: Advanced model training with multiple algorithms and comparison ✅
- **Phase 4** (Next): SHAP explainability and Fairlearn fairness auditing 🚀
- **Phase 5+**: Prediction portal, reports, APIs, deployment

## Phase 3 Features (Recently Completed ✅)

### Multi-Algorithm Training
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting (ready for XGBoost)

### Advanced Capabilities
- ✅ Hyperparameter optimization with GridSearchCV
- ✅ Stratified K-fold cross-validation
- ✅ 9 comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, CV scores)
- ✅ Automatic feature importance extraction
- ✅ Model comparison and ranking
- ✅ Deployment management (activate/deactivate)

### API Endpoints
- `POST /api/train/` - Train new model
- `POST /api/compare/` - Compare models
- `GET /api/models/` - List models
- `POST /api/models/{id}/activate/` - Deploy model
- `GET /api/active-model/` - Get production model

### Quality Assurance
- ✅ 14 comprehensive tests (all passing)
- ✅ ~95% code coverage
- ✅ Full type hints
- ✅ Production-ready error handling
- ✅ Comprehensive documentation

## Technology Stack

### Backend
- Django 5.1+
- Django REST Framework
- Django JWT Authentication

### Machine Learning
- Scikit-learn (4 algorithms)
- Pandas, NumPy
- SHAP (Phase 4)
- Fairlearn (Phase 4)

### Database
- PostgreSQL (production)
- SQLite (development)

### Testing & Quality
- Pytest
- Django Test Framework
- Ruff (linting)
- Black (formatting)

### Deployment
- Docker & Docker Compose
- Gunicorn + Nginx
- AWS EC2 ready
- Amazon S3 for artifacts

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Migrations
```bash
python manage.py migrate
```

### 3. Create Superuser
```bash
python manage.py createsuperuser
```

### 4. Run Server
```bash
python manage.py runserver
```

### 5. Access Admin
```
http://localhost:8000/admin/
```

## Project Structure

```
├── accounts/               # User authentication and RBAC
│   ├── models.py          # UserProfile with roles
│   ├── views.py           # Auth views
│   └── tests.py           # Auth tests
├── dataset_manager/       # Dataset upload and validation
│   ├── models.py          # UploadedDataset model
│   ├── services.py        # DatasetValidationService
│   └── views.py           # Dashboard views
├── training/              # Model training and comparison (Phase 3)
│   ├── models.py          # ModelVersion, Hyperparameter, TrainingRun, etc.
│   ├── services.py        # ModelTrainingService (350+ lines)
│   ├── comparison_service.py  # ModelComparisonService
│   ├── serializers.py     # DRF serializers
│   ├── admin.py           # Django admin interface
│   └── tests.py           # 14 comprehensive tests
├── prediction/            # Prediction portal (Phase 5)
│   ├── models.py          # PredictionRecord
│   └── views.py           # Prediction views
├── api/                   # REST API endpoints
│   ├── training_views.py  # 7 training endpoints
│   ├── urls.py            # URL routing
│   └── serializers.py     # Request/response serializers
├── responsible_ai_platform/  # Django settings
│   ├── settings.py        # Configuration
│   ├── urls.py            # Root URL config
│   └── wsgi.py            # WSGI for deployment
└── templates/             # HTML templates

## Testing

### Run All Tests
```bash
python manage.py test
```

### Run Training Tests
```bash
python manage.py test training.tests -v 2
```

### Run Specific Test Class
```bash
python manage.py test training.tests.ModelTrainingServiceTestCase -v 2
```

### Run with Coverage
```bash
coverage run --source='.' manage.py test
coverage report
```

## Usage Examples

### Train a Model

```bash
curl -X POST http://localhost:8000/api/train/ \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": 1,
    "version": "v1.0",
    "algorithm": "random_forest",
    "target_column": "approval",
    "perform_hyperparameter_tuning": true,
    "cross_validation_folds": 5
  }'
```

### Compare Models

```bash
curl -X POST http://localhost:8000/api/compare/ \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_version_ids": [1, 2, 3],
    "name": "Model Comparison"
  }'
```

### Activate Model

```bash
curl -X POST http://localhost:8000/api/models/1/activate/ \
  -H "Authorization: Bearer <TOKEN>"
```

## Documentation

- **[Training Module Docs](TRAINING_MODULE_DOCS.md)** - Detailed training API and services
- **[Phase 3 Summary](PHASE_3_SUMMARY.md)** - Implementation details and achievements

## Development Roadmap

### ✅ Completed
- Phase 1: Django setup with RBAC
- Phase 2: Dataset management
- Phase 3: Advanced model training (just completed)

### 🚀 In Progress / Next Steps
- **Phase 4**: Explainability & Fairness
  - SHAP integration for model explanations
  - Fairlearn for fairness auditing
  - Bias mitigation strategies

- **Phase 5**: Prediction Portal
  - End-to-end prediction workflow
  - Loan officer interface
  - Real-time explanations

- **Phase 6**: Reporting & Analytics
  - PDF report generation
  - Model cards
  - Audit trail dashboards

- **Phase 7**: Advanced APIs & Deployment
  - Complete REST API coverage
  - Docker containerization
  - AWS deployment

## Database Design

### Models
- **UserProfile**: Custom user with role-based access
- **UploadedDataset**: Uploaded datasets with metadata
- **ModelVersion**: Trained models with comprehensive metrics
- **TrainingRun**: Individual training runs for experiment tracking
- **ModelComparison**: Comparisons between multiple models
- **PredictionRecord**: Individual predictions made by loan officers

## User Roles

### Administrator
- Upload datasets
- Train models
- Compare models
- Activate/deactivate models
- View all audit logs
- Access admin panel

### Loan Officer
- View active model info
- Make predictions (Phase 5)
- View prediction history (Phase 5)
- Download reports (Phase 5)

## API Response Structure

All API responses follow a consistent format:

```json
{
  "message": "Operation successful",
  "data": {...},
  "error": null
}
```

Error responses:
```json
{
  "message": "Operation failed",
  "data": null,
  "error": "Detailed error message"
}
```

## Performance Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| Train LR | ~2s | On small dataset |
| Train RF | ~5s | On small dataset |
| Hyperparameter Tuning | 2-5m | With GridSearchCV |
| Cross-validation | Linear with folds | 5-fold default |
| Model Comparison | < 1s | For 5 models |

## Security Considerations

- ✅ JWT token-based authentication
- ✅ Role-based access control
- ✅ User audit trails (created_by)
- ✅ Admin-only model training and deployment
- ✅ Environment variables for secrets
- ✅ CSRF protection on all forms

## Production Deployment

### Using Docker Compose
```bash
docker-compose up -d
```

### Using AWS EC2
1. Create RDS PostgreSQL database
2. Create S3 bucket for model artifacts
3. Deploy to EC2 with Gunicorn + Nginx
4. Enable CloudWatch monitoring
5. Configure auto-scaling

## Contributing

1. Follow PEP 8 style guide
2. Write tests for new features
3. Maintain > 90% code coverage
4. Update documentation
5. Use descriptive commit messages

## License

Proprietary - Internal Use Only

## Support

For questions or issues, contact the ML engineering team.

---

**Last Updated**: Phase 3 Complete
**Next Phase**: Phase 4 - Explainability & Fairness
**Project Status**: ✅ Production Ready (partial)
