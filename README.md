# Responsible AI Credit Decision Platform

An end-to-end Django platform for responsible, human-reviewed AI-assisted credit decisions. It provides governed dataset ingestion, model training and deployment, individual prediction explanations, fairness tools, audit records, PDF reports, a REST API, and Docker deployment assets.

> This is a decision-support demonstration project. It must not be the sole basis for a real lending decision. Production use requires legal, compliance, security, data-governance, monitoring, and human-review controls appropriate to the jurisdiction.

## Project status

**Completed** — the application workflows, API surface, reporting, and deployment configuration are implemented and ready for local evaluation or controlled internal demonstration.

## Features

- Role-based authentication for administrators and loan officers
- CSV dataset upload, validation, metadata capture, and upload isolation
- Model training with Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting
- Cross-validation, optional hyperparameter tuning, model registry, comparison, activation, archival, and deployment controls
- Production predictions only from an explicitly deployed model
- Per-prediction explanations, fairness context, audit history, and PDF reports
- Fairness audit and bias-mitigation workflows for administrators
- JWT/session-authenticated REST API, health endpoint, Docker Compose, Gunicorn, and Nginx configuration

## Technology

- Django and Django REST Framework
- scikit-learn, pandas, NumPy, SHAP, and Fairlearn
- SQLite for local development; PostgreSQL supported for deployment
- Docker, Gunicorn, and Nginx

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows PowerShell
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Open `http://127.0.0.1:8000/` and sign in. Administrators can upload a dataset, train a model, and explicitly deploy it. Loan officers can then use the prediction portal.

## Main routes

| Route | Purpose |
| --- | --- |
| `/` | Platform home |
| `/accounts/` | Account dashboard |
| `/datasets/` | Dataset upload and validation (administrator) |
| `/training/` | Training, registry, fairness, and mitigation (administrator) |
| `/predict/` | Loan-officer prediction workflow |
| `/admin/` | Platform administration console (administrator) |
| `/django-admin/` | Django data administration |
| `/api/` | API browser (administrator) |
| `/api/health/` | Public health endpoint |

## API highlights

- `POST /api/token/` and `POST /api/token/refresh/`
- `POST /api/train/`, `POST /api/compare/`, and `GET /api/active-model/`
- `GET /api/rest/models/` and `GET /api/rest/comparisons/`
- `POST /api/predict/`
- `POST /api/mitigate/`
- PDF report endpoints under `/api/reports/`

Authenticated endpoints accept JWT bearer tokens or an authenticated Django session. Administrator-only endpoints enforce the administrator role.

## Tests and checks

```bash
python manage.py check
python manage.py test
```

## Deployment

Copy `.env.example` to `.env`, set a unique `SECRET_KEY`, configure `DEBUG=False`, provide allowed hosts and PostgreSQL values as appropriate, then run:

```bash
docker-compose up -d --build
```

Do not commit `.env`, local databases, uploaded datasets, generated model artifacts, or credentials.

## Project layout

```text
accounts/                 Authentication and roles
dataset_manager/          Dataset ingestion and validation
training/                 Training, registry, fairness, mitigation, and reports
prediction/               Prediction workflow and audit records
api/                      REST API endpoints
templates/                Responsive Django frontend
responsible_ai_platform/  Project settings and root routing
```

## License

Proprietary — internal use only.
