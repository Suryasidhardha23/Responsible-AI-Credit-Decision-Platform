import pickle
from pathlib import Path

from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from training.models import ModelVersion
from .services import ExplainabilityService


@login_required
def prediction_dashboard(request):
    latest_model = ModelVersion.objects.filter(status='trained').order_by('-created_at').first()
    result = None

    if request.method == 'POST':
        applicant_name = request.POST.get('applicant_name', '')
        income = float(request.POST.get('income', 0))
        age = int(request.POST.get('age', 0))
        credit_history = request.POST.get('credit_history', 'good')
        loan_amount = float(request.POST.get('loan_amount', 0))

        feature_row = {
            'income': income,
            'age': age,
            'credit_history': credit_history,
            'loan_amount': loan_amount,
        }

        if latest_model and Path(latest_model.artifact_path).exists():
            with open(latest_model.artifact_path, 'rb') as handle:
                model = pickle.load(handle)
            probability = float(model.predict_proba([feature_row])[0][1])
            prediction = 'Approved' if probability >= 0.5 else 'Rejected'
            explanation_service = ExplainabilityService()
            explanation = explanation_service.build_explanation(model, feature_row)
            fairness_context = explanation_service.build_fairness_context(feature_row)
            result = {
                'applicant_name': applicant_name,
                'probability': round(probability, 4),
                'prediction': prediction,
                'confidence': round(abs(probability - 0.5) * 2, 4),
                'explanation': explanation['summary'],
                'important_features': list(explanation['feature_importance'].keys()),
                'fairness_context': fairness_context,
            }

    return render(request, 'prediction/dashboard.html', {'title': 'Prediction Portal', 'result': result, 'latest_model': latest_model})
