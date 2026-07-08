import logging
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.contrib import messages
from training.models import ModelVersion
from .models import PredictionRecord
from .services import PredictionService

logger = logging.getLogger(__name__)


@login_required
def prediction_dashboard(request):
    # Retrieve the active model (or latest trained model fallback)
    active_model = PredictionService.get_active_model()
    categorical_options = {}
    result = None

    if active_model:
        # Get dynamic options for categorical dropdowns
        categorical_options = PredictionService.get_categorical_options(active_model)
        
        if request.method == 'POST':
            applicant_name = request.POST.get('applicant_name', 'Unnamed Applicant')
            
            # Retrieve values from POST data
            input_data = {}
            for key in request.POST:
                if key != 'csrfmiddlewaretoken' and key != 'applicant_name':
                    val = request.POST.get(key)
                    if val:
                        input_data[key] = val

            try:
                # Call prediction service
                service = PredictionService()
                pred_result = service.predict_and_explain(active_model, input_data)
                
                # Save prediction record with full SHAP + fairness context
                record = PredictionRecord.objects.create(
                    applicant_name=applicant_name,
                    prediction=pred_result['prediction'],
                    probability=pred_result['probability'],
                    shap_explanation=pred_result['shap_explanation'],
                    input_features=pred_result['feature_row'],
                    fairness_context=pred_result['fairness_context'],
                    model_version=active_model,
                    created_by=request.user
                )
                
                result = {
                    'record_id': record.id,
                    'applicant_name': applicant_name,
                    'prediction': pred_result['prediction'],
                    'probability': pred_result['probability'],
                    'confidence': pred_result['confidence'],
                    'explanation': pred_result['summary'],
                    'shap_explanation': pred_result['shap_explanation'],
                    'fairness_context': pred_result['fairness_context'],
                    'input_data': input_data
                }
                messages.success(request, f"Prediction successfully completed for {applicant_name}.")
            except Exception as e:
                logger.error(f"Prediction failed: {e}", exc_info=True)
                messages.error(request, f"Error running prediction: {str(e)}")

    # Fetch history for this user
    history = PredictionRecord.objects.filter(created_by=request.user).order_by('-created_at')[:10]

    return render(request, 'prediction/dashboard.html', {
        'title': 'Loan Officer Portal',
        'active_model': active_model,
        'categorical_options': categorical_options,
        'result': result,
        'history': history
    })
