from django.urls import include, path
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from .views import HealthView
from .training_views import (
    ActivateModelAPIView,
    CompareModelsAPIView,
    DeactivateModelAPIView,
    ExplainModelAPIView,
    FairnessAuditAPIView,
    GetActiveModelAPIView,
    ModelComparisonViewSet,
    ModelVersionViewSet,
    TrainModelAPIView,
)
from .prediction_views import (
    PredictAPIView,
    MitigateModelAPIView,
    PredictionReportPDFView,
    ModelCardPDFView,
    FairnessAuditReportPDFView,
)

router = DefaultRouter()
router.register(r'models', ModelVersionViewSet, basename='model')
router.register(r'comparisons', ModelComparisonViewSet, basename='comparison')

urlpatterns = [
    path('health/', HealthView.as_view(), name='health'),
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('train/', TrainModelAPIView.as_view(), name='train-model'),
    path('compare/', CompareModelsAPIView.as_view(), name='compare-models'),
    path('models/<int:model_version_id>/activate/', ActivateModelAPIView.as_view(), name='activate-model'),
    path('models/<int:model_version_id>/deactivate/', DeactivateModelAPIView.as_view(), name='deactivate-model'),
    path('models/<int:model_version_id>/explain/', ExplainModelAPIView.as_view(), name='explain-model'),
    path('models/<int:model_version_id>/fairness/', FairnessAuditAPIView.as_view(), name='fairness-audit'),
    path('active-model/', GetActiveModelAPIView.as_view(), name='get-active-model'),
    # Prediction & Responsible AI endpoints
    path('predict/', PredictAPIView.as_view(), name='live-predict'),
    path('mitigate/', MitigateModelAPIView.as_view(), name='mitigate-model'),
    # PDF Report endpoints
    path('reports/prediction/<int:record_id>/pdf/', PredictionReportPDFView.as_view(), name='prediction-report-pdf'),
    path('reports/model-card/<int:model_version_id>/pdf/', ModelCardPDFView.as_view(), name='model-card-pdf'),
    path('reports/fairness-audit/<int:model_version_id>/pdf/', FairnessAuditReportPDFView.as_view(), name='fairness-audit-pdf'),
    path('', include(router.urls)),
]
