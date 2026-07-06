from django.urls import include, path
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from .views import HealthView
from .training_views import (
    ActivateModelAPIView,
    CompareModelsAPIView,
    DeactivateModelAPIView,
    GetActiveModelAPIView,
    ModelComparisonViewSet,
    ModelVersionViewSet,
    TrainModelAPIView,
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
    path('active-model/', GetActiveModelAPIView.as_view(), name='get-active-model'),
    path('', include(router.urls)),
]
