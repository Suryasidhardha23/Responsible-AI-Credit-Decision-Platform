"""API views for model training and comparison."""

import logging
from datetime import datetime

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet

from dataset_manager.models import UploadedDataset
from training.comparison_service import ModelComparisonService
from training.explainability_service import SHAPExplainabilityService
from training.fairness_service import FairnessAuditService
from training.models import ModelComparison, ModelVersion
from training.serializers import (
    CompareModelsRequestSerializer,
    ModelComparisonSerializer,
    ModelVersionListSerializer,
    ModelVersionSerializer,
    TrainModelRequestSerializer,
)
from training.services import ModelTrainingService

logger = logging.getLogger(__name__)


def is_admin(user):
    """Check if user is an administrator."""
    return user.is_authenticated and user.role == 'admin'


class AdminOnlyPermission(IsAuthenticated):
    """Permission class for admin-only access."""

    def has_permission(self, request, view):
        return super().has_permission(request, view) and is_admin(request.user)


class TrainModelAPIView(APIView):
    """API endpoint for training models."""

    permission_classes = [AdminOnlyPermission]

    def post(self, request):
        """Train a new model."""
        serializer = TrainModelRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Get the dataset
            dataset = UploadedDataset.objects.get(id=serializer.validated_data['dataset_id'])

            if not dataset.is_valid:
                return Response(
                    {'error': 'Dataset is not valid'},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Check if version already exists
            if ModelVersion.objects.filter(version=serializer.validated_data['version']).exists():
                return Response(
                    {'error': f"Model version {serializer.validated_data['version']} already exists"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Train the model
            logger.info(f"Starting model training for dataset {dataset.id}")
            service = ModelTrainingService()

            result = service.train_model(
                dataset_path=dataset.file_path,
                target_column=serializer.validated_data['target_column'],
                algorithm=serializer.validated_data['algorithm'],
                perform_hyperparameter_tuning=serializer.validated_data['perform_hyperparameter_tuning'],
                cross_validation_folds=serializer.validated_data['cross_validation_folds'],
            )

            # Create model version record
            model_version = ModelVersion.objects.create(
                version=serializer.validated_data['version'],
                name=serializer.validated_data['name'],
                algorithm=serializer.validated_data['algorithm'],
                status='trained',
                created_by=request.user,
                dataset=dataset,
                target_column=serializer.validated_data['target_column'],
                accuracy=result['metrics']['accuracy'],
                precision=result['metrics']['precision'],
                recall=result['metrics']['recall'],
                f1_score=result['metrics']['f1_score'],
                roc_auc=result['metrics']['roc_auc'],
                pr_auc=result['metrics']['pr_auc'],
                cv_score_mean=result['cv_results']['accuracy_mean'],
                cv_score_std=result['cv_results']['accuracy_std'],
                artifact_path=result['artifact_path'],
                preprocessing_pipeline_path=result['preprocessing_pipeline_path'],
                hyperparameters=result['hyperparameters'],
                feature_list=result['feature_list'],
                confusion_matrix=result['confusion_matrix'],
                class_distribution=result['class_distribution'],
                feature_importance=result['feature_importance'],
            )

            serializer = ModelVersionSerializer(model_version)
            logger.info(f"Model {model_version.version} trained successfully")

            return Response(
                {
                    'message': 'Model trained successfully',
                    'model': serializer.data,
                },
                status=status.HTTP_201_CREATED,
            )

        except UploadedDataset.DoesNotExist:
            return Response(
                {'error': 'Dataset not found'},
                status=status.HTTP_404_NOT_FOUND,
            )
        except ValueError as e:
            logger.error(f"Validation error during training: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}", exc_info=True)
            return Response(
                {'error': f'Model training failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class CompareModelsAPIView(APIView):
    """API endpoint for comparing multiple models."""

    permission_classes = [AdminOnlyPermission]

    def post(self, request):
        """Compare multiple model versions."""
        serializer = CompareModelsRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Get all models
            models = ModelVersion.objects.filter(id__in=serializer.validated_data['model_version_ids'])

            if models.count() < 2:
                return Response(
                    {'error': 'At least 2 models are required for comparison'},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Create comparison
            comparison = ModelComparisonService.compare_models(
                model_versions=list(models),
                name=serializer.validated_data['name'],
                created_by=request.user,
            )

            logger.info(f"Comparison {comparison.id} created for {models.count()} models")

            response_serializer = ModelComparisonSerializer(comparison)
            return Response(
                {
                    'message': 'Models compared successfully',
                    'comparison': response_serializer.data,
                },
                status=status.HTTP_201_CREATED,
            )

        except Exception as e:
            logger.error(f"Error during model comparison: {str(e)}", exc_info=True)
            return Response(
                {'error': f'Model comparison failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ModelVersionViewSet(ModelViewSet):
    """ViewSet for model versions."""

    queryset = ModelVersion.objects.all()
    serializer_class = ModelVersionSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'version'

    def get_queryset(self):
        """Filter based on user role."""
        if is_admin(self.request.user):
            return ModelVersion.objects.all()
        return ModelVersion.objects.filter(is_active=True)

    def get_serializer_class(self):
        """Use different serializer for list view."""
        if self.action == 'list':
            return ModelVersionListSerializer
        return ModelVersionSerializer

    def destroy(self, request, *args, **kwargs):
        """Prevent deletion of models (use archive instead)."""
        return Response(
            {'error': 'Models cannot be deleted. Use archive instead.'},
            status=status.HTTP_405_METHOD_NOT_ALLOWED,
        )

    def get_permissions(self):
        """Admin-only for write operations."""
        if self.request.method in ['POST', 'PUT', 'PATCH', 'DELETE']:
            return [AdminOnlyPermission()]
        return super().get_permissions()


class ActivateModelAPIView(APIView):
    """API endpoint for activating a model."""

    permission_classes = [AdminOnlyPermission]

    def post(self, request, model_version_id):
        """Activate a specific model version."""
        try:
            model_version = ModelVersion.objects.get(id=model_version_id)
            model_version.activate()

            logger.info(f"Model {model_version.version} activated by {request.user.username}")

            return Response(
                {
                    'message': f'Model {model_version.version} activated successfully',
                    'model': ModelVersionSerializer(model_version).data,
                },
                status=status.HTTP_200_OK,
            )
        except ModelVersion.DoesNotExist:
            return Response(
                {'error': 'Model not found'},
                status=status.HTTP_404_NOT_FOUND,
            )


class DeactivateModelAPIView(APIView):
    """API endpoint for deactivating a model."""

    permission_classes = [AdminOnlyPermission]

    def post(self, request, model_version_id):
        """Deactivate a specific model version."""
        try:
            model_version = ModelVersion.objects.get(id=model_version_id)
            model_version.deactivate()

            logger.info(f"Model {model_version.version} deactivated by {request.user.username}")

            return Response(
                {
                    'message': f'Model {model_version.version} deactivated successfully',
                    'model': ModelVersionSerializer(model_version).data,
                },
                status=status.HTTP_200_OK,
            )
        except ModelVersion.DoesNotExist:
            return Response(
                {'error': 'Model not found'},
                status=status.HTTP_404_NOT_FOUND,
            )


class ModelComparisonViewSet(ModelViewSet):
    """ViewSet for model comparisons."""

    queryset = ModelComparison.objects.all()
    serializer_class = ModelComparisonSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Filter based on user role."""
        if is_admin(self.request.user):
            return ModelComparison.objects.all()
        return ModelComparison.objects.none()

    def get_permissions(self):
        """Admin-only for write operations."""
        if self.request.method in ['POST', 'PUT', 'PATCH', 'DELETE']:
            return [AdminOnlyPermission()]
        return super().get_permissions()


class GetActiveModelAPIView(APIView):
    """API endpoint for retrieving the currently active model."""

    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Get the active model for predictions."""
        try:
            active_model = ModelVersion.objects.get(is_active=True)
            serializer = ModelVersionSerializer(active_model)

            return Response(
                {
                    'active_model': serializer.data,
                },
                status=status.HTTP_200_OK,
            )
        except ModelVersion.DoesNotExist:
            return Response(
                {'error': 'No active model found'},
                status=status.HTTP_404_NOT_FOUND,
            )


class ExplainModelAPIView(APIView):
    """Generate SHAP explainability for a model version."""

    permission_classes = [AdminOnlyPermission]

    def post(self, request, model_version_id):
        try:
            model_version = ModelVersion.objects.get(id=model_version_id)
            service = SHAPExplainabilityService()
            explanation = service.generate_explanation(model_version, sample_size=request.data.get('sample_size', 50))
            return Response({'message': 'Explanation generated', 'explanation': explanation}, status=status.HTTP_200_OK)
        except ModelVersion.DoesNotExist:
            return Response({'error': 'Model not found'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            logger.error('Explainability generation failed: %s', exc, exc_info=True)
            return Response({'error': str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FairnessAuditAPIView(APIView):
    """Generate fairness audit results for a model version."""

    permission_classes = [AdminOnlyPermission]

    def post(self, request, model_version_id):
        try:
            model_version = ModelVersion.objects.get(id=model_version_id)
            protected_attribute = request.data.get('protected_attribute', 'sex')
            proxy_features = request.data.get('proxy_features', [])
            service = FairnessAuditService()
            audit = service.audit_model(model_version, protected_attribute=protected_attribute, proxy_features=proxy_features)
            return Response({'message': 'Fairness audit generated', 'audit': audit}, status=status.HTTP_200_OK)
        except ModelVersion.DoesNotExist:
            return Response({'error': 'Model not found'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            logger.error('Fairness audit failed: %s', exc, exc_info=True)
            return Response({'error': str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
