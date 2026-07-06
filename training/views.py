from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import render

from dataset_manager.models import UploadedDataset
from .models import ModelVersion
from .services import ModelTrainingService


def is_admin(user):
    return user.is_authenticated and user.role == 'admin'


@user_passes_test(is_admin)
def training_dashboard(request):
    if request.method == 'POST':
        version = request.POST.get('version', 'v1.0')
        dataset_id = request.POST.get('dataset_id')
        target_column = request.POST.get('target_column', 'target')
        dataset = UploadedDataset.objects.filter(id=dataset_id).first()
        if dataset:
            service = ModelTrainingService()
            result = service.train_model(dataset.file_path, target_column)
            model_version = ModelVersion.objects.create(
                version=version,
                name='Credit Model',
                status='trained',
                accuracy=result['metrics']['accuracy'],
                f1_score=result['metrics']['f1_score'],
                roc_auc=result['metrics']['roc_auc'],
                artifact_path=result['artifact_path'],
                notes=f'Trained on {dataset.name} using target column {target_column}',
            )
            return render(request, 'training/dashboard.html', {'title': 'Model Training Dashboard', 'versions': ModelVersion.objects.order_by('-created_at'), 'latest_model': model_version, 'metrics': result['metrics']})

    versions = ModelVersion.objects.order_by('-created_at')
    datasets = UploadedDataset.objects.order_by('-upload_date')
    return render(request, 'training/dashboard.html', {'title': 'Model Training Dashboard', 'versions': versions, 'datasets': datasets})
