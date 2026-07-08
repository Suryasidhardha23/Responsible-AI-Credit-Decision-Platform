from django.shortcuts import render
from training.models import ModelVersion
from dataset_manager.models import UploadedDataset
from prediction.models import PredictionRecord


def home_view(request):
    platform_stats = []
    if request.user.is_authenticated and request.user.role == 'admin':
        platform_stats = [
            ("Datasets Uploaded", "database-fill-up", "#0d3d60", UploadedDataset.objects.count()),
            ("Models Trained", "cpu-fill", "#2ec4b6", ModelVersion.objects.count()),
            ("Models Deployed", "lightning-fill", "#28a745", ModelVersion.objects.filter(is_active=True).count()),
            ("Predictions Made", "person-check-fill", "#6a1a5e", PredictionRecord.objects.count()),
        ]
    return render(request, 'home.html', {'platform_stats': platform_stats})
