from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from .models import UploadedDataset
from .services import DatasetValidationService


@login_required
def dataset_dashboard(request):
    if request.method == 'POST' and request.FILES.get('dataset_file'):
        service = DatasetValidationService()
        summary = service.validate_and_store(request.FILES['dataset_file'], request.POST.get('dataset_name', 'uploaded_dataset'))
        dataset = UploadedDataset.objects.create(
            name=request.POST.get('dataset_name', 'uploaded_dataset'),
            file_name=request.FILES['dataset_file'].name,
            file_path=str(service.upload_dir / request.FILES['dataset_file'].name),
            row_count=summary['rows'],
            column_count=summary['columns'],
            target_column=summary['target_column'] or '',
            is_valid=summary['is_valid'],
            validation_summary=f"Rows: {summary['rows']}; Columns: {summary['columns']}; Duplicate rows: {summary['duplicate_rows']}",
            missing_values=str(summary['missing_values']),
            categorical_features=str(summary['categorical_features']),
            numerical_features=str(summary['numerical_features']),
            duplicate_rows=summary['duplicate_rows'],
        )
        return render(request, 'dataset_manager/dashboard.html', {'title': 'Dataset Dashboard', 'dataset': dataset, 'summary': summary})

    datasets = UploadedDataset.objects.order_by('-upload_date')
    return render(request, 'dataset_manager/dashboard.html', {'title': 'Dataset Dashboard', 'datasets': datasets})
