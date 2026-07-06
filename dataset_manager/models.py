from django.db import models


class UploadedDataset(models.Model):
    name = models.CharField(max_length=255)
    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    upload_date = models.DateTimeField(auto_now_add=True)
    row_count = models.PositiveIntegerField(default=0)
    column_count = models.PositiveIntegerField(default=0)
    target_column = models.CharField(max_length=100, blank=True)
    is_valid = models.BooleanField(default=False)
    validation_summary = models.TextField(blank=True)
    missing_values = models.TextField(blank=True)
    categorical_features = models.TextField(blank=True)
    numerical_features = models.TextField(blank=True)
    duplicate_rows = models.PositiveIntegerField(default=0)

    def __str__(self) -> str:
        return self.name
