import csv
import os
from pathlib import Path

import pandas as pd


class DatasetValidationService:
    def __init__(self, upload_dir: str | None = None):
        self.upload_dir = Path(upload_dir or 'media/uploads')
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def validate_and_store(self, uploaded_file, dataset_name: str) -> dict:
        destination = self.upload_dir / uploaded_file.name
        with destination.open('wb+') as destination_file:
            for chunk in uploaded_file.chunks():
                destination_file.write(chunk)

        dataframe = pd.read_csv(destination)
        missing_values = dataframe.isna().sum().to_dict()
        duplicate_rows = int(dataframe.duplicated().sum())
        categorical_features = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
        numerical_features = [col for col in dataframe.columns if col not in categorical_features]

        target_column = None
        for candidate in ['target', 'label', 'risk', 'default']:
            if candidate in dataframe.columns:
                target_column = candidate
                break

        summary = {
            'rows': int(dataframe.shape[0]),
            'columns': int(dataframe.shape[1]),
            'missing_values': missing_values,
            'duplicate_rows': duplicate_rows,
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'target_column': target_column,
            'is_valid': True,
        }

        return summary
