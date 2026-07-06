"""Tests for model training service."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from django.test import TestCase

from accounts.models import UserProfile
from dataset_manager.models import UploadedDataset
from training.comparison_service import ModelComparisonService
from training.models import ModelVersion
from training.services import ModelTrainingService


class ModelTrainingServiceTestCase(TestCase):
    """Test the ModelTrainingService."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = ModelTrainingService()
        
        # Create a sample dataset
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / 'test_data.csv'
        
        # Create sample data
        np.random.seed(42)
        n_samples = 200
        data = {
            'age': np.random.randint(20, 70, n_samples),
            'income': np.random.randint(20000, 150000, n_samples),
            'credit_history': np.random.choice(['good', 'fair', 'poor'], n_samples),
            'loan_amount': np.random.randint(5000, 50000, n_samples),
            'target': np.random.choice([0, 1], n_samples),
        }
        df = pd.DataFrame(data)
        df.to_csv(self.dataset_path, index=False)

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_train_logistic_regression(self):
        """Test training a logistic regression model."""
        result = self.service.train_model(
            str(self.dataset_path),
            'target',
            algorithm='logistic_regression',
        )
        
        # Verify result structure
        assert 'metrics' in result
        assert 'cv_results' in result
        assert 'artifact_path' in result
        assert 'confusion_matrix' in result
        
        # Verify metrics
        metrics = result['metrics']
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Verify metrics are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert Path(result['artifact_path']).exists()

    def test_train_decision_tree(self):
        """Test training a decision tree model."""
        result = self.service.train_model(
            str(self.dataset_path),
            'target',
            algorithm='decision_tree',
        )
        
        assert 'metrics' in result
        assert result['metrics']['accuracy'] >= 0

    def test_train_random_forest(self):
        """Test training a random forest model."""
        result = self.service.train_model(
            str(self.dataset_path),
            'target',
            algorithm='random_forest',
        )
        
        assert 'metrics' in result
        assert 'feature_importance' in result
        assert len(result['feature_importance']) > 0

    def test_cross_validation(self):
        """Test cross-validation results."""
        result = self.service.train_model(
            str(self.dataset_path),
            'target',
            cross_validation_folds=3,
        )
        
        cv_results = result['cv_results']
        assert 'accuracy_mean' in cv_results
        assert 'accuracy_std' in cv_results
        assert cv_results['accuracy_mean'] > 0

    def test_invalid_target_column(self):
        """Test training with invalid target column."""
        with pytest.raises(ValueError):
            self.service.train_model(
                str(self.dataset_path),
                'nonexistent_column',
            )

    def test_preprocessing_pipeline_saved(self):
        """Test that preprocessing pipeline is saved."""
        result = self.service.train_model(
            str(self.dataset_path),
            'target',
        )
        
        assert 'preprocessing_pipeline_path' in result
        assert Path(result['preprocessing_pipeline_path']).exists()

    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        result = self.service.train_model(
            str(self.dataset_path),
            'target',
        )
        
        cm = result['confusion_matrix']
        assert 'true_negatives' in cm
        assert 'false_positives' in cm
        assert 'false_negatives' in cm
        assert 'true_positives' in cm


class ModelVersionTests(TestCase):
    """Test ModelVersion model."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = UserProfile.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123',
            role='admin',
        )

    def test_model_version_creation(self):
        """Test creating a model version."""
        model_version = ModelVersion.objects.create(
            version='v1.0',
            name='Credit Model',
            algorithm='logistic_regression',
            created_by=self.user,
        )
        self.assertEqual(model_version.version, 'v1.0')
        self.assertEqual(model_version.name, 'Credit Model')
        self.assertFalse(model_version.is_active)

    def test_activate_model(self):
        """Test activating a model."""
        model1 = ModelVersion.objects.create(
            version='v1.0',
            name='Model 1',
            created_by=self.user,
        )
        model2 = ModelVersion.objects.create(
            version='v1.1',
            name='Model 2',
            created_by=self.user,
        )
        
        # Activate model1
        model1.activate()
        model1.refresh_from_db()
        self.assertTrue(model1.is_active)
        self.assertEqual(model1.status, 'deployed')
        
        # Activating model2 should deactivate model1
        model2.activate()
        model1.refresh_from_db()
        model2.refresh_from_db()
        
        self.assertFalse(model1.is_active)
        self.assertTrue(model2.is_active)

    def test_deactivate_model(self):
        """Test deactivating a model."""
        model = ModelVersion.objects.create(
            version='v1.0',
            name='Model 1',
            created_by=self.user,
            is_active=True,
        )
        
        model.deactivate()
        model.refresh_from_db()
        
        self.assertFalse(model.is_active)
        self.assertEqual(model.deployment_status, 'inactive')


class ModelComparisonServiceTestCase(TestCase):
    """Test the ModelComparisonService."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test user
        self.user = UserProfile.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123',
            role='admin',
        )
        
        # Create test models
        self.model1 = ModelVersion.objects.create(
            version='v1.0',
            name='Model 1',
            algorithm='logistic_regression',
            created_by=self.user,
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            roc_auc=0.90,
            pr_auc=0.88,
            cv_score_mean=0.84,
            cv_score_std=0.02,
        )
        
        self.model2 = ModelVersion.objects.create(
            version='v1.1',
            name='Model 2',
            algorithm='random_forest',
            created_by=self.user,
            accuracy=0.88,
            precision=0.86,
            recall=0.89,
            f1_score=0.87,
            roc_auc=0.92,
            pr_auc=0.91,
            cv_score_mean=0.87,
            cv_score_std=0.01,
        )

    def test_compare_models(self):
        """Test comparing multiple models."""
        comparison = ModelComparisonService.compare_models(
            [self.model1, self.model2],
            name='Test Comparison',
            created_by=self.user,
        )
        
        self.assertEqual(comparison.name, 'Test Comparison')
        self.assertEqual(comparison.details.count(), 2)
        
        # Verify ranking (model2 should be ranked 1st due to better metrics)
        details = list(comparison.details.all().order_by('rank'))
        self.assertEqual(details[0].model_version, self.model2)
        self.assertEqual(details[0].rank, 1)

    def test_comparison_summary(self):
        """Test generating comparison summary."""
        comparison = ModelComparisonService.compare_models(
            [self.model1, self.model2],
            name='Test Comparison',
        )
        
        summary = ModelComparisonService.get_comparison_summary(comparison)
        self.assertEqual(summary['total_models_compared'], 2)
        self.assertEqual(summary['best_model']['version'], 'v1.1')

    def test_metric_differences(self):
        """Test metric differences calculation."""
        differences = ModelComparisonService.get_metric_differences(self.model2, self.model1)
        
        self.assertEqual(differences['models']['best']['version'], 'v1.1')
        self.assertIn('metric_differences', differences)
        self.assertGreater(differences['metric_differences']['accuracy']['difference'], 0)

    def test_rank_models(self):
        """Test model ranking algorithm."""
        comparison_data = [
            (self.model1, {'accuracy': 0.85, 'f1_score': 0.85, 'roc_auc': 0.90, 'precision': 0.83, 'recall': 0.87}),
            (self.model2, {'accuracy': 0.88, 'f1_score': 0.87, 'roc_auc': 0.92, 'precision': 0.86, 'recall': 0.89}),
        ]
        
        ranked = ModelComparisonService._rank_models(comparison_data)
        
        # Best model should be first
        self.assertEqual(ranked[0][0], self.model2)
        self.assertEqual(ranked[1][0], self.model1)
