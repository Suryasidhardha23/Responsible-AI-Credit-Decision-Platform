from django.test import TestCase
from accounts.models import UserProfile


class DatasetManagerTests(TestCase):
    def setUp(self):
        self.user = UserProfile.objects.create_user(
            username='testuser',
            email='testuser@example.com',
            password='testpassword123',
            role='loan_officer',
        )

    def test_dataset_dashboard_access_unauthenticated(self):
        response = self.client.get('/datasets/')
        self.assertEqual(response.status_code, 302)

    def test_dataset_dashboard_access_authenticated(self):
        self.client.login(username='testuser', password='testpassword123')
        response = self.client.get('/datasets/')
        self.assertEqual(response.status_code, 200)

