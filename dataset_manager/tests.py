from django.test import TestCase


class DatasetManagerTests(TestCase):
    def test_dataset_dashboard_access(self):
        response = self.client.get('/datasets/')
        self.assertEqual(response.status_code, 200)
