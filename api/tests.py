from django.test import TestCase


class APITests(TestCase):
    def test_health_endpoint(self):
        response = self.client.get('/api/health/')
        self.assertEqual(response.status_code, 200)
