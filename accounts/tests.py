from django.test import TestCase


class AccountTests(TestCase):
    def test_login_page_loads(self):
        response = self.client.get('/accounts/login/')
        self.assertEqual(response.status_code, 200)
