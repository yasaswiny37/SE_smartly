"""
Unit Tests - Views
Tests for Django views, HTTP responses, and API endpoints.
"""
import json
from unittest.mock import patch, MagicMock
from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone
from papers.models import Conference, Paper, AdminUser


class ViewsTest(TestCase):
    """Test Django views and HTTP responses."""

    def setUp(self):
        """Set up test data."""
        self.client = Client()
        self.user = get_user_model().objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.admin_user = get_user_model().objects.create_user(
            username='admin',
            password='admin123'
        )
        AdminUser.objects.create(user=self.admin_user, start_date=timezone.now().date())

        self.conference = Conference.objects.create(name='NeurIPS')
        self.paper = Paper.objects.create(
            title='Test Paper',
            authors='Test Author',
            abstract='Test abstract',
            conference=self.conference,
            year=2023
        )

    def test_index_view(self):
        """Test index view renders correctly."""
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'papers/index.html')
        self.assertContains(response, 'Smart Conference Assistant')

    def test_search_api_get(self):
        """Test search API GET request."""
        response = self.client.get(reverse('search_api'), {'q': 'test query'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/json')

        data = json.loads(response.content)
        self.assertIn('papers', data)
        self.assertIn('llm_answer', data)
        self.assertIn('query', data)

    def test_search_api_post(self):
        """Test search API POST request."""
        response = self.client.post(
            reverse('search_api'),
            {'q': 'test query', 'conference': 'NeurIPS', 'year': '2023'},
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.content)
        self.assertIn('papers', data)
        self.assertIn('llm_answer', data)

    @patch('papers.facade.SearchFacade.search')
    def test_search_api_with_mock(self, mock_search):
        """Test search API with mocked search facade."""
        mock_search.return_value = {
            'papers': [{'title': 'Mock Paper', 'conference': 'Mock Conf', 'year': 2023}],
            'llm_answer': 'Mock answer',
            'query': 'mock query'
        }

        response = self.client.get(reverse('search_api'), {'q': 'mock query'})
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.content)
        self.assertEqual(data['llm_answer'], 'Mock answer')
        self.assertEqual(len(data['papers']), 1)

        mock_search.assert_called_once_with('mock query', conference='', year='')

    def test_search_api_missing_query(self):
        """Test search API with missing query."""
        response = self.client.get(reverse('search_api'))
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.content)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Query is required.')

    def test_dashboard_requires_login(self):
        """Test dashboard requires authentication."""
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 302)  # Redirect to login

    def test_dashboard_admin_access(self):
        """Test admin dashboard access."""
        self.client.login(username='admin', password='admin123')
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'papers/admin_dashboard.html')
        self.assertContains(response, 'Admin Dashboard')

    def test_dashboard_regular_user_denied(self):
        """Test regular user cannot access admin dashboard."""
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 403)  # Forbidden

    def test_upload_file_requires_admin(self):
        """Test file upload requires admin permissions."""
        self.client.login(username='testuser', password='testpass123')
        response = self.client.post(reverse('upload_file'))
        self.assertEqual(response.status_code, 403)

    @patch('papers.facade.IngestionFacade.ingest_csv')
    def test_upload_csv_admin(self, mock_ingest):
        """Test CSV upload by admin."""
        mock_ingest.return_value = {'saved': 1, 'skipped': 0, 'errors': []}

        self.client.login(username='admin', password='admin123')
        csv_content = b'title,authors,abstract,conference,year\nTest,Author,Abstract,NeurIPS,2023'
        response = self.client.post(
            reverse('upload_file'),
            {'file': csv_content, 'file_type': 'csv'},
            format='multipart'
        )
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.content)
        self.assertEqual(data['saved'], 1)
        self.assertEqual(data['skipped'], 0)

    def test_upload_invalid_file_type(self):
        """Test upload with invalid file type."""
        self.client.login(username='admin', password='admin123')
        response = self.client.post(
            reverse('upload_file'),
            {'file': b'invalid', 'file_type': 'invalid'},
            format='multipart'
        )
        self.assertEqual(response.status_code, 400)

        data = json.loads(response.content)
        self.assertIn('error', data)

    def test_login_view(self):
        """Test login view."""
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'papers/admin_dashboard.html')  # Login form in dashboard

    def test_login_success(self):
        """Test successful login."""
        response = self.client.post(
            reverse('login'),
            {'username': 'admin', 'password': 'admin123'}
        )
        self.assertEqual(response.status_code, 302)  # Redirect after login

    def test_login_failure(self):
        """Test login failure."""
        response = self.client.post(
            reverse('login'),
            {'username': 'admin', 'password': 'wrongpass'}
        )
        self.assertEqual(response.status_code, 200)  # Stay on login page
        self.assertContains(response, 'Invalid credentials')

    def test_logout(self):
        """Test logout functionality."""
        self.client.login(username='admin', password='admin123')
        response = self.client.post(reverse('logout'))
        self.assertEqual(response.status_code, 302)  # Redirect after logout

    def test_admin_required_decorator(self):
        """Test admin_required decorator."""
        from papers.views import admin_required

        # Create a mock view function
        @admin_required
        def mock_view(request):
            return {'success': True}

        # Test with admin user
        from django.contrib.auth.models import AnonymousUser
        from django.test import RequestFactory

        factory = RequestFactory()
        request = factory.get('/')
        request.user = self.admin_user

        response = mock_view(request)
        self.assertEqual(response, {'success': True})

        # Test with regular user
        request.user = self.user
        response = mock_view(request)
        self.assertEqual(response.status_code, 403)

        # Test with anonymous user
        request.user = AnonymousUser()
        response = mock_view(request)
        self.assertEqual(response.status_code, 302)  # Redirect to login

    def test_error_handling(self):
        """Test error handling in views."""
        # Test with invalid JSON in POST
        response = self.client.post(
            reverse('search_api'),
            'invalid json',
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_cors_headers(self):
        """Test CORS headers for API endpoints."""
        response = self.client.get(reverse('search_api'), {'q': 'test'})
        # Check for CORS headers if implemented
        # self.assertEqual(response['Access-Control-Allow-Origin'], '*')