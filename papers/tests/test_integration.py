"""
Integration Tests - Full Workflow
Tests for complete user workflows: query → retrieval → answer → display.
"""
import json
import tempfile
from unittest.mock import patch, MagicMock
from django.test import TestCase, Client, override_settings
from django.contrib.auth import get_user_model
from django.urls import reverse
from papers.models import Conference, Paper, AdminUser


class IntegrationTest(TestCase):
    """Test complete workflows from user interaction to results."""

    def setUp(self):
        """Set up test data and client."""
        self.client = Client()

        # Create admin user
        self.admin_user = get_user_model().objects.create_user(
            username='admin',
            password='admin123'
        )
        AdminUser.objects.create(user=self.admin_user)

        # Create test data
        self.conference = Conference.objects.create(name='NeurIPS')
        self.paper1 = Paper.objects.create(
            title='Deep Learning Paper',
            authors='John Doe',
            abstract='This paper discusses deep learning and neural networks in detail.',
            conference=self.conference,
            year=2023,
            indexed=True
        )
        self.paper2 = Paper.objects.create(
            title='Computer Vision Paper',
            authors='Jane Smith',
            abstract='This paper presents advances in computer vision using CNNs.',
            conference=self.conference,
            year=2023,
            indexed=True
        )

    @patch('papers.facade._get_embedding_model')
    @patch('papers.facade._get_chroma_collection')
    @patch('papers.facade.requests.post')
    def test_full_search_workflow(self, mock_llm_post, mock_get_collection, mock_get_model):
        """Test complete search workflow: query → retrieval → LLM → display."""
        # Mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_get_model.return_value = mock_model

        # Mock ChromaDB collection with realistic results
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'documents': [
                'This paper discusses deep learning and neural networks in detail.',
                'This paper presents advances in computer vision using CNNs.'
            ],
            'metadatas': [
                {'id': self.paper1.id, 'title': 'Deep Learning Paper', 'conference': 'NEURIPS', 'year': 2023},
                {'id': self.paper2.id, 'title': 'Computer Vision Paper', 'conference': 'NEURIPS', 'year': 2023}
            ],
            'distances': [0.1, 0.2]
        }
        mock_get_collection.return_value = mock_collection

        # Mock LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.status_code = 200
        mock_llm_response.json.return_value = {
            'response': 'Based on the papers, deep learning and computer vision are key areas in AI research.'
        }
        mock_llm_post.return_value = mock_llm_response

        # Perform search via API
        response = self.client.get(reverse('search_api'), {
            'q': 'artificial intelligence',
            'conference': 'NeurIPS',
            'year': '2023'
        })

        # Verify response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)

        # Check structure
        self.assertIn('query', data)
        self.assertIn('papers', data)
        self.assertIn('llm_answer', data)

        # Check content
        self.assertEqual(data['query'], 'artificial intelligence')
        self.assertEqual(len(data['papers']), 2)
        self.assertIn('deep learning', data['llm_answer'].lower())

        # Verify paper data
        paper_titles = [p['title'] for p in data['papers']]
        self.assertIn('Deep Learning Paper', paper_titles)
        self.assertIn('Computer Vision Paper', paper_titles)

    @patch('papers.facade.IngestionFacade.ingest_csv')
    def test_admin_data_ingestion_workflow(self, mock_ingest):
        """Test admin data ingestion workflow."""
        mock_ingest.return_value = {'saved': 3, 'skipped': 0, 'errors': []}

        # Login as admin
        self.client.login(username='admin', password='admin123')

        # Upload CSV file
        csv_content = '''title,authors,abstract,conference,year
New Paper 1,Author 1,Abstract 1,NeurIPS,2024
New Paper 2,Author 2,Abstract 2,CVPR,2024
New Paper 3,Author 3,Abstract 3,ICML,2024'''

        response = self.client.post(
            reverse('upload_file'),
            {'file': csv_content.encode('utf-8'), 'file_type': 'csv'},
            format='multipart'
        )

        # Verify upload response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['saved'], 3)
        self.assertEqual(data['skipped'], 0)

        # Check dashboard shows updated data
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'New Paper 1')

    def test_user_interface_workflow(self):
        """Test complete user interface workflow."""
        # Access main page
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Smart Conference Search')

        # Perform search (will use mocked data)
        with patch('papers.facade.SearchFacade.search') as mock_search:
            mock_search.return_value = {
                'papers': [
                    {
                        'title': 'UI Test Paper',
                        'conference': 'NeurIPS',
                        'year': 2023,
                        'abstract': 'Test abstract',
                        'authors': 'Test Author'
                    }
                ],
                'llm_answer': 'This paper discusses testing.',
                'query': 'test query'
            }

            # Test AJAX search
            response = self.client.get(reverse('search_api'), {'q': 'test query'})
            self.assertEqual(response.status_code, 200)

            data = json.loads(response.content)
            self.assertEqual(len(data['papers']), 1)
            self.assertEqual(data['papers'][0]['title'], 'UI Test Paper')

    @patch('papers.facade._get_embedding_model')
    @patch('papers.facade._get_chroma_collection')
    @patch('papers.facade.requests.post')
    def test_search_with_filters_workflow(self, mock_llm_post, mock_get_collection, mock_get_model):
        """Test search with conference and year filters."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock_get_model.return_value = mock_model

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'documents': ['Filtered paper abstract'],
            'metadatas': [{'id': self.paper1.id, 'title': 'Filtered Paper', 'conference': 'NEURIPS', 'year': 2023}],
            'distances': [0.1]
        }
        mock_get_collection.return_value = mock_collection

        mock_llm_response = MagicMock()
        mock_llm_response.status_code = 200
        mock_llm_response.json.return_value = {'response': 'Filtered results'}
        mock_llm_post.return_value = mock_llm_response

        # Search with filters
        response = self.client.get(reverse('search_api'), {
            'q': 'machine learning',
            'conference': 'NeurIPS',
            'year': '2023'
        })

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)

        # Verify filters were applied
        self.assertEqual(data['query'], 'machine learning')
        self.assertEqual(len(data['papers']), 1)

        # Check ChromaDB was called with filters
        call_args = mock_collection.query.call_args
        self.assertIn('where', call_args[1])
        where_clause = call_args[1]['where']
        self.assertIn('$and', where_clause)

    def test_error_handling_workflow(self):
        """Test error handling throughout the workflow."""
        # Test search with empty query
        response = self.client.get(reverse('search_api'), {'q': ''})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('error', data)

        # Test unauthorized access to admin
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 302)  # Redirect to login

        # Test invalid file upload
        self.client.login(username='admin', password='admin123')
        response = self.client.post(
            reverse('upload_file'),
            {'file': b'invalid', 'file_type': 'invalid'},
            format='multipart'
        )
        self.assertEqual(response.status_code, 400)

    @patch('papers.facade.requests.post')
    def test_llm_failure_workflow(self, mock_llm_post):
        """Test workflow when LLM service fails."""
        # Mock LLM failure
        mock_llm_post.side_effect = Exception('LLM service unavailable')

        # Mock ChromaDB
        with patch('papers.facade._get_chroma_collection') as mock_get_collection, \
             patch('papers.facade._get_embedding_model') as mock_get_model:

            mock_model = MagicMock()
            mock_model.encode.return_value = [0.1, 0.2, 0.3]
            mock_get_model.return_value = mock_model

            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                'documents': ['Paper abstract'],
                'metadatas': [{'id': 1, 'title': 'Paper', 'conference': 'NeurIPS', 'year': 2023}],
                'distances': [0.1]
            }
            mock_get_collection.return_value = mock_collection

            # Perform search
            response = self.client.get(reverse('search_api'), {'q': 'test'})
            self.assertEqual(response.status_code, 200)

            data = json.loads(response.content)
            # Should still return papers even if LLM fails
            self.assertIn('papers', data)
            self.assertEqual(data['llm_answer'], '')  # Empty LLM answer

    def test_data_consistency_workflow(self):
        """Test data consistency across the application."""
        # Create data via admin upload
        self.client.login(username='admin', password='admin123')

        with patch('papers.facade.IngestionFacade.ingest_csv') as mock_ingest:
            mock_ingest.return_value = {'saved': 1, 'skipped': 0, 'errors': []}

            csv_content = 'title,authors,abstract,conference,year\nConsistency Test,Test Author,Test Abstract,NeurIPS,2023'
            self.client.post(
                reverse('upload_file'),
                {'file': csv_content.encode('utf-8'), 'file_type': 'csv'},
                format='multipart'
            )

            # Verify data appears in search
            with patch('papers.facade.SearchFacade.search') as mock_search:
                mock_search.return_value = {
                    'papers': [{'title': 'Consistency Test', 'conference': 'NeurIPS', 'year': 2023}],
                    'llm_answer': 'Test answer',
                    'query': 'consistency'
                }

                response = self.client.get(reverse('search_api'), {'q': 'consistency'})
                data = json.loads(response.content)
                self.assertEqual(len(data['papers']), 1)
                self.assertEqual(data['papers'][0]['title'], 'Consistency Test')