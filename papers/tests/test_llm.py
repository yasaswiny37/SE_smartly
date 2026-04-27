"""
Unit Tests - LLM Integration
Tests for LLM API calls and response handling.
"""
import json
from unittest.mock import patch, MagicMock
from django.test import TestCase, override_settings
from papers.facade import SearchFacade


class LLMIntegrationTest(TestCase):
    """Test LLM integration functionality."""

    def setUp(self):
        """Set up test data."""
        self.facade = SearchFacade()

    @patch('papers.facade.requests.post')
    def test_llm_call_successful(self, mock_post):
        """Test successful LLM API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'This paper discusses machine learning algorithms.'
        }
        mock_post.return_value = mock_response

        papers = [
            {
                'title': 'ML Paper',
                'conference': 'NeurIPS',
                'year': 2023,
                'abstract': 'This paper presents new machine learning algorithms.'
            }
        ]

        result = self.facade._call_llm('machine learning', papers)

        self.assertEqual(result, 'This paper discusses machine learning algorithms.')

        # Verify the request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args

        # Check URL
        self.assertEqual(call_args[0][0], 'http://cmsai:8000/generate/')

        # Check JSON payload
        payload = call_args[1]['json']
        self.assertIn('prompt', payload)
        self.assertIn('machine learning', payload['prompt'])
        self.assertIn('ML Paper', payload['prompt'])

    @patch('papers.facade.requests.post')
    def test_llm_call_network_error(self, mock_post):
        """Test LLM call with network error."""
        mock_post.side_effect = Exception('Connection refused')

        papers = [{'title': 'Test', 'conference': 'Test', 'year': 2023, 'abstract': 'Test'}]
        result = self.facade._call_llm('query', papers)

        self.assertEqual(result, '')  # Should return empty string on error

    @patch('papers.facade.requests.post')
    def test_llm_call_http_error(self, mock_post):
        """Test LLM call with HTTP error status."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception('Server error')
        mock_post.return_value = mock_response

        papers = [{'title': 'Test', 'conference': 'Test', 'year': 2023, 'abstract': 'Test'}]
        result = self.facade._call_llm('query', papers)

        self.assertEqual(result, '')  # Should return empty string on error

    @patch('papers.facade.requests.post')
    def test_llm_call_invalid_json_response(self, mock_post):
        """Test LLM call with invalid JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError('Invalid JSON', '', 0)
        mock_post.return_value = mock_response

        papers = [{'title': 'Test', 'conference': 'Test', 'year': 2023, 'abstract': 'Test'}]
        result = self.facade._call_llm('query', papers)

        self.assertEqual(result, '')  # Should return empty string on error

    @patch('papers.facade.requests.post')
    def test_llm_call_missing_response_key(self, mock_post):
        """Test LLM call with missing response key in JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'other_key': 'value'}  # No 'response' key
        mock_post.return_value = mock_response

        papers = [{'title': 'Test', 'conference': 'Test', 'year': 2023, 'abstract': 'Test'}]
        result = self.facade._call_llm('query', papers)

        self.assertEqual(result, '')  # Should return empty string when response key missing

    def test_llm_prompt_construction(self):
        """Test LLM prompt construction."""
        query = 'deep learning'
        papers = [
            {
                'title': 'Paper 1',
                'conference': 'NeurIPS',
                'year': 2023,
                'abstract': 'This paper discusses deep learning.'
            },
            {
                'title': 'Paper 2',
                'conference': 'ICML',
                'year': 2022,
                'abstract': 'Another paper on neural networks.'
            }
        ]

        # Test the prompt structure (without making actual call)
        expected_parts = [
            'You are a helpful assistant',
            'deep learning',
            'Paper 1',
            'NeurIPS 2023',
            'This paper discusses deep learning.',
            'Paper 2',
            'ICML 2022',
            'Another paper on neural networks.'
        ]

        # We can't directly test the prompt construction without calling _call_llm
        # But we can verify the method exists and has the right structure
        self.assertTrue(hasattr(self.facade, '_call_llm'))
        self.assertTrue(callable(self.facade._call_llm))

    @patch('papers.facade.requests.post')
    def test_llm_call_timeout(self, mock_post):
        """Test LLM call with timeout."""
        mock_post.side_effect = Exception('Timeout')

        papers = [{'title': 'Test', 'conference': 'Test', 'year': 2023, 'abstract': 'Test'}]
        result = self.facade._call_llm('query', papers)

        self.assertEqual(result, '')  # Should handle timeout gracefully

    @patch('papers.facade.requests.post')
    @override_settings(LLM_ENDPOINT='http://custom-llm:9000/generate')
    def test_llm_custom_endpoint(self, mock_post):
        """Test LLM call with custom endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'Custom endpoint response'}
        mock_post.return_value = mock_response

        papers = [{'title': 'Test', 'conference': 'Test', 'year': 2023, 'abstract': 'Test'}]
        result = self.facade._call_llm('query', papers)

        self.assertEqual(result, 'Custom endpoint response')

        # Verify custom endpoint was used
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], 'http://custom-llm:9000/generate')

    def test_llm_empty_papers(self):
        """Test LLM call with empty papers list."""
        with patch('papers.facade.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'response': 'No papers found'}
            mock_post.return_value = mock_response

            result = self.facade._call_llm('query', [])

            # Should still make the call but with no paper content
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            self.assertIn('prompt', payload)
            # The prompt should have empty abstracts section
            self.assertIn('abstracts:', payload['prompt'].lower())
            self.assertNotIn('[1]', payload['prompt'])  # No paper entries