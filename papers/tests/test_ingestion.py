"""
Unit Tests - Ingestion Facade
Tests for CSV, JSON, and BibTeX ingestion functionality.
"""
import json
from io import StringIO, BytesIO
from unittest.mock import patch
from django.test import TestCase
from django.core.files.base import ContentFile
from papers.models import Conference, Paper
from papers.facade import IngestionFacade


class IngestionFacadeTest(TestCase):
    """Test IngestionFacade functionality."""

    def setUp(self):
        """Set up test data."""
        self.facade = IngestionFacade()

    @patch.object(IngestionFacade, '_index_paper')
    def test_csv_ingestion_valid(self, mock_index):
        """Test valid CSV ingestion."""
        csv_data = """title,authors,abstract,conference,year,doi_url
Test Paper,John Doe,This is a test abstract.,NeurIPS,2023,https://doi.org/10.1234/test
Another Paper,Jane Smith,Another abstract here.,CVPR,2022,"""
        csv_file = ContentFile(csv_data.encode('utf-8'), name='test.csv')

        result = self.facade.ingest_csv(csv_file)

        self.assertEqual(result['saved'], 2)
        self.assertEqual(result['skipped'], 0)
        self.assertEqual(len(result['errors']), 0)
        self.assertEqual(Paper.objects.count(), 2)
        self.assertEqual(Conference.objects.count(), 2)

    @patch.object(IngestionFacade, '_index_paper')
    def test_csv_ingestion_invalid_data(self, mock_index):
        """Test CSV with invalid data."""
        csv_data = """title,authors,abstract,conference,year
Incomplete Paper,Author,Abstract,NeurIPS,invalid_year
Valid Paper,Author,Abstract,NeurIPS,2023"""
        csv_file = ContentFile(csv_data.encode('utf-8'), name='test.csv')

        result = self.facade.ingest_csv(csv_file)

        self.assertEqual(result['saved'], 1)
        self.assertEqual(result['skipped'], 1)
        self.assertEqual(len(result['errors']), 1)
        self.assertIn('Invalid year', result['errors'][0])

    @patch.object(IngestionFacade, '_index_paper')
    def test_csv_ingestion_duplicate(self, mock_index):
        """Test CSV with duplicate papers."""
        conference = Conference.objects.create(name='NEURIPS')
        Paper.objects.create(
            title='Existing Paper',
            authors='Author',
            abstract='Abstract',
            conference=conference,
            year=2023
        )

        csv_data = """title,authors,abstract,conference,year
Existing Paper,Author,Abstract,NeurIPS,2023
New Paper,Author,Abstract,NeurIPS,2023"""
        csv_file = ContentFile(csv_data.encode('utf-8'), name='test.csv')

        result = self.facade.ingest_csv(csv_file)

        self.assertEqual(result['saved'], 1)
        self.assertEqual(result['skipped'], 1)
        self.assertEqual(len(result['errors']), 1)
        self.assertIn('Duplicate', result['errors'][0])

    @patch.object(IngestionFacade, '_index_paper')
    def test_csv_ingestion_missing_required_fields(self, mock_index):
        """Test CSV with missing required fields."""
        csv_data = """title,authors,abstract,conference,year
Missing Authors,,Abstract,NeurIPS,2023
Missing Abstract,Author,,NeurIPS,2023
Valid Paper,Author,Abstract,NeurIPS,2023"""
        csv_file = ContentFile(csv_data.encode('utf-8'), name='test.csv')

        result = self.facade.ingest_csv(csv_file)

        self.assertEqual(result['saved'], 1)
        self.assertEqual(result['skipped'], 2)
        self.assertEqual(len(result['errors']), 2)

    @patch.object(IngestionFacade, '_index_paper')
    def test_json_ingestion_valid(self, mock_index):
        """Test valid JSON ingestion."""
        json_data = [
            {
                'title': 'JSON Paper 1',
                'authors': 'Author 1',
                'abstract': 'Abstract 1',
                'conference': 'NeurIPS',
                'year': 2023,
                'doi_url': 'https://doi.org/10.1234/json1'
            },
            {
                'title': 'JSON Paper 2',
                'authors': 'Author 2',
                'abstract': 'Abstract 2',
                'conference': 'CVPR',
                'year': 2022
            }
        ]
        json_file = ContentFile(json.dumps(json_data).encode('utf-8'), name='test.json')

        result = self.facade.ingest_json(json_file)

        self.assertEqual(result['saved'], 2)
        self.assertEqual(result['skipped'], 0)
        self.assertEqual(len(result['errors']), 0)
        self.assertEqual(Paper.objects.count(), 2)

    @patch.object(IngestionFacade, '_index_paper')
    def test_json_ingestion_invalid(self, mock_index):
        """Test JSON with invalid data."""
        json_data = [
            {
                'title': 'Valid Paper',
                'authors': 'Author',
                'abstract': 'Abstract',
                'conference': 'NeurIPS',
                'year': 2023
            },
            {
                'title': 'Invalid Paper',
                'authors': '',
                'abstract': 'Abstract',
                'conference': 'NeurIPS',
                'year': 2023
            }
        ]
        json_file = ContentFile(json.dumps(json_data).encode('utf-8'), name='test.json')

        result = self.facade.ingest_json(json_file)

        self.assertEqual(result['saved'], 1)
        self.assertEqual(result['skipped'], 1)
        self.assertEqual(len(result['errors']), 1)

    @patch.object(IngestionFacade, '_index_paper')
    def test_bibtex_ingestion(self, mock_index):
        """Test BibTeX ingestion (basic test)."""
        bibtex_data = """@article{test2023,
  title={BibTeX Test Paper},
  author={Test Author},
  abstract={This is a test abstract from BibTeX.},
  journal={Test Journal},
  year={2023}
}"""
        bibtex_file = ContentFile(bibtex_data.encode('utf-8'), name='test.bib')

        result = self.facade.ingest_bibtex(bibtex_file)

        self.assertEqual(result['saved'], 1)
        self.assertEqual(result['skipped'], 0)
        self.assertEqual(len(result['errors']), 0)

    @patch.object(IngestionFacade, '_index_paper')
    def test_ingestion_with_bytes_content(self, mock_index):
        """Test ingestion with bytes content (file upload simulation)."""
        csv_data = b"""title,authors,abstract,conference,year
Bytes Test,Byte Author,Byte abstract,NeurIPS,2023"""
        csv_file = BytesIO(csv_data)

        result = self.facade.ingest_csv(csv_file)

        self.assertEqual(result['saved'], 1)
        self.assertEqual(result['skipped'], 0)
        self.assertEqual(len(result['errors']), 0)