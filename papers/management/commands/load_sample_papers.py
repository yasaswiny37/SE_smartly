"""
Usage:
  python manage.py load_sample_papers
"""
from django.core.management.base import BaseCommand
from papers.facade import IngestionFacade
import os


class Command(BaseCommand):
    help = 'Load the sample IEEE papers CSV into the database'

    def handle(self, *args, **options):
        csv_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data', 'sample_papers.csv'
        )
        csv_path = os.path.abspath(csv_path)

        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f'Sample CSV not found at: {csv_path}'))
            return

        facade = IngestionFacade()
        with open(csv_path, 'rb') as f:
            result = facade.ingest_csv(f)

        self.stdout.write(self.style.SUCCESS(
            f"Loaded {result['saved']} papers. Skipped: {result['skipped']}."
        ))
        if result['errors']:
            for err in result['errors'][:5]:
                self.stdout.write(self.style.WARNING(f"  {err}"))