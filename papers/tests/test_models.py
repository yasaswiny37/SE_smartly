"""
Unit Tests - Database Models
Tests for Paper, Conference, User models and their relationships.
"""
import os
from django.test import TestCase
from django.utils import timezone
from papers.models import Conference, Paper, User, AdminUser


class ConferenceModelTest(TestCase):
    """Test Conference model."""

    def test_conference_creation(self):
        """Test creating a conference."""
        conf = Conference.objects.create(name='NeurIPS', full_name='Neural Information Processing Systems')
        self.assertEqual(conf.name, 'NeurIPS')
        self.assertEqual(conf.full_name, 'Neural Information Processing Systems')
        self.assertEqual(str(conf), 'NeurIPS')

    def test_conference_unique_name(self):
        """Test conference name uniqueness."""
        Conference.objects.create(name='CVPR')
        with self.assertRaises(Exception):  # IntegrityError
            Conference.objects.create(name='CVPR')

    def test_conference_ordering(self):
        """Test conference ordering by name."""
        c2 = Conference.objects.create(name='ICML')
        c1 = Conference.objects.create(name='CVPR')
        conferences = list(Conference.objects.all())
        self.assertEqual(conferences, [c1, c2])  # CVPR before ICML alphabetically


class PaperModelTest(TestCase):
    """Test Paper model."""

    def setUp(self):
        """Set up test data."""
        self.conference = Conference.objects.create(name='NeurIPS')

    def test_paper_creation(self):
        """Test creating a paper."""
        paper = Paper.objects.create(
            title='Test Paper',
            authors='John Doe',
            abstract='This is a test abstract.',
            conference=self.conference,
            year=2023,
            doi_url='https://doi.org/10.1234/test'
        )
        self.assertEqual(paper.title, 'Test Paper')
        self.assertEqual(paper.authors, 'John Doe')
        self.assertEqual(paper.year, 2023)
        self.assertEqual(paper.indexed, False)  # Default value
        self.assertIsNotNone(paper.added_at)
        self.assertEqual(str(paper), f'Test Paper (NeurIPS 2023)')

    def test_paper_ordering(self):
        """Test paper ordering by added_at descending."""
        from django.utils import timezone
        import time

        # Create papers with explicit time difference
        paper2 = Paper.objects.create(
            title='Paper 2', authors='Author 2', abstract='Abstract 2',
            conference=self.conference, year=2023
        )
        time.sleep(0.01)  # Small delay to ensure different timestamps
        paper1 = Paper.objects.create(
            title='Paper 1', authors='Author 1', abstract='Abstract 1',
            conference=self.conference, year=2023
        )

        papers = list(Paper.objects.all())
        self.assertEqual(papers[0], paper1)  # Most recent first
        self.assertEqual(papers[1], paper2)

    def test_paper_foreign_key(self):
        """Test paper-conference relationship."""
        paper = Paper.objects.create(
            title='Test', authors='Test', abstract='Test',
            conference=self.conference, year=2023
        )
        self.assertEqual(paper.conference, self.conference)
        self.assertIn(paper, self.conference.papers.all())


class UserModelTest(TestCase):
    """Test User and AdminUser models."""

    def test_user_creation(self):
        """Test creating a user."""
        user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            first_name='Test',
            last_name='User'
        )
        self.assertEqual(user.username, 'testuser')
        self.assertEqual(user.first_name, 'Test')
        self.assertEqual(user.last_name, 'User')
        self.assertTrue(user.check_password('testpass123'))
        self.assertEqual(str(user), 'testuser')

    def test_admin_user_creation(self):
        """Test creating an admin user."""
        user = User.objects.create_user(
            username='admin',
            password='admin123',
            first_name='Admin',
            last_name='User'
        )
        admin_profile = AdminUser.objects.create(
            user=user,
            start_date=timezone.now().date()
        )
        self.assertEqual(admin_profile.user, user)
        self.assertIsNotNone(admin_profile.start_date)
        self.assertEqual(str(admin_profile), f'Admin: admin')

    def test_user_permissions(self):
        """Test user permission methods."""
        user = User.objects.create_user(username='user', password='pass')
        # Regular user has no staff/superuser permissions
        self.assertFalse(user.is_staff)
        self.assertFalse(user.has_perm('any_perm'))
        self.assertFalse(user.has_module_perms('any_app'))