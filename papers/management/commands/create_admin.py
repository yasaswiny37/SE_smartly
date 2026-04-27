"""
Usage:
  python manage.py create_admin --username admin --password admin123 --first-name Admin --last-name User
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from papers.models import User, AdminUser


class Command(BaseCommand):
    help = 'Create an admin user for SmartConf'

    def add_arguments(self, parser):
        parser.add_argument('--username',   default='admin')
        parser.add_argument('--password',   default='admin123')
        parser.add_argument('--first-name', default='Admin')
        parser.add_argument('--last-name',  default='User')

    def handle(self, *args, **options):
        username = options['username']
        if User.objects.filter(username=username).exists():
            self.stdout.write(self.style.WARNING(f'User "{username}" already exists.'))
            return

        user = User.objects.create_user(
            username   = username,
            password   = options['password'],
            first_name = options['first_name'],
            last_name  = options['last_name'],
        )
        user.is_staff = True
        user.save()

        AdminUser.objects.create(user=user, start_date=timezone.now().date())

        self.stdout.write(self.style.SUCCESS(
            f'Admin user "{username}" created. Password: {options["password"]}'
        ))