from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.utils import timezone


# ─── Conference ───────────────────────────────────────────────
class Conference(models.Model):
    name = models.CharField(max_length=100, unique=True)          # e.g. "NeurIPS"
    full_name = models.CharField(max_length=255, blank=True)      # e.g. "Neural Information Processing Systems"
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name


# ─── Paper ────────────────────────────────────────────────────
class Paper(models.Model):
    title       = models.CharField(max_length=500)
    authors     = models.CharField(max_length=500)
    abstract    = models.TextField()
    conference  = models.ForeignKey(Conference, on_delete=models.CASCADE, related_name='papers')
    year        = models.PositiveIntegerField()
    doi_url     = models.URLField(blank=True)
    indexed     = models.BooleanField(default=False)   # True once embedded in ChromaDB
    added_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-added_at']

    def __str__(self):
        return f"{self.title} ({self.conference} {self.year})"


# ─── User (supertype) ──────────────────────────────────────────
class UserManager(BaseUserManager):
    def create_user(self, username, password, first_name='', last_name=''):
        user = self.model(username=username, first_name=first_name, last_name=last_name)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, password, first_name='Admin', last_name='User'):
        user = self.create_user(username, password, first_name, last_name)
        AdminUser.objects.create(user=user, start_date=timezone.now().date())
        return user


class User(AbstractBaseUser):
    username   = models.CharField(max_length=150, unique=True)
    first_name = models.CharField(max_length=100)
    last_name  = models.CharField(max_length=100)
    # password is inherited from AbstractBaseUser
    # last_login is inherited from AbstractBaseUser

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['first_name', 'last_name']

    objects = UserManager()

    # Required for Django admin compatibility
    is_active = models.BooleanField(default=True)
    is_staff  = models.BooleanField(default=False)

    def has_perm(self, perm, obj=None):
        return self.is_staff

    def has_module_perms(self, app_label):
        return self.is_staff

    def __str__(self):
        return self.username


# ─── AdminUser (subtype) ───────────────────────────────────────
class AdminUser(models.Model):
    user       = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True, related_name='admin_profile')
    start_date = models.DateField()

    def __str__(self):
        return f"Admin: {self.user.username}"


# ─── RegularUser (subtype) ────────────────────────────────────
class RegularUser(models.Model):
    user              = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True, related_name='regular_profile')
    registration_date = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"Regular: {self.user.username}"


# ─── UserSession ──────────────────────────────────────────────
class UserSession(models.Model):
    session_id   = models.CharField(max_length=200, unique=True)
    session_date = models.DateTimeField(auto_now_add=True)
    user         = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')

    class Meta:
        ordering = ['-session_date']

    def __str__(self):
        return f"Session {self.session_id} — {self.user.username}"