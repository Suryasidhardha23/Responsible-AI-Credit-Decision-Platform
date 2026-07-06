from django.contrib.auth.models import AbstractUser
from django.db import models


class UserProfile(AbstractUser):
    ROLE_CHOICES = [
        ('admin', 'Administrator'),
        ('loan_officer', 'Loan Officer'),
    ]

    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='loan_officer')
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
