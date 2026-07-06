from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import UserProfile


class CustomUserCreationForm(UserCreationForm):
    role = forms.ChoiceField(choices=[('admin', 'Administrator'), ('loan_officer', 'Loan Officer')])

    class Meta(UserCreationForm.Meta):
        model = UserProfile
        fields = ('username', 'email', 'role')
