from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import UserProfile


class CustomUserCreationForm(UserCreationForm):
    role = forms.ChoiceField(choices=[('admin', 'Administrator'), ('loan_officer', 'Loan Officer')])

    class Meta(UserCreationForm.Meta):
        model = UserProfile
        fields = ('username', 'email', 'role')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            if field_name == 'role':
                field.widget.attrs.update({'class': 'form-select bg-light border-0'})
            else:
                field.widget.attrs.update({'class': 'form-control bg-light border-0'})
