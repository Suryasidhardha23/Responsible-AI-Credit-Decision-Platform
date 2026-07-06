from django.db import models

from accounts.models import UserProfile


class PredictionRecord(models.Model):
    applicant_name = models.CharField(max_length=255, blank=True)
    prediction = models.CharField(max_length=50)
    probability = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(UserProfile, on_delete=models.SET_NULL, null=True, blank=True, related_name='predictions')

    def __str__(self) -> str:
        return f"{self.applicant_name or 'Applicant'} - {self.prediction}"
