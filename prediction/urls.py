from django.urls import path
from . import views

urlpatterns = [
    path('', views.prediction_dashboard, name='prediction_dashboard'),
]
