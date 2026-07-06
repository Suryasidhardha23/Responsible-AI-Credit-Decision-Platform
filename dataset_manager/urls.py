from django.urls import path
from . import views

urlpatterns = [
    path('', views.dataset_dashboard, name='dataset_dashboard'),
]
