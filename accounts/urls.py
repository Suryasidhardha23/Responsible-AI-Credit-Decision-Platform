from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='account_dashboard'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
]
