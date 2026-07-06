from django.contrib import admin
from django.urls import include, path

from .views import home_view

urlpatterns = [
    path('', home_view, name='home'),
    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),
    path('datasets/', include('dataset_manager.urls')),
    path('training/', include('training.urls')),
    path('predict/', include('prediction.urls')),
    path('api/', include('api.urls')),
]
