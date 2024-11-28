from django.conf import settings
from django.urls import path
from . import views
from django.conf.urls.static import static

urlpatterns = [
    path('', views.search_view, name='search'),
]+ static(settings.DATA_URL, document_root=settings.DATA_ROOT)
