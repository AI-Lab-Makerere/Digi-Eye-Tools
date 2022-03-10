from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'pattern'
urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_images, name='upload'),
    path('downloadcsv/', views.download_csv, name='download_csv'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
