from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'pattern'
urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_images, name='upload'),
    path('rgb/', views.rgb_page, name='rgb_name'),
    path('upload_sample', views.upload_sample, name='upload_sample'),
    path('mealiness/', views.mealiness_page, name='mealiness'),
    path('predict_mealiness/', views.predict_mealiness, name='predict_mealiness'),
    path('mealiness/downloadcsv/', views.download_csv, name='download_csv'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

urlpatterns+= static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)