from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home page
    path('get-started/', views.get_started, name='get_started'),
    path('explore-doctors/', views.doctors_page, name='doctors_page'),
]
