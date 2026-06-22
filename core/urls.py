from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('login/', auth_views.LoginView.as_view(template_name='core/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('register/', views.register_view, name='register'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('accidents/', views.accident_list, name='accident_list'),
    path('accidents/add/', views.accident_add, name='accident_add'),
    path('accidents/<int:pk>/', views.accident_detail, name='accident_detail'),
    path('clustering/', views.clustering_view, name='clustering'),
    path('clustering/run/', views.run_clustering_view, name='run_clustering'),
    path('ambulances/', views.ambulance_list, name='ambulance_list'),
    path('ambulances/<int:pk>/dispatch/', views.dispatch_ambulance, name='dispatch_ambulance'),
    path('eda/', views.eda_view, name='eda'),
    path('notifications/', views.notifications_view, name='notifications'),
    path('notifications/mark-read/<int:pk>/', views.mark_notification_read, name='mark_notification_read'),
    path('upload/', views.upload_data, name='upload_data'),
    path('api/accidents/', views.api_accidents, name='api_accidents'),
    path('api/ambulances/', views.api_ambulances, name='api_ambulances'),
    path('api/stats/', views.api_stats, name='api_stats'),
]
