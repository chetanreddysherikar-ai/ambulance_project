from django.contrib import admin
from .models import AccidentRecord, AmbulanceLocation, ClusteringResult, DispatchLog, Notification, DataUpload

@admin.register(AccidentRecord)
class AccidentAdmin(admin.ModelAdmin):
    list_display = ['id', 'latitude', 'longitude', 'severity', 'weather_condition', 'casualties', 'cluster_id', 'date_recorded']
    list_filter = ['severity', 'weather_condition', 'road_type']
    search_fields = ['severity']

@admin.register(AmbulanceLocation)
class AmbulanceAdmin(admin.ModelAdmin):
    list_display = ['name', 'latitude', 'longitude', 'cluster_id', 'status', 'is_optimal', 'distance_score']
    list_filter = ['status', 'is_optimal']

@admin.register(ClusteringResult)
class ClusteringResultAdmin(admin.ModelAdmin):
    list_display = ['algorithm', 'n_clusters', 'silhouette_score', 'distance_score', 'accuracy', 'is_active', 'created_at']
    list_filter = ['algorithm', 'is_active']

@admin.register(DispatchLog)
class DispatchLogAdmin(admin.ModelAdmin):
    list_display = ['id', 'ambulance', 'status', 'distance_km', 'response_time_min', 'dispatch_time']
    list_filter = ['status']

@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ['title', 'notification_type', 'is_read', 'created_at']
    list_filter = ['notification_type', 'is_read']

@admin.register(DataUpload)
class DataUploadAdmin(admin.ModelAdmin):
    list_display = ['filename', 'status', 'records_processed', 'uploaded_at']
