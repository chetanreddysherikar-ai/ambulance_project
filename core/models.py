from django.db import models
from django.contrib.auth.models import User


class AccidentRecord(models.Model):
    SEVERITY_CHOICES = [('low', 'Low'), ('medium', 'Medium'), ('high', 'High'), ('fatal', 'Fatal')]
    WEATHER_CHOICES = [('clear', 'Clear'), ('rain', 'Rain'), ('fog', 'Fog'), ('storm', 'Storm')]
    ROAD_CHOICES = [('highway', 'Highway'), ('urban', 'Urban'), ('rural', 'Rural'), ('intersection', 'Intersection')]

    latitude = models.FloatField()
    longitude = models.FloatField()
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default='medium')
    weather_condition = models.CharField(max_length=20, choices=WEATHER_CHOICES, default='clear')
    road_type = models.CharField(max_length=20, choices=ROAD_CHOICES, default='urban')
    time_of_day = models.IntegerField(default=12)
    day_of_week = models.IntegerField(default=1)
    casualties = models.IntegerField(default=0)
    fatalities = models.IntegerField(default=0)
    vehicles_involved = models.IntegerField(default=1)
    speed_limit = models.IntegerField(default=60)
    date_recorded = models.DateTimeField(auto_now_add=True)
    cluster_id = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"Accident at ({self.latitude:.4f}, {self.longitude:.4f}) - {self.severity}"

    class Meta:
        ordering = ['-date_recorded']


class AmbulanceLocation(models.Model):
    STATUS_CHOICES = [('available', 'Available'), ('dispatched', 'Dispatched'), ('maintenance', 'Maintenance')]

    name = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()
    cluster_id = models.IntegerField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='available')
    coverage_radius_km = models.FloatField(default=5.0)
    avg_response_time_min = models.FloatField(default=10.0)
    distance_score = models.FloatField(null=True, blank=True)
    is_optimal = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} - Cluster {self.cluster_id} ({self.status})"

    class Meta:
        ordering = ['cluster_id']


class ClusteringResult(models.Model):
    ALGORITHM_CHOICES = [
        ('DEC', 'Deep Embedded Clustering'),
        ('KMeans', 'K-Means'),
        ('GMM', 'Gaussian Mixture Model'),
        ('Agglomerative', 'Agglomerative Clustering'),
    ]

    algorithm = models.CharField(max_length=30, choices=ALGORITHM_CHOICES)
    n_clusters = models.IntegerField()
    accuracy = models.FloatField(null=True, blank=True)
    silhouette_score = models.FloatField(null=True, blank=True)
    davies_bouldin_score = models.FloatField(null=True, blank=True)
    calinski_harabasz_score = models.FloatField(null=True, blank=True)
    distance_score = models.FloatField(null=True, blank=True)
    training_time_sec = models.FloatField(null=True, blank=True)
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.algorithm} - {self.n_clusters} clusters (Score: {self.distance_score})"

    class Meta:
        ordering = ['-created_at']


class DispatchLog(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'), ('dispatched', 'Dispatched'),
        ('en_route', 'En Route'), ('arrived', 'Arrived'), ('completed', 'Completed'),
    ]

    accident = models.ForeignKey(AccidentRecord, on_delete=models.CASCADE, related_name='dispatches')
    ambulance = models.ForeignKey(AmbulanceLocation, on_delete=models.SET_NULL, null=True, related_name='dispatches')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    dispatch_time = models.DateTimeField(auto_now_add=True)
    arrival_time = models.DateTimeField(null=True, blank=True)
    response_time_min = models.FloatField(null=True, blank=True)
    distance_km = models.FloatField(null=True, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-dispatch_time']


class Notification(models.Model):
    TYPE_CHOICES = [('alert', 'Alert'), ('info', 'Info'), ('warning', 'Warning'), ('success', 'Success')]

    title = models.CharField(max_length=200)
    message = models.TextField()
    notification_type = models.CharField(max_length=20, choices=TYPE_CHOICES, default='info')
    is_read = models.BooleanField(default=False)
    recipient = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='notifications')
    created_at = models.DateTimeField(auto_now_add=True)
    related_accident = models.ForeignKey(AccidentRecord, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ['-created_at']


class DataUpload(models.Model):
    STATUS_CHOICES = [('pending', 'Pending'), ('processing', 'Processing'), ('completed', 'Completed'), ('failed', 'Failed')]

    file = models.FileField(upload_to='uploads/')
    filename = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    records_processed = models.IntegerField(default=0)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    error_message = models.TextField(blank=True)
