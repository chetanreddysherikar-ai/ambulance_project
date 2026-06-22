import json
import sys
import os
import numpy as np
import csv
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.http import JsonResponse
from django.db.models import Count, Avg
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

from .models import AccidentRecord, AmbulanceLocation, ClusteringResult, DispatchLog, Notification, DataUpload


def landing(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'core/landing.html')


def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created successfully!')
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'core/register.html', {'form': form})


@login_required
def dashboard(request):
    total_accidents = AccidentRecord.objects.count()
    total_ambulances = AmbulanceLocation.objects.count()
    available_ambulances = AmbulanceLocation.objects.filter(status='available').count()
    active_dispatches = DispatchLog.objects.filter(status__in=['dispatched', 'en_route']).count()
    recent_accidents = AccidentRecord.objects.order_by('-date_recorded')[:5]
    recent_dispatches = DispatchLog.objects.select_related('accident', 'ambulance').order_by('-dispatch_time')[:5]
    unread_notifications = Notification.objects.filter(is_read=False).count()
    latest_clustering = ClusteringResult.objects.filter(is_active=True).first()

    severity_counts = {}
    for sev, label in AccidentRecord.SEVERITY_CHOICES:
        severity_counts[label] = AccidentRecord.objects.filter(severity=sev).count()

    context = {
        'total_accidents': total_accidents,
        'total_ambulances': total_ambulances,
        'available_ambulances': available_ambulances,
        'active_dispatches': active_dispatches,
        'recent_accidents': recent_accidents,
        'recent_dispatches': recent_dispatches,
        'unread_notifications': unread_notifications,
        'latest_clustering': latest_clustering,
        'severity_counts': json.dumps(severity_counts),
    }
    return render(request, 'core/dashboard.html', context)


@login_required
def accident_list(request):
    accidents = AccidentRecord.objects.all()
    severity_filter = request.GET.get('severity', '')
    if severity_filter:
        accidents = accidents.filter(severity=severity_filter)
    accidents_json = [
        {'lat': a.latitude, 'lon': a.longitude, 'severity': a.severity, 'id': a.id}
        for a in accidents
    ]
    context = {
        'accidents': accidents[:100],
        'accidents_json': json.dumps(accidents_json),
        'severity_choices': AccidentRecord.SEVERITY_CHOICES,
        'selected_severity': severity_filter,
    }
    return render(request, 'core/accident_list.html', context)


@login_required
def accident_add(request):
    if request.method == 'POST':
        try:
            accident = AccidentRecord(
                latitude=float(request.POST['latitude']),
                longitude=float(request.POST['longitude']),
                severity=request.POST.get('severity', 'medium'),
                weather_condition=request.POST.get('weather_condition', 'clear'),
                road_type=request.POST.get('road_type', 'urban'),
                time_of_day=int(request.POST.get('time_of_day', 12)),
                day_of_week=int(request.POST.get('day_of_week', 1)),
                casualties=int(request.POST.get('casualties', 0)),
                fatalities=int(request.POST.get('fatalities', 0)),
                vehicles_involved=int(request.POST.get('vehicles_involved', 1)),
                speed_limit=int(request.POST.get('speed_limit', 60)),
            )
            accident.save()

            # Notify
            Notification.objects.create(
                title=f'New {accident.severity.upper()} accident reported',
                message=f'Accident at ({accident.latitude:.4f}, {accident.longitude:.4f}). Casualties: {accident.casualties}',
                notification_type='alert' if accident.severity in ['high', 'fatal'] else 'info',
                recipient=request.user,
                related_accident=accident,
            )
            messages.success(request, 'Accident recorded successfully!')
            return redirect('accident_list')
        except Exception as e:
            messages.error(request, f'Error: {str(e)}')
    
    context = {
        'severity_choices': AccidentRecord.SEVERITY_CHOICES,
        'weather_choices': AccidentRecord.WEATHER_CHOICES,
        'road_choices': AccidentRecord.ROAD_CHOICES,
    }
    return render(request, 'core/accident_add.html', context)


@login_required
def accident_detail(request, pk):
    accident = get_object_or_404(AccidentRecord, pk=pk)
    dispatches = DispatchLog.objects.filter(accident=accident).select_related('ambulance')
    ambulances = AmbulanceLocation.objects.filter(status='available')
    
    # Find nearest ambulance
    nearest_ambulance = None
    min_dist = float('inf')
    for amb in ambulances:
        dlat = (amb.latitude - accident.latitude) * 111
        dlon = (amb.longitude - accident.longitude) * 111 * np.cos(np.radians(accident.latitude))
        dist = np.sqrt(dlat**2 + dlon**2)
        if dist < min_dist:
            min_dist = dist
            nearest_ambulance = amb
    
    context = {
        'accident': accident,
        'dispatches': dispatches,
        'ambulances': ambulances,
        'nearest_ambulance': nearest_ambulance,
        'nearest_distance': round(min_dist, 2),
    }
    return render(request, 'core/accident_detail.html', context)


@login_required
def clustering_view(request):
    results = ClusteringResult.objects.all().order_by('-created_at')
    ambulances = AmbulanceLocation.objects.filter(is_optimal=True)
    ambulances_json = [
        {'lat': a.latitude, 'lon': a.longitude, 'name': a.name, 'cluster': a.cluster_id,
         'status': a.status, 'score': a.distance_score}
        for a in ambulances
    ]
    context = {
        'results': results,
        'ambulances_json': json.dumps(ambulances_json),
        'algorithm_choices': ClusteringResult.ALGORITHM_CHOICES,
    }
    return render(request, 'core/clustering.html', context)


@login_required
def run_clustering_view(request):
    if request.method != 'POST':
        return redirect('clustering')
    
    algorithm = request.POST.get('algorithm', 'DEC')
    n_clusters = int(request.POST.get('n_clusters', 8))
    
    try:
        from ml_models.dec_model import run_clustering, generate_nairobi_data
        
        # Use existing accident data if available
        acc_count = AccidentRecord.objects.count()
        
        if acc_count < 50:
            data = generate_nairobi_data(500)
            # Save generated data to DB
            accidents_to_create = []
            for _, row in data.iterrows():
                accidents_to_create.append(AccidentRecord(
                    latitude=row['latitude'], longitude=row['longitude'],
                    severity=row['severity'], weather_condition=row['weather_condition'],
                    road_type=row['road_type'], time_of_day=int(row['time_of_day']),
                    day_of_week=int(row['day_of_week']), casualties=int(row['casualties']),
                    fatalities=int(row['fatalities']), vehicles_involved=int(row['vehicles_involved']),
                    speed_limit=int(row['speed_limit']),
                ))
            AccidentRecord.objects.bulk_create(accidents_to_create)
            result = run_clustering(algorithm=algorithm, n_clusters=n_clusters, data=data)
        else:
            import pandas as pd
            qs = AccidentRecord.objects.all().values(
                'latitude', 'longitude', 'severity', 'weather_condition',
                'road_type', 'time_of_day', 'day_of_week', 'casualties',
                'fatalities', 'vehicles_involved', 'speed_limit'
            )
            data = pd.DataFrame(list(qs))
            result = run_clustering(algorithm=algorithm, n_clusters=n_clusters, data=data)
        
        # Update accident cluster IDs
        labels = result['labels']
        accidents = list(AccidentRecord.objects.all())
        for i, acc in enumerate(accidents[:len(labels)]):
            acc.cluster_id = int(labels[i])
        AccidentRecord.objects.bulk_update(accidents[:len(labels)], ['cluster_id'])
        
        # Deactivate previous results, save new
        ClusteringResult.objects.update(is_active=False)
        cr = ClusteringResult.objects.create(
            algorithm=algorithm,
            n_clusters=n_clusters,
            accuracy=result.get('accuracy'),
            silhouette_score=result.get('silhouette_score'),
            davies_bouldin_score=result.get('davies_bouldin_score'),
            calinski_harabasz_score=result.get('calinski_harabasz_score'),
            distance_score=result.get('distance_score'),
            training_time_sec=result.get('training_time_sec'),
            is_active=True,
            notes=f"Ran on {result['n_samples']} accident records",
        )
        
        # Create optimal ambulance positions
        AmbulanceLocation.objects.filter(is_optimal=True).delete()
        for i, (lat, lon, count) in enumerate(result['cluster_centers']):
            AmbulanceLocation.objects.create(
                name=f'{algorithm} Ambulance Station {i+1}',
                latitude=lat,
                longitude=lon,
                cluster_id=i,
                status='available',
                coverage_radius_km=max(3.0, count * 0.05),
                avg_response_time_min=max(5.0, count * 0.1),
                distance_score=result.get('distance_score'),
                is_optimal=True,
            )
        
        Notification.objects.create(
            title=f'{algorithm} Clustering Completed',
            message=f'{n_clusters} optimal ambulance positions identified. Distance score: {result.get("distance_score", 0):.3f} km',
            notification_type='success',
            recipient=request.user,
        )
        
        messages.success(request, f'{algorithm} clustering completed! {n_clusters} optimal locations found.')
    except Exception as e:
        messages.error(request, f'Clustering error: {str(e)}')
    
    return redirect('clustering')


@login_required
def ambulance_list(request):
    ambulances = AmbulanceLocation.objects.all()
    ambulances_json = [
        {'lat': a.latitude, 'lon': a.longitude, 'name': a.name, 'cluster': a.cluster_id, 'status': a.status}
        for a in ambulances
    ]
    dispatches = DispatchLog.objects.select_related('accident', 'ambulance').order_by('-dispatch_time')[:20]
    context = {
        'ambulances': ambulances,
        'ambulances_json': json.dumps(ambulances_json),
        'dispatches': dispatches,
    }
    return render(request, 'core/ambulance_list.html', context)


@login_required
def dispatch_ambulance(request, pk):
    ambulance = get_object_or_404(AmbulanceLocation, pk=pk)
    accident_id = request.POST.get('accident_id')
    
    if not accident_id:
        messages.error(request, 'Please select an accident.')
        return redirect('ambulance_list')
    
    accident = get_object_or_404(AccidentRecord, pk=accident_id)
    
    dlat = (ambulance.latitude - accident.latitude) * 111
    dlon = (ambulance.longitude - accident.longitude) * 111 * np.cos(np.radians(accident.latitude))
    distance_km = round(np.sqrt(dlat**2 + dlon**2), 2)
    response_time = round(distance_km / 60 * 60, 1)
    
    DispatchLog.objects.create(
        accident=accident,
        ambulance=ambulance,
        status='dispatched',
        distance_km=distance_km,
        response_time_min=response_time,
    )
    
    ambulance.status = 'dispatched'
    ambulance.save()
    
    Notification.objects.create(
        title=f'Ambulance {ambulance.name} Dispatched',
        message=f'Dispatched to accident at ({accident.latitude:.4f}, {accident.longitude:.4f}). ETA: {response_time} min',
        notification_type='alert',
        recipient=request.user,
        related_accident=accident,
    )
    
    messages.success(request, f'{ambulance.name} dispatched! ETA: {response_time} minutes')
    return redirect('ambulance_list')


@login_required
def eda_view(request):
    accidents = AccidentRecord.objects.all()
    total = accidents.count()
    
    severity_data = {}
    for sev, label in AccidentRecord.SEVERITY_CHOICES:
        severity_data[label] = accidents.filter(severity=sev).count()
    
    weather_data = {}
    for w, label in AccidentRecord.WEATHER_CHOICES:
        weather_data[label] = accidents.filter(weather_condition=w).count()
    
    road_data = {}
    for r, label in AccidentRecord.ROAD_CHOICES:
        road_data[label] = accidents.filter(road_type=r).count()
    
    hour_data = [0] * 24
    for acc in accidents:
        hour_data[acc.time_of_day] += 1
    
    context = {
        'total': total,
        'severity_data': json.dumps(severity_data),
        'weather_data': json.dumps(weather_data),
        'road_data': json.dumps(road_data),
        'hour_data': json.dumps(hour_data),
        'avg_casualties': accidents.aggregate(Avg('casualties'))['casualties__avg'] or 0,
        'avg_fatalities': accidents.aggregate(Avg('fatalities'))['fatalities__avg'] or 0,
    }
    return render(request, 'core/eda.html', context)


@login_required
def notifications_view(request):
    notifs = Notification.objects.filter(recipient=request.user) | Notification.objects.filter(recipient=None)
    notifs = notifs.order_by('-created_at')[:50]
    return render(request, 'core/notifications.html', {'notifications': notifs})


@login_required
def mark_notification_read(request, pk):
    notif = get_object_or_404(Notification, pk=pk)
    notif.is_read = True
    notif.save()
    return redirect('notifications')


@login_required
def upload_data(request):
    if request.method == 'POST' and request.FILES.get('data_file'):
        f = request.FILES['data_file']
        upload = DataUpload.objects.create(
            file=f,
            filename=f.name,
            status='processing',
            uploaded_by=request.user,
        )
        try:
            content = f.read().decode('utf-8')
            reader = csv.DictReader(io.StringIO(content))
            count = 0
            for row in reader:
                try:
                    AccidentRecord.objects.create(
                        latitude=float(row.get('latitude', row.get('lat', 0))),
                        longitude=float(row.get('longitude', row.get('lon', row.get('lng', 0)))),
                        severity=row.get('severity', 'medium'),
                        weather_condition=row.get('weather_condition', row.get('weather', 'clear')),
                        road_type=row.get('road_type', 'urban'),
                        time_of_day=int(float(row.get('time_of_day', 12))),
                        day_of_week=int(float(row.get('day_of_week', 1))),
                        casualties=int(float(row.get('casualties', 0))),
                        fatalities=int(float(row.get('fatalities', 0))),
                        vehicles_involved=int(float(row.get('vehicles_involved', 1))),
                        speed_limit=int(float(row.get('speed_limit', 60))),
                    )
                    count += 1
                except Exception:
                    continue
            
            upload.status = 'completed'
            upload.records_processed = count
            upload.save()
            messages.success(request, f'Successfully imported {count} accident records!')
        except Exception as e:
            upload.status = 'failed'
            upload.error_message = str(e)
            upload.save()
            messages.error(request, f'Upload failed: {str(e)}')
    
    uploads = DataUpload.objects.order_by('-uploaded_at')[:10]
    return render(request, 'core/upload.html', {'uploads': uploads})


# API Views
def api_accidents(request):
    accidents = AccidentRecord.objects.all().values(
        'id', 'latitude', 'longitude', 'severity', 'weather_condition', 'cluster_id'
    )
    return JsonResponse({'accidents': list(accidents)})


def api_ambulances(request):
    ambulances = AmbulanceLocation.objects.all().values(
        'id', 'name', 'latitude', 'longitude', 'status', 'cluster_id', 'distance_score'
    )
    return JsonResponse({'ambulances': list(ambulances)})


def api_stats(request):
    return JsonResponse({
        'total_accidents': AccidentRecord.objects.count(),
        'total_ambulances': AmbulanceLocation.objects.count(),
        'available_ambulances': AmbulanceLocation.objects.filter(status='available').count(),
        'active_dispatches': DispatchLog.objects.filter(status__in=['dispatched', 'en_route']).count(),
        'unread_notifications': Notification.objects.filter(is_read=False).count(),
    })
