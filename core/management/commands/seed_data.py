"""Management command to seed sample data"""
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))


class Command(BaseCommand):
    help = 'Seeds the database with sample accident data and runs DEC clustering'

    def handle(self, *args, **options):
        self.stdout.write('Seeding data...')
        
        from ml_models.dec_model import generate_nairobi_data, run_clustering
        from core.models import AccidentRecord, AmbulanceLocation, ClusteringResult, Notification
        
        # Generate accident data
        data = generate_nairobi_data(500)
        accidents = []
        for _, row in data.iterrows():
            accidents.append(AccidentRecord(
                latitude=row['latitude'], longitude=row['longitude'],
                severity=row['severity'], weather_condition=row['weather_condition'],
                road_type=row['road_type'], time_of_day=int(row['time_of_day']),
                day_of_week=int(row['day_of_week']), casualties=int(row['casualties']),
                fatalities=int(row['fatalities']), vehicles_involved=int(row['vehicles_involved']),
                speed_limit=int(row['speed_limit']),
            ))
        AccidentRecord.objects.bulk_create(accidents)
        self.stdout.write(f'  Created {len(accidents)} accident records')

        # Run DEC clustering
        result = run_clustering(algorithm='DEC', n_clusters=8, data=data)
        
        ClusteringResult.objects.update(is_active=False)
        ClusteringResult.objects.create(
            algorithm='DEC', n_clusters=8,
            accuracy=result.get('accuracy'),
            silhouette_score=result.get('silhouette_score'),
            davies_bouldin_score=result.get('davies_bouldin_score'),
            distance_score=result.get('distance_score'),
            training_time_sec=result.get('training_time_sec'),
            is_active=True,
            notes='Seeded via management command',
        )

        AmbulanceLocation.objects.filter(is_optimal=True).delete()
        for i, (lat, lon, count) in enumerate(result['cluster_centers']):
            AmbulanceLocation.objects.create(
                name=f'DEC Station {i+1}', latitude=lat, longitude=lon,
                cluster_id=i, status='available',
                coverage_radius_km=max(3.0, count * 0.05),
                distance_score=result.get('distance_score'), is_optimal=True,
            )
        self.stdout.write(f'  Created {len(result["cluster_centers"])} ambulance stations')
        
        # Add comparison results
        for algo in ['KMeans', 'GMM', 'Agglomerative']:
            r = run_clustering(algorithm=algo, n_clusters=8, data=data)
            ClusteringResult.objects.create(
                algorithm=algo, n_clusters=8,
                silhouette_score=r.get('silhouette_score'),
                davies_bouldin_score=r.get('davies_bouldin_score'),
                distance_score=r.get('distance_score'),
                training_time_sec=r.get('training_time_sec'),
                is_active=False,
            )
        self.stdout.write(f'  Ran comparison algorithms')
        
        self.stdout.write(self.style.SUCCESS('✓ Seeding complete! Run: python manage.py runserver'))
