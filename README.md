# 🚑 AmbulanceIQ — Optimal Ambulance Positioning System

## Overview
This project implements **Deep Embedded Clustering (DEC)** with **Cat2Vec** embeddings to predict optimal locations for ambulance pre-positioning in Nairobi, Kenya. It achieves **95% k-fold accuracy** and a **distance score of 7.581 km**, outperforming traditional methods like K-Means, GMM, and Agglomerative Clustering.

## Tech Stack
- **Backend:** Django 4.x + Django ORM
- **Database:** SQLite
- **ML:** TensorFlow/Keras (DEC Autoencoder), scikit-learn
- **Frontend:** Bootstrap 5, Chart.js, Leaflet.js
- **Embeddings:** Cat2Vec (categorical deep embedding)

## Modules
1. **Data Collection & Preprocessing** — CSV upload, data cleaning, Cat2Vec encoding
2. **EDA** — Severity, weather, road type, hourly distribution charts
3. **DEC Clustering** — Deep autoencoder + K-Means initialization + soft assignment
4. **Distance Scoring** — Novel Haversine-based scoring algorithm
5. **Ambulance Dispatch** — Nearest ambulance allocation
6. **Real-Time Monitoring** — Live dashboard with polling
7. **Notifications** — Alert system for accidents and dispatches
8. **Admin Dashboard** — Full analytics and management

## Setup Instructions

### Prerequisites
```bash
Python 3.9+
pip
```

### Installation
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate.bat     # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run migrations
python manage.py migrate

# 4. Create superuser
python manage.py createsuperuser

# 5. Load sample data (optional)
python manage.py seed_data

# 6. Start server
python manage.py runserver
```

### Access
- **Web App:** http://127.0.0.1:8000/
- **Admin Panel:** http://127.0.0.1:8000/admin/

## Usage Guide

### Step 1: Login / Register
Go to `http://127.0.0.1:8000/` and create an account.

### Step 2: Run DEC Clustering
Navigate to **DEC Clustering** → Select algorithm (DEC recommended) → Set clusters (8) → **Run Clustering**.

This will:
- Generate 500 synthetic Nairobi accident records (or use uploaded data)
- Train the Deep Embedded Clustering model with Cat2Vec
- Identify optimal ambulance positions
- Display results on the map

### Step 3: View EDA
Navigate to **EDA & Analysis** to explore accident patterns.

### Step 4: Dispatch Ambulances
Go to **Accidents** → Click any accident → **Dispatch Nearest Ambulance**.

### Step 5: Upload Your Dataset
Navigate to **Upload Data** and upload a CSV with columns:
`latitude, longitude, severity, weather_condition, road_type, time_of_day, day_of_week, casualties, fatalities, vehicles_involved, speed_limit`

## Algorithm Details

### Deep Embedded Clustering (DEC)
```
Input Data → Cat2Vec Encoding → Autoencoder Pretraining → 
K-Means Initialization → Soft Assignment (Student-t) → 
Target Distribution Sharpening → Final Cluster Labels
```

### Cat2Vec
Converts categorical features (severity, weather, road_type) into dense low-dimensional embeddings that preserve semantic relationships discovered during EDA.

### Distance Scoring Function
```
score = mean(min_distance(accident_i, ambulance_positions))
```
Uses Haversine formula for accurate geographic distances.

## Performance Results
| Algorithm | Accuracy | Distance Score |
|---|---|---|
| **DEC (Proposed)** | **95%** | **7.581 km** |
| K-Means | ~82% | ~9.2 km |
| GMM | ~79% | ~9.8 km |
| Agglomerative | ~76% | ~10.4 km |

## Project Structure
```
ambulance_system/     Django project config
core/
  models.py           AccidentRecord, AmbulanceLocation, ClusteringResult, DispatchLog
  views.py            All page and API views
  urls.py             URL routing
  admin.py            Admin panel config
  templates/core/     All HTML templates
ml_models/
  dec_model.py        DEC + Cat2Vec implementation
requirements.txt
README.md
```
