# 🚖 NYC Taxi Fare Prediction - MLOps Project

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.124-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Heroku](https://img.shields.io/badge/Heroku-Deployed-purple.svg)](https://heroku.com)

Proyek MLOps end-to-end untuk memprediksi tarif taksi NYC menggunakan data NYC TLC Trip Record.

**🔗 Live Demo:** [https://mlops-nyc-taxi-7eaf5edf4a58.herokuapp.com](https://mlops-nyc-taxi-7eaf5edf4a58.herokuapp.com)

---

## 📋 Fitur Utama

| Komponen       | Deskripsi                                              |
| -------------- | ------------------------------------------------------ |
| **ML Model**   | Random Forest & Gradient Boosting dengan Optuna tuning |
| **API**        | FastAPI dengan auto-documentation (Swagger)            |
| **Dashboard**  | HTML/CSS/JS dengan prediksi real-time                  |
| **Monitoring** | Drift detection untuk Distance & Target (Fare)         |
| **Registry**   | Blue/Green deployment dengan MLflow                    |
| **Deployment** | Docker + Heroku dengan auto-deploy dari GitHub         |

---

## 🏗️ Arsitektur

```
┌─────────────────────────────────────────────────────────┐
│                    BROWSER                               │
│   HTML Dashboard (Prediction + Monitoring)               │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP REST API
┌─────────────────────▼───────────────────────────────────┐
│                  FastAPI Server                          │
│  ├── POST /predict        → Prediksi tarif              │
│  ├── GET  /health         → Status server               │
│  ├── GET  /model/info     → Info model aktif            │
│  ├── GET  /monitoring/drift → Drift metrics             │
│  └── GET  /docs           → Swagger UI                  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              ML Model Layer                              │
│  ├── production_model.joblib (Model aktif)              │
│  ├── MLflow Registry (Version control)                  │
│  └── Reference Stats (Baseline untuk drift)             │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Struktur Project

```
mlops/
├── src/
│   ├── serving/
│   │   ├── api.py              # FastAPI server
│   │   ├── static/index.html   # Dashboard UI
│   │   └── reference_stats.json
│   ├── models/
│   │   ├── train.py            # Training pipeline
│   │   └── registry.py         # MLflow registry
│   └── features/
│       └── engineering.py      # Feature engineering
├── cli/
│   └── main.py                 # CLI commands
├── models/
│   └── production_model.joblib # Model production
├── Dockerfile                  # Docker config
├── heroku.yml                  # Heroku config
└── requirements.txt
```

---

## 🚀 Quick Start

### Lokal (Development)

```bash
# Clone repo
git clone https://github.com/kemasverii/mlops-nyc-taxi.git
cd mlops-nyc-taxi

# Install dependencies
pip install -r requirements.txt

# Jalankan server
python cli/main.py serve start

# Buka browser: http://localhost:8000
```

### Docker

```bash
# Build image
docker build -t mlops-api .

# Run container
docker run -p 8000:8000 mlops-api
```

---

## 🔧 CLI Commands

```bash
# Server
python cli/main.py serve start              # Start API server
python cli/main.py serve start -p 8080      # Custom port
python cli/main.py serve mlflow             # Start MLflow UI

# Model Registry
python cli/main.py registry status          # Lihat Blue/Green status
python cli/main.py registry list            # List semua versi
python cli/main.py registry promote 2       # Promote versi ke production
python cli/main.py registry rollback 1      # Rollback ke versi sebelumnya
python cli/main.py registry runs            # List MLflow runs
python cli/main.py registry register <run_id>  # Register model dari run

# Training
python cli/main.py train quick              # Quick training (no registry)
python cli/main.py train pipeline           # Production training + registry
python cli/main.py train compare-algos      # Compare semua algoritma
python cli/main.py train tune               # Hyperparameter tuning (Optuna)

# Data
python cli/main.py data --help              # Data operations

# Monitoring
python cli/main.py monitor --help           # Monitoring operations

# Model Testing
python cli/main.py model test <version>     # Test model tertentu
python cli/main.py model compare 1 2        # Compare dua versi
```

---

## 📊 Model Registry (Blue/Green)

| Stage       | Deskripsi                   |
| ----------- | --------------------------- |
| 🔵 **BLUE**  | Model production yang aktif |
| 🟢 **GREEN** | Model staging untuk testing |

Promote model dari GREEN ke BLUE:
```bash
python cli/main.py registry promote <version>
```

---

## 📈 Monitoring Dashboard

Dashboard menampilkan:
- **Data Drift (Input)** - Pergeseran distribusi fitur input
- **Prediction Drift (Output)** - Pergeseran distribusi prediksi model
- **Drifted Features** - Jumlah fitur yang terdeteksi drift
- **Model Info** - Nama & versi model aktif
- **Charts** - Visualisasi distribusi data

---

## 🔍 Drift Detection

### Jenis Drift yang Dideteksi

| Jenis                | Yang Dibandingkan      | Library      |
| -------------------- | ---------------------- | ------------ |
| **Data Drift**       | Input features (X)     | Evidently AI |
| **Prediction Drift** | Output predictions (Ŷ) | scipy.stats  |

### Data Drift Detection

| Komponen           | Deskripsi                                                     |
| ------------------ | ------------------------------------------------------------- |
| **Reference Data** | `reference_sample.parquet` (10K dari training)                |
| **Current Data**   | `prediction_logs.json` (100 terakhir)                         |
| **Fitur dicek**    | trip_distance, passenger_count, pickup_hour, pickup_dayofweek |
| **Algoritma**      | Wasserstein Distance                                          |
| **Threshold**      | 0.3 per fitur, 50% dataset                                    |

### Prediction Drift Detection

| Komponen      | Deskripsi                                      |
| ------------- | ---------------------------------------------- |
| **Reference** | `reference_predictions.json` (10K prediksi)    |
| **Current**   | `prediction_logs.json` (100 prediksi terakhir) |
| **Algoritma** | Wasserstein Distance / std                     |
| **Threshold** | 0.3                                            |

### Simulation Modes

| Mode       | Sumber Data                     | Distribusi            | Hasil            |
| ---------- | ------------------------------- | --------------------- | ---------------- |
| **Normal** | `test.parquet` (sample 50 rows) | Sama dengan training  | ✅ No Drift       |
| **Drift**  | Synthetic (trip_distance 8-25)  | Berbeda dari training | ⚠️ Drift Detected |

---

## 🔧 Feature Engineering

### 18 Fitur untuk Training & Prediksi

| #   | Fitur                   | Kategori    | Cara Input/Hitung                |
| --- | ----------------------- | ----------- | -------------------------------- |
| 1   | `trip_distance`         | Numerik     | Auto dari lookup / user input    |
| 2   | `passenger_count`       | Numerik     | User input                       |
| 3   | `trip_duration_minutes` | Numerik     | Dihitung: distance/11*60         |
| 4   | `avg_speed_mph`         | Numerik     | Dihitung: distance/(duration/60) |
| 5   | `pickup_hour`           | Numerik     | User input                       |
| 6   | `pickup_dayofweek`      | Numerik     | User input                       |
| 7   | `pickup_month`          | Numerik     | Random 1-5                       |
| 8   | `hour_sin`              | Cyclical    | sin(2π × hour/24)                |
| 9   | `hour_cos`              | Cyclical    | cos(2π × hour/24)                |
| 10  | `dow_sin`               | Cyclical    | sin(2π × dow/7)                  |
| 11  | `dow_cos`               | Cyclical    | cos(2π × dow/7)                  |
| 12  | `PULocationID`          | Kategorikal | User pilih dropdown              |
| 13  | `DOLocationID`          | Kategorikal | User pilih dropdown              |
| 14  | `VendorID`              | Kategorikal | Fixed=2 atau random              |
| 15  | `is_weekend`            | Binary      | 1 if dow >= 5                    |
| 16  | `is_rush_hour`          | Binary      | 1 if hour in [7,8,9,16,17,18,19] |
| 17  | `same_location`         | Binary      | 1 if PU == DO                    |
| 18  | `has_tolls`             | Binary      | Fixed = 0                        |

### Input dari User (Web Form)

| Fitur              | Range          | Deskripsi        |
| ------------------ | -------------- | ---------------- |
| `PULocationID`     | Dropdown (263) | Zona pickup NYC  |
| `DOLocationID`     | Dropdown (263) | Zona dropoff NYC |
| `passenger_count`  | 1 - 6          | Jumlah penumpang |
| `pickup_hour`      | 0 - 23         | Jam pickup       |
| `pickup_dayofweek` | 0 - 6          | Hari (0=Senin)   |

---

## 📐 Route Distance Lookup

### Cara Kerja

Sistem menggunakan **pre-computed lookup table** untuk estimasi jarak:

1. User pilih Pickup Location (dropdown 263 zona)
2. User pilih Dropoff Location (dropdown 263 zona)
3. Sistem query lookup table → auto-fill `trip_distance`

### Data Lookup

| Statistik              | Nilai          |
| ---------------------- | -------------- |
| **Total Routes**       | 39,307         |
| **Sumber**             | 11M trips      |
| **Algoritma**          | Mean per route |
| **Default (jika N/A)** | 3.0 miles      |

---

## 📁 Reference Files

| File                         | Isi              | Fungsi                     |
| ---------------------------- | ---------------- | -------------------------- |
| `reference_sample.parquet`   | 10K rows input   | Baseline Data Drift        |
| `reference_predictions.json` | 10K prediksi     | Baseline Prediction Drift  |
| `reference_stats.json`       | Statistik fitur  | Dashboard visualization    |
| `route_distances.json`       | 39K rute         | Auto-fill trip_distance    |
| `test_sample.parquet`        | 10K rows test    | Generate Normal simulation |
| `prediction_logs.json`       | History prediksi | Current data untuk compare |

### Scripts untuk Generate Reference Files

```bash
# Generate reference sample (10K input dari training)
python src/scripts/create_reference_sample.py

# Generate reference predictions (10K prediksi)
python src/scripts/create_reference_predictions.py

# Generate reference stats (statistik untuk dashboard)
python src/scripts/compute_reference_stats.py

# Generate test sample (10K untuk deployment)
python src/scripts/create_test_sample.py
```

---

## 📐 Statistical Justification for 10K Sampling

### Cochran's Sample Size Formula

```
n = (Z² × p × (1-p)) / e²

Dimana:
- Z = 2.58 (confidence level 99%)
- p = 0.5 (variasi maksimal)
- e = 0.013 (margin error 1.3%)

Hasil: 10,000 sample cukup untuk 99% confidence, 1.3% margin error
```

### Perbandingan Populasi vs Sample

| Fitur              | Populasi (2.4M) | Sample (10K) | Perbedaan |
| ------------------ | --------------- | ------------ | --------- |
| `trip_distance`    | 3.3642          | 3.4253       | 1.82%     |
| `passenger_count`  | 1.3183          | 1.3184       | **0.01%** |
| `pickup_hour`      | 14.3625         | 14.4166      | **0.38%** |
| `pickup_dayofweek` | 3.0246          | 3.0095       | **0.50%** |

### Kolmogorov-Smirnov Test

| Fitur              | P-value | Kesimpulan              |
| ------------------ | ------- | ----------------------- |
| `trip_distance`    | 0.14    | ✅ Tidak beda signifikan |
| `passenger_count`  | 1.00    | ✅ Identik               |
| `pickup_hour`      | 0.90    | ✅ Tidak beda signifikan |
| `pickup_dayofweek` | 0.93    | ✅ Tidak beda signifikan |

**P-value > 0.05 = Sample berasal dari distribusi yang SAMA dengan populasi!**

### Kesimpulan

✅ **10K sample TERBUKTI secara statistik mewakili populasi 2.4 juta dengan akurasi >98%**

---

## 📊 Design Decisions

### Mengapa AVG_SPEED = 11 mph?

Dihitung dari **11 juta trips** seluruh NYC:
- Mean actual: 11.11 mph
- Dibulatkan: 11 mph

### Mengapa is_rush_hour = [7,8,9,16,17,18,19]?

Sesuai dengan definisi di training data yang mencakup:
- Pagi: 7-9 AM
- Sore: 4-7 PM

### Mengapa VendorID = 2 (atau random)?

Training data: 77% VendorID=2, 23% VendorID=1

Training data hanya berisi bulan Januari-Mei:
- Jan: 1.9 juta trips
- Feb: 2.0 juta trips
- Mar: 2.4 juta trips
- Apr: 2.3 juta trips
- Mei: 2.5 juta trips
---

## 🌐 API Endpoints

| Method | Endpoint            | Deskripsi           |
| ------ | ------------------- | ------------------- |
| GET    | `/`                 | Dashboard HTML      |
| GET    | `/health`           | Health check        |
| POST   | `/predict`          | Prediksi tarif      |
| GET    | `/model/info`       | Info model          |
| GET    | `/zones`            | List 263 zona NYC   |
| GET    | `/route-distance`   | Estimasi jarak rute |
| GET    | `/monitoring/drift` | Drift metrics       |
| GET    | `/docs`             | Swagger UI          |

### Contoh Request

```bash
curl -X POST https://mlops-nyc-taxi-7eaf5edf4a58.herokuapp.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trip_distance": 5.0,
    "passenger_count": 2,
    "pickup_hour": 14,
    "pickup_dayofweek": 2,
    "PULocationID": 161,
    "DOLocationID": 237,
    "pickup_month": 1
  }'
```

---

## 🚢 Deployment

Project ini di-deploy ke Heroku dengan **auto-deploy** dari GitHub:

1. Push ke GitHub → Heroku auto-rebuild
2. Zero-downtime deployment
3. Docker-based containerization

---

## 📝 Tech Stack

- **Backend:** FastAPI, Uvicorn
- **ML:** Scikit-learn, Pandas, NumPy
- **Tracking:** MLflow
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **Deployment:** Docker, Heroku
- **CI/CD:** GitHub (auto-deploy)

---

## 👤 Author

**Kemas Veriandra Ramadhan**

**Ahmad Sahidin Akbar**

**Eli Dwi Putra Berema**

**Nisrina Nur Afifah**

**⁠Khaalishah Zuhrah Alyaa V.**

---

## 📄 License

MIT License
