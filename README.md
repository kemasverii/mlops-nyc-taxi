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
- **Distance Drift** - Pergeseran distribusi jarak trip
- **Target Drift** - Pergeseran prediksi fare
- **Model Info** - Nama & versi model aktif
- **Charts** - Visualisasi distribusi data

---

## 🌐 API Endpoints

| Method | Endpoint            | Deskripsi      |
| ------ | ------------------- | -------------- |
| GET    | `/`                 | Dashboard HTML |
| GET    | `/health`           | Health check   |
| POST   | `/predict`          | Prediksi tarif |
| GET    | `/model/info`       | Info model     |
| GET    | `/monitoring/drift` | Drift metrics  |
| GET    | `/docs`             | Swagger UI     |

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

**Ahmad Sahidin**

**Bereme Putra**

**Nisrina Nur Afifah**

**Khalishah Alya Vanefi**

---

## 📄 License

MIT License
