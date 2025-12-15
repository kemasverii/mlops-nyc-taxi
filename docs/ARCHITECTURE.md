# 🏗️ Architecture - NYC Taxi MLOps Project

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                               │
│                                                                  │
│   Browser (localhost:8501)                                      │
│   └── Streamlit Dashboard (src/mlops_dashboard.py)             │
│       ├── Tab 1: Production (Prediction UI)                     │
│       ├── Tab 2: CI/CD Visualization                           │
│       └── Tab 3: Monitoring & Drift Detection                   │
│                                                                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ HTTP/REST API
                       │ (requests.post/get)
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                     API LAYER                                    │
│                                                                  │
│   FastAPI Server (localhost:8000)                              │
│   └── src/api.py                                                │
│       ├── POST /predict    → Make prediction                    │
│       ├── GET  /health     → Check API status                   │
│       ├── GET  /model/info → Get model metadata                 │
│       └── GET  /docs       → Swagger documentation              │
│                                                                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ joblib.load()
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                     MODEL LAYER                                  │
│                                                                  │
│   ML Model (models/production_model.joblib)                     │
│   └── Random Forest Regressor                                   │
│       ├── 18 features                                           │
│       ├── Trained on NYC Taxi data                             │
│       └── Version: 1.0.0                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Verifikasi: 100% Via API

### Apa yang Via API? ✅

| Aksi | Metode | Endpoint | File |
|------|--------|----------|------|
| **Prediksi Fare** | `requests.post()` | `/predict` | `mlops_dashboard.py:252` |
| **Health Check** | `requests.get()` | `/health` | `mlops_dashboard.py:80` |
| **Model Info** | Available | `/model/info` | API ready |

### Apa yang TIDAK Via API? ❌

| Aksi | Alasan | Lokasi |
|------|--------|--------|
| Load training data untuk monitoring | Data historis, bukan prediksi | Tab Monitoring |
| Simpan prediction log | Local logging | `prediction_logs.json` |

---

## 🔄 Request Flow: User → Prediction

```
1. User Input (Dashboard)
   └── trip_distance: 5.0
   └── pickup_hour: 17
   └── passenger_count: 2

2. Dashboard Prepare Request
   └── payload = {
         "trip_distance": 5.0,
         "pickup_hour": 17,
         ...
       }

3. HTTP POST to API
   └── requests.post("http://localhost:8000/predict", json=payload)

4. API Receives Request
   └── FastAPI validates with Pydantic

5. API Calculates Features
   └── calculate_derived_features()
       ├── trip_duration_minutes (calculated)
       ├── avg_speed_mph (calculated)
       ├── hour_sin, hour_cos (calculated)
       ├── is_weekend, is_rush_hour (calculated)
       └── Total: 18 features

6. API Loads Model
   └── model = joblib.load("production_model.joblib")

7. Model Predicts
   └── prediction = model.predict(features)

8. API Returns JSON
   └── {
         "predicted_fare": 18.50,
         "model_name": "random_forest",
         "model_version": "1.0.0",
         "timestamp": "2025-12-13T..."
       }

9. Dashboard Displays Result
   └── Show: $18.50
   └── Log prediction
```

---

## 📝 Code Evidence

### Dashboard Calls API (NOT Direct Model Loading)

**File:** `src/mlops_dashboard.py`

```python
# Line 252: Prediction via API
response = requests.post(
    f"{API_URL}/predict",
    json=payload,
    timeout=5
)

# Line 80: Health check via API
response = requests.get(f"{API_URL}/health", timeout=2)
```

**NO** `model.predict()` in dashboard code! ✅

---

### API Loads Model and Serves

**File:** `src/api.py`

```python
# Line 37: Load model on startup
@app.on_event("startup")
def load_model():
    global model_package
    model_package = joblib.load(MODEL_PATH)

# Line 155: Make prediction
model = model_package['model']
prediction = model.predict(df)[0]
```

---

## 🎯 Why This Architecture?

| Benefit | Explanation |
|---------|-------------|
| **Separation of Concerns** | UI (Streamlit) ≠ Business Logic (FastAPI) |
| **Scalability** | Multiple frontends can use same API |
| **Security** | Model file not exposed to frontend |
| **Testability** | Can test API independently |
| **Industry Standard** | Same pattern as real production systems |

---

## 🔍 How to Verify It's Using API

### Test 1: Stop API, Dashboard Fails

```bash
# Terminal 1: Start ONLY dashboard (no API)
streamlit run src/mlops_dashboard.py

# Expected: Dashboard shows "❌ API Offline" in sidebar
# Expected: Prediction button shows error
```

### Test 2: API Logs Show Requests

```bash
# Terminal 1: Start API
uvicorn src.api:app --reload --port 8000

# Terminal 2: Start Dashboard & make prediction
streamlit run src/mlops_dashboard.py

# Expected in Terminal 1:
# INFO: 127.0.0.1:XXXXX - "POST /predict HTTP/1.1" 200 OK
```

### Test 3: Check Network Traffic

```python
# In dashboard code, add print before API call:
print(f"Calling API: {API_URL}/predict")
print(f"Payload: {payload}")

# You'll see this in terminal when predicting
```

---

## 📊 Component Responsibilities

| Component | Role | Responsibility |
|-----------|------|----------------|
| **Streamlit Dashboard** | Frontend/UI | • Display forms<br>• Show results<br>• Visualize data<br>• Call API |
| **FastAPI** | Backend/API | • Load model<br>• Validate input<br>• Calculate features<br>• Make predictions |
| **Model** | ML Core | • Store trained model<br>• Make predictions |
| **Data Files** | Storage | • Historical data<br>• Logs |

---

## ✅ Summary

**YES**, dashboard **100% menggunakan FastAPI** untuk prediksi!

- ✅ Prediction: Via API `/predict`
- ✅ Health check: Via API `/health`
- ✅ No direct `model.predict()` in dashboard
- ✅ API owns the model
- ✅ Dashboard is pure UI layer

Ini adalah **proper microservices architecture** sesuai best practice MLOps! 🚀
