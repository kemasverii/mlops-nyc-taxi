# Dokumentasi Drift Detection - NYC Taxi Fare Predictor

## 1. Pendahuluan

### Apa itu Data Drift?
Data drift adalah perubahan distribusi data input yang diterima model di production dibandingkan dengan data yang digunakan saat training. Drift dapat menyebabkan degradasi performa model karena model memprediksi data yang "belum pernah dilihat" sebelumnya.

### Mengapa Drift Detection Penting?
- **Early Warning**: Mendeteksi masalah sebelum berdampak ke pengguna
- **Model Maintenance**: Mengetahui kapan model perlu di-retrain
- **Quality Assurance**: Memastikan prediksi model tetap reliable

---

## 2. Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DRIFT DETECTION FLOW                        │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Training Data   │───►│ Reference Sample │    │  Prediction Logs │
│ (11M+ rows)      │    │ (10,000 rows)    │    │ (real-time)      │
└──────────────────┘    └────────┬─────────┘    └────────┬─────────┘
                                 │                       │
                                 └───────────┬───────────┘
                                             │
                                             ▼
                                 ┌──────────────────────┐
                                 │   Evidently AI       │
                                 │   Drift Detection    │
                                 │                      │
                                 │ • Wasserstein Dist.  │
                                 │ • Statistical Tests  │
                                 └──────────┬───────────┘
                                            │
                                            ▼
                                 ┌──────────────────────┐
                                 │   Dashboard UI       │
                                 │ • Drift Status       │
                                 │ • Per-column Details │
                                 │ • Histograms         │
                                 └──────────────────────┘
```

---

## 3. Komponen Sistem

### 3.1 Reference Data (Baseline)

**File**: `src/serving/reference_sample.parquet`

**Dibuat oleh**: `src/scripts/create_reference_sample.py`

**Isi**: 10,000 baris random sample dari data training

**Fitur yang dipantau** (hanya yang user input):
| Fitur              | Deskripsi                | Contoh Nilai      |
| ------------------ | ------------------------ | ----------------- |
| `trip_distance`    | Jarak perjalanan (miles) | 0.1 - 100         |
| `passenger_count`  | Jumlah penumpang         | 1 - 6             |
| `pickup_hour`      | Jam pickup (0-23)        | 0, 8, 14, 22      |
| `pickup_dayofweek` | Hari dalam minggu (0-6)  | 0=Senin, 6=Minggu |

**Statistik Reference**:
```
trip_distance:     mean=3.40, std=4.34
passenger_count:   mean=1.31, std=0.79
pickup_hour:       mean=14.40, std=5.81
pickup_dayofweek:  mean=3.01, std=1.90
```

### 3.2 Current Data (Production)

**File**: `src/serving/prediction_logs.json`

**Update**: Setiap kali `/predict` dipanggil

**Format**:
```json
{
  "timestamp": "2025-12-15T15:30:00",
  "inputs": {
    "trip_distance": 5.5,
    "passenger_count": 2,
    "pickup_hour": 14,
    "pickup_dayofweek": 2
  },
  "prediction": 25.50
}
```

---

## 4. Algoritma Drift Detection

### 4.1 Wasserstein Distance (Earth Mover's Distance)

Untuk data numerik, Evidently menggunakan **Wasserstein Distance**.

**Intuisi**: Bayangkan distribusi data sebagai tumpukan tanah. Wasserstein distance mengukur "biaya minimum" untuk memindahkan satu tumpukan menjadi bentuk tumpukan lainnya.

**Formula**:
```
W(P, Q) = ∫|F_P(x) - F_Q(x)| dx

Dimana:
- F_P = CDF (Cumulative Distribution Function) dari Reference
- F_Q = CDF dari Current
```

**Contoh Perhitungan Manual**:
```python
Reference: [1.0, 2.0, 3.0, 4.0, 5.0]  # Sudah diurutkan
Current:   [8.0, 9.0, 10.0, 11.0, 12.0]

# Hitung jarak per pasangan (quantile matching)
Jarak = [|1-8|, |2-9|, |3-10|, |4-11|, |5-12|]
      = [7, 7, 7, 7, 7]

Wasserstein = mean(Jarak) = 7.0

# Normalisasi (dibagi range gabungan)
Range = max(12) - min(1) = 11
Normalized = 7.0 / 11 = 0.636

# Bandingkan dengan threshold
Threshold = 0.3
0.636 > 0.3 → DRIFT DETECTED!
```

### 4.2 Threshold dan Keputusan

| Parameter       | Nilai | Keterangan                          |
| --------------- | ----- | ----------------------------------- |
| `num_threshold` | 0.3   | Threshold per kolom numerik         |
| `cat_threshold` | 0.3   | Threshold per kolom kategorikal     |
| `drift_share`   | 0.5   | Dataset drift jika >50% kolom drift |

**Keputusan Drift**:
```
Per-Column: score > 0.3 → Column Drifted
Dataset:    drifted_columns / total_columns > 0.5 → Dataset Drifted
```

---

## 5. Kode Implementasi

### 5.1 Generate Reference Sample

**File**: `src/scripts/create_reference_sample.py`

```python
DRIFT_FEATURES = [
    'trip_distance',
    'passenger_count',
    'pickup_hour',
    'pickup_dayofweek'
]

def create_reference_sample():
    # Load training data (11M+ rows)
    df = pd.read_parquet("data/processed/train.parquet")
    
    # Random sample 10,000 rows
    sample_df = df[DRIFT_FEATURES].sample(n=10000, random_state=42)
    
    # Save untuk drift detection
    sample_df.to_parquet("src/serving/reference_sample.parquet")
```

### 5.2 Drift Detection API

**File**: `src/serving/api.py` (Line 205-385)

```python
@app.get("/monitoring/drift")
async def get_drift_metrics():
    # 1. Load Reference Data
    ref_df = pd.read_parquet("reference_sample.parquet")
    
    # 2. Load Current Data (100 prediksi terakhir)
    with open("prediction_logs.json", 'r') as f:
        logs = json.load(f)
    curr_df = pd.DataFrame([l['inputs'] for l in logs[-100:]])
    
    # 3. Jalankan Evidently
    from evidently import Report
    from evidently.presets import DataDriftPreset
    
    report = Report(metrics=[DataDriftPreset(
        drift_share=0.5,
        num_threshold=0.3,
        cat_threshold=0.3
    )])
    snapshot = report.run(reference_data=ref_df, current_data=curr_df)
    result = snapshot.dict()
    
    # 4. Parse hasil
    for metric in result["metrics"]:
        if "DriftedColumnsCount" in metric["metric_name"]:
            n_drifted = metric["value"]["count"]
            drift_share = metric["value"]["share"]
        
        elif "ValueDrift" in metric["metric_name"]:
            column = metric["config"]["column"]
            score = metric["value"]
            is_drifted = score > 0.3
    
    return drift_results
```

### 5.3 Dashboard UI

**File**: `src/serving/static/index.html` (Line 175-320)

```html
<!-- Drift Metrics Grid -->
<div id="drift-dataset-val">⚠️ YES / ✅ NO</div>
<div id="drift-features-val">2/4</div>

<!-- Per-Column Details -->
<div id="drift-columns-list">
    <!-- Generated by JavaScript -->
    <div class="bg-red-50">
        ⚠️ trip_distance: 1.587 (threshold: 0.3)
    </div>
</div>
```

---

## 6. Cara Penggunaan

### 6.1 Melihat Drift Dashboard
1. Buka `http://localhost:8000`
2. Klik tab **Monitoring**
3. Lihat status drift di bagian atas

### 6.2 Simulasi Drift
1. Klik tombol **Clear Logs** (reset data)
2. Klik **Generate Drift Data** (data berbeda dari training)
3. Klik **Refresh** - Dashboard akan menunjukkan DRIFT

### 6.3 Simulasi Normal
1. Klik tombol **Clear Logs**
2. Klik **Generate Normal Data** (data mirip training)
3. Klik **Refresh** - Dashboard akan menunjukkan NO DRIFT

---

## 7. Interpretasi Hasil

### 7.1 Per-Column Drift Details

| Status | Warna | Arti                                  |
| ------ | ----- | ------------------------------------- |
| ⚠️      | Merah | Kolom ini drifted (score > threshold) |
| ✅      | Hijau | Kolom ini stabil (score ≤ threshold)  |

### 7.2 Contoh Interpretasi

```
📋 Per-Column Drift Details

⚠️ trip_distance     1.587  threshold: 0.3
   → Jarak rata-rata prediksi SANGAT BERBEDA dari training
   → Training: 3.4 miles, Current: 16.5 miles
   → 1.587 >> 0.3 → DRIFT

✅ passenger_count   0.181  threshold: 0.3
   → Jumlah penumpang masih mirip training
   → 0.181 < 0.3 → STABIL

⚠️ pickup_hour       0.399  threshold: 0.3
   → Jam pickup berbeda dari training
   → Training: variatif (0-23), Current: selalu jam 14
   → 0.399 > 0.3 → DRIFT

✅ pickup_dayofweek  0.110  threshold: 0.3
   → Hari pickup masih mirip training
   → 0.110 < 0.3 → STABIL

Dataset Status: 2/4 = 50% → DRIFTED (karena ≥50%)
```

---

## 8. FAQ

### Q: Kapan drift dihitung?
**A**: Hanya saat endpoint `/monitoring/drift` dipanggil (Refresh di dashboard), BUKAN setiap prediksi.

### Q: Berapa minimum prediksi untuk analisis?
**A**: Minimal 30 prediksi untuk hasil yang bermakna secara statistik.

### Q: Mengapa threshold 0.3?
**A**: Default Evidently adalah 0.1, tetapi terlalu ketat untuk data produksi. 0.3 memberikan toleransi lebih untuk variasi normal.

### Q: Apa yang harus dilakukan jika drift terdeteksi?
**A**: 
1. Analisis kolom mana yang drift
2. Investigasi penyebab (data source berubah? bug?)
3. Pertimbangkan retrain model dengan data baru

---

## 9. File Referensi

| File                                     | Fungsi                         |
| ---------------------------------------- | ------------------------------ |
| `src/scripts/create_reference_sample.py` | Generate reference data        |
| `src/serving/reference_sample.parquet`   | File reference data (10K rows) |
| `src/serving/prediction_logs.json`       | Log prediksi real-time         |
| `src/serving/api.py`                     | API endpoint drift detection   |
| `src/serving/static/index.html`          | Dashboard UI                   |
| `src/serving/reference_stats.json`       | Statistik lengkap reference    |
