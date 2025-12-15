# 🚀 NYC Taxi MLOps - CLI Reference

Panduan lengkap command `cli/main.py` untuk mengelola lifecycle model.

## 🔑 Essential Workflow (Wajib Tahu)

Command yang paling sering dipakai untuk maintenance sehari-hari.

| Fitur          | Command                                                         | Fungsi                                                         |
| -------------- | --------------------------------------------------------------- | -------------------------------------------------------------- |
| **Cek Status** | `python cli/main.py registry status`                            | Cek model aktif di **Production (Blue)** & **Staging (Green)** |
| **Lihat Runs** | `python cli/main.py registry runs`                              | Lihat daftar hasil training untuk mengambil `run_id`           |
| **Register**   | `python cli/main.py registry register <RUN_ID> --stage Staging` | Mendaftarkan model baru ke Staging                             |
| **Deploy**     | `python cli/main.py registry promote <VERSION>`                 | Promote model Staging ke Production (Update API)               |
| **Monitoring** | `python cli/main.py serve mlflow`                               | Buka Dashboard MLflow UI                                       |

---

## 🛠️ Model Maintenance

Command untuk manajemen registry dan rollback.

```bash
# Lihat semua versi model yang terdaftar
python cli/main.py registry list

# Rollback production ke versi sebelumnya
python cli/main.py registry rollback <VERSION>
```

---

## 🧠 Training & Experimentation

Command untuk melatih model baru.

```bash
# 1. Bandingkan 5 algoritma sekaligus (Riset)
python cli/main.py train compare-algos

# 2. Tuning hyperparameter (Optuna)
python cli/main.py train tune --model ridge

# 3. Training pipeline otomatis (Train + Register)
python cli/main.py train pipeline --model gradient_boosting --register --stage Staging
```

---

## 🚀 Serving (API)

Command untuk menjalankan aplikasi prediksi.

```bash
# Jalankan API Server (Load Production Model)
# Default port: 8000
python cli/main.py serve start --model models/production_model.joblib
```

---

## 📊 Data Preparation

Command untuk download dan proses data.

```bash
# 1. Download data baru
python cli/main.py data download --year 2024 --months 1,2,3

# 2. Preprocess data (Raw -> Cleaned -> Features)
python cli/main.py data preprocess
```
