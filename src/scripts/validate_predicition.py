"""
Validate Model Predictions vs Actual Fare
Script untuk memvalidasi prediksi model dengan fare asli dari test data

Cara pakai:
    python src/scripts/validate_predictions.py

Atau dengan filter custom:
    python src/scripts/validate_predictions.py --pu 161 --do 237 --hour 14
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# PATHS - Tidak perlu diubah
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "production_model.joblib"


def validate_predictions(
    pu_location: int = None,      # GANTI: ID lokasi pickup (1-263)
    do_location: int = None,      # GANTI: ID lokasi dropoff (1-263)
    passenger_count: int = None,  # GANTI: Jumlah penumpang (1-6)
    pickup_hour: int = None,      # GANTI: Jam pickup (0-23)
    pickup_dayofweek: int = None, # GANTI: Hari (0=Senin, 6=Minggu)
    vendor_id: int = None,        # GANTI: VendorID (1 atau 2)
    sample_size: int = 100        # GANTI: Jumlah sample untuk validasi
):
    """
    Validasi prediksi model vs actual fare dari test data.
    
    Args:
        pu_location: Filter PULocationID (opsional)
        do_location: Filter DOLocationID (opsional)
        passenger_count: Filter passenger_count (opsional)
        pickup_hour: Filter pickup_hour (opsional)
        pickup_dayofweek: Filter pickup_dayofweek (opsional)
        vendor_id: Filter VendorID (opsional)
        sample_size: Jumlah sample untuk validasi
    """
    
    # Load test data
    logger.info(f"Loading test data from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Total test data: {len(df):,} rows")
    
    # Apply filters
    # ==========================================
    # GANTI NILAI DI BAWAH INI SESUAI KEBUTUHAN
    # Set None jika tidak ingin filter
    # ==========================================
    
    filtered = df.copy()
    filters_applied = []
    
    if pu_location is not None:
        filtered = filtered[filtered['PULocationID'] == pu_location]
        filters_applied.append(f"PULocationID={pu_location}")
    
    if do_location is not None:
        filtered = filtered[filtered['DOLocationID'] == do_location]
        filters_applied.append(f"DOLocationID={do_location}")
    
    if passenger_count is not None:
        filtered = filtered[filtered['passenger_count'] == passenger_count]
        filters_applied.append(f"passenger_count={passenger_count}")
    
    if pickup_hour is not None:
        filtered = filtered[filtered['pickup_hour'] == pickup_hour]
        filters_applied.append(f"pickup_hour={pickup_hour}")
    
    if pickup_dayofweek is not None:
        filtered = filtered[filtered['pickup_dayofweek'] == pickup_dayofweek]
        filters_applied.append(f"pickup_dayofweek={pickup_dayofweek}")
    
    if vendor_id is not None:
        filtered = filtered[filtered['VendorID'] == vendor_id]
        filters_applied.append(f"VendorID={vendor_id}")
    
    logger.info(f"\nFilters: {', '.join(filters_applied) if filters_applied else 'None'}")
    logger.info(f"Matched: {len(filtered):,} rows")
    
    if len(filtered) == 0:
        logger.error("No data matched the filters!")
        return None
    
    # Sample data
    sample_size = min(sample_size, len(filtered))
    sample = filtered.sample(n=sample_size, random_state=42)
    
    # Load model
    logger.info(f"\nLoading model from: {MODEL_PATH}")
    model_pkg = joblib.load(MODEL_PATH)
    model = model_pkg['model']
    features = list(model.feature_names_in_)
    
    # Predict
    X = sample[features]
    y_actual = sample['fare_amount'].values
    y_predicted = model.predict(X)
    
    # Calculate metrics
    errors = np.abs(y_actual - y_predicted)
    mae = errors.mean()
    rmse = np.sqrt(((y_actual - y_predicted) ** 2).mean())
    mape = (errors / y_actual).mean() * 100
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("STATISTIK DATA ASLI")
    logger.info("=" * 60)
    logger.info(f"  Fare Amount:")
    logger.info(f"    Mean:   ${sample['fare_amount'].mean():.2f}")
    logger.info(f"    Median: ${sample['fare_amount'].median():.2f}")
    logger.info(f"    Min:    ${sample['fare_amount'].min():.2f}")
    logger.info(f"    Max:    ${sample['fare_amount'].max():.2f}")
    logger.info(f"  Trip Distance:")
    logger.info(f"    Mean:   {sample['trip_distance'].mean():.2f} miles")
    
    logger.info("\n" + "=" * 60)
    logger.info("METRICS VALIDASI")
    logger.info("=" * 60)
    logger.info(f"  MAE  (Mean Absolute Error):  ${mae:.2f}")
    logger.info(f"  RMSE (Root Mean Square Error): ${rmse:.2f}")
    logger.info(f"  MAPE (Mean % Error):          {mape:.1f}%")
    
    logger.info("\n" + "=" * 60)
    logger.info("DISTRIBUSI ERROR")
    logger.info("=" * 60)
    logger.info(f"  Error < $1:   {(errors < 1).sum() / len(errors) * 100:.1f}%")
    logger.info(f"  Error < $2:   {(errors < 2).sum() / len(errors) * 100:.1f}%")
    logger.info(f"  Error < $5:   {(errors < 5).sum() / len(errors) * 100:.1f}%")
    logger.info(f"  Error < $10:  {(errors < 10).sum() / len(errors) * 100:.1f}%")
    
    logger.info("\n" + "=" * 60)
    logger.info("CONTOH PREDIKSI vs ACTUAL (10 sampel)")
    logger.info("=" * 60)
    logger.info("\n   Distance | Actual | Predicted | Error")
    logger.info("-" * 50)
    for i in range(min(10, len(sample))):
        dist = sample['trip_distance'].iloc[i]
        actual = y_actual[i]
        pred = y_predicted[i]
        err = abs(actual - pred)
        logger.info(f"   {dist:6.2f}  | ${actual:6.2f} | ${pred:6.2f}   | ${err:.2f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'sample_size': len(sample),
        'y_actual': y_actual,
        'y_predicted': y_predicted
    }


def main():
    parser = argparse.ArgumentParser(description='Validate model predictions vs actual fare')
    
    # ==========================================
    # ARGUMEN COMMAND LINE
    # Contoh: python validate_predictions.py --pu 161 --do 237 --hour 14
    # ==========================================
    parser.add_argument('--pu', type=int, help='PULocationID filter (1-263)')
    parser.add_argument('--do', type=int, help='DOLocationID filter (1-263)')
    parser.add_argument('--passenger', type=int, help='Passenger count filter (1-6)')
    parser.add_argument('--hour', type=int, help='Pickup hour filter (0-23)')
    parser.add_argument('--dow', type=int, help='Day of week filter (0=Mon, 6=Sun)')
    parser.add_argument('--vendor', type=int, help='VendorID filter (1 or 2)')
    parser.add_argument('--sample', type=int, default=100, help='Sample size (default: 100)')
    
    args = parser.parse_args()
    
    validate_predictions(
        pu_location=args.pu,
        do_location=args.do,
        passenger_count=args.passenger,
        pickup_hour=args.hour,
        pickup_dayofweek=args.dow,
        vendor_id=args.vendor,
        sample_size=args.sample
    )


if __name__ == "__main__":
    main()
