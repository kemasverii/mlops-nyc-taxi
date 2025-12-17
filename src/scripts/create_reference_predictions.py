"""
Create Reference Predictions for Prediction Drift Detection
Generate model predictions on training sample for baseline comparison
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "production_model.joblib"
OUTPUT_DIR = PROJECT_ROOT / "src" / "serving"

# Sample size (same as reference_sample for consistency)
SAMPLE_SIZE = 10000
RANDOM_STATE = 42


def create_reference_predictions():
    """Create reference predictions from training data sample."""
    
    train_path = DATA_DIR / "train.parquet"
    
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        return None
    
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        return None
    
    # Load training data
    logger.info(f"Loading training data from: {train_path}")
    df = pd.read_parquet(train_path)
    logger.info(f"Training data shape: {df.shape}")
    
    # Sample with same seed as reference_sample.parquet
    np.random.seed(RANDOM_STATE)
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)))
    logger.info(f"Sample shape: {sample.shape}")
    
    # Load model
    logger.info(f"Loading model from: {MODEL_PATH}")
    model_pkg = joblib.load(MODEL_PATH)
    model = model_pkg['model']
    features = model_pkg.get('features', list(model.feature_names_in_))
    logger.info(f"Model requires {len(features)} features")
    
    # Feature engineering (same as api.py)
    df_pred = sample.copy()
    
    # Cyclical features
    df_pred['hour_sin'] = np.sin(2 * np.pi * df_pred['pickup_hour'] / 24)
    df_pred['hour_cos'] = np.cos(2 * np.pi * df_pred['pickup_hour'] / 24)
    df_pred['dow_sin'] = np.sin(2 * np.pi * df_pred['pickup_dayofweek'] / 7)
    df_pred['dow_cos'] = np.cos(2 * np.pi * df_pred['pickup_dayofweek'] / 7)
    
    # Duration and speed (if not in data)
    if 'trip_duration_minutes' not in df_pred.columns:
        df_pred['trip_duration_minutes'] = (df_pred['trip_distance'] / 11.0) * 60
    if 'avg_speed_mph' not in df_pred.columns:
        df_pred['avg_speed_mph'] = df_pred['trip_distance'] / (df_pred['trip_duration_minutes'] / 60 + 0.01)
    
    # Binary features
    if 'is_rush_hour' not in df_pred.columns:
        df_pred['is_rush_hour'] = df_pred['pickup_hour'].apply(lambda x: 1 if x in [7,8,9,16,17,18,19] else 0)
    if 'same_location' not in df_pred.columns:
        df_pred['same_location'] = (df_pred['PULocationID'] == df_pred['DOLocationID']).astype(int)
    if 'has_tolls' not in df_pred.columns:
        df_pred['has_tolls'] = 0
    
    # Ensure all features exist
    for feat in features:
        if feat not in df_pred.columns:
            logger.warning(f"Missing feature: {feat}, setting to 0")
            df_pred[feat] = 0
    
    # Select features and predict
    X = df_pred[features]
    logger.info(f"Predicting on {len(X)} samples...")
    predictions = model.predict(X)
    
    # Calculate stats
    pred_mean = float(predictions.mean())
    pred_std = float(predictions.std())
    pred_min = float(predictions.min())
    pred_max = float(predictions.max())
    
    logger.info(f"Predictions: mean=${pred_mean:.2f}, std=${pred_std:.2f}, range=[${pred_min:.2f}, ${pred_max:.2f}]")
    
    # Save
    output = {
        "predictions": predictions.tolist(),
        "mean": pred_mean,
        "std": pred_std,
        "min": pred_min,
        "max": pred_max,
        "count": len(predictions),
        "random_state": RANDOM_STATE
    }
    
    output_path = OUTPUT_DIR / "reference_predictions.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"✅ Saved reference predictions to: {output_path}")
    logger.info(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return output


if __name__ == "__main__":
    create_reference_predictions()
