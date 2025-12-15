"""
Create Reference Sample for Evidently Drift Detection
Generate 10,000 row sample from training data for drift comparison
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "src" / "serving"

# Sample size (statistically calculated: 1% margin of error, 95% confidence)
SAMPLE_SIZE = 10000

# Features to include for drift detection (only user-input features)
DRIFT_FEATURES = [
    'trip_distance',
    'passenger_count',
    'pickup_hour',
    'pickup_dayofweek'
    # pickup_month removed - fixed value, not user input
]


def create_reference_sample():
    """Create reference sample from training data."""
    
    train_path = DATA_DIR / "train.parquet"
    
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        return None
    
    logger.info(f"Loading training data from: {train_path}")
    df = pd.read_parquet(train_path)
    logger.info(f"Training data shape: {df.shape}")
    
    # Select only drift features
    available_features = [f for f in DRIFT_FEATURES if f in df.columns]
    logger.info(f"Available features for drift: {available_features}")
    
    # Random sample
    sample_size = min(SAMPLE_SIZE, len(df))
    sample_df = df[available_features].sample(n=sample_size, random_state=42)
    logger.info(f"Sample shape: {sample_df.shape}")
    
    # Save
    output_path = OUTPUT_DIR / "reference_sample.parquet"
    sample_df.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved reference sample to: {output_path}")
    logger.info(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return sample_df


if __name__ == "__main__":
    create_reference_sample()
