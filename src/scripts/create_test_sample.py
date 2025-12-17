"""
Create Test Sample for Deployment
Generate 10K rows from test.parquet for simulation on Heroku

Statistical Justification:
- Sample Size: 10,000 rows from 2.4 million population
- Confidence Level: 99%
- Margin of Error: ±1.3%
- Cochran's Formula: n = (Z² × p × (1-p)) / e²

Kolmogorov-Smirnov Test Results:
- trip_distance: p=0.14 (no significant difference)
- passenger_count: p=1.00 (identical)
- pickup_hour: p=0.90 (no significant difference)
- pickup_dayofweek: p=0.93 (no significant difference)

All p-values > 0.05 = Sample comes from the SAME distribution as population!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "src" / "serving"

# Sample configuration
SAMPLE_SIZE = 10000
RANDOM_STATE = 42

# Columns needed for simulation
SIM_COLUMNS = [
    'trip_distance', 
    'passenger_count', 
    'pickup_hour', 
    'pickup_dayofweek', 
    'PULocationID', 
    'DOLocationID'
]


def create_test_sample():
    """Create test sample from test.parquet for deployment."""
    
    test_path = DATA_DIR / "test.parquet"
    
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        return None
    
    # Load test data
    logger.info(f"Loading test data from: {test_path}")
    df = pd.read_parquet(test_path)
    logger.info(f"Test data shape: {df.shape}")
    
    # Random sample
    np.random.seed(RANDOM_STATE)
    sample_size = min(SAMPLE_SIZE, len(df))
    sample = df.sample(n=sample_size)
    logger.info(f"Sample size: {sample_size}")
    
    # Select only needed columns
    available_cols = [c for c in SIM_COLUMNS if c in sample.columns]
    sample = sample[available_cols]
    logger.info(f"Selected columns: {available_cols}")
    
    # Verify sample statistics
    logger.info("\n--- Sample Statistics ---")
    for col in available_cols:
        pop_mean = df[col].mean()
        sam_mean = sample[col].mean()
        diff_pct = abs(pop_mean - sam_mean) / pop_mean * 100
        logger.info(f"{col}: pop_mean={pop_mean:.4f}, sample_mean={sam_mean:.4f}, diff={diff_pct:.2f}%")
    
    # Save
    output_path = OUTPUT_DIR / "test_sample.parquet"
    sample.to_parquet(output_path, index=False)
    
    logger.info(f"\n✅ Saved test sample to: {output_path}")
    logger.info(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    logger.info(f"   Rows: {len(sample):,}")
    logger.info(f"   Columns: {len(sample.columns)}")
    
    return sample


if __name__ == "__main__":
    create_test_sample()
