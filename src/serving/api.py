"""
Model Serving API
FastAPI application for NYC Taxi Fare prediction
"""

import logging
import os
import datetime
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="MLOps Project - Predict taxi fares in New York City",
    version="1.0.0"
)

# ==========================================
# Pydantic Models
# ==========================================

class TripFeatures(BaseModel):
    """Input features for a single trip prediction."""
    trip_distance: float = Field(..., ge=0.1, le=100, description="Trip distance in miles")
    passenger_count: int = Field(..., ge=1, le=6, description="Number of passengers")
    PULocationID: int = Field(161, ge=1, description="Pickup location ID")
    DOLocationID: int = Field(237, ge=1, description="Dropoff location ID")
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of pickup (0-23)")
    pickup_dayofweek: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    pickup_month: int = Field(1, ge=1, le=12, description="Month (1-12)")
    is_weekend: int = Field(0, ge=0, le=1, description="Is weekend (0 or 1)")
    trip_duration_minutes: float = Field(15.0, ge=1, le=180, description="Trip duration in minutes")

    class Config:
        json_schema_extra = {
            "example": {
                "trip_distance": 2.5,
                "passenger_count": 1,
                "pickup_hour": 14,
                "pickup_dayofweek": 2,
                "pickup_month": 6,
                "is_weekend": 0,
                "trip_duration_minutes": 15.0
            }
        }


class BatchTripFeatures(BaseModel):
    """Input features for batch prediction."""
    trips: List[TripFeatures]


class PredictionResponse(BaseModel):
    """Response for single prediction."""
    predicted_fare: float
    currency: str = "USD"
    model_name: str
    model_version: str
    timestamp: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[float]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool

# ==========================================
# Global Variables & Configuration
# ==========================================

current_dir = Path(__file__).parent
static_dir = current_dir / "static"
static_dir.mkdir(exist_ok=True)

# Mount static directory
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

MONITORING_FILE = current_dir / "prediction_logs.json"
REFERENCE_STATS_FILE = current_dir / "reference_stats.json"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/production_model.joblib"))

model = None

# ==========================================
# Helper Functions
# ==========================================

def load_model():
    """Load model at startup."""
    global model
    
    logger.info(f"Attempting to load model from: {MODEL_PATH}")
    if MODEL_PATH.exists():
        loaded = joblib.load(MODEL_PATH)
        # Check if loaded object is a dict (new format) or model (old format)
        if isinstance(loaded, dict) and "model" in loaded:
            model = loaded
            logger.info(f"Model package loaded from {MODEL_PATH} (v{model.get('version', 'unknown')})")
        else:
            # Wrap legacy model in dict structure
            # Dynamically detect model type from class name
            model_type_name = type(loaded).__name__
            model = {
                "model": loaded,
                "model_name": model_type_name,
                "version": "1.0.0",
                "model_type": model_type_name,
                "features": getattr(loaded, 'feature_names_in_', []).tolist() if hasattr(loaded, 'feature_names_in_') else []
            }
            logger.info(f"Legacy model ({model_type_name}) loaded from {MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}")

def log_prediction(inputs: dict, prediction: float):
    """Log prediction for monitoring."""
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "inputs": inputs,
            "prediction": prediction
        }
        
        logs = []
        if MONITORING_FILE.exists():
            try:
                with open(MONITORING_FILE, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        logs.append(log_entry)
        logs = logs[-1000:] # Keep last 1000
        
        with open(MONITORING_FILE, 'w') as f:
            json.dump(logs, f)
            
    except Exception as e:
        logger.error(f"Logging failed: {e}")

# ==========================================
# Endpoints
# ==========================================

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/", tags=["General"])
async def root():
    return FileResponse(static_dir / "index.html")

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )

@app.get("/model/info", tags=["General"])
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get metrics if available
    metrics = model.get("metrics", {})
    
    return {
        "model_name": model.get("model_name", "nyc-taxi-fare"),
        "model_type": model.get("model_type", "unknown"),
        "version": model.get("version", "unknown"),
        "num_features": len(model.get("features", [])),
        "features": model.get("features", []),
        "metrics": {
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mse": metrics.get("mse"),  # loss
            "mape": metrics.get("mape"),
            "r2": metrics.get("r2")
        },
        "created_at": model.get("created_at"),
        "mlflow_version": model.get("mlflow_version")
    }

@app.get("/monitoring/drift", tags=["Monitoring"])
async def get_drift_metrics():
    """Get drift metrics using Evidently AI."""
    
    # Load Reference Sample (10K rows)
    REFERENCE_SAMPLE_FILE = current_dir / "reference_sample.parquet"
    
    # Default response if no data
    default_response = {
        "dataset_drift": False,
        "drift_share": 0.0,
        "n_features": 0,
        "n_drifted": 0,
        "features": {},
        "reference": {},
        "current": {},
        "total_predictions": 0,
        "model_info": {
            "name": model.get("model_name", "unknown") if model else "Not Loaded",
            "version": model.get("version", "unknown") if model else "N/A"
        },
        "evidently_available": False
    }
    
    # Check if reference sample exists
    if not REFERENCE_SAMPLE_FILE.exists():
        logger.warning("Reference sample not found for Evidently drift detection")
        default_response["error"] = "Reference sample not found. Run create_reference_sample.py"
        return default_response
    
    # Load prediction logs
    logs = []
    if MONITORING_FILE.exists():
        try:
            with open(MONITORING_FILE, 'r') as f:
                logs = json.load(f)
        except:
            logs = []
    
    # Need at least 30 predictions for meaningful drift analysis
    if len(logs) < 30:
        default_response["total_predictions"] = len(logs)
        default_response["error"] = f"Need at least 30 predictions for drift analysis (current: {len(logs)})"
        return default_response
    
    try:
        # Import Evidently (0.7+ API)
        from evidently import Report
        from evidently.presets import DataDriftPreset
        
        # Load reference data
        ref_df = pd.read_parquet(REFERENCE_SAMPLE_FILE)
        
        # Build current DataFrame from recent predictions
        recent_logs = logs[-100:]  # Last 100 predictions
        curr_df = pd.DataFrame([l['inputs'] for l in recent_logs])
        
        # Only use columns that exist in both
        common_cols = list(set(ref_df.columns) & set(curr_df.columns))
        if not common_cols:
            default_response["error"] = "No common features between reference and current data"
            return default_response
        
        ref_df = ref_df[common_cols]
        curr_df = curr_df[common_cols]
        
        # Run Evidently drift report with adjusted thresholds (more tolerant)
        # num_threshold=0.3 for numerical columns, cat_threshold=0.3 for categorical
        report = Report(metrics=[DataDriftPreset(drift_share=0.5, num_threshold=0.3, cat_threshold=0.3)])
        snapshot = report.run(reference_data=ref_df, current_data=curr_df)
        
        # Extract results (Evidently 0.7+ - run() returns snapshot)
        result_dict = snapshot.dict()
        
        # Parse Evidently 0.7+ results
        drift_results = {
            "dataset_drift": False,
            "drift_share": 0.0,
            "n_features": len(common_cols),
            "n_drifted": 0,
            "features": {},
            "reference": {},
            "current": {},
            "total_predictions": len(logs),
            "model_info": {
                "name": model.get("model_name", "unknown") if model else "Not Loaded",
                "version": model.get("version", "unknown") if model else "N/A"
            },
            "evidently_available": True
        }
        
        # Parse Evidently 0.7+ metric structure
        metrics = result_dict.get("metrics", [])
        for metric in metrics:
            metric_name = metric.get("metric_name", "")
            config = metric.get("config", {})
            value = metric.get("value", {})
            
            # DriftedColumnsCount contains overall drift info
            if "DriftedColumnsCount" in metric_name:
                if isinstance(value, dict):
                    n_drifted = int(value.get("count", 0))
                    drift_share = float(value.get("share", 0.0))
                    drift_results["n_drifted"] = n_drifted
                    drift_results["drift_share"] = drift_share
                    # Dataset is drifted if >50% columns drift
                    drift_results["dataset_drift"] = drift_share > 0.5
            
            # ValueDrift contains per-column drift score
            elif "ValueDrift" in metric_name:
                col_name = config.get("column", "")
                threshold = config.get("threshold", 0.1)
                method = config.get("method", "unknown")
                drift_score = float(value) if isinstance(value, (int, float)) else 0.0
                
                if col_name:
                    drift_results["features"][col_name] = {
                        "drift_detected": drift_score > threshold,
                        "drift_score": drift_score,
                        "stattest_name": method,
                        "stattest_threshold": threshold
                    }
        
        # Add simple stats for backward compatibility
        for col in common_cols:
            drift_results["reference"][col] = {"mean": float(ref_df[col].mean())}
            drift_results["current"][col] = {"mean": float(curr_df[col].mean())}
        
        # Add histogram data for overlapping histogram visualization
        drift_results["histograms"] = {}
        for col in ["trip_distance", "passenger_count", "pickup_hour"]:
            if col in common_cols:
                try:
                    # Create histogram bins
                    combined = pd.concat([ref_df[col], curr_df[col]])
                    hist_min = float(combined.min())
                    hist_max = float(combined.max())
                    bins = np.linspace(hist_min, hist_max, 11)  # 10 bins
                    
                    ref_counts, _ = np.histogram(ref_df[col], bins=bins)
                    curr_counts, _ = np.histogram(curr_df[col], bins=bins)
                    
                    # Normalize to percentage
                    ref_pct = (ref_counts / ref_counts.sum() * 100).tolist()
                    curr_pct = (curr_counts / curr_counts.sum() * 100).tolist()
                    
                    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
                    
                    drift_results["histograms"][col] = {
                        "labels": bin_labels,
                        "reference": ref_pct,
                        "current": curr_pct
                    }
                except Exception:
                    pass
        
        # Add fare predictions stats
        predictions = [l['prediction'] for l in recent_logs]
        drift_results["current"]["fare_amount"] = {"mean": float(np.mean(predictions))}
        
        # Add reference fare_amount from reference_stats.json
        REFERENCE_STATS_FILE = current_dir / "reference_stats.json"
        if REFERENCE_STATS_FILE.exists():
            try:
                with open(REFERENCE_STATS_FILE, 'r') as f:
                    ref_stats = json.load(f)
                if "fare_amount" in ref_stats:
                    drift_results["reference"]["fare_amount"] = {"mean": ref_stats["fare_amount"]["mean"]}
            except:
                pass
        
        return drift_results
        
    except ImportError as e:
        logger.error(f"Evidently not installed: {e}")
        default_response["error"] = "Evidently library not installed"
        return default_response
    except Exception as e:
        logger.error(f"Evidently drift detection error: {e}")
        default_response["error"] = str(e)
        default_response["total_predictions"] = len(logs)
        return default_response

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fare(trip: TripFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to DataFrame
    df = pd.DataFrame([trip.model_dump()])
    
    # Feature Engineering
    df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['pickup_dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['pickup_dayofweek'] / 7)
    
    df['VendorID'] = 2 
    df['avg_speed_mph'] = df['trip_distance'] / (df['trip_duration_minutes'] / 60)
    df.loc[df['trip_duration_minutes'] <= 0, 'avg_speed_mph'] = 12.0
    df['avg_speed_mph'] = df['avg_speed_mph'].clip(1, 60)
    
    df['has_tolls'] = 0
    df['is_rush_hour'] = df['pickup_hour'].apply(lambda x: 1 if 16 <= x <= 19 else 0)
    df['same_location'] = (df['PULocationID'] == df['DOLocationID']).astype(int)
    
    # Column Reordering
    model_obj = model["model"]
    if hasattr(model_obj, "feature_names_in_"):
        required_features = model_obj.feature_names_in_
        for col in required_features:
            if col not in df.columns:
                df[col] = 0
        df = df[required_features]
    
    # Prediction
    try:
        prediction = model_obj.predict(df)[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    prediction = max(0, prediction)
    
    # Logging
    log_prediction(trip.model_dump(), float(prediction))
    
    return PredictionResponse(
        predicted_fare=round(prediction, 2),
        currency="USD",
        model_name=model.get("model_name", "nyc-taxi-fare"),
        model_version=model.get("version", "unknown"),
        timestamp=datetime.datetime.now().isoformat()
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchTripFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    df = pd.DataFrame([trip.model_dump() for trip in batch.trips])
    try:
        predictions = model["model"].predict(df)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    predictions = np.maximum(predictions, 0)
    
    return BatchPredictionResponse(
        predictions=[round(p, 2) for p in predictions],
        count=len(predictions)
    )

@app.post("/reload-model", tags=["Admin"])
async def reload_model_endpoint():
    load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Failed to load model")
    return {
        "message": "Model reloaded successfully",
        "version": model.get("version", "unknown")
    }

@app.post("/monitoring/simulate", tags=["Monitoring"])
async def simulate_data(mode: str = "normal"):
    """
    Generate 50 simulated predictions for drift visualization.
    mode: 'normal' = similar to training data, 'drift' = different from training
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    np.random.seed(42)
    count = 50
    
    if mode == "drift":
        # Generate data DIFFERENT from training (causes drift)
        simulated_inputs = [
            {
                "trip_distance": float(np.random.uniform(8, 25)),  # Training mean ~3.4
                "passenger_count": int(np.random.choice([1, 1, 1, 2])),
                "pickup_hour": int(np.random.choice([14, 15, 16])),  # Not varied
                "pickup_dayofweek": int(np.random.choice([2, 3])),  # Not varied
                "pickup_month": 1,  # Always January
                "PULocationID": 161,
                "DOLocationID": 237,
                "is_weekend": 0,
                "trip_duration_minutes": float(np.random.uniform(20, 60))
            }
            for _ in range(count)
        ]
    else:  # normal
        # Generate data SIMILAR to training (no drift)
        # Match reference: trip_distance~3.4, pickup_hour~14.4, pickup_dayofweek~3.0, pickup_month~3.15
        simulated_inputs = [
            {
                "trip_distance": float(max(0.1, np.random.exponential(3.0) + 0.4)),  # ~3.4
                "passenger_count": int(np.random.choice([1, 1, 1, 1, 2, 2, 3])),  # ~1.3
                "pickup_hour": int(np.clip(np.random.normal(14.4, 5.8), 0, 23)),  # ~14.4
                "pickup_dayofweek": int(np.random.randint(0, 7)),  # ~3.0
                "pickup_month": int(np.random.choice([1, 2, 3, 3, 3, 4, 5])),  # ~3.15
                "PULocationID": 161,
                "DOLocationID": 237,
                "is_weekend": int(np.random.choice([0, 0, 0, 0, 0, 1, 1])),  # ~30%
                "trip_duration_minutes": float(np.random.uniform(5, 40))
            }
            for _ in range(count)
        ]
    
    # Make predictions and log them
    predictions = []
    for inputs in simulated_inputs:
        try:
            df = pd.DataFrame([inputs])
            df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['pickup_dayofweek'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['pickup_dayofweek'] / 7)
            df['avg_speed_mph'] = df['trip_distance'] / (df['trip_duration_minutes'] / 60 + 0.01)
            df['is_rush_hour'] = df['pickup_hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)
            df['same_location'] = (df['PULocationID'] == df['DOLocationID']).astype(int)
            df['has_tolls'] = 0
            df['VendorID'] = 1
            
            features = model.get("features", [])
            X = df[features]
            prediction = float(model["model"].predict(X)[0])
            predictions.append(prediction)
            log_prediction(inputs, prediction)
        except Exception as e:
            logger.error(f"Simulation error: {e}")
    
    return {
        "message": f"Generated {count} {mode} predictions",
        "mode": mode,
        "count": count,
        "avg_distance": float(np.mean([i['trip_distance'] for i in simulated_inputs])),
        "avg_prediction": float(np.mean(predictions)) if predictions else 0
    }

@app.post("/monitoring/clear", tags=["Monitoring"])
async def clear_logs():
    """Clear all prediction logs to reset drift detection."""
    try:
        with open(MONITORING_FILE, 'w') as f:
            json.dump([], f)
        return {"message": "Prediction logs cleared", "count": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
