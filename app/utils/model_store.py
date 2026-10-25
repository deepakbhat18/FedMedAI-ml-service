import os
import joblib
import json
import numpy as np
from datetime import datetime
from config import MODEL_STORAGE_DIR

def save_model(model, scaler, disease_type, version, metrics=None):
    """
    Save trained model and scaler to disk.

    Args:
        model: Trained sklearn model
        scaler: Fitted StandardScaler
        disease_type: e.g., 'heart_disease'
        version: e.g., 'v1.0'
        metrics: Dictionary of training metrics
    """
    disease_dir = os.path.join(MODEL_STORAGE_DIR, disease_type)
    os.makedirs(disease_dir, exist_ok=True)

    model_path = os.path.join(disease_dir, f"model_{version}.joblib")
    joblib.dump(model, model_path)

    scaler_path = os.path.join(disease_dir, f"scaler_{version}.joblib")
    joblib.dump(scaler, scaler_path)

    metadata = {
        "disease_type": disease_type,
        "version": version,
        "model_type": type(model).__name__,
        "saved_at": datetime.now().isoformat(),
        "model_path": model_path,
        "scaler_path": scaler_path,
        "metrics": metrics,
    }

    metadata_path = os.path.join(disease_dir, f"metadata_{version}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    joblib.dump(model, os.path.join(disease_dir, "model_latest.joblib"))
    joblib.dump(scaler, os.path.join(disease_dir, "scaler_latest.joblib"))

    return model_path

def load_model(disease_type, version=None):
    """
    Load model and scaler from disk.

    Args:
        disease_type: e.g., 'heart_disease'
        version: e.g., 'v1.0' (None = latest)

    Returns:
        Tuple of (model, scaler) or (None, None)
    """
    disease_dir = os.path.join(MODEL_STORAGE_DIR, disease_type)

    if version:
        model_path = os.path.join(disease_dir, f"model_{version}.joblib")
        scaler_path = os.path.join(disease_dir, f"scaler_{version}.joblib")
    else:
        model_path = os.path.join(disease_dir, "model_latest.joblib")
        scaler_path = os.path.join(disease_dir, "scaler_latest.joblib")

    if not os.path.exists(model_path):
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    return model, scaler

def model_exists(disease_type):
    """Check if a model exists for given disease type."""
    disease_dir = os.path.join(MODEL_STORAGE_DIR, disease_type)
    model_path = os.path.join(disease_dir, "model_latest.joblib")
    return os.path.exists(model_path)

def get_model_weights(disease_type, version=None):
    """
    Extract model weights (coefficients) for FL.

    Returns:
        Dictionary with weights and intercept
    """
    model, scaler = load_model(disease_type, version)

    if model is None:
        return None

    weights = {
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }

    if scaler is not None:
        weights["scaler_mean"] = scaler.mean_.tolist()
        weights["scaler_scale"] = scaler.scale_.tolist()

    return weights

def set_model_weights(model, weights):
    """
    Set model weights from FL aggregation.

    Args:
        model: sklearn model
        weights: Dictionary with coefficients and intercept

    Returns:
        Model with updated weights
    """
    model.coef_ = np.array(weights["coefficients"])
    model.intercept_ = np.array(weights["intercept"])
    return model

def get_all_model_status():
    """Get status of all disease models."""
    from config import DISEASE_TYPES, DISEASE_DISPLAY_NAMES

    status = {}
    for disease_type in DISEASE_TYPES:
        disease_dir = os.path.join(MODEL_STORAGE_DIR, disease_type)
        model_path = os.path.join(disease_dir, "model_latest.joblib")
        metadata_path = os.path.join(disease_dir, "metadata_latest.json")

        if os.path.exists(model_path):
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

            latest_metadata = None
            if os.path.exists(disease_dir):
                for file in sorted(os.listdir(disease_dir)):
                    if file.startswith("metadata_v"):
                        file_path = os.path.join(disease_dir, file)
                        with open(file_path, "r") as f:
                            latest_metadata = json.load(f)

            if latest_metadata:
                metadata = latest_metadata

            status[disease_type] = {
                "name": DISEASE_DISPLAY_NAMES.get(disease_type, disease_type),
                "available": True,
                "version": metadata.get("version", "unknown"),
                "accuracy": metadata.get("metrics", {}).get("accuracy") if metadata.get("metrics") else None,
                "saved_at": metadata.get("saved_at"),
            }
        else:
            status[disease_type] = {
                "name": DISEASE_DISPLAY_NAMES.get(disease_type, disease_type),
                "available": False,
                "version": None,
                "accuracy": None,
                "saved_at": None,
            }

    return status