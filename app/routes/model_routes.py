

from fastapi import APIRouter, HTTPException
import numpy as np
from app.utils.model_store import (
    get_all_model_status,
    model_exists,
    get_model_weights,
    load_model
)
from app.federated.server import fl_server
from app.flower.flower_server import flower_server
from app.data.preprocessor import preprocessor
from app.utils.metrics import calculate_metrics
from config import DISEASE_TYPES, DISEASE_DISPLAY_NAMES

router = APIRouter()


@router.get("/model/status")
async def get_model_status():
    status = get_all_model_status()
    available_count = sum(1 for s in status.values() if s["available"])
    return {
        "total_diseases": len(DISEASE_TYPES),
        "models_available": available_count,
        "models": status
    }


@router.get("/model/weights/{disease_type}")
async def get_weights(disease_type: str):
    if disease_type not in DISEASE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown disease: {disease_type}")
    if not model_exists(disease_type):
        raise HTTPException(status_code=404, detail=f"No model for {disease_type}")
    weights = get_model_weights(disease_type)
    return {"disease_type": disease_type, "weights": weights}


@router.get("/model/weights")
async def get_all_weights():
    all_weights = {}
    for disease_type in DISEASE_TYPES:
        if model_exists(disease_type):
            weights = get_model_weights(disease_type)
            if weights:
                all_weights[disease_type] = weights
    return {"models_count": len(all_weights), "weights": all_weights}


@router.get("/training/history")
async def get_training_history():
    custom_history = fl_server.get_training_history()
    flower_history = flower_server.get_history()
    return {
        "total_rounds": len(custom_history) + len(flower_history),
        "custom_rounds": len(custom_history),
        "flower_rounds": len(flower_history),
        "custom_history": custom_history,
        "flower_history": flower_history
    }


@router.get("/analytics/full")
async def get_full_analytics():
    """
    Returns REAL dynamic analytics data:
    - Live model accuracy per disease
    - Feature importance from model weights
    - Training history
    - Engine comparison data
    """
    analytics = {
        "disease_metrics": {},
        "feature_importance": {},
        "engine_comparison": {},
        "training_rounds": [],
        "summary": {}
    }

    total_accuracy = []

    for disease_type in DISEASE_TYPES:
        try:
            model, scaler = load_model(disease_type)
            if model is None:
                continue

            # Evaluate model on full dataset
            X, y, feature_names = preprocessor.load_disease_data(disease_type)
            X, y = preprocessor.balance_data(X, y, disease_type)

            from sklearn.preprocessing import StandardScaler
            eval_scaler = StandardScaler()
            X_scaled = eval_scaler.fit_transform(X)

            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1]
            metrics = calculate_metrics(y, y_pred, y_prob)

            analytics["disease_metrics"][disease_type] = {
                "disease_name": DISEASE_DISPLAY_NAMES.get(disease_type, disease_type),
                "accuracy": round(metrics["accuracy"] * 100, 2),
                "f1_score": round(metrics["f1_score"] * 100, 2),
                "precision": round(metrics["precision"] * 100, 2),
                "recall": round(metrics["recall"] * 100, 2),
                "loss": round(metrics.get("loss", 0) or 0, 4),
                "total_samples": metrics["total_samples"],
                "positive_count": metrics["positive_count"],
                "negative_count": metrics["negative_count"],
                "confusion_matrix": metrics["confusion_matrix"]
            }

            total_accuracy.append(metrics["accuracy"])

            # Feature importance from model coefficients
            if hasattr(model, 'coef_'):
                coef = model.coef_[0]
                importance_data = []

                for i, (feat_name, weight) in enumerate(zip(feature_names, coef)):
                    importance_data.append({
                        "feature": feat_name,
                        "weight": round(float(weight), 4),
                        "abs_weight": round(float(abs(weight)), 4),
                        "direction": "risk" if weight > 0 else "protective",
                        "impact": "HIGH" if abs(weight) > 0.5 else "MEDIUM" if abs(weight) > 0.2 else "LOW"
                    })

                # Sort by absolute weight
                importance_data.sort(key=lambda x: x["abs_weight"], reverse=True)
                analytics["feature_importance"][disease_type] = {
                    "disease_name": DISEASE_DISPLAY_NAMES.get(disease_type, disease_type),
                    "features": importance_data[:10],
                    "total_features": len(importance_data)
                }

        except Exception as e:
            print(f"Analytics error for {disease_type}: {e}")

    # Summary
    analytics["summary"] = {
        "total_diseases": len(DISEASE_TYPES),
        "models_available": len(analytics["disease_metrics"]),
        "avg_accuracy": round(float(np.mean(total_accuracy) * 100), 2) if total_accuracy else 0,
        "avg_accuracy_decimal": round(float(np.mean(total_accuracy)), 4) if total_accuracy else 0,
        "best_disease": max(analytics["disease_metrics"].items(),
                           key=lambda x: x[1]["accuracy"])[0] if analytics["disease_metrics"] else None,
        "worst_disease": min(analytics["disease_metrics"].items(),
                            key=lambda x: x[1]["accuracy"])[0] if analytics["disease_metrics"] else None,
    }

    # Training history
    custom_history = fl_server.get_training_history()
    flower_history = flower_server.get_history()
    analytics["training_rounds"] = {
        "custom": custom_history,
        "flower": flower_history,
        "total": len(custom_history) + len(flower_history)
    }

    return analytics


@router.get("/health")
async def health_check():
    import flwr
    status = get_all_model_status()
    available = sum(1 for s in status.values() if s["available"])
    return {
        "status": "UP",
        "service": "ML Service (Multi-Disease FL)",
        "models_available": available,
        "total_diseases": len(DISEASE_TYPES),
        "diseases": list(DISEASE_TYPES),
        "engines": {
            "custom_fedavg": "available",
            "flower": f"available (v{flwr.__version__})"
        }
    }