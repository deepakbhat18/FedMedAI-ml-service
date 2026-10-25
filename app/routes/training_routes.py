
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.federated.server import fl_server
from app.flower.flower_server import flower_server
from app.federated.differential_privacy import DifferentialPrivacy

router = APIRouter()

class TrainRequest(BaseModel):
    hospital_id: int
    data: Optional[List[List[float]]] = None
    labels: Optional[List[int]] = None
    epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    global_weights: Optional[Dict] = None
    dp_epsilon: Optional[float] = None

class TrainAllRequest(BaseModel):
    hospital_ids: List[int] = [1, 2, 3]
    epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    engine: str = "custom"
    num_rounds: int = 3
    dp_epsilon: Optional[float] = None

class AggregateRequest(BaseModel):
    hospital_weights: List[Dict]
    aggregation_method: str = "fedavg"

@router.post("/train")
async def train_local(request: TrainRequest):
    """Train all disease models locally for a single hospital."""
    try:
        from app.federated.client import HospitalClient

        client = HospitalClient(request.hospital_id)
        result = client.train_all_diseases(
            global_weights=request.global_weights,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            dp_epsilon=request.dp_epsilon
        )

        response = {
            "hospital_id": request.hospital_id,
            "local_weights": result["disease_weights"],
            "metrics": {},
            "samples_count": result["samples_count"],
            "training_time_seconds": 0,
            "dp_applied": result.get("dp_applied", False),
            "dp_epsilon": result.get("dp_epsilon"),
        }

        total_time = 0
        for disease, metrics in result["disease_metrics"].items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                response["metrics"][disease] = metrics
                total_time += metrics.get("training_time", 0)

        all_accuracies = [
            m.get("accuracy", 0)
            for m in response["metrics"].values()
            if isinstance(m, dict) and "accuracy" in m
        ]
        if all_accuracies:
            response["metrics"]["accuracy"] = round(
                sum(all_accuracies) / len(all_accuracies), 4)
            response["metrics"]["loss"] = round(
                1 - response["metrics"]["accuracy"], 4)
            response["metrics"]["precision"] = round(
                sum(m.get("precision", 0) for m in response["metrics"].values()
                    if isinstance(m, dict) and "precision" in m) / len(all_accuracies), 4)
            response["metrics"]["recall"] = round(
                sum(m.get("recall", 0) for m in response["metrics"].values()
                    if isinstance(m, dict) and "recall" in m) / len(all_accuracies), 4)
            response["metrics"]["f1_score"] = round(
                sum(m.get("f1_score", 0) for m in response["metrics"].values()
                    if isinstance(m, dict) and "f1_score" in m) / len(all_accuracies), 4)

        response["training_time_seconds"] = round(total_time, 3)
        return response

    except Exception as e:
        raise HTTPException(status_code=500,
            detail=f"Training failed for hospital {request.hospital_id}: {str(e)}")

@router.post("/train/all")
async def train_all_hospitals(request: TrainAllRequest):
    """
    Run a COMPLETE Federated Learning round.
    Supports: custom FedAvg, Flower framework, Differential Privacy.
    """
    try:
        dp_info = None
        if request.dp_epsilon is not None and request.dp_epsilon > 0:
            dp = DifferentialPrivacy(epsilon=request.dp_epsilon)
            dp_info = dp.get_privacy_report()

        if request.engine == "flower":
            result = flower_server.run_flower_round(
                hospital_ids=request.hospital_ids,
                num_rounds=request.num_rounds,
                epochs=request.epochs
            )
        else:
            result = fl_server.run_training_round(
                hospital_ids=request.hospital_ids,
                epochs=request.epochs,
                learning_rate=request.learning_rate,
                batch_size=request.batch_size,
                dp_epsilon=request.dp_epsilon
            )

        if dp_info:
            result["differential_privacy"] = dp_info

        return result

    except Exception as e:
        raise HTTPException(status_code=500,
            detail=f"Federated learning round failed: {str(e)}")

@router.post("/aggregate")
async def aggregate_weights(request: AggregateRequest):
    """Aggregate hospital weights using FedAvg."""
    try:
        from app.federated.fed_avg import federated_averaging_all_diseases

        result = federated_averaging_all_diseases(request.hospital_weights)
        return {
            "global_weights": result,
            "model_version": f"v{fl_server.training_round + 1}.0",
            "participating_hospitals": len(request.hospital_weights),
            "aggregation_method": request.aggregation_method,
            "diseases_aggregated": list(result.keys())
        }

    except Exception as e:
        raise HTTPException(status_code=500,
            detail=f"Aggregation failed: {str(e)}")

@router.get("/training/engines")
async def get_available_engines():
    """Get available FL training engines."""
    flower_ver = "unknown"
    try:
        import flwr
        flower_ver = flwr.__version__
    except Exception:
        pass
    flower_desc = f"Flower FL framework v{flower_ver}" if flower_ver != "unknown" else "Flower FL framework (simulation)"
    return {
        "engines": [
            {
                "id": "custom",
                "name": "Custom FedAvg",
                "description": "Hand-implemented Federated Averaging algorithm",
                "status": "available"
            },
            {
                "id": "flower",
                "name": "Flower Framework",
                "description": flower_desc,
                "version": flower_ver,
                "status": "available"
            }
        ],
        "privacy": {
            "differential_privacy": "available",
            "description": "Add dp_epsilon parameter (0.1 to 10.0) to enable"
        }
    }

@router.get("/privacy/info")
async def get_privacy_info():
    """Get differential privacy information."""
    examples = [
        {"epsilon": 0.1, "level": "VERY HIGH", "note": "Maximum privacy, lower accuracy"},
        {"epsilon": 0.5, "level": "HIGH", "note": "Strong privacy"},
        {"epsilon": 1.0, "level": "HIGH", "note": "Good balance (recommended)"},
        {"epsilon": 5.0, "level": "MODERATE", "note": "Moderate privacy"},
        {"epsilon": 10.0, "level": "LOW", "note": "Minimal privacy, highest accuracy"},
    ]
    return {
        "mechanism": "Laplace Differential Privacy",
        "parameter": "dp_epsilon",
        "description": "Lower epsilon = more privacy, less accuracy",
        "examples": examples,
        "usage": 'POST /ml/train/all with {"dp_epsilon": 1.0}'
    }