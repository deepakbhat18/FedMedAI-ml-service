from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)
import numpy as np


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate all classification metrics.

    Args:
        y_true: Actual labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    # Calculate loss if probabilities available
    if y_prob is not None:
        try:
            metrics["loss"] = round(log_loss(y_true, y_prob), 4)
        except Exception:
            metrics["loss"] = None
    else:
        metrics["loss"] = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Counts
    metrics["total_samples"] = len(y_true)
    metrics["positive_count"] = int(np.sum(y_true == 1))
    metrics["negative_count"] = int(np.sum(y_true == 0))

    return metrics


def get_risk_level(confidence):
    """
    Convert confidence score to risk level.

    Args:
        confidence: Float between 0 and 1

    Returns:
        Risk level string
    """
    if confidence >= 0.75:
        return "HIGH"
    elif confidence >= 0.50:
        return "MODERATE"
    elif confidence >= 0.25:
        return "LOW"
    else:
        return "MINIMAL"