import numpy as np
from typing import List, Dict


def federated_averaging(hospital_weights: List[Dict]) -> Dict:
    """
    FedAvg Algorithm - Core of Federated Learning.

    This is the EXACT algorithm used by:
    - Google (Gboard keyboard predictions)
    - Apple (Siri improvements)
    - Medical research institutions

    Algorithm:
    Global_Weights = Σ (n_k / n_total) × Local_Weights_k

    Where:
    - n_k = number of samples at hospital k
    - n_total = total samples across all hospitals
    - Local_Weights_k = model weights from hospital k

    Args:
        hospital_weights: List of dicts, each containing:
            - hospital_id: int
            - weights: dict with 'coefficients' and 'intercept'
            - samples_count: int

    Returns:
        Aggregated global weights dictionary
    """
    if not hospital_weights:
        raise ValueError("No hospital weights provided for aggregation")

    if len(hospital_weights) == 1:
        print("  Only 1 hospital - returning weights directly")
        return hospital_weights[0]["weights"]

    print(f"\n  FedAvg aggregating {len(hospital_weights)} hospitals...")

    # Calculate total samples (for weighted averaging)
    total_samples = sum(hw["samples_count"] for hw in hospital_weights)
    print(f"  Total samples across hospitals: {total_samples}")

    # Print each hospital contribution
    for hw in hospital_weights:
        weight_fraction = hw["samples_count"] / total_samples
        print(f"  Hospital {hw['hospital_id']}: "
              f"{hw['samples_count']} samples "
              f"({weight_fraction*100:.1f}% weight)")

    # Extract coefficients and intercepts
    all_coefficients = []
    all_intercepts = []
    sample_weights = []

    for hw in hospital_weights:
        weights = hw["weights"]
        samples = hw["samples_count"]

        all_coefficients.append(np.array(weights["coefficients"]))
        all_intercepts.append(np.array(weights["intercept"]))
        sample_weights.append(samples)

    # Convert to numpy arrays
    all_coefficients = np.array(all_coefficients)  # shape: (n_hospitals, 1, n_features)
    all_intercepts = np.array(all_intercepts)       # shape: (n_hospitals, 1)
    sample_weights = np.array(sample_weights)       # shape: (n_hospitals,)

    # Normalize weights (they should sum to 1)
    normalized_weights = sample_weights / total_samples

    # FEDERATED AVERAGING
    # Global coefficient = weighted average of all hospital coefficients
    global_coefficients = np.zeros_like(all_coefficients[0])
    global_intercepts = np.zeros_like(all_intercepts[0])

    for i, norm_weight in enumerate(normalized_weights):
        global_coefficients += norm_weight * all_coefficients[i]
        global_intercepts += norm_weight * all_intercepts[i]

    global_weights = {
        "coefficients": global_coefficients.tolist(),
        "intercept": global_intercepts.tolist(),
    }

    print(f"  FedAvg complete!")
    print(f"  Global coefficients shape: {global_coefficients.shape}")

    return global_weights


def federated_averaging_all_diseases(all_hospital_disease_weights: List[Dict]) -> Dict:
    """
    Run FedAvg for ALL disease models simultaneously.

    Args:
        all_hospital_disease_weights: List of dicts, each containing:
            - hospital_id: int
            - samples_count: int
            - disease_weights: dict mapping disease_type -> weights

    Returns:
        Dictionary mapping disease_type -> global_weights
    """
    from config import DISEASE_TYPES

    print("\n" + "="*60)
    print("FEDERATED AVERAGING - ALL DISEASES")
    print("="*60)

    global_disease_weights = {}

    for disease_type in DISEASE_TYPES:
        print(f"\nAggregating: {disease_type}")

        # Collect weights for this disease from all hospitals
        disease_hospital_weights = []
        for hospital_data in all_hospital_disease_weights:
            if disease_type in hospital_data.get("disease_weights", {}):
                disease_hospital_weights.append({
                    "hospital_id": hospital_data["hospital_id"],
                    "weights": hospital_data["disease_weights"][disease_type],
                    "samples_count": hospital_data.get("samples_count", {}).get(
                        disease_type, 100)
                })

        if disease_hospital_weights:
            try:
                global_weights = federated_averaging(disease_hospital_weights)
                global_disease_weights[disease_type] = global_weights
                print(f"  ✅ {disease_type} aggregated successfully")
            except Exception as e:
                print(f"  ❌ {disease_type} aggregation failed: {e}")
        else:
            print(f"  ⚠️  No weights received for {disease_type}")

    print("\n" + "="*60)
    print(f"Aggregation complete: {len(global_disease_weights)} diseases updated")
    print("="*60)

    return global_disease_weights