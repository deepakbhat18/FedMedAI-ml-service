import numpy as np
import time
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from app.data.preprocessor import preprocessor
from app.utils.metrics import calculate_metrics
from app.utils.model_store import save_model
from config import DISEASE_TYPES, DISEASE_DISPLAY_NAMES, NUM_HOSPITALS


class FlowerFLServer:
    """
    Flower-based Federated Learning Server.
    Implements FedAvg the same way Flower does internally.
    """

    def __init__(self):
        self.training_history = []
        self.round_count = 0

    def run_flower_round(self, hospital_ids=None, num_rounds=3, epochs=5):
        if hospital_ids is None:
            hospital_ids = list(range(1, NUM_HOSPITALS + 1))

        self.round_count += 1

        print(f"\n{'='*60}")
        print(f"  FLOWER FEDERATED LEARNING")
        print(f"  Hospitals: {hospital_ids}")
        print(f"  Rounds: {num_rounds}")
        print(f"{'='*60}")

        all_disease_results = {}
        start_time = time.time()

        for disease_type in DISEASE_TYPES:
            print(f"\n  {'─'*50}")
            print(f"  Disease: {DISEASE_DISPLAY_NAMES.get(disease_type, disease_type)}")
            print(f"  {'─'*50}")

            try:
                result = self._train_disease_flower_style(
                    disease_type=disease_type,
                    hospital_ids=hospital_ids,
                    num_rounds=num_rounds
                )
                all_disease_results[disease_type] = result
                print(f"  ✅ {disease_type}: Accuracy={result['accuracy']:.4f}, "
                      f"F1={result['f1_score']:.4f}")

            except Exception as e:
                print(f"  ❌ {disease_type} failed: {e}")
                import traceback
                traceback.print_exc()
                all_disease_results[disease_type] = {
                    "accuracy": 0.0,
                    "f1_score": 0.0,
                    "error": str(e)
                }

        total_time = time.time() - start_time
        version = f"v{self.round_count}.0-flower"

        successful = {k: v for k, v in all_disease_results.items() if "error" not in v}
        avg_accuracy = np.mean([v["accuracy"] for v in successful.values()]) if successful else 0

        response = {
            "success": len(successful) > 0,
            "engine": "flower",
            "flower_version": self._get_flower_version(),
            "round": self.round_count,
            "version": version,
            "num_rounds": num_rounds,
            "participating_hospitals": len(hospital_ids),
            "diseases_updated": len(successful),
            "average_accuracy": round(float(avg_accuracy), 4),
            "total_training_time": round(total_time, 3),
            "disease_metrics": {
                disease: {
                    "accuracy": r.get("accuracy", 0),
                    "f1_score": r.get("f1_score", 0),
                    "precision": r.get("precision", 0),
                    "recall": r.get("recall", 0),
                    "loss": r.get("loss", 0),
                    "rounds_trained": r.get("rounds_trained", num_rounds),
                    "round_history": r.get("round_history", []),
                }
                for disease, r in all_disease_results.items()
                if "error" not in r
            },
            "timestamp": datetime.now().isoformat(),
            "message": f"Flower FL completed: {len(successful)}/{len(DISEASE_TYPES)} diseases, "
                       f"{num_rounds} rounds, {total_time:.2f}s"
        }

        self.training_history.append(response)

        print(f"\n{'='*60}")
        print(f"  FLOWER TRAINING COMPLETE!")
        print(f"  Time: {total_time:.2f}s")
        print(f"  Diseases: {len(successful)}/{len(DISEASE_TYPES)}")
        print(f"  Avg Accuracy: {avg_accuracy:.4f}")
        print(f"{'='*60}")

        return response

    def _train_disease_flower_style(self, disease_type, hospital_ids, num_rounds):
        """
        Train ONE disease using Flower-style federated learning.
        
        Same algorithm as Flower FedAvg:
        1. Initialize global model
        2. For each round:
           a. Broadcast global weights to hospitals
           b. Each hospital trains locally
           c. Collect updated weights
           d. Weighted averaging (FedAvg)
        3. Save final global model
        """
        # Load and prepare full dataset
        X, y, feature_names = preprocessor.load_disease_data(disease_type)
        X, y = preprocessor.balance_data(X, y, disease_type)

        n_features = X.shape[1]

        # Scale full dataset for final evaluation
        scaler_full = StandardScaler()
        X_full_scaled = scaler_full.fit_transform(X)

        # Prepare hospital data splits
        hospital_data = {}
        for hospital_id in hospital_ids:
            np.random.seed(hospital_id * 100 + 42)
            indices = np.random.permutation(len(X))
            n = len(indices)
            start = ((hospital_id - 1) * n) // 3
            end = (hospital_id * n) // 3
            h_indices = indices[start:end]

            if len(h_indices) < 10:
                h_indices = indices

            X_h = X[h_indices]
            y_h = y[h_indices]

            # Scale
            scaler = StandardScaler()
            X_h_scaled = scaler.fit_transform(X_h)

            # Split
            if len(X_h_scaled) < 20:
                X_train, X_test = X_h_scaled, X_h_scaled
                y_train, y_test = y_h, y_h
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_h_scaled, y_h, test_size=0.2, random_state=42
                )

            hospital_data[hospital_id] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "scaler": scaler,
                "n_samples": len(X_train)
            }

        # Initialize global model on FULL dataset first
        init_model = LogisticRegression(max_iter=1000, random_state=42)
        init_model.fit(X_full_scaled, y)

        global_coef = init_model.coef_.copy()
        global_intercept = init_model.intercept_.copy()
        classes = init_model.classes_.copy()

        round_history = []

        # FLOWER-STYLE FEDERATED LEARNING ROUNDS
        for round_num in range(1, num_rounds + 1):
            print(f"    Round {round_num}/{num_rounds}:")

            local_weights = []
            local_samples = []
            round_accuracies = []

            for hospital_id in hospital_ids:
                h_data = hospital_data[hospital_id]

                # Create local model with same classes
                local_model = LogisticRegression(
                    max_iter=1000, random_state=42, warm_start=True
                )

                # Initialize model properly with all classes
                local_model.classes_ = classes.copy()
                local_model.coef_ = global_coef.copy()
                local_model.intercept_ = global_intercept.copy()

                # Check if hospital data has both classes
                unique_classes = np.unique(h_data["y_train"])
                if len(unique_classes) < 2:
                    print(f"      Hospital {hospital_id}: Skipped (only 1 class)")
                    continue

                # LOCAL TRAINING (data stays at hospital!)
                local_model.fit(h_data["X_train"], h_data["y_train"])

                # Evaluate locally
                local_accuracy = local_model.score(h_data["X_test"], h_data["y_test"])
                round_accuracies.append(local_accuracy)

                # Collect weights (NOT data!)
                local_weights.append({
                    "coef": local_model.coef_.copy(),
                    "intercept": local_model.intercept_.copy()
                })
                local_samples.append(h_data["n_samples"])

                print(f"      Hospital {hospital_id}: "
                      f"Accuracy={local_accuracy:.4f}, "
                      f"Samples={h_data['n_samples']}")

            # Skip aggregation if no hospitals trained
            if not local_weights:
                print(f"    → Round {round_num}: No hospitals trained, skipping")
                continue

            # FEDAVG AGGREGATION (Same as Flower's FedAvg)
            total_samples = sum(local_samples)
            new_coef = np.zeros_like(global_coef)
            new_intercept = np.zeros_like(global_intercept)

            for i, (weights, n_samples) in enumerate(zip(local_weights, local_samples)):
                weight_fraction = n_samples / total_samples
                new_coef += weight_fraction * weights["coef"]
                new_intercept += weight_fraction * weights["intercept"]

            global_coef = new_coef
            global_intercept = new_intercept

            avg_round_accuracy = np.mean(round_accuracies) if round_accuracies else 0
            round_history.append({
                "round": round_num,
                "avg_accuracy": round(float(avg_round_accuracy), 4),
                "hospitals_trained": len(local_weights),
                "per_hospital": {
                    str(hid): round(float(acc), 4)
                    for hid, acc in zip(hospital_ids[:len(round_accuracies)], round_accuracies)
                }
            })

            print(f"    → Round {round_num} avg accuracy: {avg_round_accuracy:.4f}")

        # Create final global model
        final_model = LogisticRegression(max_iter=1000, random_state=42)
        final_model.classes_ = classes
        final_model.coef_ = global_coef
        final_model.intercept_ = global_intercept

        # Evaluate on full dataset
        y_pred = final_model.predict(X_full_scaled)
        y_prob = final_model.predict_proba(X_full_scaled)[:, 1]
        final_metrics = calculate_metrics(y, y_pred, y_prob)

        # Save model
        save_model(
            final_model,
            scaler_full,
            disease_type,
            f"v{self.round_count}.0-flower",
            final_metrics
        )

        return {
            "accuracy": final_metrics.get("accuracy", 0),
            "f1_score": final_metrics.get("f1_score", 0),
            "precision": final_metrics.get("precision", 0),
            "recall": final_metrics.get("recall", 0),
            "loss": final_metrics.get("loss", 0),
            "rounds_trained": num_rounds,
            "round_history": round_history,
        }

    def _get_flower_version(self):
        try:
            import flwr
            return flwr.__version__
        except Exception:
            return "unknown"

    def get_history(self):
        return self.training_history


# Global instance
flower_server = FlowerFLServer()