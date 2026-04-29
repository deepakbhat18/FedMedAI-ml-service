

import os
import json
import numpy as np
from datetime import datetime
from app.federated.fed_avg import federated_averaging, federated_averaging_all_diseases
from app.federated.client import HospitalClient
from app.models.disease_models import get_model
from app.utils.model_store import save_model, load_model
from app.utils.metrics import calculate_metrics
from app.data.preprocessor import preprocessor
from config import DISEASE_TYPES, DISEASE_DISPLAY_NAMES, NUM_HOSPITALS, MODEL_STORAGE_DIR


class FederatedServer:
    """Central Federated Learning Server with Before/After tracking."""

    def __init__(self):
        self.global_models = {}
        self.global_scalers = {}
        self.training_round = 0
        self.training_history = []

    def run_training_round(self, hospital_ids=None,
                           epochs=5, learning_rate=0.01,
                           batch_size=32, dp_epsilon=None):
        if hospital_ids is None:
            hospital_ids = list(range(1, NUM_HOSPITALS + 1))

        self.training_round += 1
        round_number = self.training_round

        print(f"\n{'='*60}")
        print(f"FEDERATED LEARNING - ROUND {round_number}")
        print(f"Participating hospitals: {hospital_ids}")
        if dp_epsilon:
            print(f"🔒 Differential Privacy: ε={dp_epsilon}")
        print(f"{'='*60}")

        # ===== STEP 0: Get BEFORE metrics =====
        before_metrics = self._evaluate_current_models()
        print(f"\n📊 BEFORE Training:")
        for disease, metrics in before_metrics.items():
            if metrics:
                print(f"  {disease}: Accuracy={metrics.get('accuracy', 'N/A')}")
            else:
                print(f"  {disease}: No model yet")

        # Get current global weights
        current_global_weights = self._get_global_weights()

        # ===== STEP 1: Each hospital trains locally =====
        all_hospital_results = []
        successful_hospitals = 0

        for hospital_id in hospital_ids:
            try:
                client = HospitalClient(hospital_id)
                result = client.train_all_diseases(
                    global_weights=current_global_weights,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    dp_epsilon=dp_epsilon
                )
                all_hospital_results.append(result)
                successful_hospitals += 1
                print(f"\n  ✅ Hospital {hospital_id} completed")
            except Exception as e:
                print(f"\n  ❌ Hospital {hospital_id} failed: {e}")

        if not all_hospital_results:
            return {"success": False, "error": "All hospitals failed", "round": round_number}

        # ===== STEP 2: FedAvg Aggregation =====
        print(f"\n{'='*60}")
        print("CENTRAL SERVER - FEDERATED AVERAGING")
        print(f"{'='*60}")

        new_global_weights = federated_averaging_all_diseases(all_hospital_results)

        # ===== STEP 3: Update and save global models =====
        after_metrics = {}
        version = f"v{round_number}.0"

        for disease_type in DISEASE_TYPES:
            if disease_type not in new_global_weights:
                continue
            try:
                model = get_model(disease_type)
                model.set_weights(new_global_weights[disease_type])

                X, y, _ = preprocessor.load_disease_data(disease_type)
                X, y = preprocessor.balance_data(X, y, disease_type)

                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                y_pred = model.model.predict(X_scaled)
                y_prob = model.model.predict_proba(X_scaled)[:, 1]
                metrics = calculate_metrics(y, y_pred, y_prob)
                after_metrics[disease_type] = metrics

                save_model(model.model, scaler, disease_type, version, metrics)
                self.global_models[disease_type] = model
                self.global_scalers[disease_type] = scaler

                print(f"\n  ✅ {disease_type}: Accuracy={metrics['accuracy']:.4f}")
            except Exception as e:
                print(f"\n  ❌ {disease_type} update failed: {e}")

        # ===== STEP 4: Calculate improvement =====
        comparison = {}
        for disease_type in DISEASE_TYPES:
            before_acc = before_metrics.get(disease_type, {}).get("accuracy", 0) if before_metrics.get(disease_type) else 0
            after_acc = after_metrics.get(disease_type, {}).get("accuracy", 0) if after_metrics.get(disease_type) else 0
            improvement = after_acc - before_acc

            comparison[disease_type] = {
                "before": round(before_acc, 4),
                "after": round(after_acc, 4),
                "improvement": round(improvement, 4),
                "improved": improvement > 0,
                "disease_name": DISEASE_DISPLAY_NAMES.get(disease_type, disease_type)
            }

        # Print comparison
        print(f"\n{'='*60}")
        print("📊 BEFORE vs AFTER COMPARISON")
        print(f"{'='*60}")
        for disease, comp in comparison.items():
            arrow = "↑" if comp["improvement"] > 0 else "↓" if comp["improvement"] < 0 else "="
            color = "🟢" if comp["improvement"] > 0 else "🔴" if comp["improvement"] < 0 else "🟡"
            print(f"  {color} {comp['disease_name']}: "
                  f"{comp['before']*100:.1f}% → {comp['after']*100:.1f}% "
                  f"({arrow} {comp['improvement']*100:+.2f}%)")

        # ===== STEP 5: Build response =====
        avg_before = np.mean([c["before"] for c in comparison.values()])
        avg_after = np.mean([c["after"] for c in comparison.values()])
        avg_improvement = avg_after - avg_before

        round_summary = {
            "round": round_number,
            "version": version,
            "engine": "custom",
            "timestamp": datetime.now().isoformat(),
            "participating_hospitals": successful_hospitals,
            "total_hospitals": len(hospital_ids),
            "dp_applied": dp_epsilon is not None,
            "dp_epsilon": dp_epsilon,
            "before_metrics": {
                d: {"accuracy": c["before"], "disease_name": c["disease_name"]}
                for d, c in comparison.items()
            },
            "after_metrics": {
                d: {
                    "accuracy": after_metrics.get(d, {}).get("accuracy", 0),
                    "f1_score": after_metrics.get(d, {}).get("f1_score", 0),
                    "precision": after_metrics.get(d, {}).get("precision", 0),
                    "recall": after_metrics.get(d, {}).get("recall", 0),
                }
                for d in DISEASE_TYPES
                if d in after_metrics
            },
            "comparison": comparison,
            "overall": {
                "avg_before": round(float(avg_before), 4),
                "avg_after": round(float(avg_after), 4),
                "avg_improvement": round(float(avg_improvement), 4),
            }
        }

        self.training_history.append(round_summary)
        self._save_training_history()

        print(f"\n{'='*60}")
        print(f"ROUND {round_number} COMPLETE!")
        print(f"Average: {avg_before*100:.1f}% → {avg_after*100:.1f}% "
              f"({'↑' if avg_improvement > 0 else '↓'} {avg_improvement*100:+.2f}%)")
        print(f"{'='*60}")

        return {
            "success": True,
            "round": round_number,
            "version": version,
            "engine": "custom",
            "participating_hospitals": successful_hospitals,
            "diseases_updated": len(after_metrics),
            "dp_applied": dp_epsilon is not None,
            "dp_epsilon": dp_epsilon,
            "disease_metrics": {
                d: {
                    "accuracy": m.get("accuracy", 0),
                    "f1_score": m.get("f1_score", 0),
                    "precision": m.get("precision", 0),
                    "recall": m.get("recall", 0),
                    "loss": m.get("loss", 0),
                }
                for d, m in after_metrics.items()
            },
            "comparison": comparison,
            "overall": {
                "avg_before": round(float(avg_before), 4),
                "avg_after": round(float(avg_after), 4),
                "avg_improvement": round(float(avg_improvement), 4),
            },
            "summary": round_summary,
            "message": f"Round {round_number}: "
                       f"{avg_before*100:.1f}% → {avg_after*100:.1f}% "
                       f"({'↑' if avg_improvement > 0 else '↓'} {avg_improvement*100:+.2f}%)"
        }

    def _evaluate_current_models(self):
        """Evaluate current global models to get BEFORE metrics."""
        before_metrics = {}
        for disease_type in DISEASE_TYPES:
            try:
                model, scaler = load_model(disease_type)
                if model is None:
                    before_metrics[disease_type] = None
                    continue

                X, y, _ = preprocessor.load_disease_data(disease_type)
                X, y = preprocessor.balance_data(X, y, disease_type)

                from sklearn.preprocessing import StandardScaler
                eval_scaler = StandardScaler()
                X_scaled = eval_scaler.fit_transform(X)

                y_pred = model.predict(X_scaled)
                y_prob = model.predict_proba(X_scaled)[:, 1]
                metrics = calculate_metrics(y, y_pred, y_prob)
                before_metrics[disease_type] = metrics
            except Exception as e:
                before_metrics[disease_type] = None
        return before_metrics

    def _get_global_weights(self):
        from app.utils.model_store import get_model_weights
        global_weights = {}
        for disease_type in DISEASE_TYPES:
            weights = get_model_weights(disease_type)
            if weights:
                global_weights[disease_type] = weights
        return global_weights if global_weights else None

    def _save_training_history(self):
        history_path = os.path.join(MODEL_STORAGE_DIR, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2, default=str)

    def get_training_history(self):
        history_path = os.path.join(MODEL_STORAGE_DIR, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                return json.load(f)
        return self.training_history


# Global server instance
fl_server = FederatedServer()