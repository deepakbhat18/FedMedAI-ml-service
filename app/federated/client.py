import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from app.models.disease_models import get_model
from app.data.preprocessor import preprocessor
from app.utils.metrics import calculate_metrics
from config import DISEASE_TYPES

class HospitalClient:
    """
    Simulates a hospital's local training process.
    Patient data NEVER leaves this environment.
    Only model weights are extracted and sent.
    """

    def __init__(self, hospital_id: int):
        self.hospital_id = hospital_id
        self.local_models = {}
        self.local_scalers = {}

    def train_all_diseases(self, global_weights=None, epochs=5,
                           learning_rate=0.01, batch_size=32,
                           dp_epsilon=None):
        """
        Train models for ALL diseases on this hospital's local data.
        
        Args:
            global_weights: Dict of disease_type -> weights
            epochs: Training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            dp_epsilon: Differential privacy epsilon (None = no DP)
        """
        print(f"\n{'='*60}")
        print(f"HOSPITAL {self.hospital_id} - LOCAL TRAINING")
        if dp_epsilon:
            print(f"🔒 Differential Privacy: ε={dp_epsilon}")
        print(f"{'='*60}")

        results = {
            "hospital_id": self.hospital_id,
            "disease_weights": {},
            "disease_metrics": {},
            "samples_count": {},
            "dp_applied": dp_epsilon is not None,
            "dp_epsilon": dp_epsilon,
            "status": "success"
        }

        for disease_type in DISEASE_TYPES:
            try:
                print(f"\n  Training: {disease_type}")

                X, y, feature_names = preprocessor.load_disease_data(disease_type)
                X, y = preprocessor.balance_data(X, y, disease_type)

                np.random.seed(self.hospital_id * 100 + 42)
                indices = np.random.permutation(len(X))
                n = len(indices)
                start = ((self.hospital_id - 1) * n) // 3
                end = (self.hospital_id * n) // 3
                hospital_indices = indices[start:end]

                if len(hospital_indices) < 10:
                    hospital_indices = indices

                X_hospital = X[hospital_indices]
                y_hospital = y[hospital_indices]

                print(f"    Hospital data: {len(X_hospital)} samples, "
                      f"Disease: {y_hospital.sum()}, "
                      f"Healthy: {(y_hospital==0).sum()}")

                if len(X_hospital) < 20:
                    X_train, X_test = X_hospital, X_hospital
                    y_train, y_test = y_hospital, y_hospital
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_hospital, y_hospital,
                        test_size=0.2, random_state=42
                    )

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = get_model(disease_type)

                if global_weights and disease_type in global_weights:
                    try:
                        model.set_weights(global_weights[disease_type])
                        print(f"    Initialized with global weights")
                    except Exception as e:
                        print(f"    Could not load global weights: {e}")

                start_time = time.time()
                metrics = model.train(X_train_scaled, y_train,
                                      X_test_scaled, y_test)
                training_time = time.time() - start_time

                metrics["training_time"] = round(training_time, 3)
                metrics["hospital_id"] = self.hospital_id

                self.local_models[disease_type] = model
                self.local_scalers[disease_type] = scaler

                weights = model.get_weights()

                if dp_epsilon is not None and dp_epsilon > 0:
                    from app.federated.differential_privacy import DifferentialPrivacy
                    dp_module = DifferentialPrivacy(epsilon=dp_epsilon)
                    weights = dp_module.add_noise_to_weights(weights)
                    print(f"    🔒 DP noise added (ε={dp_epsilon})")

                results["disease_weights"][disease_type] = weights
                results["disease_metrics"][disease_type] = metrics
                results["samples_count"][disease_type] = len(X_train)

                print(f"    ✅ Done in {training_time:.2f}s | "
                      f"Accuracy: {metrics['accuracy']:.4f}")

            except Exception as e:
                print(f"    ❌ Failed: {e}")
                results["disease_metrics"][disease_type] = {"error": str(e)}

        print(f"\n  Hospital {self.hospital_id} training complete!")
        return results

    def train_single_disease(self, disease_type: str,
                              global_weights=None, epochs=5,
                              dp_epsilon=None):
        """Train model for a SINGLE disease."""
        print(f"\n  Hospital {self.hospital_id} training {disease_type}...")

        X, y, feature_names = preprocessor.load_disease_data(disease_type)
        X, y = preprocessor.balance_data(X, y, disease_type)

        np.random.seed(self.hospital_id * 100 + 42)
        indices = np.random.permutation(len(X))
        n = len(indices)
        start = ((self.hospital_id - 1) * n) // 3
        end = (self.hospital_id * n) // 3
        hospital_indices = indices[start:end]

        if len(hospital_indices) < 10:
            hospital_indices = indices

        X_h = X[hospital_indices]
        y_h = y[hospital_indices]

        if len(X_h) < 20:
            X_train, X_test = X_h, X_h
            y_train, y_test = y_h, y_h
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_h, y_h, test_size=0.2, random_state=42
            )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = get_model(disease_type)
        if global_weights:
            try:
                model.set_weights(global_weights)
            except Exception:
                pass

        metrics = model.train(X_train_s, y_train, X_test_s, y_test)

        self.local_models[disease_type] = model
        self.local_scalers[disease_type] = scaler

        weights = model.get_weights()

        if dp_epsilon is not None and dp_epsilon > 0:
            from app.federated.differential_privacy import DifferentialPrivacy
            dp_module = DifferentialPrivacy(epsilon=dp_epsilon)
            weights = dp_module.add_noise_to_weights(weights)

        return weights, metrics, len(X_train)