import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import flwr as fl
from app.data.preprocessor import preprocessor
from app.utils.metrics import calculate_metrics
from config import DISEASE_TYPES


class HospitalFlowerClient(fl.client.NumPyClient):
    """
    Flower client representing ONE hospital.
    
    In real Federated Learning:
    - This code runs ON the hospital's own server
    - Patient data NEVER leaves this client
    - Only model weights are sent to server
    
    Flower handles all the communication automatically.
    """

    def __init__(self, hospital_id, disease_type, X_train, y_train, X_test, y_test):
        self.hospital_id = hospital_id
        self.disease_type = disease_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Create and initialize model
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            warm_start=True
        )
        # Initial fit to set model shape
        self.model.fit(self.X_train, self.y_train)

    def get_parameters(self, config):
        """
        Send model weights TO the Flower server.
        This is what gets aggregated (NOT patient data).
        """
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        """
        Receive aggregated global weights FROM the Flower server.
        Update local model with new global weights.
        """
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        """
        Train model locally on THIS hospital's private data.
        
        Steps:
        1. Receive global weights from server
        2. Update local model
        3. Train on LOCAL data (data stays here!)
        4. Return ONLY updated weights + sample count
        """
        # Update model with global weights
        self.set_parameters(parameters)
        
        # Train on LOCAL private data
        self.model.fit(self.X_train, self.y_train)
        
        # Return updated weights, sample count, and metrics
        # ONLY weights leave the hospital, NOT data!
        return (
            self.get_parameters(config),
            len(self.X_train),
            {
                "hospital_id": self.hospital_id,
                "disease": self.disease_type,
            }
        )

    def evaluate(self, parameters, config):
        """
        Evaluate the global model on local test data.
        Returns metrics to server for monitoring.
        """
        self.set_parameters(parameters)
        
        # Predict on local test set
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(self.y_test, y_pred, y_prob)
        
        loss = metrics.get("loss", 0.0) or 0.0
        accuracy = metrics.get("accuracy", 0.0)
        
        return (
            float(loss),
            len(self.X_test),
            {
                "accuracy": float(accuracy),
                "f1_score": float(metrics.get("f1_score", 0.0)),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "hospital_id": self.hospital_id,
            }
        )


def create_flower_client(hospital_id, disease_type):
    """
    Factory function to create a Flower client for a specific hospital.
    
    Loads data, splits for this hospital, and returns client instance.
    """
    # Load disease data
    X, y, feature_names = preprocessor.load_disease_data(disease_type)
    X, y = preprocessor.balance_data(X, y, disease_type)
    
    # Split data for this specific hospital
    np.random.seed(hospital_id * 100 + 42)
    indices = np.random.permutation(len(X))
    n = len(indices)
    start = ((hospital_id - 1) * n) // 3
    end = (hospital_id * n) // 3
    hospital_indices = indices[start:end]
    
    if len(hospital_indices) < 10:
        hospital_indices = indices
    
    X_hospital = X[hospital_indices]
    y_hospital = y[hospital_indices]
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hospital)
    
    # Train/test split
    if len(X_scaled) < 20:
        X_train, X_test = X_scaled, X_scaled
        y_train, y_test = y_hospital, y_hospital
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_hospital,
            test_size=0.2,
            random_state=42
        )
    
    return HospitalFlowerClient(
        hospital_id=hospital_id,
        disease_type=disease_type,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    ), scaler