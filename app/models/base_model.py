import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from app.utils.metrics import calculate_metrics
from app.utils.model_store import save_model, load_model, get_model_weights, set_model_weights

class BaseDiseaseModel:
    """
    Base class for all disease models.
    Uses Logistic Regression for Federated Learning compatibility.
    Logistic Regression is perfect for FL because:
    - Weights (coefficients) are easy to extract
    - FedAvg works directly on coefficients
    - Fast training on small hospital datasets
    - Interpretable results
    """

    def __init__(self, disease_type):
        self.disease_type = disease_type
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.version = "v1.0"
        self.is_trained = False

    def build_model(self):
        """Create Logistic Regression model."""
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            C=1.0
        )
        return self.model

    def train(self, X_train, y_train, X_test, y_test):
        """
        Train model and return metrics.

        Args:
            X_train: Scaled training features
            y_train: Training labels
            X_test: Scaled test features
            y_test: Test labels

        Returns:
            Dictionary of training metrics
        """
        if self.model is None:
            self.build_model()

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_prob)

        print(f"  [{self.disease_type}] "
              f"Accuracy: {metrics['accuracy']:.4f}, "
              f"F1: {metrics['f1_score']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}")

        return metrics

    def predict(self, X):
        """
        Make prediction on features.

        Args:
            X: Feature array (already scaled)

        Returns:
            prediction (0/1), confidence (0-1), probabilities
        """
        if self.model is None:
            raise ValueError(f"Model not trained for {self.disease_type}")

        prediction = int(self.model.predict(X)[0])
        probabilities = self.model.predict_proba(X)[0]
        confidence = float(probabilities[prediction])

        return prediction, confidence, probabilities.tolist()

    def get_weights(self):
        """
        Extract model weights for Federated Learning.

        Returns:
            Dictionary with coefficients and intercept
        """
        if self.model is None or not self.is_trained:
            return None

        return {
            "coefficients": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
            "disease_type": self.disease_type,
            "n_features": len(self.model.coef_[0])
        }

    def set_weights(self, weights):
        """
        Set model weights from FL aggregation.

        Args:
            weights: Dictionary with coefficients and intercept
        """
        if self.model is None:
            self.build_model()
            n_features = len(weights["coefficients"][0])
            dummy_X = np.zeros((2, n_features))
            dummy_y = np.array([0, 1])
            self.model.fit(dummy_X, dummy_y)

        self.model.coef_ = np.array(weights["coefficients"])
        self.model.intercept_ = np.array(weights["intercept"])
        self.is_trained = True

    def save(self, version=None, metrics=None):
        """Save model to disk."""
        if version:
            self.version = version
        save_model(
            self.model,
            self.scaler,
            self.disease_type,
            self.version,
            metrics
        )

    def load(self, version=None):
        """Load model from disk."""
        model, scaler = load_model(self.disease_type, version)
        if model is not None:
            self.model = model
            self.scaler = scaler
            self.is_trained = True
            return True
        return False