import numpy as np
from typing import Dict, List

class DifferentialPrivacy:
    """
    Differential Privacy for Federated Learning.
    
    Adds calibrated noise to model weights before sharing.
    Uses Laplace mechanism for epsilon-differential privacy.
    
    Privacy Budget (epsilon):
    - Lower epsilon = MORE privacy, LESS accuracy
    - Higher epsilon = LESS privacy, MORE accuracy
    - Typical range: 0.1 (very private) to 10.0 (less private)
    - Recommended: 1.0 to 5.0 for medical data
    """

    def __init__(self, epsilon=1.0, delta=1e-5):
        """
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Probability of privacy breach (very small)
        """
        self.epsilon = epsilon
        self.delta = delta

    def add_noise_to_weights(self, weights: Dict) -> Dict:
        """
        Add Laplace noise to model weights.
        
        Laplace mechanism:
        noise = Laplace(0, sensitivity/epsilon)
        
        Args:
            weights: Dict with 'coefficients' and 'intercept'
            
        Returns:
            Noisy weights (same structure)
        """
        noisy_weights = {}

        if "coefficients" in weights:
            coef = np.array(weights["coefficients"])
            sensitivity = self._calculate_sensitivity(coef)
            noise_scale = sensitivity / self.epsilon
            noise = np.random.laplace(0, noise_scale, coef.shape)
            noisy_coef = coef + noise
            noisy_weights["coefficients"] = noisy_coef.tolist()
        
        if "intercept" in weights:
            intercept = np.array(weights["intercept"])
            sensitivity = self._calculate_sensitivity(intercept)
            noise_scale = sensitivity / self.epsilon
            noise = np.random.laplace(0, noise_scale, intercept.shape)
            noisy_intercept = intercept + noise
            noisy_weights["intercept"] = noisy_intercept.tolist()

        for key in weights:
            if key not in noisy_weights:
                noisy_weights[key] = weights[key]

        return noisy_weights

    def add_noise_to_all_diseases(self, disease_weights: Dict) -> Dict:
        """
        Add noise to weights for ALL diseases.
        
        Args:
            disease_weights: Dict mapping disease_type -> weights
            
        Returns:
            Dict with noisy weights for each disease
        """
        noisy_disease_weights = {}

        for disease_type, weights in disease_weights.items():
            noisy_weights = self.add_noise_to_weights(weights)
            noisy_disease_weights[disease_type] = noisy_weights
            
            if "coefficients" in weights and "coefficients" in noisy_weights:
                original = np.array(weights["coefficients"])
                noisy = np.array(noisy_weights["coefficients"])
                noise_magnitude = np.mean(np.abs(noisy - original))
                print(f"    [{disease_type}] Noise added: "
                      f"avg magnitude={noise_magnitude:.6f}, "
                      f"epsilon={self.epsilon}")

        return noisy_disease_weights

    def _calculate_sensitivity(self, array: np.ndarray) -> float:
        """
        Calculate sensitivity of the data.
        Sensitivity = max possible change in output when one record changes.
        
        For logistic regression weights, we use L2 sensitivity.
        """
        sensitivity = float(np.std(array)) if np.std(array) > 0 else 1.0
        return min(max(sensitivity, 0.001), 10.0)

    def get_privacy_report(self) -> Dict:
        """
        Get privacy configuration report.
        """
        privacy_level = "UNKNOWN"
        if self.epsilon <= 0.5:
            privacy_level = "VERY HIGH"
        elif self.epsilon <= 1.0:
            privacy_level = "HIGH"
        elif self.epsilon <= 5.0:
            privacy_level = "MODERATE"
        elif self.epsilon <= 10.0:
            privacy_level = "LOW"
        else:
            privacy_level = "MINIMAL"

        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "privacy_level": privacy_level,
            "mechanism": "Laplace",
            "description": f"ε={self.epsilon} differential privacy. "
                          f"Privacy level: {privacy_level}. "
                          f"Lower epsilon = more privacy but less accuracy."
        }

dp = DifferentialPrivacy(epsilon=1.0)