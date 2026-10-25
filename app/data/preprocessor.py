import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import RAW_DATA_DIR, HOSPITAL_DATA_DIRS, NUM_HOSPITALS

class DiseaseDataPreprocessor:
    """Handles loading, cleaning, splitting and balancing data for all diseases."""

    def __init__(self):
        self.scalers = {}

    def load_heart_data(self):
        path = os.path.join(RAW_DATA_DIR, "heart.csv")
        df = pd.read_csv(path)

        feature_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        target_col = "target"

        X = df[feature_cols].values.astype(float)
        y = df[target_col].values.astype(int)
        X = np.nan_to_num(X, nan=0.0)

        return X, y, feature_cols

    def load_diabetes_data(self):
        path = os.path.join(RAW_DATA_DIR, "diabetes.csv")
        df = pd.read_csv(path)

        feature_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        target_col = "Outcome"

        X = df[feature_cols].values.astype(float)
        y = df[target_col].values.astype(int)

        zero_replace_cols = [1, 2, 3, 4, 5]
        for col_idx in zero_replace_cols:
            col_mean = X[X[:, col_idx] != 0, col_idx].mean()
            X[X[:, col_idx] == 0, col_idx] = col_mean

        X = np.nan_to_num(X, nan=0.0)
        return X, y, feature_cols

    def load_kidney_data(self):
        path = os.path.join(RAW_DATA_DIR, "kidney_disease.csv")

        if not os.path.exists(path):
            from app.data.data_generator import generate_kidney_dataset
            generate_kidney_dataset()

        df = pd.read_csv(path)

        feature_cols = ["age", "blood_pressure", "specific_gravity", "albumin",
                        "sugar", "blood_glucose_random", "blood_urea", "serum_creatinine",
                        "sodium", "potassium", "hemoglobin", "packed_cell_volume",
                        "white_blood_cell_count", "red_blood_cell_count",
                        "hypertension", "diabetes_mellitus", "appetite", "anemia"]
        target_col = "classification"

        X = df[feature_cols].values.astype(float)
        y = df[target_col].values.astype(int)
        X = np.nan_to_num(X, nan=0.0)

        return X, y, feature_cols

    def load_liver_data(self):
        path = os.path.join(RAW_DATA_DIR, "indian_liver_patient.csv")
        df = pd.read_csv(path)

        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0)
        df["target"] = df["Dataset"].apply(lambda x: 1 if x == 1 else 0)

        feature_cols = ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
                        "Alkaline_Phosphotase", "Alamine_Aminotransferase",
                        "Aspartate_Aminotransferase", "Total_Protiens",
                        "Albumin", "Albumin_and_Globulin_Ratio"]

        df = df.fillna(df.median(numeric_only=True))

        X = df[feature_cols].values.astype(float)
        y = df["target"].values.astype(int)
        X = np.nan_to_num(X, nan=0.0)

        return X, y, feature_cols

    def load_stroke_data(self):
        path = os.path.join(RAW_DATA_DIR, "healthcare-dataset-stroke-data.csv")
        df = pd.read_csv(path)

        df = df.drop(columns=["id"], errors="ignore")

        df["gender"] = df["gender"].map(
            {"Male": 1, "Female": 0, "Other": 0}).fillna(0)
        df["ever_married"] = df["ever_married"].map(
            {"Yes": 1, "No": 0}).fillna(0)
        df["work_type"] = df["work_type"].map({
            "Private": 0, "Self-employed": 1,
            "Govt_job": 2, "children": 3, "Never_worked": 4
        }).fillna(0)
        df["Residence_type"] = df["Residence_type"].map(
            {"Urban": 1, "Rural": 0}).fillna(0)
        df["smoking_status"] = df["smoking_status"].map({
            "formerly smoked": 1, "smokes": 2,
            "never smoked": 0, "Unknown": 0
        }).fillna(0)

        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())

        feature_cols = ["gender", "age", "hypertension", "heart_disease",
                        "ever_married", "work_type", "Residence_type",
                        "avg_glucose_level", "bmi", "smoking_status"]

        X = df[feature_cols].values.astype(float)
        y = df["stroke"].values.astype(int)
        X = np.nan_to_num(X, nan=0.0)

        return X, y, feature_cols

    def load_disease_data(self, disease_type):
        loaders = {
            "heart_disease": self.load_heart_data,
            "diabetes": self.load_diabetes_data,
            "kidney_disease": self.load_kidney_data,
            "liver_disease": self.load_liver_data,
            "stroke": self.load_stroke_data,
        }
        if disease_type not in loaders:
            raise ValueError(f"Unknown disease type: {disease_type}")
        return loaders[disease_type]()

    def balance_data(self, X, y, disease_type="unknown"):
        """
        Balance imbalanced dataset using SMOTE.
        Only applies if imbalance ratio > 1.5

        Args:
            X: Features
            y: Labels
            disease_type: For logging

        Returns:
            Balanced X, y
        """
        class_counts = np.bincount(y)
        majority = class_counts.max()
        minority = class_counts.min()
        ratio = majority / minority

        if ratio > 1.5:
            print(f"  [{disease_type}] Imbalance ratio: {ratio:.2f} - Applying SMOTE")
            try:
                
                k_neighbors = min(5, minority - 1)
                if k_neighbors < 1:
                    print(f"  [{disease_type}] Too few minority samples, skipping SMOTE")
                    return X, y

                smote = SMOTE(
                    random_state=42,
                    k_neighbors=k_neighbors
                )
                X_balanced, y_balanced = smote.fit_resample(X, y)
                print(f"  [{disease_type}] After SMOTE: {len(X_balanced)} samples")
                return X_balanced, y_balanced
            except Exception as e:
                print(f"  [{disease_type}] SMOTE failed: {e}, using original data")
                return X, y
        else:
            print(f"  [{disease_type}] Balanced enough (ratio: {ratio:.2f}), no SMOTE needed")
            return X, y

    def prepare_training_data(self, disease_type, balance=True):
        """
        Full pipeline: load → balance → scale → split.

        Args:
            disease_type: Disease name
            balance: Whether to apply SMOTE

        Returns:
            X_train, X_test, y_train, y_test, scaler, feature_names
        """
        print(f"\nPreparing data for: {disease_type}")

        X, y, feature_names = self.load_disease_data(disease_type)
        print(f"  Loaded: {X.shape[0]} rows, {X.shape[1]} features")
        print(f"  Disease: {y.sum()}, Healthy: {(y==0).sum()}")

        if balance:
            X, y = self.balance_data(X, y, disease_type)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers[disease_type] = scaler

        print(f"  Train: {X_train_scaled.shape[0]}, Test: {X_test_scaled.shape[0]}")

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

    def split_for_hospitals(self, X, y, num_hospitals=NUM_HOSPITALS):
        """
        Split dataset for each hospital.
        Each hospital gets different portion.

        Returns:
            List of (X_hospital, y_hospital) tuples
        """
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        split_sizes = np.array_split(indices, num_hospitals)

        hospital_data = []
        for i, split_idx in enumerate(split_sizes):
            X_h = X[split_idx]
            y_h = y[split_idx]
            hospital_data.append((X_h, y_h))
            print(f"  Hospital {i+1}: {len(X_h)} samples, "
                  f"Disease: {y_h.sum()}, Healthy: {(y_h==0).sum()}")

        return hospital_data

    def scale_features(self, X, disease_type):
        """Scale features using cached scaler."""
        if disease_type in self.scalers:
            return self.scalers[disease_type].transform(X)
        else:
            
            scaler = StandardScaler()
            return scaler.fit_transform(X)

preprocessor = DiseaseDataPreprocessor()