import numpy as np
import pandas as pd
import os
from config import RAW_DATA_DIR


def generate_kidney_dataset(n_samples=500):
    """
    Generate realistic chronic kidney disease dataset.
    Based on UCI CKD dataset features.
    """
    np.random.seed(42)

    age = np.random.randint(20, 80, n_samples)
    blood_pressure = np.random.randint(60, 180, n_samples)
    specific_gravity = np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n_samples)
    albumin = np.random.randint(0, 5, n_samples)
    sugar = np.random.randint(0, 5, n_samples)
    blood_glucose = np.random.randint(70, 490, n_samples)
    blood_urea = np.random.randint(10, 200, n_samples)
    serum_creatinine = np.round(np.random.uniform(0.4, 15.0, n_samples), 1)
    sodium = np.random.randint(111, 163, n_samples)
    potassium = np.round(np.random.uniform(2.5, 47.0, n_samples), 1)
    hemoglobin = np.round(np.random.uniform(3.1, 17.8, n_samples), 1)
    packed_cell_volume = np.random.randint(16, 54, n_samples)
    white_blood_cell_count = np.random.randint(3800, 26400, n_samples)
    red_blood_cell_count = np.round(np.random.uniform(2.1, 8.0, n_samples), 1)
    hypertension = np.random.randint(0, 2, n_samples)
    diabetes_mellitus = np.random.randint(0, 2, n_samples)
    appetite = np.random.randint(0, 2, n_samples)
    anemia = np.random.randint(0, 2, n_samples)

    # Generate realistic target based on risk factors
    risk_score = (
        (blood_pressure > 140).astype(float) * 0.15 +
        (albumin > 2).astype(float) * 0.2 +
        (blood_glucose > 200).astype(float) * 0.1 +
        (blood_urea > 50).astype(float) * 0.15 +
        (serum_creatinine > 3).astype(float) * 0.2 +
        (hemoglobin < 10).astype(float) * 0.1 +
        (hypertension == 1).astype(float) * 0.05 +
        (diabetes_mellitus == 1).astype(float) * 0.05
    )
    target = (risk_score + np.random.normal(0, 0.08, n_samples) > 0.35).astype(int)

    df = pd.DataFrame({
        "age": age,
        "blood_pressure": blood_pressure,
        "specific_gravity": specific_gravity,
        "albumin": albumin,
        "sugar": sugar,
        "blood_glucose_random": blood_glucose,
        "blood_urea": blood_urea,
        "serum_creatinine": serum_creatinine,
        "sodium": sodium,
        "potassium": potassium,
        "hemoglobin": hemoglobin,
        "packed_cell_volume": packed_cell_volume,
        "white_blood_cell_count": white_blood_cell_count,
        "red_blood_cell_count": red_blood_cell_count,
        "hypertension": hypertension,
        "diabetes_mellitus": diabetes_mellitus,
        "appetite": appetite,
        "anemia": anemia,
        "classification": target
    })

    # Save to raw directory
    output_path = os.path.join(RAW_DATA_DIR, "kidney_disease.csv")
    df.to_csv(output_path, index=False)
    print(f"Kidney dataset generated: {len(df)} rows")
    print(f"Disease cases: {df.classification.sum()}")
    print(f"Healthy cases: {(df.classification == 0).sum()}")
    print(f"Saved to: {output_path}")

    return df


if __name__ == "__main__":
    generate_kidney_dataset()