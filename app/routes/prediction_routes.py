from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from app.models.disease_models import get_model
from app.utils.model_store import load_model, model_exists
from app.utils.metrics import get_risk_level
from config import DISEASE_TYPES, DISEASE_DISPLAY_NAMES
from fastapi.responses import StreamingResponse
import io
from datetime import datetime

router = APIRouter()


# ==================== REQUEST/RESPONSE MODELS ====================

class PredictRequest(BaseModel):
    features: List[float]
    disease_type: Optional[str] = None 

class SinglePrediction(BaseModel):
    disease_type: str
    disease_name: str
    prediction: int
    prediction_label: str
    confidence: float
    risk_level: str
    probabilities: List[float]

class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    predictions: List[SinglePrediction]
    model_version: str
    total_diseases_checked: int

class PredictSingleRequest(BaseModel):
    features: List[float]

class PredictFromDBRequest(BaseModel):
    """Request format from Spring Boot"""
    features: List[float]


# ==================== FEATURE MAPPING ====================


DISEASE_FEATURE_MAP = {
    "heart_disease": {
        "count": 13,
        "names": ["age", "sex", "cp", "trestbps", "chol", "fbs",
                  "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    },
    "diabetes": {
        "count": 8,
        "names": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    },
    "kidney_disease": {
        "count": 18,
        "names": ["age", "blood_pressure", "specific_gravity", "albumin",
                  "sugar", "blood_glucose_random", "blood_urea", "serum_creatinine",
                  "sodium", "potassium", "hemoglobin", "packed_cell_volume",
                  "white_blood_cell_count", "red_blood_cell_count",
                  "hypertension", "diabetes_mellitus", "appetite", "anemia"]
    },
    "liver_disease": {
        "count": 10,
        "names": ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
                  "Alkaline_Phosphotase", "Alamine_Aminotransferase",
                  "Aspartate_Aminotransferase", "Total_Protiens",
                  "Albumin", "Albumin_and_Globulin_Ratio"]
    },
    "stroke": {
        "count": 10,
        "names": ["gender", "age", "hypertension", "heart_disease",
                  "ever_married", "work_type", "Residence_type",
                  "avg_glucose_level", "bmi", "smoking_status"]
    }
}

# General input form → mapped to each disease's features
GENERAL_INPUT_FIELDS = [
    "age", "gender", "blood_pressure", "cholesterol", "heart_rate",
    "glucose", "bmi", "smoking", "diabetes_history", "family_history",
    "creatinine", "hemoglobin", "bilirubin", "alt_enzyme",
    "albumin", "potassium", "sodium", "urea"
]


def map_general_to_disease(general_features: List[float], disease_type: str) -> List[float]:
    """
    Map general input features to disease-specific features.
    Fills missing features with safe default values.
    """
    # Build a dict from general features
    feature_dict = {}
    for i, name in enumerate(GENERAL_INPUT_FIELDS):
        if i < len(general_features):
            feature_dict[name] = general_features[i]
        else:
            feature_dict[name] = 0.0

    # Map to disease-specific features
    if disease_type == "heart_disease":
        return [
            feature_dict.get("age", 50),
            feature_dict.get("gender", 0),
            0,  # cp (chest pain type - default)
            feature_dict.get("blood_pressure", 120),
            feature_dict.get("cholesterol", 200),
            1 if feature_dict.get("glucose", 100) > 120 else 0,  # fbs
            0,  # restecg
            feature_dict.get("heart_rate", 75),
            0,  # exang
            0,  # oldpeak
            1,  # slope
            0,  # ca
            2,  # thal
        ]

    elif disease_type == "diabetes":
        return [
            0,  # pregnancies
            feature_dict.get("glucose", 100),
            feature_dict.get("blood_pressure", 70),
            20,  # skin thickness default
            80,  # insulin default
            feature_dict.get("bmi", 25),
            0.5,  # diabetes pedigree
            feature_dict.get("age", 50),
        ]

    elif disease_type == "kidney_disease":
        return [
            feature_dict.get("age", 50),
            feature_dict.get("blood_pressure", 80),
            1.020,  # specific gravity
            0,  # albumin
            0,  # sugar
            feature_dict.get("glucose", 100),
            feature_dict.get("urea", 30),
            feature_dict.get("creatinine", 1.0),
            feature_dict.get("sodium", 140),
            feature_dict.get("potassium", 4.5),
            feature_dict.get("hemoglobin", 14),
            44,  # packed cell volume
            8000,  # wbc
            5.0,  # rbc
            1 if feature_dict.get("blood_pressure", 120) > 140 else 0,  # hypertension
            feature_dict.get("diabetes_history", 0),
            1,  # appetite (good)
            0,  # anemia
        ]

    elif disease_type == "liver_disease":
        return [
            feature_dict.get("age", 50),
            feature_dict.get("gender", 0),
            feature_dict.get("bilirubin", 1.0),
            0.3,  # direct bilirubin
            200,  # alkaline phosphotase
            feature_dict.get("alt_enzyme", 25),
            25,  # aspartate aminotransferase
            6.5,  # total proteins
            feature_dict.get("albumin", 3.5),
            1.0,  # albumin/globulin ratio
        ]

    elif disease_type == "stroke":
        return [
            feature_dict.get("gender", 0),
            feature_dict.get("age", 50),
            1 if feature_dict.get("blood_pressure", 120) > 140 else 0,  # hypertension
            feature_dict.get("diabetes_history", 0),  # heart_disease proxy
            1,  # ever married
            0,  # work type
            1,  # residence type
            feature_dict.get("glucose", 100),
            feature_dict.get("bmi", 25),
            feature_dict.get("smoking", 0),
        ]

    return general_features


# ==================== ENDPOINTS ====================

@router.post("/predict", response_model=dict)
async def predict_all_diseases(request: PredictRequest):
    """
    Predict all diseases or a specific disease.
    This is called by Spring Boot backend.
    """
    predictions = []
    model_version = "unknown"

    diseases_to_check = [request.disease_type] if request.disease_type else DISEASE_TYPES

    for disease_type in diseases_to_check:
        try:
            if not model_exists(disease_type):
                continue

            
            model, scaler = load_model(disease_type)
            if model is None:
                continue

            
            disease_features = map_general_to_disease(request.features, disease_type)

            
            features_array = np.array(disease_features).reshape(1, -1)
            if scaler is not None:
                features_array = scaler.transform(features_array)

            # Predict
            prediction = int(model.predict(features_array)[0])
            probabilities = model.predict_proba(features_array)[0].tolist()
            confidence = float(probabilities[1])  # Probability of disease

            risk_level = get_risk_level(confidence)
            prediction_label = (
                f"{DISEASE_DISPLAY_NAMES.get(disease_type, disease_type)} Detected"
                if prediction == 1
                else "Healthy"
            )

            predictions.append({
                "disease_type": disease_type,
                "disease_name": DISEASE_DISPLAY_NAMES.get(disease_type, disease_type),
                "prediction": prediction,
                "prediction_label": prediction_label,
                "confidence": round(confidence, 4),
                "risk_level": risk_level,
                "probabilities": [round(p, 4) for p in probabilities]
            })

        except Exception as e:
            print(f"Error predicting {disease_type}: {e}")
            continue

    # Sort by confidence (highest risk first)
      # Sort by confidence (highest risk first)
    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    # Generate AI recommendations
    recommendations = None
    try:
        from app.utils.recommendation_engine import recommendation_engine

        # Build feature dict from input list
        feature_names = [
            "age", "gender", "bloodPressure", "cholesterol",
            "heartRate", "glucose", "bmi", "smoking",
            "diabetesHistory", "familyHistory", "creatinine",
            "hemoglobin", "bilirubin", "altEnzyme",
            "albumin", "potassium", "sodium", "urea"
        ]
        features_dict = {}
        for i, name in enumerate(feature_names):
            if i < len(request.features):
                features_dict[name] = request.features[i]

        recommendations = recommendation_engine.generate(
            patient_features=features_dict,
            predictions=predictions
        )
        print(f"  Recommendations: {recommendations.get('source', 'Generated')}")

    except Exception as e:
        print(f"  Recommendations failed: {e}")

    return {
        "predictions": predictions,
        "model_version": "v1.0",
        "total_diseases_checked": len(predictions),
        "recommendations": recommendations
    }


@router.post("/predict/single")
async def predict_single_disease(disease_type: str, request: PredictSingleRequest):
    """Predict a single specific disease."""
    if disease_type not in DISEASE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown disease: {disease_type}. Available: {DISEASE_TYPES}"
        )

    if not model_exists(disease_type):
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for {disease_type}. Run training first."
        )

    model, scaler = load_model(disease_type)

    
    disease_features = map_general_to_disease(request.features, disease_type)
    features_array = np.array(disease_features).reshape(1, -1)

    if scaler is not None:
        features_array = scaler.transform(features_array)

    prediction = int(model.predict(features_array)[0])
    probabilities = model.predict_proba(features_array)[0].tolist()
    confidence = float(probabilities[1])

    return {
        "disease_type": disease_type,
        "disease_name": DISEASE_DISPLAY_NAMES.get(disease_type, disease_type),
        "prediction": prediction,
        "prediction_label": f"{DISEASE_DISPLAY_NAMES[disease_type]} Detected" if prediction == 1 else "Healthy",
        "confidence": round(confidence, 4),
        "risk_level": get_risk_level(confidence),
        "probabilities": [round(p, 4) for p in probabilities],
        "model_version": "v1.0"
    }
@router.post("/recommend")
async def get_recommendations_only(request: PredictRequest):
    """
    Get AI recommendations without making new predictions.
    Use existing predictions and generate recommendations.
    """
    from app.utils.recommendation_engine import recommendation_engine

    # Map features
    feature_names = [
        "age", "gender", "bloodPressure", "cholesterol",
        "heartRate", "glucose", "bmi", "smoking",
        "diabetesHistory", "familyHistory", "creatinine",
        "hemoglobin", "bilirubin", "altEnzyme",
        "albumin", "potassium", "sodium", "urea"
    ]
    features_dict = {}
    for i, name in enumerate(feature_names):
        if i < len(request.features):
            features_dict[name] = request.features[i]

    # First predict
    predictions = []
    for disease_type in DISEASE_TYPES:
        try:
            if not model_exists(disease_type):
                continue
            model, scaler = load_model(disease_type)
            if model is None:
                continue

            disease_features = map_general_to_disease(request.features, disease_type)
            features_array = np.array(disease_features).reshape(1, -1)
            if scaler is not None:
                features_array = scaler.transform(features_array)

            prediction = int(model.predict(features_array)[0])
            probabilities = model.predict_proba(features_array)[0].tolist()
            confidence = float(probabilities[1])
            risk_level = get_risk_level(confidence)

            predictions.append({
                "disease_type": disease_type,
                "disease_name": DISEASE_DISPLAY_NAMES.get(disease_type, disease_type),
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "risk_level": risk_level,
                "probabilities": [round(p, 4) for p in probabilities]
            })
        except Exception as e:
            continue

    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    recommendations = recommendation_engine.generate(
        patient_features=features_dict,
        predictions=predictions
    )

    return {
        "predictions": predictions,
        "recommendations": recommendations,
        "features_analyzed": features_dict
    }
class PDFReportRequest(BaseModel):
    patient_data: Dict
    predictions: List[Dict]
    recommendations: Optional[Dict] = None
    hospital_data: Optional[Dict] = None
    features: Optional[Dict] = None


@router.post("/report/generate")
async def generate_pdf_report(request: PDFReportRequest):
    """
    Generate professional PDF medical report.
    Returns PDF as downloadable file.
    """
    try:
        from app.utils.pdf_generator import pdf_generator

        pdf_bytes = pdf_generator.generate(
            patient_data=request.patient_data,
            predictions=request.predictions,
            recommendations=request.recommendations,
            hospital_data=request.hospital_data,
            features=request.features,
        )

        patient_code = request.patient_data.get('patientCode', 'report')
        filename = f"medical_report_{patient_code}_{datetime.now().strftime('%Y%m%d')}.pdf"

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(pdf_bytes)),
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")