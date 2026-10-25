from app.models.base_model import BaseDiseaseModel

class HeartDiseaseModel(BaseDiseaseModel):
    """
    Heart Disease Prediction Model
    Dataset: UCI Heart Disease
    Features: age, sex, cp, trestbps, chol, fbs,
              restecg, thalach, exang, oldpeak, slope, ca, thal
    Target: 0=Healthy, 1=Heart Disease
    """
    def __init__(self):
        super().__init__("heart_disease")
        self.feature_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]

class DiabetesModel(BaseDiseaseModel):
    """
    Diabetes Prediction Model
    Dataset: Pima Indians Diabetes
    Features: Pregnancies, Glucose, BloodPressure, SkinThickness,
              Insulin, BMI, DiabetesPedigreeFunction, Age
    Target: 0=Healthy, 1=Diabetes
    """
    def __init__(self):
        super().__init__("diabetes")
        self.feature_names = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]

class KidneyDiseaseModel(BaseDiseaseModel):
    """
    Chronic Kidney Disease Prediction Model
    Dataset: Generated (based on UCI CKD features)
    Features: age, blood_pressure, specific_gravity, albumin, sugar,
              blood_glucose_random, blood_urea, serum_creatinine,
              sodium, potassium, hemoglobin, packed_cell_volume,
              white_blood_cell_count, red_blood_cell_count,
              hypertension, diabetes_mellitus, appetite, anemia
    Target: 0=Healthy, 1=CKD
    """
    def __init__(self):
        super().__init__("kidney_disease")
        self.feature_names = [
            "age", "blood_pressure", "specific_gravity", "albumin",
            "sugar", "blood_glucose_random", "blood_urea", "serum_creatinine",
            "sodium", "potassium", "hemoglobin", "packed_cell_volume",
            "white_blood_cell_count", "red_blood_cell_count",
            "hypertension", "diabetes_mellitus", "appetite", "anemia"
        ]

class LiverDiseaseModel(BaseDiseaseModel):
    """
    Liver Disease Prediction Model
    Dataset: Indian Liver Patient Dataset (ILPD)
    Features: Age, Gender, Total_Bilirubin, Direct_Bilirubin,
              Alkaline_Phosphotase, Alamine_Aminotransferase,
              Aspartate_Aminotransferase, Total_Protiens,
              Albumin, Albumin_and_Globulin_Ratio
    Target: 0=Healthy, 1=Liver Disease
    """
    def __init__(self):
        super().__init__("liver_disease")
        self.feature_names = [
            "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
            "Alkaline_Phosphotase", "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase", "Total_Protiens",
            "Albumin", "Albumin_and_Globulin_Ratio"
        ]

class StrokeModel(BaseDiseaseModel):
    """
    Stroke Prediction Model
    Dataset: Healthcare Stroke Dataset
    Features: gender, age, hypertension, heart_disease,
              ever_married, work_type, Residence_type,
              avg_glucose_level, bmi, smoking_status
    Target: 0=No Stroke, 1=Stroke
    """
    def __init__(self):
        super().__init__("stroke")
        self.feature_names = [
            "gender", "age", "hypertension", "heart_disease",
            "ever_married", "work_type", "Residence_type",
            "avg_glucose_level", "bmi", "smoking_status"
        ]

MODEL_REGISTRY = {
    "heart_disease": HeartDiseaseModel,
    "diabetes": DiabetesModel,
    "kidney_disease": KidneyDiseaseModel,
    "liver_disease": LiverDiseaseModel,
    "stroke": StrokeModel,
}

def get_model(disease_type):
    """
    Get model instance for a disease type.

    Args:
        disease_type: e.g., 'heart_disease'

    Returns:
        Model instance
    """
    if disease_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown disease: {disease_type}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[disease_type]()

def get_all_models():
    """Get instances of all disease models."""
    return {
        disease_type: model_class()
        for disease_type, model_class in MODEL_REGISTRY.items()
    }