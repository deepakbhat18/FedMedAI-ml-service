import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
MODEL_STORAGE_DIR = os.path.join(STORAGE_DIR, "models")

# Hospital data directories
HOSPITAL_DATA_DIRS = {
    1: os.path.join(DATA_DIR, "hospital_1"),
    2: os.path.join(DATA_DIR, "hospital_2"),
    3: os.path.join(DATA_DIR, "hospital_3"),
}

# Disease types
DISEASE_TYPES = [
    "heart_disease",
    "diabetes",
    "kidney_disease",
    "liver_disease",
    "stroke"
]

# Dataset file mapping (raw files)
RAW_DATASET_FILES = {
    "heart_disease": "heart.csv",
    "diabetes": "diabetes.csv",
    "kidney_disease": None,  # Will be generated
    "liver_disease": "indian_liver_patient.csv",
    "stroke": "healthcare-dataset-stroke-data.csv",
}

# Disease display names
DISEASE_DISPLAY_NAMES = {
    "heart_disease": "Heart Disease",
    "diabetes": "Diabetes",
    "kidney_disease": "Chronic Kidney Disease",
    "liver_disease": "Liver Disease",
    "stroke": "Stroke",
}

# Model settings
MODEL_SETTINGS = {
    "default_epochs": 5,
    "default_learning_rate": 0.01,
    "default_batch_size": 32,
    "test_size": 0.2,
    "random_state": 42,
}

# Number of hospitals
NUM_HOSPITALS = 3

# Server settings
ML_SERVER_HOST = "0.0.0.0"
ML_SERVER_PORT = 8000

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, STORAGE_DIR, MODEL_STORAGE_DIR]:
    os.makedirs(directory, exist_ok=True)

for hospital_dir in HOSPITAL_DATA_DIRS.values():
    os.makedirs(hospital_dir, exist_ok=True)