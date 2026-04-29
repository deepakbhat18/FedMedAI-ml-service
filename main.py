
from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import prediction_routes, training_routes, model_routes
from config import ML_SERVER_HOST, ML_SERVER_PORT

app = FastAPI(
    title="Federated Learning ML Service",
    description="Multi-Disease Prediction with Federated Learning + AI Recommendations. "
                "Supports: Heart Disease, Diabetes, Kidney Disease, "
                "Liver Disease, Stroke.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_routes.router, prefix="/ml", tags=["Prediction"])
app.include_router(training_routes.router, prefix="/ml", tags=["Training"])
app.include_router(model_routes.router, prefix="/ml", tags=["Model Management"])


@app.get("/")
async def root():
    return {
        "service": "Federated Learning ML Service",
        "version": "1.0.0",
        "features": [
            "Multi-Disease Prediction (5 diseases)",
            "Federated Learning (Custom FedAvg + Flower)",
            "Differential Privacy",
            "AI Medical Recommendations (OpenRouter)",
        ],
        "diseases": [
            "Heart Disease", "Diabetes",
            "Kidney Disease", "Liver Disease", "Stroke"
        ],
        "docs": "http://localhost:8000/docs"
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  FEDERATED LEARNING ML SERVICE")
    print("  Multi-Disease Medical Diagnosis + AI Recommendations")
    print("="*60)
    print(f"  Server: http://{ML_SERVER_HOST}:{ML_SERVER_PORT}")
    print(f"  Docs:   http://localhost:{ML_SERVER_PORT}/docs")
    print("="*60 + "\n")

    uvicorn.run(
        "main:app",
        host=ML_SERVER_HOST,
        port=ML_SERVER_PORT,
        reload=True
    )
