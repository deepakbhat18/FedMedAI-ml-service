from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import prediction_routes, training_routes, model_routes
from config import ML_SERVER_HOST, ML_SERVER_PORT

app = FastAPI(
    title="Federated Learning ML Service",
    description="Multi-Disease Prediction with Federated Learning + AI Recommendations.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "status": "UP"
    }

@app.get("/ml/health")
async def health_check():
    return {"status": "UP"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=ML_SERVER_HOST,
        port=ML_SERVER_PORT,
        reload=True
    )