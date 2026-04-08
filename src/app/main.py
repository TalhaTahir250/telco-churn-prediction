# src/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.serving.inference import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Request schema ─────────────────────────────────────────────────────────────
class CustomerRequest(BaseModel):
    gender:           str
    SeniorCitizen:    int
    Partner:          str
    Dependents:       str
    tenure:           float
    PhoneService:     str
    MultipleLines:    str
    InternetService:  str
    OnlineSecurity:   str
    OnlineBackup:     str
    DeviceProtection: str
    TechSupport:      str
    StreamingTV:      str
    StreamingMovies:  str
    Contract:         str
    PaperlessBilling: str
    PaymentMethod:    str
    MonthlyCharges:   float
    Total_Charges:    float


# ── Response schema ────────────────────────────────────────────────────────────
class ChurnResponse(BaseModel):
    churn:             bool
    churn_probability: float
    threshold_used:    float
    risk_level:        str


# ── Startup ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifact...")
    # Warm up model on startup
    try:
        import joblib
        joblib.load("model.pkl")
        logger.info("Model loaded and ready.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    logger.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Telco Churn Prediction API",
    description = "Predicts customer churn probability for telecom customers.",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Telco Churn Prediction API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=ChurnResponse)
def churn_predict(request: CustomerRequest):
    try:
        result = predict(request.model_dump())

        # Add human-readable risk level
        prob = result["churn_probability"]
        if prob >= 0.7:
            risk = "HIGH"
        elif prob >= 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        logger.info(f"Prediction: churn={result['churn']} prob={prob:.3f} risk={risk}")

        return ChurnResponse(
            churn             = result["churn"],
            churn_probability = result["churn_probability"],
            threshold_used    = result["threshold_used"],
            risk_level        = risk,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    