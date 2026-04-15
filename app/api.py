from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.utils import get_logger, load_model
from config.settings import Settings

settings = Settings()
logger = get_logger(__name__)

app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

# Lightweight request model: accepts a mapping of feature name -> value
class PredictionRequest(BaseModel):
    features: dict

# Try to load a model at startup; it's optional
_model = None

@app.on_event("startup")
def _startup():
    global _model
    try:
        _model = load_model("best_model.joblib")
        logger.info("Model loaded into API")
    except Exception:
        logger.warning("No model found at startup; /predict will return 503 until a model is saved")


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    # Convert features to a single-row DataFrame; model code should accept this shape
    df = pd.DataFrame([req.features])
    try:
        pred = _model.predict(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"prediction": float(pred[0])}
