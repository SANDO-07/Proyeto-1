from __future__ import annotations
import io
import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest_fire_risk.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"
FORECAST_PATH = ROOT / "data" / "outputs" / "calendario_incendios_2026.csv"

app = FastAPI(
    title="Train-LLM Guatemala Fire Detection API",
    description="API MVP para análisis y predicción de incendios forestales en Guatemala.",
    version="1.0.0",
)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
forecast_df = pd.read_csv(FORECAST_PATH)

class TabularInput(BaseModel):
    month: int = Field(..., ge=1, le=12)
    department: str
    temperature_c: float
    humidity_pct: float = Field(..., ge=0, le=100)
    wind_kmh: float = Field(..., ge=0)
    rain_last_7d_mm: float = Field(..., ge=0)
    ndvi: float = Field(..., ge=0, le=1)
    brightness_k: float = Field(..., ge=250, le=500)
    frp_mw: float = Field(..., ge=0)
    confidence_pct: float = Field(..., ge=0, le=100)

@app.get("/health")
def health():
    return {"status": "ok", "service": "fire-detection-mvp"}

@app.get("/metrics")
def get_metrics():
    return metrics

@app.get("/forecast/2026")
def get_forecast():
    return forecast_df.to_dict(orient="records")

@app.post("/predict/tabular")
def predict_tabular(payload: TabularInput):
    df = pd.DataFrame([payload.model_dump()])
    prob = float(model.predict_proba(df)[0, 1])
    pred = int(prob >= 0.5)
    return {
        "prediction": pred,
        "label": "Riesgo alto" if pred == 1 else "Riesgo bajo",
        "probability": round(prob, 4),
        "input": payload.model_dump(),
    }

@app.post("/predict/vision")
async def predict_vision(file: UploadFile = File(...)):
    """
    MVP visual:
    analiza brillo promedio y dominancia rojiza como una aproximación simple.
    Esto NO reemplaza un modelo multimodal real, pero deja listo el flujo API.
    """
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    arr = np.asarray(image).astype(np.float32)

    mean_r = float(arr[:, :, 0].mean())
    mean_g = float(arr[:, :, 1].mean())
    mean_b = float(arr[:, :, 2].mean())
    brightness = float(arr.mean())

    fire_score = (mean_r - mean_g) * 0.7 + (brightness - 110) * 0.04
    probable_fire = fire_score > 18

    explanation = (
        "Se detectan patrones compatibles con focos calientes o zonas rojizas intensas."
        if probable_fire
        else "La imagen no muestra un patrón visual fuerte de foco térmico en este MVP."
    )

    return {
        "filename": file.filename,
        "mean_rgb": {"r": round(mean_r, 2), "g": round(mean_g, 2), "b": round(mean_b, 2)},
        "brightness_score": round(brightness, 2),
        "vision_score": round(fire_score, 2),
        "prediction": "Posible incendio" if probable_fire else "Sin evidencia fuerte",
        "explanation": explanation,
        "note": "Este endpoint es un stub MVP listo para ser reemplazado por un VLM real.",
    }
