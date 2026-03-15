from __future__ import annotations
import pickle
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def root_path() -> Path:
    return ROOT

def load_model():
    model_path = ROOT / "models" / "random_forest_fire_risk.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

def load_forecast() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "outputs" / "calendario_incendios_2026.csv")

def load_metrics() -> dict:
    import json
    return json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))

def load_dataset() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "fires_model_dataset.csv")
