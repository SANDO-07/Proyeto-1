from __future__ import annotations
import json
import pickle
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "fires_model_dataset.csv"
MODEL_PATH = ROOT / "models" / "random_forest_fire_risk.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"

def main() -> None:
    df = pd.read_csv(DATA_PATH)
    feature_cols_num = [
        "month", "temperature_c", "humidity_pct", "wind_kmh",
        "rain_last_7d_mm", "ndvi", "brightness_k", "frp_mw", "confidence_pct"
    ]
    feature_cols_cat = ["department"]

    X = df[feature_cols_num + feature_cols_cat]
    y = df["fire_risk"]

    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), feature_cols_num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), feature_cols_cat),
    ])

    model = Pipeline([
        ("prep", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=220,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Modelo y métricas actualizadas correctamente.")

if __name__ == "__main__":
    main()
