from __future__ import annotations
import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Guatemala Fire Detection MVP", layout="wide")
st.title("Train-LLM - Guatemala Fire Detection & Prediction")
st.caption("Dashboard MVP para el curso de Inteligencia Artificial")

metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
df = pd.read_csv(ROOT / "data" / "processed" / "fires_model_dataset.csv")
forecast = pd.read_csv(ROOT / "data" / "outputs" / "calendario_incendios_2026.csv")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", metrics["accuracy"])
col2.metric("ROC AUC", metrics["roc_auc"])
col3.metric("Registros", len(df))

st.subheader("Vista general del dataset")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Estacionalidad mensual")
monthly = df.groupby("month")["fire_risk"].sum().reset_index()
st.line_chart(monthly, x="month", y="fire_risk")

st.subheader("Forecast 2026")
st.dataframe(forecast, use_container_width=True)
st.line_chart(forecast, x="mes", y="prediccion_incendios")

st.subheader("Predicción manual")
with open(ROOT / "models" / "random_forest_fire_risk.pkl", "rb") as f:
    model = pickle.load(f)

with st.form("predict_form"):
    month = st.slider("Mes", 1, 12, 4)
    department = st.selectbox("Departamento", sorted(df["department"].unique()))
    temperature_c = st.number_input("Temperatura °C", value=33.0)
    humidity_pct = st.number_input("Humedad %", value=40.0)
    wind_kmh = st.number_input("Viento km/h", value=18.0)
    rain_last_7d_mm = st.number_input("Lluvia 7 días mm", value=2.0)
    ndvi = st.number_input("NDVI", value=0.35, min_value=0.0, max_value=1.0)
    brightness_k = st.number_input("Brillo térmico K", value=340.0)
    frp_mw = st.number_input("FRP MW", value=45.0)
    confidence_pct = st.number_input("Confianza %", value=82.0)
    submitted = st.form_submit_button("Predecir")

if submitted:
    input_df = pd.DataFrame([{
        "month": month,
        "department": department,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "wind_kmh": wind_kmh,
        "rain_last_7d_mm": rain_last_7d_mm,
        "ndvi": ndvi,
        "brightness_k": brightness_k,
        "frp_mw": frp_mw,
        "confidence_pct": confidence_pct,
    }])
    proba = float(model.predict_proba(input_df)[0, 1])
    st.success(f"Probabilidad de riesgo alto: {proba:.2%}")
