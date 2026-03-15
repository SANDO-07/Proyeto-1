from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "sample_firms_guatemala_2014_2024.csv"

def main() -> None:
    rng = np.random.default_rng(42)
    departments = {
        "Petén": (16.9, -89.9),
        "Alta Verapaz": (15.5, -90.4),
        "Quiché": (15.0, -91.0),
        "Huehuetenango": (15.3, -91.5),
        "Izabal": (15.6, -89.0),
        "Zacapa": (14.9, -89.5),
        "Jutiapa": (14.3, -89.9),
        "Escuintla": (14.1, -90.8),
        "Santa Rosa": (14.2, -90.3),
        "Chiquimula": (14.8, -89.5),
    }
    rows = []
    for year in range(2014, 2025):
        for month in range(1, 13):
            seasonal = 2.2 if month in [3, 4, 5] else 0.8
            n = max(12, int(rng.normal(45 * seasonal, 8)))
            for _ in range(n):
                dept = random.choice(list(departments))
                lat0, lon0 = departments[dept]
                temp = rng.normal(28, 4) + (4.5 if month in [3, 4, 5] else 0)
                humidity = rng.normal(58, 12) - (18 if month in [3, 4, 5] else 0)
                wind = max(0.1, rng.normal(12, 4))
                rain = max(0, rng.normal(3 if month in [3, 4, 5] else 10, 5))
                ndvi = float(np.clip(rng.normal(0.53, 0.16), 0.05, 0.95))
                brightness = 310 + temp * 0.9 + wind * 0.7 - humidity * 0.18 + rng.normal(0, 6)
                frp = max(1, rng.normal(25, 12) + (temp - 25) * 1.1)
                confidence = float(np.clip(40 + frp * 1.3 + (brightness - 320) * 0.8, 0, 100))
                risk = int((temp > 31 and humidity < 48 and brightness > 330) or (frp > 40))
                rows.append({
                    "year": year,
                    "month": month,
                    "department": dept,
                    "latitude": round(float(rng.normal(lat0, 0.22)), 5),
                    "longitude": round(float(rng.normal(lon0, 0.25)), 5),
                    "temperature_c": round(float(temp), 2),
                    "humidity_pct": round(float(np.clip(humidity, 10, 98)), 2),
                    "wind_kmh": round(float(np.clip(wind, 0.1, 45)), 2),
                    "rain_last_7d_mm": round(float(np.clip(rain, 0, 120)), 2),
                    "ndvi": round(ndvi, 3),
                    "brightness_k": round(float(np.clip(brightness, 290, 380)), 2),
                    "frp_mw": round(float(np.clip(frp, 1, 220)), 2),
                    "confidence_pct": round(confidence, 2),
                    "fire_risk": risk,
                })
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"Archivo generado: {OUT_PATH}")

if __name__ == "__main__":
    main()
