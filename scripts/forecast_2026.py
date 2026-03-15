from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "monthly_fire_summary.csv"
OUT_PATH = ROOT / "data" / "outputs" / "calendario_incendios_2026.csv"

def main() -> None:
    monthly = pd.read_csv(IN_PATH)
    monthly_avg = monthly.groupby("month")["fire_events"].mean()
    yearly_totals = monthly.groupby("year")["fire_events"].sum()
    trend_per_year = np.polyfit(yearly_totals.index.astype(int), yearly_totals.values, 1)[0]
    projected_total = max(12, yearly_totals.iloc[-1] + trend_per_year * 2)
    shares = monthly.groupby("month")["fire_events"].sum() / monthly["fire_events"].sum()

    rows = []
    for m in range(1, 13):
        pred = projected_total * shares.loc[m]
        rows.append({
            "fecha": f"2026-{m:02d}-01",
            "mes": m,
            "prediccion_incendios": round(float(pred), 2),
            "intervalo_inferior": round(float(pred * 0.82), 2),
            "intervalo_superior": round(float(pred * 1.18), 2),
            "nivel_riesgo": "Alto" if m in [3, 4, 5] else ("Medio" if m in [2, 6, 11, 12] else "Bajo"),
        })
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"Forecast guardado en {OUT_PATH}")

if __name__ == "__main__":
    main()
