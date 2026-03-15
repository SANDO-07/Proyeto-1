"""
Plantilla para descargar datos reales de NASA FIRMS.
Requiere registrar una API key propia en:
https://firms.modaps.eosdis.nasa.gov/api/area/

Uso:
    export FIRMS_MAP_KEY=tu_api_key
    python scripts/download_firms_template.py
"""
from __future__ import annotations
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "nasa_firms_guatemala_real.csv"

def main() -> None:
    api_key = os.getenv("FIRMS_MAP_KEY", "").strip()
    if not api_key:
        raise SystemExit("No se encontró FIRMS_MAP_KEY en variables de entorno.")

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/VIIRS_SNPP_NRT/country/GTM/1/2024-01-01"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    OUT_PATH.write_bytes(response.content)
    df = pd.read_csv(OUT_PATH)
    print("Descarga completada.")
    print(df.head())

if __name__ == "__main__":
    main()
