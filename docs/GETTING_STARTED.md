# Guía rápida

## 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 2. Ver resultados existentes
- `models/metrics.json`
- `visualizations/`
- `data/outputs/calendario_incendios_2026.csv`

## 3. Levantar API
```bash
uvicorn api.main:app --reload
```

## 4. Levantar dashboard
```bash
streamlit run dashboard/app.py
```

## 5. Volver a entrenar el modelo
```bash
python scripts/train_model.py
```
