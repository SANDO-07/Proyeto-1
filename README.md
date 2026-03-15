# Train-LLM - Guatemala Fire Detection & Prediction (MVP)

Proyecto del curso de **Inteligencia Artificial** enfocado en un MVP para detección y predicción de incendios forestales en Guatemala.

## Objetivo
Construir una base técnica reproducible que combine:

- análisis exploratorio de incendios históricos,
- un modelo de **Machine Learning** para riesgo de incendio,
- un módulo base para análisis **multimodal** de imágenes,
- una **API FastAPI**,
- y un **dashboard Streamlit**.

> Nota importante: este repositorio incluye un **dataset de muestra sintético** con formato similar a datos satelitales / FIRMS para que el proyecto sea demostrable desde ya. El código deja listo el flujo para sustituirlo por datos reales de NASA FIRMS y un modelo multimodal real.

## Estructura del repositorio

```text
train-llm-guatemala-fire-mvp/
├── api/                  # API FastAPI
├── dashboard/            # Dashboard Streamlit
├── data/
│   ├── raw/              # Datos fuente
│   ├── processed/        # Datos limpios / modelado
│   └── outputs/          # Forecast y reportes
├── docs/                 # Informe técnico y guía
├── models/               # Modelo entrenado y métricas
├── notebooks/            # Exploración y modelado
├── scripts/              # Utilidades de generación / entrenamiento
├── visualizations/       # Gráficas exportadas
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## Estado del avance
Este repositorio ya incluye:

- dataset base `sample_firms_guatemala_2014_2024.csv`
- dataset procesado para modelado
- modelo Random Forest entrenado
- métricas y visualizaciones
- forecast mensual 2026
- API MVP
- dashboard MVP
- documentación para presentar el avance

## Cómo ejecutar

### 1) Crear entorno e instalar dependencias
```bash
pip install -r requirements.txt
```

### 2) Ejecutar API
```bash
uvicorn api.main:app --reload
```

API docs:
- `http://127.0.0.1:8000/docs`

### 3) Ejecutar dashboard
```bash
streamlit run dashboard/app.py
```

## Endpoints disponibles
- `GET /health`
- `GET /metrics`
- `POST /predict/tabular`
- `GET /forecast/2026`
- `POST /predict/vision`

## Qué falta para la versión final
- conectar NASA FIRMS real
- integrar imágenes reales satelitales
- reemplazar heurística visual por un LLM/VLM real
- entrenar un modelo multimodal con imágenes anotadas de Guatemala

## Sugerencia para entrega en GitHub
1. Crear repositorio nuevo.
2. Subir todo el contenido de esta carpeta.
3. Invitar como colaborador a `rotizs@miumg.edu.gt`.
4. En el README dejar claro que es un **MVP con avance funcional**.

## Autor
Proyecto académico preparado para entrega del curso de Inteligencia Artificial.
