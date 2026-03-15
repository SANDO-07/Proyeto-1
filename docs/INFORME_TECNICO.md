# Informe Técnico
## Sistema Predictivo de Incendios Forestales con LLM Multimodal
### Proyecto: Train-LLM - Guatemala Fire Detection & Prediction - MVP

## 1. Descripción del problema
Los incendios forestales representan una amenaza ambiental y social en Guatemala. Detectarlos tarde incrementa pérdidas ecológicas, económicas y humanas. Por ello, se propone un sistema capaz de analizar datos históricos y generar alertas de riesgo con apoyo de inteligencia artificial.

## 2. Objetivo general
Desarrollar un MVP que permita estimar el riesgo de incendio forestal y proyectar comportamiento futuro mediante análisis de datos, aprendizaje automático y una arquitectura preparada para un componente multimodal.

## 3. Objetivos específicos
1. Organizar una base de datos histórica de eventos de incendio.
2. Entrenar un modelo de clasificación para estimar riesgo.
3. Generar un forecast mensual para 2026.
4. Publicar una API para inferencia.
5. Crear un dashboard de visualización.
6. Dejar lista la integración futura con un modelo multimodal real.

## 4. Arquitectura del MVP
- **Capa de datos**: archivos CSV con estructura tipo FIRMS.
- **Capa analítica**: Python + pandas + matplotlib.
- **Capa predictiva**: Random Forest.
- **Capa temporal**: forecast mensual derivado de histórico 2014-2024.
- **Capa de servicio**: FastAPI.
- **Capa visual**: Streamlit.
- **Capa multimodal futura**: endpoint base para análisis de imagen.

## 5. Dataset utilizado
El repositorio incluye un dataset sintético con formato académico y reproducible:
- período: 2014-2024
- cobertura: departamentos priorizados de Guatemala
- variables: temperatura, humedad, viento, lluvia, brillo térmico, FRP, confianza y riesgo

> Para la versión final del proyecto se sustituye este dataset por datos reales de NASA FIRMS y/o imágenes satelitales.

## 6. Variables principales
- `temperature_c`
- `humidity_pct`
- `wind_kmh`
- `rain_last_7d_mm`
- `ndvi`
- `brightness_k`
- `frp_mw`
- `confidence_pct`
- `department`
- `fire_risk`

## 7. Modelo predictivo
Se entrenó un **Random Forest Classifier** para estimar la probabilidad de riesgo de incendio.

### Justificación
Este algoritmo es apropiado para un MVP porque:
- maneja relaciones no lineales,
- tolera bien variables mixtas,
- es interpretable mediante importancia de variables,
- y permite buenos resultados sin una fase compleja de ajuste.

## 8. Resultados obtenidos
Las métricas del modelo se guardan en `models/metrics.json`.
Además, el proyecto ya exporta:
- matriz de correlación,
- estacionalidad mensual,
- importancia de variables,
- forecast mensual 2026.

## 9. Componente multimodal
El título del proyecto menciona LLM multimodal. En este MVP se deja preparada la estructura para ello:
- endpoint `/predict/vision`
- lógica base para análisis de imagen
- documentación para futura integración con Qwen-VL, LLaVA o modelos similares

## 10. Limitaciones actuales
- el dataset incluido es sintético para fines de demostración del avance,
- el módulo visual actual es heurístico y no un VLM real,
- no se ha conectado aún la descarga automatizada de NASA FIRMS.

## 11. Próximos pasos
1. Descargar eventos reales de NASA FIRMS.
2. Etiquetar imágenes satelitales.
3. Ajustar modelo temporal más robusto.
4. Integrar modelo multimodal real.
5. Publicar versión final con datos reales y validación externa.

## 12. Conclusión
El repositorio entregado representa un avance funcional y presentable para el curso. Incluye estructura profesional, modelo predictivo, forecast, API, dashboard y documentación suficiente para revisión por GitHub.
