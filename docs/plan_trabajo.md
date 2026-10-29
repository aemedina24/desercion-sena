# Plan de trabajo

**Equipo:** Andrés (datos y modelo) · David (ingeniería y calidad)
**Entrega:** 11 de noviembre de 2026

Cada tarea tiene un **autor** (A) y un **revisor** (R). El autor la hace en su propia rama y abre un Pull Request; el revisor la prueba y la aprueba antes del merge. Para marcar una tarea, cambia `- [ ]` por `- [x]`.

---

## ✅ Hecho

- [x] A: Andrés · Repositorio, uv, `.gitignore`, estructura y DVC
- [x] A: Andrés · Versionar `data/raw/aprendices.parquet` con DVC
- [x] A: Andrés · `process.py` y etapa `process` en `dvc.yaml` (PR #1)
- [x] A: Andrés · `.gitattributes` para saltos de línea LF (PR #2)
- [x] A: David · Clonar el repositorio y reproducir `dvc repro` con los mismos resultados

## 29–30 oct · Validación de datos y diccionario

- [ ] A: David · Limpiar su carpeta tras el PR #2 y comprobar `dvc status`
- [ ] A: Juntos · Acordar el contrato: las 43 columnas de `data/processed/aprendices.parquet`
- [ ] A: David · R: Andrés · `validate.py`: esquema de Pandera de los datos procesados
- [ ] A: David · R: Andrés · `tests/test_validate.py`: acepta datos buenos y rechaza datos malos
- [ ] A: David · R: Andrés · `tests/test_process.py`: sin traslados ni columnas de fuga
- [ ] A: Andrés · R: David · `docs/datos.md`: diccionario de variables con su rol

## 31 oct–2 nov · Exploración, variables y CI

- [ ] A: Andrés · R: David · `notebooks/01_eda.ipynb`: exploración y conclusiones
- [ ] A: Andrés · R: David · `features.py`, `config/features/seguimiento.yaml` y etapa `features`
- [ ] A: David · R: Andrés · `tests/test_features.py`: sin fuga temporal en las variables
- [ ] A: David · R: Andrés · `.github/workflows/ci.yaml`: Ruff y pytest en cada push y PR
- [ ] A: David · Regla de protección de `main`: exigir 1 aprobación y CI en verde

## 3–4 nov · Entrenamiento y experimentos

- [ ] A: Andrés · R: David · División temporal por cohorte y `train_model.py` (modelo base en MLflow)
- [ ] A: Andrés · R: David · `notebooks/02_modelo_base.ipynb`
- [ ] A: Andrés · R: David · `tune.py`: LightGBM con Optuna y `GroupKFold`; registrar `desercion-seguimiento`
- [ ] A: David · R: Andrés · `tests/test_model_behavior.py`: rendimiento mínimo, invariancia y fuga

## 5–6 nov · Evaluación, equidad y base de la API

- [ ] A: Andrés · R: David · `evaluate.py` y `explain.py` (métricas por modalidad y SHAP)
- [ ] A: Andrés · R: David · `notebooks/03_comparacion_modelos.ipynb` (lee los experimentos de MLflow)
- [ ] A: David · R: Andrés · `notebooks/04_evaluacion_y_equidad.ipynb` (subgrupos con la tabla de auditoría)
- [ ] A: David · R: Andrés · `api/schemas.py` y endpoint `/health`

## 7–8 nov · Recomendaciones, API y Docker

- [ ] A: Andrés · R: David · `medidas.yaml`, `recomendaciones.py` y función de predicción para la API
- [ ] A: David · R: Andrés · Endpoints `/aprendices/{id}/riesgo` y `/fichas/{id}/riesgo`
- [ ] A: David · R: Andrés · `tests/test_api.py` y `Dockerfile`

## 9 nov · Monitoreo y documentación

- [ ] A: David · R: Andrés · `monitor.py`: reporte de drift con Evidently
- [ ] A: Andrés · R: David · `docs/model_card.md`
- [ ] A: Juntos · Completar el README

## 10–11 nov · Cierre

- [ ] A: Juntos · Ejecutar todo desde cero: `git clone` → `uv sync` → `dvc repro` → `pytest`
- [ ] A: Juntos · Ejecutar los 4 cuadernos con **Restart → Run All** y subirlos con resultados
- [ ] A: Juntos · Actualizar la presentación con resultados reales y ensayarla (12 min)
- [ ] A: Juntos · Crear el tag de entrega `v1.0.0`
