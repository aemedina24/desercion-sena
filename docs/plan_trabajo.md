# Plan de trabajo

**Equipo:** Andrés (datos y modelo) · David (ingeniería y calidad)
**Entrega:** 11 de noviembre de 2026

Para marcar una tarea como hecha, cambia `- [ ]` por `- [x]` y haz commit. GitHub muestra las casillas marcadas.

**Regla para todas las tareas:** se trabaja en una rama propia, se abre un Pull Request y el otro lo revisa antes de unirlo a `main`.

---

## Día 1 · 28 oct · Fundaciones

- [x] Andrés · Crear el repositorio en GitHub e invitar a David
- [x] Andrés · Proyecto con uv, `.gitignore` y estructura de carpetas
- [x] Andrés · Instalar numpy, pandas, pyarrow, Ruff, pytest y DVC
- [x] Andrés · Versionar `data/raw/aprendices.parquet` con DVC
- [x] Andrés · Primer commit y push
- [ ] David · Clonar el repositorio, `uv sync` y poner el Parquet en `data/raw/`
- [ ] David · Verificar con `uv run dvc status` que los datos coinciden

## 29–30 oct · Limpieza y validación de datos

- [ ] Andrés · `process.py`: excluir traslados, separar auditoría y quitar columnas de fuga
- [ ] Andrés · Etapa `process` en `dvc.yaml` y ejecutarla con `uv run dvc repro`
- [ ] Juntos · Acordar el contrato: las 43 columnas de los datos procesados
- [ ] David · `validate.py`: esquema de Pandera de los datos procesados
- [ ] David · `tests/test_validate.py`: el esquema acepta datos buenos y rechaza datos malos
- [ ] David · `tests/test_process.py`: no quedan traslados ni columnas de fuga
- [ ] Andrés · `docs/datos.md`: diccionario de variables con su rol

## 31 oct–2 nov · Variables y automatización

- [ ] Andrés · Exploración rápida en `notebooks/01_eda.ipynb`
- [ ] Andrés · `features.py`: variables finales y lista en `config/features/seguimiento.yaml`
- [ ] Andrés · Etapa `features` en `dvc.yaml`
- [ ] David · `.github/workflows/ci.yaml`: Ruff y pytest en cada push y Pull Request
- [ ] David · Verificar el check verde en GitHub

## 3–4 nov · Entrenamiento

- [ ] Andrés · Instalar scikit-learn, LightGBM, Optuna y MLflow
- [ ] Andrés · División temporal: cohortes antiguas para entrenar, la más reciente para prueba
- [ ] Andrés · `train_model.py`: modelo base (regresión logística) registrado en MLflow
- [ ] Andrés · `tune.py`: LightGBM con Optuna y `GroupKFold` por aprendiz
- [ ] Andrés · Registrar el mejor modelo en el Model Registry como `desercion-seguimiento`
- [ ] David · `tests/test_model_behavior.py`: rendimiento mínimo e invariancia
- [ ] David · Prueba de fuga: ninguna columna del resultado llega al entrenamiento

## 5–6 nov · Evaluación y base de la API

- [ ] Andrés · `evaluate.py`: recall, F1 y AUC global y por modalidad
- [ ] Andrés · Métricas por subgrupo usando la tabla de auditoría
- [ ] Andrés · `explain.py`: factores principales de cada predicción con SHAP
- [ ] David · `api/schemas.py`: modelos de entrada y salida con Pydantic
- [ ] David · `api/main.py`: endpoint `/health`

## 7–8 nov · Recomendaciones, API y Docker

- [ ] Andrés · `config/recomendaciones/medidas.yaml`: tabla factor → medida
- [ ] Andrés · `recomendaciones.py`: convertir factores en medidas sugeridas
- [ ] Andrés · Función de predicción para la API: probabilidad, factores y medidas
- [ ] David · Endpoints `/aprendices/{id}/riesgo` y `/fichas/{id}/riesgo`
- [ ] David · `tests/test_api.py` con TestClient
- [ ] David · `Dockerfile` y comprobar que la API corre en el contenedor

## 9 nov · Monitoreo y documentación

- [ ] David · `monitor.py`: reporte de drift con Evidently en `reports/drift/`
- [ ] David · Completar el README
- [ ] Andrés · `docs/model_card.md`: métricas, límites y equidad del modelo

## 10–11 nov · Cierre

- [ ] Juntos · Ejecutar todo desde cero en un computador limpio (`git clone` → `uv sync` → `dvc repro`)
- [ ] Juntos · Actualizar la presentación con resultados reales
- [ ] Juntos · Ensayar la presentación con cronómetro (12 minutos)
- [ ] Juntos · Crear el tag de entrega `v1.0.0`
