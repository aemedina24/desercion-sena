# Plan de trabajo

**Equipo:** Andrés (datos y modelo) · David (ingeniería y calidad)
**Entrega:** 11 de noviembre de 2026 (etiqueta `v1.0.0`) · **Sustentación:** 25 de noviembre de 2026

Cada tarea tiene un **autor** (A) y un **revisor** (R). El autor la hace en su propia rama y abre un Pull Request; el revisor la prueba y la aprueba antes del merge. Para marcar una tarea, cambia `- [ ]` por `- [x]`.

---

## ✅ Hecho

- [x] A: Andrés · Repositorio, uv, `.gitignore`, estructura y DVC
- [x] A: Andrés · Versionar `data/raw/aprendices.parquet` con DVC
- [x] A: Andrés · `process.py` y etapa `process` en `dvc.yaml` (PR #1)
- [x] A: Andrés · `.gitattributes` para saltos de línea LF (PR #2)
- [x] A: David · Clonar el repositorio y reproducir `dvc repro` con los mismos resultados

## 29–30 oct · Validación de datos y diccionario

- [x] A: David · Limpiar su carpeta tras el PR #2 y comprobar `dvc status`
- [x] A: Juntos · Acordar el contrato: las 43 columnas de `data/processed/aprendices.parquet`
- [x] A: David · R: Andrés · `validate.py`: esquema de Pandera de los datos procesados
- [x] A: David · R: Andrés · `tests/test_validate.py`: acepta datos buenos y rechaza datos malos
- [x] A: David · R: Andrés · `tests/test_process.py`: sin traslados ni columnas de fuga
- [x] A: Andrés · R: David · `docs/datos.md`: diccionario de variables con su rol (PR #6)

## 31 oct–2 nov · Exploración, variables y CI

- [x] A: Andrés · R: David · `notebooks/01_eda.ipynb`: exploración y conclusiones
- [x] A: Andrés · R: David · `features.py`, `config/features/seguimiento.yaml` y etapa `features`
- [x] A: David · R: Andrés · `tests/test_features.py`: sin fuga temporal en las variables
- [x] A: David · R: Andrés · `.github/workflows/ci.yaml`: Ruff y pytest en cada push y PR
- [x] A: David · Regla de protección de `main`: exigir 1 aprobación y CI en verde (ruleset `proteger-main`, configurado por Andrés)

## 3–4 nov · Entrenamiento y experimentos

- [x] A: Andrés · R: David · División temporal por cohorte y `train_model.py` (modelo base en MLflow)
- [x] A: Andrés · R: David · `notebooks/02_modelo_base.ipynb` (PR #18)
- [x] A: Andrés · R: David · `tune.py`: LightGBM con Optuna y `GroupKFold` (experimento; el modelo final es la logística con umbral out-of-fold, ADR 0002, PR #17)
- [x] A: David · R: Andrés · `tests/test_model_behavior.py`: rendimiento mínimo, invariancia y fuga

## 5–6 nov · Evaluación, equidad y base de la API

- [x] A: Andrés · R: David · `evaluate.py` y `explain.py` (métricas por modalidad y coeficientes)
- [x] A: Andrés · R: David · `notebooks/03_comparacion_modelos.ipynb` (lee los experimentos de MLflow)
- [x] A: David · R: Andrés · `notebooks/04_evaluacion_y_equidad.ipynb` (subgrupos con la tabla de auditoría)
- [x] A: David · R: Andrés · `api/schemas.py` y endpoint `/health`

## 7–8 nov · Recomendaciones, API y Docker

- [x] A: Andrés · R: David · `medidas.yaml`, `recomendaciones.py` y función de predicción para la API (PR #28)
- [x] A: David · R: Andrés · Endpoints `/aprendices/{id}/riesgo` y `/cohortes/{id}/riesgo` con filtros (ADR 0004, PR #26 y #27)
- [x] A: David · R: Andrés · `tests/test_api.py`
- [x] A: Andrés · R: David · `Dockerfile` y `.dockerignore` (imagen de la API)
- [x] A: David · R: Andrés · Conectar `predecir_riesgo` en `crear_app` (factores y medidas en la API)

## 9 nov · Monitoreo y documentación

- [x] A: David · R: Andrés · `monitor.py`: drift con PSI y recall por cohorte (sin Evidently: no es compatible con pandas 3, [ADR 0005](decisiones/0005-monitoreo-sin-evidently.md))
- [x] A: Andrés · R: David · `docs/model_card.md` (equidad, explicación, monitoreo y limitaciones)
- [x] A: Andrés · R: David · Completar el README y escribir `docs/index.md`, `arquitectura.md`, `api.md` y el ADR 0001

## 10–11 nov · Cierre

- [ ] A: Juntos · Ejecutar todo desde cero: `git clone` → `uv sync` → `dvc repro` → `pytest`
- [x] A: Juntos · Ejecutar los 4 cuadernos con **Restart → Run All** y subirlos con resultados
- [x] A: Juntos · Actualizar la presentación con resultados reales y ensayarla (19 diapositivas)
- [x] A: Juntos · Crear el tag de entrega `v1.0.0`

---

## Después de la entrega · Datos v2 (12–25 nov)

Trabajo adicional a la entrega. Los datos de la v1 se recibieron generados con ChatGPT y sin código, así que no se podía explicar cómo se asignó la deserción. El 19 de noviembre decidimos adoptar un generador propio (v2). La versión entregada sigue intacta en la etiqueta `v1.0.0`.

### 12–15 nov · Generador propio

- [x] A: Andrés · R: David · `generar_datos.py`, `config/data/generador.yaml` y etapa `generate` (reemplaza a `data/raw/aprendices.parquet.dvc`); `data/raw/verdad.parquet` con la probabilidad real de cada aprendiz (PR #33)
- [x] A: Andrés · R: David · Ejecución oficial del pipeline con la v2 y `dvc.lock` (PR #33)

### 16–18 nov · Plan de acción y calibración

- [x] A: Andrés · R: David · Plan de acción para el instructor por cada medida, API 0.4.0 ([ADR 0006](decisiones/0006-plan-de-accion.md), PR #34)
- [x] A: Andrés · R: David · Correcciones de la revisión: variables con signo invertido sin medida, prueba de dirección y factores con dato vacío sin medida (PR #34)
- [x] A: Andrés · R: David · Etapa `calibrate`: la probabilidad está inflada y se corrige sin reentrenar; `verdad.parquet` como dependencia (PR #35)

### 19 nov · Decisión: adoptar la v2

- [x] A: Juntos · Decidir entre la v1 y la v2: se adopta la v2
- [x] A: Andrés · R: David · Cifras v2 en README, model card, `datos.md`, `api.md`, `arquitectura.md`, ADR 0002 y 0004; cuadernos 01 a 03 (PR #33)

### 20–21 nov · Equidad, monitoreo y experimentos

- [x] A: David · R: Andrés · Equidad con la v2: cuaderno 04, ADR 0003 y model card; la brecha de presencial viene sobre todo de los datos (PR #37)
- [x] A: David · R: Andrés · Monitoreo: estado ok / vigilar / revisar según el IC del recall ([ADR 0005](decisiones/0005-monitoreo-sin-evidently.md), PR #36)
- [x] A: Andrés · `tune`, `comparar_modelos` y cuaderno 03 registrados en el MLflow de DagsHub con la v2
- [x] A: Andrés · Actualizar este plan de trabajo

### 22–25 nov · Cierre y sustentación

- [x] A: Andrés · R: David · Fusionar `datos-v2` en `main` (PR #33)
- [x] A: Andrés · R: David · Notas de la revisión del #33 y prueba desde cero en Linux: mismos datos y métricas (PR #38)
- [ ] A: Juntos · Ejecutar todo desde cero con la v2 en Windows: `git clone` → `uv sync` → `dvc repro` → `pytest`
- [ ] A: Andrés · R: David · Valor del aprendiz y promedio junto a cada factor, API 0.5.0
- [x] A: Andrés · R: David · Vista del instructor servida por la API (`/instructor` y `/cohortes`) (PR #40)
- [ ] A: Andrés · R: David · Notas de David en el #40: el motivo es el primer factor con dato y no se muestran factores del lado bueno del promedio
- [ ] A: Juntos · Actualizar la presentación con la v2 (techo conocido, plan de acción, calibración, equidad y monitoreo) y ensayarla
- [ ] A: Juntos · Crear la etiqueta `v2.0.0` después de fusionar
- [ ] A: Juntos · Sustentación (25 nov)

### Para después (no entran en la sustentación)

- [ ] Umbral por modalidad: decisión institucional ([ADR 0003](decisiones/0003-umbral-y-equidad.md))
- [ ] Calibración: aplicar la corrección en la API o mostrar bandas; probar Platt o isotónica con out-of-fold
- [ ] Plan de acción: validar los guiones con instructores y Bienestar
- [ ] Sugerencias de revisión pendientes: frase de inactividad "noté poca actividad", tamaño de `/cohortes` con planes, redondear `calibracion.json`
