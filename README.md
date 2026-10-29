# Alerta temprana de deserción · SENA

Sistema de machine learning que estima el riesgo de deserción de aprendices del SENA en formación **virtual, presencial y mixta**, y entrega al instructor una API con la lista de aprendices ordenada por riesgo, los factores que lo explican y medidas de acompañamiento sugeridas.

El proyecto está construido con prácticas de **MLOps**: entorno reproducible, datos versionados, experimentos registrados, pruebas automáticas y monitoreo del modelo.

**Equipo:** Andrés (datos y modelo) · David (ingeniería y calidad)

---

## Estado del proyecto

Detalle de tareas en [`docs/plan_trabajo.md`](docs/plan_trabajo.md).

- [x] **1. Definición del problema:** variable objetivo, métricas y criterios éticos
- [x] **2. Preparación del entorno:** repositorio, uv, Ruff y DVC
- [ ] **3. Carga y validación de datos:** limpieza, esquema con Pandera y versionado con DVC
- [ ] **4. Exploración y construcción de variables**
- [ ] **5. Entrenamiento y evaluación:** LightGBM, Optuna, MLflow, SHAP y métricas por modalidad
- [ ] **6. Pruebas y automatización:** pytest y GitHub Actions
- [ ] **7. Despliegue:** API con FastAPI y Docker
- [ ] **8. Monitoreo:** detección de drift con Evidently

---

## Problema

| | |
|---|---|
| **Tipo** | Clasificación binaria |
| **Variable objetivo** | `deserto`: 1 si el aprendiz desertó después de la fecha de corte, 0 si no |
| **Momento de la predicción** | Alrededor de la semana 4 de formación |
| **Métrica principal** | Recall de la clase "deserta", acompañado de F1 y ROC-AUC |
| **Exclusiones** | Los traslados no se consideran deserción y se excluyen |

**Criterios éticos:** no se usan datos sensibles (etnia, religión, orientación sexual, biométricos). El sexo se guarda aparte y solo se usa para auditar la equidad del modelo. Cada instructor consulta únicamente sus fichas.

---

## Herramientas

| Etapa | Herramientas |
|---|---|
| Entorno y calidad de código | VS Code, uv, Python 3.11, Ruff (PEP 8) |
| Versionado | Git y GitHub (código), DVC (datos), MLflow Model Registry (modelos) |
| Datos | pandas, Pandera |
| Modelado | scikit-learn, LightGBM, Optuna, SHAP, MLflow Tracking |
| Pruebas y CI | pytest, GitHub Actions |
| Despliegue y monitoreo | FastAPI, Docker, Evidently |

---

## Estructura

```
desercion-sena/
├── config/           # configuración: variables, modelo, tuning y medidas
├── data/             # datos (versionados con DVC, no con Git)
│   ├── raw/          # datos originales
│   ├── processed/    # datos limpios
│   └── final/        # variables listas para entrenar
├── docs/             # diccionario de datos, arquitectura, model card y plan
├── models/           # modelos entrenados
├── notebooks/        # solo exploración
├── reports/          # figuras y reportes de drift
├── src/desercion/    # código del proyecto (un módulo por etapa) y la API
├── tests/            # pruebas con pytest
├── dvc.yaml          # etapas del pipeline
├── pyproject.toml    # dependencias
└── uv.lock           # versiones exactas
```

---

## Instalación

Requisitos: Git, [uv](https://docs.astral.sh/uv/) y Python 3.11 (uv lo instala si no está).

```powershell
git clone https://github.com/aemedina24/desercion-sena.git
cd desercion-sena
uv sync
```

### Datos

El archivo de datos no se incluye en el repositorio: se versiona con DVC y se comparte por fuera de Git. Una vez recibido, se ubica en `data/raw/aprendices.parquet` y se verifica que coincide con la versión registrada:

```powershell
uv run dvc status
```

Debe responder *"Data and pipelines are up to date"*.

---

## Uso

```powershell
uv run dvc repro       # ejecuta el pipeline completo
uv run ruff check .    # revisa la calidad del código
uv run pytest          # ejecuta las pruebas
```

---

## Problemas comunes

**DVC: "attempt to write a readonly database" (Windows).** DVC intenta escribir en una carpeta del sistema sin permisos. Solución:

```powershell
uv run dvc config --local core.site_cache_dir "$env:LOCALAPPDATA\iterative\dvc\site_cache"
```

**La terminal muestra `(base)`.** Hay un entorno de Conda activo. Ejecute `conda deactivate` antes de trabajar en el proyecto.
