# Alerta temprana de deserción · SENA

Sistema de machine learning que estima el riesgo de deserción de aprendices del SENA en formación **virtual, presencial y mixta**, y entrega al instructor una API con la lista de aprendices ordenada por riesgo, los factores que lo explican y medidas de acompañamiento sugeridas.

El proyecto está construido con prácticas de **MLOps**: entorno reproducible, datos versionados, experimentos registrados, pruebas automáticas y monitoreo del modelo.

**Equipo:** Andrés (datos y modelo) · David (ingeniería y calidad)

---

## Estado del proyecto

Detalle de tareas en [`docs/plan_trabajo.md`](docs/plan_trabajo.md).

- [x] **1. Definición del problema:** variable objetivo, métricas y criterios éticos
- [x] **2. Preparación del entorno:** repositorio, uv, Ruff y DVC
- [x] **3. Carga y validación de datos:** limpieza, contrato con Pandera y versionado con DVC
- [x] **4. Exploración y construcción de variables**
- [x] **5. Entrenamiento y evaluación:** regresión logística comparada con LightGBM (Optuna) y una red neuronal, experimentos en MLflow, métricas por modalidad y auditoría de equidad
- [x] **6. Pruebas y automatización:** pytest y GitHub Actions
- [x] **7. Despliegue:** API con FastAPI y Docker, con factores y medidas por aprendiz
- [x] **8. Monitoreo:** drift con PSI y recall por cohorte

---

## Resultados

Regresión logística con umbral 0.461, evaluada en las dos cohortes más recientes (2025-T3 y 2025-T4, 1.544 aprendices), que el modelo nunca vio:

| Métrica | Valor |
|---|---|
| Recall (desertores detectados) | **0.686**: 338 de 493 |
| Precisión | 0.481 (al azar, 0.319) |
| ROC-AUC · PR-AUC | 0.735 · 0.576 |
| Recall por modalidad | mixta 0.756 · virtual 0.719 · presencial 0.525 |

- Tres modelos distintos (logística, LightGBM y red neuronal) llegan al mismo techo de PR-AUC (0.56 a 0.58); se eligió la logística por ser igual de buena, más simple y explicable ([ADR 0002](docs/decisiones/0002-eleccion-modelo.md)).
- Por sexo el modelo cumple la regla de los cuatro quintos (0.96); por modalidad no (0.69), por la brecha en presencial ([ADR 0003](docs/decisiones/0003-umbral-y-equidad.md)).
- El monitoreo no encuentra drift en las cohortes de prueba (PSI máximo 0.02).

Detalle, limitaciones y uso previsto en la [model card](docs/model_card.md).

---

## Problema

| | |
|---|---|
| **Tipo** | Clasificación binaria |
| **Variable objetivo** | `deserto`: 1 si el aprendiz desertó después de la fecha de corte, 0 si no |
| **Momento de la predicción** | Alrededor de la semana 4 de formación |
| **Métrica principal** | Recall de la clase "deserta", acompañado de F1 y ROC-AUC |
| **Exclusiones** | Los traslados y las cancelaciones no se consideran deserción y se excluyen |

**Criterios éticos:** no se usan datos sensibles (etnia, religión, orientación sexual, biométricos). El sexo se guarda aparte y solo se usa para auditar la equidad del modelo. La API no tiene un id de ficha (no existe en los datos): el instructor consulta su cohorte y la acota con filtros por modalidad, área y nivel. La demo no tiene autenticación: cualquiera puede consultar cualquier cohorte; en producción cada instructor debería ver solo sus grupos (ADR 0004).

---

## Herramientas

| Etapa | Herramientas |
|---|---|
| Entorno y calidad de código | VS Code, uv, Python 3.11, Ruff (PEP 8) |
| Versionado | Git y GitHub (código), DVC con remoto en DagsHub (datos y modelo) |
| Datos | pandas, Pandera |
| Modelado | scikit-learn (modelo final), LightGBM y Optuna (comparación), skops (guardar el modelo), MLflow Tracking en DagsHub (experimentos) |
| Explicación | Aportes por coeficientes de la logística (equivalentes a SHAP en un modelo lineal, sin la librería `shap`) |
| Pruebas y CI | pytest, GitHub Actions |
| Despliegue | FastAPI, Pydantic, uvicorn, Docker |
| Monitoreo | PSI y pruebas estadísticas con scipy (Evidently no es compatible con pandas 3: [ADR 0005](docs/decisiones/0005-monitoreo-sin-evidently.md)) |

---

## Estructura

```
desercion-sena/
├── config/           # configuración: datos, variables, modelos, tuning y medidas
├── data/             # datos (versionados con DVC, no con Git)
│   ├── raw/          # datos originales
│   ├── processed/    # datos limpios y tabla de auditoría
│   └── features/     # train y test listos para el modelo
├── docs/             # documentación y decisiones (ADR)
├── models/           # modelo y referencia para los factores (DVC)
├── notebooks/        # cuadernos que muestran los resultados
├── reports/          # métricas, importancia de variables y monitoreo (drift/)
├── src/desercion/    # código del proyecto (un módulo por etapa) y la API (api/)
├── tests/            # pruebas con pytest
├── Dockerfile        # imagen de la API
├── dvc.yaml          # etapas del pipeline
├── pyproject.toml    # dependencias
└── uv.lock           # versiones exactas
```

---

## Cuadernos

Los cuadernos muestran y explican los resultados; la lógica vive en `src/desercion/` y se importa desde ellos.

| Cuaderno | Contenido |
|---|---|
| `notebooks/01_eda.ipynb` | Exploración de los datos y conclusiones para construir las variables |
| `notebooks/02_modelo_base.ipynb` | Modelo base: regresión logística, elección del umbral y métricas en test |
| `notebooks/03_comparacion_modelos.ipynb` | Comparación en los mismos folds, leída de MLflow: logística, LightGBM tuneado con Optuna y red neuronal |
| `notebooks/04_evaluacion_y_equidad.ipynb` | Métricas por modalidad, equidad por modalidad y sexo con intervalos bootstrap, y efecto de un umbral por modalidad |

---

## Documentación

| Documento | Contenido |
|---|---|
| [Arquitectura](docs/arquitectura.md) | Pipeline, dónde vive cada artefacto, API, calidad y monitoreo |
| [API](docs/api.md) | Endpoints, parámetros, ejemplos de respuesta y errores |
| [Model card](docs/model_card.md) | Uso previsto, rendimiento, equidad y limitaciones del modelo |
| [Datos](docs/datos.md) | Diccionario de variables, roles y reglas de limpieza |
| [Decisiones (ADR)](docs/decisiones/) | uv, elección del modelo, umbral y equidad, diseño de la API, monitoreo |
| [Plan de trabajo](docs/plan_trabajo.md) | Tareas con autor y revisor |

---

## Cómo trabajamos

- Cada tarea es un Issue con autor y revisor ([plan de trabajo](docs/plan_trabajo.md)).
- Una rama por tarea; nadie trabaja directamente en `main`.
- Todo cambio entra por Pull Request, con la aprobación del otro integrante y la CI en verde.

---

## Instalación

Requisitos: Git, [uv](https://docs.astral.sh/uv/) y Python 3.11 (uv lo instala si no está).

```powershell
git clone https://github.com/aemedina24/desercion-sena.git
cd desercion-sena
uv sync
```

### Datos y modelo (DVC en DagsHub)

Los datos y el modelo no están en Git: se versionan con DVC y se guardan en el remoto de DagsHub. Requiere acceso al repositorio en DagsHub y un token personal (DagsHub → *Your settings* → *Tokens*).

Configure sus credenciales una sola vez. Quedan en `.dvc/config.local`, que no se sube a Git. El token se escribe oculto para que no quede en pantalla ni en el historial de la terminal:

```powershell
uv run dvc remote modify origin --local auth basic
uv run dvc remote modify origin --local user SU_USUARIO_DAGSHUB
$t = Read-Host "Token DagsHub" -AsSecureString
uv run dvc remote modify origin --local password ([System.Net.NetworkCredential]::new('', $t).Password)
Remove-Variable t
```

Descargue los datos y el modelo, y verifique:

```powershell
uv run dvc pull
uv run dvc status
```

`dvc status` debe responder *"Data and pipelines are up to date"*. Repita `uv run dvc pull` después de cada `git pull`.

### Credenciales de MLflow (DagsHub)

La etapa `train` registra cada experimento en el MLflow de DagsHub. Requiere acceso al repositorio en DagsHub y un token personal (DagsHub → *Your settings* → *Tokens*). Copie la plantilla y complete sus datos:

```powershell
Copy-Item .env.example .env
```

Ábralo con `notepad .env` y pegue ahí el token; nunca lo escriba en la terminal. `.env` está en `.gitignore`: nunca se sube a Git. Cada integrante usa su propio token.

---


## Uso

```powershell
uv run dvc repro       # ejecuta el pipeline completo
uv run ruff check .    # revisa la calidad del código
uv run pytest          # ejecuta las pruebas
```

### API con Docker

La imagen copia el modelo y los datos que versiona DVC, así que primero hay que descargarlos. Requiere Docker Desktop.

```powershell
uv run dvc pull
docker build -t desercion-api .
docker run --rm -d -p 8000:8000 --name api desercion-api
```

Espere unos segundos a que cargue el modelo. Compruebe que responde en <http://127.0.0.1:8000/health> (`"estado": "ok"`) y explore los endpoints en <http://127.0.0.1:8000/docs>; la [documentación de la API](docs/api.md) explica cada uno. Para detenerla: `docker stop api`. Sin Docker, la misma API se levanta con `uv run uvicorn desercion.api.main:app --reload`.

---

## Problemas comunes

**DVC: "attempt to write a readonly database" (Windows).** DVC intenta escribir en una carpeta del sistema sin permisos. Solución:

```powershell
uv run dvc config --local core.site_cache_dir "$env:LOCALAPPDATA\iterative\dvc\site_cache"
```

**`dvc pull` deja `data/raw/aprendices.parquet` fuera de la carpeta.** El archivo sigue en la caché local de DVC. Restáurelo y revise el estado antes de ejecutar el pipeline:

```powershell
uv run dvc checkout data/raw/aprendices.parquet.dvc
uv run dvc status
```

Si `dvc status` muestra otras etapas como cambiadas porque faltan sus salidas, use `uv run dvc checkout <etapa>` en lugar de `dvc repro`, para no volver a ejecutarlas.

**`docker run`: "port is already allocated" o "Empty reply from server".** Otro contenedor o programa usa el puerto 8000 (véalo con `docker ps`), o la API todavía está arrancando: espere unos segundos y repita la consulta.

**La terminal muestra `(base)`.** Hay un entorno de Conda activo. Ejecute `conda deactivate` antes de trabajar en el proyecto.

**DVC vuelve a ejecutar etapas sin cambios o modifica `dvc.lock` (Windows).** Git convirtió los saltos de línea a CRLF y cambió el hash de los archivos. El repositorio incluye `.gitattributes` para evitarlo; si ocurre en una copia anterior:

```powershell
git rm -r --cached -q .
git reset --hard
```
