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
| **Exclusiones** | Los traslados y las cancelaciones no se consideran deserción y se excluyen |

**Criterios éticos:** no se usan datos sensibles (etnia, religión, orientación sexual, biométricos). El sexo se guarda aparte y solo se usa para auditar la equidad del modelo. La API no tiene un id de ficha (no existe en los datos): el instructor consulta su cohorte y la acota con filtros por modalidad, área y nivel. La demo no tiene autenticación: cualquiera puede consultar cualquier cohorte; en producción cada instructor debería ver solo sus grupos (ADR 0004).

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

## Cuadernos

Los cuadernos muestran y explican los resultados; la lógica vive en `src/desercion/` y se importa desde ellos.

| Cuaderno | Contenido |
|---|---|
| `notebooks/01_eda.ipynb` | Exploración de los datos y conclusiones para construir las variables |
| `notebooks/02_modelo_base.ipynb` | Primer experimento: regresión logística |
| `notebooks/03_comparacion_modelos.ipynb` | Comparación de experimentos registrados en MLflow (modelo base, LightGBM y Optuna) |
| `notebooks/04_evaluacion_y_equidad.ipynb` | Métricas por modalidad y subgrupo, y factores de riesgo con SHAP |

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

Compruebe que responde en <http://127.0.0.1:8000/health> (`"estado": "ok"`) y explore los endpoints en <http://127.0.0.1:8000/docs>. Para detenerla: `docker stop api`. Sin Docker, la misma API se levanta con `uv run uvicorn desercion.api.main:app --reload`.

---

## Problemas comunes

**DVC: "attempt to write a readonly database" (Windows).** DVC intenta escribir en una carpeta del sistema sin permisos. Solución:

```powershell
uv run dvc config --local core.site_cache_dir "$env:LOCALAPPDATA\iterative\dvc\site_cache"
```

**La terminal muestra `(base)`.** Hay un entorno de Conda activo. Ejecute `conda deactivate` antes de trabajar en el proyecto.

**DVC vuelve a ejecutar etapas sin cambios o modifica `dvc.lock` (Windows).** Git convirtió los saltos de línea a CRLF y cambió el hash de los archivos. El repositorio incluye `.gitattributes` para evitarlo; si ocurre en una copia anterior:

```powershell
git rm -r --cached -q .
git reset --hard
```
