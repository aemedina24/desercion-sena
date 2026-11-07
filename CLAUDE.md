# Contexto del proyecto para Claude

Este archivo resume el proyecto para que cualquier sesión de Claude (de Andrés o de David) trabaje con el mismo contexto. Si cambia una decisión, actualízalo en un PR.

## Qué es

Proyecto académico de MLOps (Universidad de Medellín). Sistema de alerta temprana que estima el riesgo de deserción de aprendices del SENA en modalidad **virtual, presencial y mixta**, y entrega al instructor una API con la lista de aprendices ordenada por riesgo, los factores que lo explican (aportes de los coeficientes de la logística, equivalentes a SHAP lineal) y medidas sugeridas.

- **Entrega:** 11 de noviembre de 2026.
- **Equipo:** Andrés (`aemedina24`, datos y modelo) · David (ingeniería y calidad).
- **Plan, tareas y responsables:** `docs/plan_trabajo.md`.

## Entorno

- Windows, PowerShell y VS Code. Los comandos se dan en PowerShell.
- Python 3.11 con **uv**: `uv sync`, `uv add`, `uv run`. No usar Conda, pip suelto ni `requirements.txt`.
- No abrir la terminal como administrador. Si aparece `(base)`, ejecutar `conda deactivate`.

## Decisiones tomadas (no cambiar sin acordarlo)

- **Problema:** clasificación binaria. Etiqueta `deserto` (1 = desertó después de la fecha de corte).
- **Un solo modelo de seguimiento**, con corte alrededor de la semana 4. El modelo de ingreso queda como trabajo futuro.
- **Métrica principal:** recall de la clase "deserta"; además F1 y ROC-AUC, reportadas por modalidad.
- **Traslados y cancelaciones** se excluyen (no son la deserción que se predice).
- **Columnas que nunca entran al modelo:** `estado_final` y `fecha_desercion` (fuga de datos); `sexo` solo va en `data/processed/auditoria.parquet` para auditar equidad; identificadores y fechas solo para agrupar y dividir.
- **Columnas descartadas:** `nombre_programa`, `barrio_residencia`, `municipio_nacimiento`, `tasa_abandono_previa`.
- **Validación:** entrenar con las cohortes antiguas y evaluar con las dos más recientes, 2025-T3 y 2025-T4 (`COHORTES_TEST` en `features.py`), para tener más casos por modalidad. La validación cruzada usa `GroupKFold` por `id_cohorte`: hay una fila por aprendiz, así que agrupar por `id_aprendiz` no aportaba nada.
- **Modelo final:** regresión logística con umbral elegido con predicciones out-of-fold en train (recall ≥ 0.70 con la mayor precisión). LightGBM y la red neuronal no la superaban de forma significativa ([ADR 0002](docs/decisiones/0002-eleccion-modelo.md)).
- **Equidad:** criterio principal igualdad de oportunidades, medida con la regla de los cuatro quintos (recall mínimo / máximo ≥ 0.80) y siempre con IC por bootstrap. `sexo` solo se audita. Umbral global en la v1; umbral por modalidad como siguiente versión ([ADR 0003](docs/decisiones/0003-umbral-y-equidad.md)).
- **Sin Hydra:** configuración en YAML simple dentro de `config/`.
- **Ética:** Ley 1581 de 2012. No se usan datos sensibles (etnia, religión, orientación sexual, biométricos).
- **Explicación:** aporte = coef × (valor − media de train), agrupado por variable original (`explain.py`). Es SHAP exacto para un modelo lineal; no se usa la librería `shap`.

## Datos y pipeline

- `data/raw/aprendices.parquet`: 10.000 × 50, versionado con DVC. No está en Git: se descarga del remoto de DagsHub con `uv run dvc pull` (credenciales en `.dvc/config.local`, ver README).
- Etapas de `dvc.yaml`, en orden:
  - `process` → `data/processed/aprendices.parquet` (9.518 × 43) y `auditoria.parquet`.
  - `validate` → `reports/validacion.json` (contrato de Pandera).
  - `features` → `data/features/train.parquet` y `test.parquet`.
  - `train` → `models/modelo.skops` (el modelo que usan `evaluate`, `explain` y la API) y `reports/metricas_modelo_base.json` (incluye el `umbral`). Registra el run en MLflow de DagsHub: requiere `.env`.
  - `evaluate` → `reports/evaluacion.json` (métricas por modalidad).
  - `explain` → `reports/importancia_variables.json`.
- El pipeline se ejecuta con `uv run dvc repro`. Cada etapa nueva se declara en `dvc.yaml`. Después de `git pull`, ejecutar `uv run dvc pull` para tener los datos y el modelo de esa versión.

## Convenciones de código

- La lógica vive en `src/desercion/` (un módulo por etapa); los cuadernos importan desde ahí y no contienen la lógica principal.
- Funciones pequeñas y puras; solo `main()` lee y escribe archivos.
- PEP 8 con Ruff (`uv run ruff check .`); pruebas con `uv run pytest`.
- Saltos de línea LF (`.gitattributes`).
- Nunca subir datos, modelos, `.env` ni secretos a Git.

## Cuadernos

`01_eda`, `02_modelo_base`, `03_comparacion_modelos` (lee MLflow, no reentrena) y `04_evaluacion_y_equidad`. Se suben ejecutados con Restart → Run All.

## Flujo de trabajo en GitHub

1. Una tarea = un Issue; una rama por Issue desde `main` actualizado.
2. Pull Request con la plantilla `.github/pull_request_template.md`, revisor = el otro integrante.
3. Merge solo con aprobación y CI en verde; luego se borra la rama.
4. `main` está protegida (ruleset `proteger-main`): todo entra por PR, con 1 aprobación, el check "Ruff y pytest" en verde y la rama al día con `main`. Si GitHub indica que la rama está desactualizada: `git merge origin/main`, `uv run pytest -q` y `git push` (o el botón *Update branch*).

## Problemas conocidos

- DVC "attempt to write a readonly database": `uv run dvc config --local core.site_cache_dir "$env:LOCALAPPDATA\iterative\dvc\site_cache"`.
- DVC detecta cambios falsos por saltos de línea CRLF: resuelto con `.gitattributes`. Comprobar con `git ls-files --eol <archivo>` (debe decir `w/lf`).
- **Regla:** si `dvc repro` o `dvc status` modifican `dvc.lock` sin que nadie haya cambiado código ni datos, no hacer commit; primero investigar la causa. Un `dvc.lock` con hashes equivocados se propaga a todo el equipo.
- Tras `dvc repro` en Windows, `dvc.lock` puede aparecer modificado solo porque DVC lo reescribe con CRLF. Comprobar con `git diff --ignore-cr-at-eol dvc.lock` (debe salir vacío) y descartar con `git restore dvc.lock`.
- Los archivos que escribe Python (JSON de `reports/`) deben usar `write_text(..., newline="\n")`: en Windows, `write_text` convierte `\n` en CRLF y `dvc.lock` queda con un hash que nadie más reproduce. Ojo: `tests/test_dvc_lock.py` solo revisa las dependencias de código (`src/`, `config/`), no estas salidas.
- `models/modelo.skops` no es idéntico byte a byte al reentrenar (skops nombra sus archivos internos con identificadores de memoria), aunque las predicciones sí lo son. Si `train` se vuelve a ejecutar, `dvc.lock` cambia sin cambio real: el modelo se compara por sus métricas, no por su hash. Para no reentrenar, hacer `uv run dvc pull` antes de `dvc repro`.

## Cómo debe ayudar Claude

Actuar como mentor senior de MLOps: explicar el concepto y el porqué antes del código; dar código funcional indicando archivo, comando para ejecutarlo y cómo verificar el resultado; ante un error, diagnosticar primero (qué significa, hipótesis, comando para comprobarla) y luego proponer la solución; distinguir entre "funciona" y "es buena práctica"; evitar sobreingeniería; nunca inventar resultados. Respetar el reparto de tareas de `docs/plan_trabajo.md` y las decisiones de este archivo.
