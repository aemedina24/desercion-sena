# Contexto del proyecto para Claude

Este archivo resume el proyecto para que cualquier sesión de Claude (de Andrés o de David) trabaje con el mismo contexto. Si cambia una decisión, actualízalo en un PR.

## Qué es

Proyecto académico de MLOps (Universidad de Medellín). Sistema de alerta temprana que estima el riesgo de deserción de aprendices del SENA en modalidad **virtual, presencial y mixta**, y entrega al instructor una API con la lista de aprendices ordenada por riesgo, los factores que lo explican (SHAP) y medidas sugeridas.

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
- **Traslados** se excluyen (no son deserción).
- **Columnas que nunca entran al modelo:** `estado_final` y `fecha_desercion` (fuga de datos); `sexo` solo va en `data/processed/auditoria.parquet` para auditar equidad; identificadores y fechas solo para agrupar y dividir.
- **Columnas descartadas:** `nombre_programa`, `barrio_residencia`, `municipio_nacimiento`, `tasa_abandono_previa`.
- **Validación:** entrenar con las cohortes antiguas y evaluar con la más reciente (`id_cohorte`); en Optuna, `GroupKFold` por `id_aprendiz`.
- **Sin Hydra:** configuración en YAML simple dentro de `config/`.
- **Ética:** Ley 1581 de 2012. No se usan datos sensibles (etnia, religión, orientación sexual, biométricos).

## Datos y pipeline

- `data/raw/aprendices.parquet`: 10.000 × 50, versionado con DVC (no está en Git; se comparte por fuera).
- Etapa `process` (`src/desercion/process.py`) → `data/processed/aprendices.parquet` (9.789 × 43) y `auditoria.parquet`.
- El pipeline se ejecuta con `uv run dvc repro`. Cada etapa nueva se declara en `dvc.yaml`.

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

## Problemas conocidos

- DVC "attempt to write a readonly database": `uv run dvc config --local core.site_cache_dir "$env:LOCALAPPDATA\iterative\dvc\site_cache"`.
- DVC detecta cambios falsos por saltos de línea CRLF: resuelto con `.gitattributes`. Comprobar con `git ls-files --eol <archivo>` (debe decir `w/lf`).
- **Regla:** si `dvc repro` o `dvc status` modifican `dvc.lock` sin que nadie haya cambiado código ni datos, no hacer commit; primero investigar la causa. Un `dvc.lock` con hashes equivocados se propaga a todo el equipo.

## Cómo debe ayudar Claude

Actuar como mentor senior de MLOps: explicar el concepto y el porqué antes del código; dar código funcional indicando archivo, comando para ejecutarlo y cómo verificar el resultado; ante un error, diagnosticar primero (qué significa, hipótesis, comando para comprobarla) y luego proponer la solución; distinguir entre "funciona" y "es buena práctica"; evitar sobreingeniería; nunca inventar resultados. Respetar el reparto de tareas de `docs/plan_trabajo.md` y las decisiones de este archivo.
