# Guía de desarrollo

Resume el proyecto, las decisiones tomadas y la forma de trabajar del equipo, para que cualquier persona que trabaje en el repositorio (o el asistente de código que use) parta del mismo contexto. Si cambia una decisión, se actualiza aquí en un PR.

## Qué es

Proyecto académico de MLOps (Universidad de Medellín). Sistema de alerta temprana que estima el riesgo de deserción de aprendices del SENA en modalidad **virtual, presencial y mixta**, y entrega al instructor una API con la lista de aprendices ordenada por riesgo, los factores que lo explican (aportes de los coeficientes de la logística, equivalentes a SHAP lineal) y medidas sugeridas.

- **Entrega:** 11 de noviembre de 2026.
- **Equipo:** Andrés (`aemedina24`, datos y modelo) · David (ingeniería y calidad).
- **Plan, tareas y responsables:** [plan de trabajo](plan_trabajo.md).

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
- **Modelo final:** regresión logística con umbral elegido con predicciones out-of-fold en train (recall ≥ 0.70 con la mayor precisión). LightGBM y la red neuronal no la superaban de forma significativa ([ADR 0002](decisiones/0002-eleccion-modelo.md)).
- **Equidad:** criterio principal igualdad de oportunidades, medida con la regla de los cuatro quintos (recall mínimo / máximo ≥ 0.80) y siempre con IC por bootstrap. `sexo` solo se audita. Umbral global; umbral por modalidad propuesto como decisión institucional para la siguiente versión (con la v2 cumple en el 85 % de las réplicas, pero redistribuye la detección) ([ADR 0003](decisiones/0003-umbral-y-equidad.md)).
- **Sin Hydra:** configuración en YAML simple dentro de `config/`.
- **Ética:** Ley 1581 de 2012. No se usan datos sensibles (etnia, religión, orientación sexual, biométricos).
- **API:** consulta por aprendiz y por cohorte con filtros (modalidad, área, nivel, solo alertas, límite); no hay `id_ficha` en los datos. Al arrancar carga los aprendices activos (`data/features/test.parquet` en la demo) y precalcula su riesgo ([ADR 0004](decisiones/0004-api-consulta-y-datos.md)).
- **Explicación:** aporte = coef × (valor − media de train), agrupado por variable original (`explain.py`). Es SHAP exacto para un modelo lineal; no se usa la librería `shap`.
- **Predicción para la API:** `prediccion.predecir_riesgo(aprendices, *, modelo, referencia, umbral)` no carga nada; la API carga modelo, umbral y referencia al arrancar. Factores con aporte ≥ 0.05; medidas solo con alerta, definidas en `config/recomendaciones/medidas.yaml` (cada variable del modelo tiene medida o está en `sin_medida`; nunca medidas por características personales).

## Datos y pipeline

- `data/raw/aprendices.parquet`: 10.000 × 50, versionado con DVC. No está en Git: se descarga del remoto de DagsHub con `uv run dvc pull` (credenciales en `.dvc/config.local`, ver el [README](../README.md)).
- Etapas de `dvc.yaml`, en orden:
  - `process` → `data/processed/aprendices.parquet` (9.475 × 43) y `auditoria.parquet`.
  - `validate` → `reports/validacion.json` (contrato de Pandera).
  - `features` → `data/features/train.parquet` y `test.parquet`.
  - `train` → `models/modelo.skops` (el modelo que usan `evaluate`, `explain` y la API) y `reports/metricas_modelo_base.json` (incluye el `umbral`). Registra el run en MLflow de DagsHub: requiere `.env`.
  - `evaluate` → `reports/evaluacion.json` (métricas por modalidad).
  - `explain` → `reports/importancia_variables.json` y `models/referencia.json` (aprendiz promedio de train; la API calcula los factores contra él).
  - `monitor` → `reports/drift/drift.json` y `drift.html`: PSI por variable de cada cohorte de test frente a train (> 0.20 = drift) y recall por cohorte con IC bootstrap ("ok" si llega a 0.70, "vigilar" si queda debajo pero el IC incluye 0.70, "revisar" si todo el IC queda debajo; ningún estado hace fallar el repro). Sin Evidently: no funciona con pandas 3 ([ADR 0005](decisiones/0005-monitoreo-sin-evidently.md)).
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
- Tras `dvc repro` en Windows, `dvc.lock` puede aparecer modificado solo porque DVC lo reescribe con CRLF. Comprobar con `git diff --ignore-cr-at-eol dvc.lock`:
  - **Sale vacío:** el cambio es solo de saltos de línea; descartarlo con `git restore dvc.lock`.
  - **Muestra cambios:** **no restaurar**. Son cambios reales (por ejemplo, cambió el código o los parámetros de una etapa); revisar el diff y, si corresponden a lo que se cambió, hacer commit de `dvc.lock`.
- Los archivos que escribe Python (JSON de `reports/`) deben usar `write_text(..., newline="\n")`: en Windows, `write_text` convierte `\n` en CRLF y `dvc.lock` queda con un hash que nadie más reproduce. Ojo: `tests/test_dvc_lock.py` solo revisa las dependencias de código (`src/`, `config/`), no estas salidas.
- **Colinealidad entre entregadas y aprobadas:** `porcentaje_actividades_entregadas`, `porcentaje_actividades_aprobadas` y `tasa_aprobacion_entregas` dicen casi lo mismo, y a una de las dos primeras le sale el signo invertido. Cuál de las dos depende de los datos: con la v1 era entregadas (factor de riesgo en 623 aprendices de test que entregaban 91 % frente a 75 %); con la v2 es aprobadas (277 aprendices que aprueban 92 % frente a 64 %). La que sale invertida se muestra como factor, pero va en `sin_medida` de `medidas.yaml`, y la otra es la que se prueba en `DIRECCIONES` de `tests/test_model_behavior.py`. Si se regeneran los datos hay que revisar ambas cosas. Con la v2 pasa lo mismo en el historial: por la colinealidad con `n_cursos_abandonados`, `tasa_finalizacion_previa` (+0.037) y `abandono_ultimo_curso` (−0.043) salen invertidas, y la medida "antecedentes de abandono" le llegaba a 40 alertas de aprendices que terminaron más cursos que el promedio; ambas pasaron a `sin_medida`. `test_factores_con_medida_van_en_la_direccion_esperada` (`tests/test_prediccion.py`) lo detecta: con el modelo real, falla si una variable con medida sale como factor en un aprendiz que está mejor que el promedio, según la dirección declarada en `DIRECCION_RIESGO` (`prediccion.py`). Además, `prediccion.quitar_contrarios` no muestra un factor cuyo valor está del lado bueno del promedio (con la v2, 40 de `tasa_finalizacion_previa` y 1 de `porcentaje_actividades_aprobadas`), y `test_solo_se_quitan_las_variables_que_el_modelo_invierte` verifica con el modelo real que lo que se quita son solo variables de `sin_medida`. Si se regeneran los datos y otra variable sale invertida, hay que agregarla a `DIRECCION_RIESGO`. La solución de fondo (agrupar los factores por tema) sigue pendiente.
- **Datos vacíos como factor:** el imputador agrega un indicador de faltante por variable numérica, y su aporte se suma al de la variable. `asistencia_porcentaje` es nula en todos los virtuales y su indicador sube el riesgo (+0.234), así que sale como factor en casi todos los virtuales con alerta. `prediccion.factores_con_dato` impide que un factor con dato vacío active una medida; `test_ninguna_medida_sale_de_un_dato_vacio` lo verifica con el modelo real. En la vista del instructor, el motivo de la lista es el primer factor con dato (antes, 103 virtuales tenían «Asistencia: sin dato» como motivo) y, en el detalle, la asistencia vacía dice «No aplica (sin sesiones presenciales)».
- `models/modelo.skops` no es idéntico byte a byte al reentrenar (skops nombra sus archivos internos con identificadores de memoria), aunque las predicciones sí lo son. Si `train` se vuelve a ejecutar, `dvc.lock` cambia sin cambio real: el modelo se compara por sus métricas, no por su hash. Para no reentrenar, hacer `uv run dvc pull` antes de `dvc repro`.

## Uso de asistentes de IA

El equipo usó Claude (Anthropic) como asistente de código y de revisión. Esta guía es el contexto que se le da al iniciar cada sesión, y estas son las instrucciones de cómo debe ayudar: actuar como mentor senior de MLOps: explicar el concepto y el porqué antes del código; dar código funcional indicando archivo, comando para ejecutarlo y cómo verificar el resultado; ante un error, diagnosticar primero (qué significa, hipótesis, comando para comprobarla) y luego proponer la solución; distinguir entre "funciona" y "es buena práctica"; evitar sobreingeniería; nunca inventar resultados. Respetar el reparto de tareas del plan de trabajo y las decisiones de esta guía. Todo cambio, venga de una persona o del asistente, entra por Pull Request con la revisión del otro integrante.
