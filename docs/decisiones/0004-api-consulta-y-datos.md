# ADR 004: Consulta por cohorte y datos de la API

- **Estado:** Aceptada
- **Fecha:** 2026-11-08
- **Decisores:** Andrés Medina, David Montilla

## Contexto

La API debe entregar al instructor la lista de aprendices ordenada por riesgo,
con los factores que lo explican y medidas sugeridas. Había dos preguntas
abiertas:

1. **Cómo agrupar a los aprendices.** El plan hablaba de `/fichas/{id}/riesgo`,
   pero los datos no tienen `id_ficha`. Las cohortes de test (2025-T3 y 2025-T4)
   tienen unos 770 aprendices cada una, de 16 programas y 3 modalidades: una
   lista así no le sirve a un instructor. Construir una "ficha" con cohorte +
   área + nivel + modalidad da 84 grupos de unos 16, pero sería un id inventado.
2. **De dónde salen los aprendices.** La API recibe un id y necesita las
   variables de ese aprendiz.

## Decisión 1: consulta por cohorte con filtros sobre columnas reales

- `GET /aprendices/{id_aprendiz}/riesgo`: riesgo de un aprendiz (404 si no está
  entre los activos).
- `GET /cohortes/{id_cohorte}/riesgo`: aprendices de la cohorte, ordenados de
  mayor a menor probabilidad, con filtros opcionales:
  - `modalidad`, `area_formacion`, `nivel_formacion`: solo los valores del
    contrato de datos (`validate.py`); otro valor responde 422.
  - `solo_alertas`: solo quienes superan el umbral.
  - `limite`: cuántos devolver (1–500, por defecto 50).
- La respuesta incluye `total_aprendices` y `total_alertas` después de filtrar y
  antes de aplicar el límite.
- El esquema `RiesgoFicha` pasa a ser `RiesgoCohorte`.
- Si algún día hay `id_ficha`, se agrega como un filtro más, sin cambiar el
  contrato.

## Decisión 2: aprendices activos y riesgo precalculado al arrancar

- Al arrancar (lifespan), la API carga el modelo (`models/modelo.skops`), el
  umbral (`reports/metricas_modelo_base.json`) y los **aprendices activos**.
- **En la demo, los aprendices activos son `data/features/test.parquet`**
  (1.544 aprendices de 2025-T3 y 2025-T4). En producción vendrían de la base del
  SENA; la carga está en un solo lugar para cambiarla allí.
- El riesgo de todos (probabilidad, alerta, factores y medidas) se calcula **una
  sola vez** con `predecir_riesgo(df)`. Los endpoints solo consultan esa tabla.
- `predecir_riesgo` debe devolver una fila por aprendiz con las columnas
  `id_aprendiz`, `probabilidad`, `alerta`, `factores` y `medidas`. Si no cumple
  ese contrato, la API arranca igual y `/health` responde 503 con el motivo.
- Los factores se calculan contra `models/referencia.json` (medias de train), que
  escribe la etapa `explain`: la API no carga los datos de entrenamiento.

## Consecuencias

- **La API no tiene autenticación.** En la demo, cualquiera que llegue a la API
  puede consultar cualquier aprendiz y cualquier cohorte. Antes el README decía
  que cada instructor solo vería sus fichas; eso ya no se cumple. En producción
  habría que autenticar a cada instructor y restringir sus consultas a sus
  grupos (por ejemplo, con un filtro obligatorio según su asignación), además de
  servir la API solo por HTTPS. Es aceptable en la demo solo porque los datos
  son sintéticos.
- **La etiqueta nunca sale de la API.** `test.parquet` incluye `deserto`; la API
  solo expone las columnas de filtro y las de riesgo. Una prueba lo verifica en
  todas las respuestas.
- **Son aprendices con resultado conocido.** En la demo se puede mostrar a quién
  alertó el modelo y si desertó, pero no es una predicción a futuro. Los datos
  son sintéticos.
- **La API reproduce el pipeline.** Con el modelo de DVC, las dos cohortes suman
  702 alertas, las mismas de `reports/evaluacion.json`.
- **La imagen Docker necesita `uv run dvc pull` antes del build**: modelo, umbral,
  referencia y aprendices activos.
- **`/health` distingue el motivo:** `sin_modelo` (no cargó el modelo o el
  umbral) o `sin_datos` (el modelo cargó, pero no los aprendices o su riesgo).
- **Factores y medidas:** la API usa `prediccion.predecir_riesgo` con el modelo,
  el umbral y `models/referencia.json` cargados al arrancar. Si falta la
  referencia o es de otro modelo, `/health` responde `sin_datos`.
- **Equidad:** el umbral es global (ADR 0003); la brecha por modalidad sigue
  siendo una limitación conocida.
