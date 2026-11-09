# ADR 005: Monitoreo de drift con PSI propio, sin Evidently

- **Estado:** Aceptada
- **Fecha:** 2026-11-08
- **Decisores:** Andrés Medina, David Montilla

## Contexto

El plan de trabajo decía "`monitor.py`: reporte de drift con Evidently". Al
probarlo:

- `uv add evidently` resuelve la versión 0.7.23, la más reciente publicada.
- Al importarlo falla: `ImportError: cannot import name 'infer_dtype_from_object'
  from 'pandas.core.dtypes.common'`. Evidently usa una función interna que pandas 3
  eliminó, y el proyecto usa pandas 3.1.0. Andrés reprodujo el error.

Alternativas descartadas:

- **Bajar pandas a 2.x:** cambia una dependencia de todo el pipeline; los parquet
  y el modelo podrían cambiar de hash y habría que reentrenar a tres días de la
  entrega.
- **Parchear pandas para que Evidently importe:** frágil y difícil de explicar.

## Decisión

`src/desercion/monitor.py` calcula el drift con `scipy` (dependencia directa,
antes solo llegaba a través de scikit-learn; no cambia versiones) y es la etapa
`monitor` de `dvc.yaml`. Compara cada cohorte de test con train:

1. **PSI por variable como criterio de alerta.** PSI > 0.20 = drift; 0.10–0.20 =
   moderado. Una sola escala para todas las variables:
   - numéricas: 10 bins por cuantiles de train (los valores fuera de rango caen
     en el primer o el último bin);
   - discretas (numéricas o booleanas con 10 valores o menos) y categóricas:
     cada valor es un bin;
   - los nulos son un bin propio (si una fuente deja de reportar, se nota);
   - proporción mínima de 1e-4 para los bins vacíos (evita log(0));
   - las categorías que no existían en train se marcan aparte.
2. **KS y chi-cuadrado solo informativos.** Con 7.974 aprendices de train contra
   unos 770 por cohorte y 39 variables, dan p-valores minúsculos por cambios
   triviales; no deciden la alerta.
3. **Recall por cohorte con IC bootstrap** (`equidad.bootstrap_recall`, como en
   el cuaderno 04). Si queda por debajo de 0.70 el reporte marca "revisar", pero
   `dvc repro` no falla: es una señal para el equipo, no un error.
4. **Fuera del drift:** `sexo` (atributo protegido, solo se audita) y `deserto`
   (la etiqueta). Se monitorean solo las 39 variables del modelo.

Salidas, en Git (`cache: false`) y escritas con LF:
`reports/drift/drift.json` y `reports/drift/drift.html` (las mismas tablas, para
la demo). El reporte no lleva fecha: los mismos datos dan el mismo hash.

## Resultado con el modelo de DVC

| Cohorte | Drift (PSI) | Recall (IC 95 %) | Estado |
|---|---|---|---|
| 2025-T3 | estable: PSI máximo 0.022 | 0.668 (0.609–0.727) | revisar |
| 2025-T4 | estable: PSI máximo 0.017 | 0.703 (0.643–0.760) | ok |

Las alertas (340 + 362) y los desertores (247 + 246) suman lo mismo que
`reports/evaluacion.json` (recall global 0.686). Con unos 770 aprendices, un PSI
de 0.01–0.02 es el ruido esperable del muestreo: los datos sintéticos no tienen
drift.

## Consecuencias

- **El recall de 2025-T3 queda en "revisar".** El IC incluye 0.70, así que no es
  una caída comprobada, pero es la señal que en producción llevaría a revisar el
  modelo con la cohorte siguiente.
- **En producción el recall llega tarde:** la deserción se conoce al terminar la
  cohorte. El PSI sí se puede calcular desde la semana 4, con las mismas
  variables que usa el modelo.
- **Se pierde el tablero interactivo de Evidently.** El HTML propio es más simple,
  pero no agrega dependencias ni depende de su compatibilidad con pandas.
- **Si Evidently publica una versión compatible con pandas 3**, se puede volver a
  evaluar sin cambiar el criterio (PSI > 0.20).
- El README deja de mencionar Evidently (PR de Andrés).
