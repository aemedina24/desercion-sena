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
   el cuaderno 04). El estado compara el objetivo de 0.70 con el intervalo (ver
   "Actualización con los datos v2"): "ok", "vigilar" o "revisar". Ningún estado
   hace fallar `dvc repro`: son señales para el equipo, no errores.
4. **Fuera del drift:** `sexo` (atributo protegido, solo se audita) y `deserto`
   (la etiqueta). Se monitorean solo las 39 variables del modelo.

Salidas, en Git (`cache: false`) y escritas con LF:
`reports/drift/drift.json` y `reports/drift/drift.html` (las mismas tablas, para
la demo). El reporte no lleva fecha: los mismos datos dan el mismo hash.

## Resultado con el modelo de DVC (datos v1)

| Cohorte | Drift (PSI) | Recall (IC 95 %) | Estado |
|---|---|---|---|
| 2025-T3 | estable: PSI máximo 0.022 | 0.668 (0.609–0.727) | revisar |
| 2025-T4 | estable: PSI máximo 0.017 | 0.703 (0.643–0.760) | ok |

Las alertas (340 + 362) y los desertores (247 + 246) suman lo mismo que
`reports/evaluacion.json` (recall global 0.686). Con unos 770 aprendices, un PSI
de 0.01–0.02 es el ruido esperable del muestreo: los datos sintéticos no tienen
drift.

## Actualización con los datos v2: el estado mira el intervalo

- **Fecha:** 2026-11-20 · **Decisores:** Andrés Medina, David Montilla

Con la v2 las dos cohortes de test quedaron en "revisar" con la regla original
(recall < 0.70): 0.691 (IC 0.632–0.747) y 0.696 (IC 0.639–0.755). Pero
`data/raw/verdad.parquet` muestra que **0.70 está en el techo de los datos**: con
las mismas 716 alertas, un modelo que conociera la probabilidad real de cada
aprendiz llegaría a un recall de 0.692. Con la regla del punto, cerca de la mitad
de las cohortes saldría en "revisar" solo por azar, y una alarma que suena siempre
deja de mirarse.

**Decisión:** se mantiene el objetivo de 0.70 (es el mismo con que se eligió el
umbral, y bajarlo después de ver los resultados sería mover la vara) y cambia la
regla de decisión, con el mismo bootstrap que ya usa la equidad:

| Estado | Regla | Qué hacer |
|---|---|---|
| ok | recall ≥ 0.70 | Nada |
| vigilar | recall < 0.70, pero el IC 95 % llega a 0.70 | Mirar la cohorte siguiente; puede ser azar |
| revisar | todo el IC queda debajo de 0.70 | Caída clara: revisar el modelo (drift, reentrenar) |

**Resultado con los datos v2:** 2025-T3 y 2025-T4 pasan de "revisar" a
**"vigilar"**. Las cifras del reporte no cambian; solo el estado.
`tests/test_monitor.py` prueba la regla con casos límite (el objetivo exacto es
"ok"; un IC que toca 0.70 es "vigilar") y con una caída clara simulada que sale en
"revisar".

## Consecuencias

- **Con la v1, el recall de 2025-T3 quedaba en "revisar"** con la regla original.
  Su IC incluía 0.70, así que con la regla actual habría salido en "vigilar": no
  era una caída comprobada.
- **"vigilar" es más tolerante con cohortes pequeñas:** con pocos desertores el IC
  es ancho y casi cualquier recall sale en "vigilar". Es correcto (no hay evidencia
  de caída), pero conviene mirar varias cohortes seguidas: dos o tres en "vigilar"
  son una señal aunque ninguna llegue a "revisar".
- **En producción el recall llega tarde:** la deserción se conoce al terminar la
  cohorte. El PSI sí se puede calcular desde la semana 4, con las mismas
  variables que usa el modelo.
- **Se pierde el tablero interactivo de Evidently.** El HTML propio es más simple,
  pero no agrega dependencias ni depende de su compatibilidad con pandas.
- **Si Evidently publica una versión compatible con pandas 3**, se puede volver a
  evaluar sin cambiar el criterio (PSI > 0.20).
- El README deja de mencionar Evidently (PR de Andrés).
