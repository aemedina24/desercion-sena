# ADR 003: Umbral de alerta y criterio de equidad

- **Estado:** Aceptada
- **Fecha:** 2026-11-06
- **Decisores:** Andrés Medina, David Montilla

## Contexto

El modelo final (ADR 002) usa un único umbral, 0.461, elegido con predicciones
out-of-fold en train para lograr un recall de al menos 0.70. En test, el recall
cambia mucho entre modalidades:

| Modalidad | Desertores en test | Recall | Precisión | % alertado |
|---|---|---|---|---|
| Mixta | 90 | 0.756 | 0.486 | 46 % |
| Presencial | 101 | 0.525 | 0.434 | 26 % |
| Virtual | 302 | 0.719 | 0.493 | 57 % |

Hacía falta acordar qué significa que el sistema sea equitativo y si conviene un
umbral distinto por modalidad. El análisis completo está en
`notebooks/04_evaluacion_y_equidad.ipynb`; la lógica, en `src/desercion/equidad.py`.

## Decisión 1: criterio de equidad

- **Principal: igualdad de oportunidades.** Todo aprendiz que va a desertar debe
  tener la misma probabilidad de recibir una alerta, sea del grupo que sea. Se
  mide con la **regla de los cuatro quintos** sobre el recall:
  `recall mínimo / recall máximo ≥ 0.80`.
- **Siempre con intervalo de confianza por bootstrap** (estratificado por grupo,
  2000 réplicas, IC 95 %). Una tolerancia fija en puntos (por ejemplo, 10 puntos)
  no se puede verificar: el recall de presencial en test tiene un IC de 0.43–0.62
  con solo 101 desertores.
- **Secundarias: precisión y tasa de alerta.** Con tasas de deserción de 22, 29 y
  39 % no se pueden igualar a la vez que el recall.
- **`sexo` es un atributo protegido:** solo se audita, desde
  `data/processed/auditoria.parquet`. Nunca entra al modelo ni a los umbrales.

Por qué el recall: el sistema es de apoyo, no de sanción. Lo más grave es que un
desertor no reciba ayuda por su grupo; una falsa alarma cuesta una llamada.

## Resultados con el umbral global

| Grupo | Ratio min/max | IC 95 % | Réplicas ≥ 0.80 |
|---|---|---|---|
| Modalidad | 0.695 | 0.55–0.84 | 7 % |
| Sexo | 0.964 | 0.85–1.00 | 99.9 % |

- **Por modalidad no se cumple.** La brecha es sistemática: también aparece en
  train con out-of-fold (presencial 0.50, virtual 0.79).
- **Por sexo se cumple.**

## Decisión 2: umbral global en la v1; umbral por modalidad como siguiente versión

Se evaluó un umbral por modalidad, elegido con out-of-fold en train y el mismo
criterio (recall ≥ 0.70 con la mayor precisión): mixta 0.440, presencial 0.343,
virtual 0.530.

| | Umbral global | Umbral por modalidad |
|---|---|---|
| Ratio modalidad (IC 95 %) | 0.695 (0.55–0.84) | 0.826 (0.71–0.93) |
| Réplicas que cumplen | 7 % | 61 % |
| Detectados: mixta / presencial / virtual | 68 / 53 / 217 | 70 / 69 / 194 |
| Alertas totales | 702 | 708 |
| Alertas en presencial | 122 | 206 |
| Alertas por acierto en presencial | 2.30 | 2.99 |
| Ratio sexo | 0.964 | 0.938 |

Es una **redistribución, no una mejora**: con casi las mismas alertas totales,
presencial gana 16 desertores detectados y virtual pierde 23. Además, solo el
61 % de las réplicas cumple la regla, así que la mejora no es concluyente.

**Se mantiene el umbral global en la v1.** El umbral por modalidad queda como
siguiente versión, condicionado a:

1. Confirmarlo con más cohortes (que el ratio cumpla con un IC por encima de 0.80).
2. Acordar con quienes usan el sistema el costo: más alertas por acierto en
   presencial y menos detección en virtual.

## Consecuencias

- La API usa un solo umbral (`reports/metricas_modelo_base.json`).
- La brecha por modalidad es una limitación conocida y debe ser visible: las
  métricas se reportan por modalidad (`reports/evaluacion.json`) y la model card
  la menciona.
- La equidad se vuelve a medir con cada cohorte nueva, con las funciones de
  `equidad.py` (monitoreo).
- Datos sintéticos: las brechas describen el generador; lo que se lleva a
  producción es el método.

## Cómo reproducir

```powershell
uv sync
uv run dvc pull
uv run dvc repro
# Abrir notebooks/04_evaluacion_y_equidad.ipynb y ejecutar Restart → Run All
```
