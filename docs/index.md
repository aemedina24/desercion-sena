# Documentación

Sistema de alerta temprana que estima, en la semana 4 de formación, el riesgo de
deserción de los aprendices del SENA en modalidad virtual, presencial y mixta, y le
entrega al instructor una API con el riesgo, los factores que lo explican y medidas
sugeridas. La presentación general y la instalación están en el
[README](../README.md).

## Por dónde empezar

| Si quiere… | Lea |
|---|---|
| Entender cómo encajan las piezas | [Arquitectura](arquitectura.md) |
| Saber qué tan bueno es el modelo y cuándo no confiar en él | [Model card](model_card.md) |
| Consultar la API | [API](api.md) |
| Conocer cada variable y cómo se limpió | [Datos](datos.md) |
| Ver quién hizo cada tarea | [Plan de trabajo](plan_trabajo.md) |

## Decisiones (ADR)

Cada decisión importante quedó registrada con su contexto, las opciones evaluadas y sus
consecuencias.

| ADR | Decisión |
|---|---|
| [001](decisiones/0001-uso-de-uv.md) | uv para el entorno y las dependencias |
| [002](decisiones/0002-eleccion-modelo.md) | Regresión logística como modelo final: LightGBM y la red neuronal no la superan de forma significativa |
| [003](decisiones/0003-umbral-y-equidad.md) | Umbral global en la v1; equidad medida con la regla de los cuatro quintos y umbral por modalidad como siguiente versión |
| [004](decisiones/0004-api-consulta-y-datos.md) | API por aprendiz y por cohorte con filtros, con el riesgo calculado al arrancar |
| [005](decisiones/0005-monitoreo-sin-evidently.md) | Monitoreo con PSI y recall por cohorte, sin Evidently |

## Resultados

| Reporte | Contenido |
|---|---|
| `reports/evaluacion.json` | Métricas en test, globales y por modalidad |
| `reports/importancia_variables.json` | Importancia media de cada variable |
| `reports/drift/drift.json` y `drift.html` | Drift por variable y recall por cohorte |
| `notebooks/01` a `04` | Exploración, modelo base, comparación de modelos y equidad |
