# Model card: alerta temprana de deserción

Ficha del modelo para quien lo vaya a usar o revisar: qué hace, con qué datos se
entrenó, cómo rinde y dónde no conviene confiar en él. Se actualiza en el mismo PR
que cambie el modelo, el umbral o los datos.

**Versión descrita:** `models/modelo.skops` con md5 `f0dd9683c0396c196eb6cd409b605c29`
(el de `dvc.lock`; la API lo reporta en `/health` como `version_modelo`). Actualizada el
9 de noviembre de 2026.

## Resumen

| | |
|---|---|
| **Tarea** | Clasificación binaria: ¿desertará el aprendiz después de la fecha de corte? |
| **Modelo** | Regresión logística (`class_weight="balanced"`, `C = 1.0`) |
| **Momento de la predicción** | Alrededor de la semana 4 de la ficha (`fecha_corte`) |
| **Salida** | Probabilidad de deserción y alerta (sí/no) con umbral **0.461** |
| **Código** | `src/desercion/train_model.py`, `src/desercion/modelado.py` |
| **Configuración** | `config/model/logistic.yaml` |
| **Registro** | MLflow (DagsHub), experimento `desercion-seguimiento`, run `modelo-base-logistica` |
| **Decisión del modelo** | [ADR 0002](decisiones/0002-eleccion-modelo.md) |
| **Explicación** | Aporte de cada variable por coeficientes (`explain.py`), equivalente a SHAP en un modelo lineal |
| **Servicio** | API FastAPI en Docker ([api.md](api.md)) |
| **Monitoreo** | Etapa `monitor`: PSI por variable y recall por cohorte ([ADR 0005](decisiones/0005-monitoreo-sin-evidently.md)) |

## Uso previsto

- **Para qué sirve:** priorizar a qué aprendices contactar primero con acciones de
  apoyo (bienestar, tutoría, seguimiento del instructor) en las primeras semanas.
- **Quién lo usa:** equipos de bienestar y coordinación académica del SENA.
- **Para qué NO sirve:** tomar decisiones sobre el aprendiz sin intervención humana
  (sanciones, cancelación de matrícula, asignación de cupos). Una alerta es una
  invitación a revisar el caso, no un diagnóstico.

## Datos

- **Origen:** `data/raw/aprendices.parquet`, versionado con DVC. Limpieza y roles de
  cada variable en [datos.md](datos.md).
- **Exclusiones:** traslados y cancelaciones (482 registros), porque no son la
  deserción que se predice.
- **Variables:** 39 (perfil, historial en el SENA, conectividad, seguimiento hasta la
  fecha de corte y 3 derivadas en `features.py`). `id_aprendiz` e `id_cohorte` no
  entran al modelo. `sexo` se excluye y solo se usa para auditar equidad.
- **División temporal por cohorte:**

| Conjunto | Cohortes | Filas | Desertores |
|---|---|---|---|
| Train | 2023-T1 a 2025-T2 | 7974 | 31.0 % |
| Test | 2025-T3 y 2025-T4 | 1544 | 493 (31.9 %) |

## Umbral de decisión

El umbral se eligió **sin mirar test**, con predicciones out-of-fold en train
(`GroupKFold`, 5 folds por cohorte): entre los umbrales que alcanzan recall ≥ 0.70, el
de mayor precisión. Resultado: **0.461** (recall out-of-fold 0.701, precisión 0.472).

Se prioriza el recall porque en alerta temprana perder a un desertor cuesta más que
una alerta de más. El objetivo se cambia en `config/model/logistic.yaml`
(`recall_objetivo`).

## Rendimiento en test

| Métrica | Valor |
|---|---|
| Recall | 0.686 |
| Precisión | 0.481 |
| F1 | 0.566 |
| ROC-AUC | 0.735 |
| PR-AUC | 0.576 (azar: 0.319) |

En la práctica, de los 493 desertores de test el modelo alerta a unos **338** y deja
sin alertar a unos 155. De cada 100 alertas, unas 48 corresponden a desertores reales
(al azar serían 32).

### Por modalidad

| Modalidad | Aprendices | Desertores | Alertas | Recall | Precisión |
|---|---|---|---|---|---|
| Mixta | 307 | 90 | 140 | 0.756 | 0.486 |
| Virtual | 774 | 302 | 440 | 0.719 | 0.493 |
| Presencial | 463 | 101 | 122 | **0.525** | 0.434 |

## Equidad

Criterio: **igualdad de oportunidades**. Un aprendiz que va a desertar debe tener la
misma probabilidad de recibir una alerta, sea cual sea su grupo. Se mide con la regla de
los cuatro quintos (recall mínimo / recall máximo ≥ 0.80) y con intervalos de confianza
por bootstrap, porque los grupos son pequeños. Detalle en el cuaderno 04 y en el
[ADR 0003](decisiones/0003-umbral-y-equidad.md).

| Grupo | Razón de recall (IC 95 %) | ¿Cumple? |
|---|---|---|
| Sexo (solo se audita, no entra al modelo) | 0.96 (0.85–1.00) | Sí |
| Modalidad, umbral global (v1) | 0.69 (0.55–0.84) | No: presencial queda atrás |
| Modalidad, umbral por modalidad (evaluado) | 0.83 (0.71–0.93) | No concluyente: cumple el 61 % de las réplicas |

El umbral por modalidad (mixta 0.44, presencial 0.34, virtual 0.53) redistribuye la
detección en vez de mejorarla: presencial gana 16 desertores detectados y virtual pierde
23, y las alertas en presencial pasan de 122 a 206. Por eso la v1 mantiene un umbral
global y deja el umbral por modalidad como siguiente versión. La brecha en presencial
aparece también en train: no es casualidad de las cohortes de prueba.

## Explicación de cada alerta y medidas

Para cada aprendiz, el aporte de una variable es su coeficiente por la diferencia entre
su valor y el promedio de train (`models/referencia.json`), sumado por variable original.
En un modelo lineal es exactamente el valor SHAP, y los aportes suman la predicción
menos la del aprendiz promedio (`explain.py` lo verifica).

La API muestra hasta 3 factores con aporte de al menos 0.05 log-odds y, solo si hay
alerta, hasta 3 medidas de `config/recomendaciones/medidas.yaml`. Las 10 medidas cubren
26 variables; las otras 13 explican el riesgo, pero no generan medida. Nunca se sugiere
una medida por características personales (edad, nivel educativo, modalidad).

Importancia media en test (|aporte| en log-odds): actividades aprobadas 0.60,
actividades entregadas 0.25, promedio de calificaciones 0.15, evaluaciones no
presentadas 0.15, ingresos a la plataforma en 7 días 0.15, asistencia 0.14.

## Limitaciones y riesgos

- **Techo de rendimiento:** tres modelos distintos (logística, LightGBM, red neuronal)
  llegan a la misma PR-AUC (~0.56–0.58). El límite está en la información disponible,
  no en el modelo; mejorar exige variables nuevas.
- **Brecha en presencial:** el modelo detecta a casi la mitad menos de desertores en
  presencial que en las otras modalidades. En esa modalidad conviene complementar la
  alerta con el criterio del instructor.
- **Cambio en el tiempo:** en test el recall (0.686) queda por debajo del alcanzado en
  validación cruzada (0.701). El monitoreo no encuentra drift en ninguna variable (PSI
  máximo 0.02) y el recall por cohorte es 0.668 en 2025-T3 (IC 0.61–0.73, «revisar») y
  0.703 en 2025-T4: la diferencia es variación de muestreo, no un cambio en los datos.
  Hay que seguir midiendo cada cohorte nueva y reentrenar si el recall cae de forma
  sostenida.
- **Colinealidad en los factores:** `porcentaje_actividades_entregadas`,
  `porcentaje_actividades_aprobadas` y `tasa_aprobacion_entregas` dicen casi lo mismo.
  Con las otras dos fijas, entregar más termina sumando riesgo: en test, 623 aprendices
  tienen actividades entregadas como factor con una media de 91 % entregado, frente al
  75 % general. La variable se muestra como factor, pero no genera medida, para no
  pedirle «entregar las pendientes» a quien ya entregó. La solución de fondo es agrupar
  los factores por tema (v2).
- **Datos de encuesta incompletos:** `horas_disponibles_estudio`,
  `dificultad_economica_reportada` y `calidad_internet` faltan en 679 aprendices; el
  modelo trata la ausencia como información.
- **Falsos positivos:** algo más de la mitad de las alertas son aprendices que no
  desertan (364 de 702 en test). Las acciones de apoyo deben ser útiles también para
  ellos.
- **Datos sintéticos:** las cifras describen al generador de datos, no al SENA real. Lo
  que se transfiere es el método; con datos reales hay que reentrenar y volver a auditar
  la equidad.
- **Sin autenticación en la API:** en la demo cualquiera puede consultar cualquier
  cohorte. En producción habría que restringir cada instructor a sus grupos
  ([ADR 0004](decisiones/0004-api-consulta-y-datos.md)).

## Cómo reproducir

```powershell
uv sync
uv run dvc pull
uv run dvc status
uv run dvc repro
```

`dvc pull` descarga el modelo de esta versión, así que `dvc repro` no reentrena si nada
cambió (`dvc status` debe decir que todo está al día). Si se reentrena, el archivo del
modelo cambia de hash aunque las predicciones sean idénticas (skops nombra sus archivos
internos de forma distinta en cada ejecución): los modelos se comparan por sus métricas,
no por el hash.
