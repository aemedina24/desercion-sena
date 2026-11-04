# Model card: alerta temprana de deserción

Ficha del modelo para quien lo vaya a usar o revisar: qué hace, con qué datos se
entrenó, cómo rinde y dónde no conviene confiar en él. Se actualiza en el mismo PR
que cambie el modelo, el umbral o los datos.

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

| Modalidad | Aprendices | Desertores | Recall |
|---|---|---|---|
| Mixta | 307 | 90 | 0.756 |
| Virtual | 774 | 302 | 0.719 |
| Presencial | 463 | 101 | **0.525** |

## Limitaciones y riesgos

- **Techo de rendimiento:** tres modelos distintos (logística, LightGBM, red neuronal)
  llegan a la misma PR-AUC (~0.56–0.58). El límite está en la información disponible,
  no en el modelo; mejorar exige variables nuevas.
- **Brecha en presencial:** el modelo detecta a casi la mitad menos de desertores en
  presencial que en las otras modalidades. En esa modalidad conviene complementar la
  alerta con el criterio del instructor.
- **Cambio en el tiempo:** en test el recall (0.686) queda por debajo del alcanzado en
  validación cruzada (0.701), y la tasa de deserción sube de 31.0 % a 31.9 %. Hay que
  monitorear el recall con cada cohorte nueva y reentrenar si cae.
- **Datos de encuesta incompletos:** `horas_disponibles_estudio`,
  `dificultad_economica_reportada` y `calidad_internet` faltan en 679 aprendices; el
  modelo trata la ausencia como información.
- **Falsos positivos:** algo más de la mitad de las alertas son aprendices que no
  desertan. Las acciones de apoyo deben ser útiles también para ellos.

## Cómo reproducir

```powershell
uv sync
uv run dvc pull
uv run dvc repro train
uv run dvc metrics show
```
