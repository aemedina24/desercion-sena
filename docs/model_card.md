# Model card: alerta temprana de deserción

Ficha del modelo para quien lo vaya a usar o revisar: qué hace, con qué datos se
entrenó, cómo rinde y dónde no conviene confiar en él. Se actualiza en el mismo PR
que cambie el modelo, el umbral o los datos.

**Versión descrita:** `models/modelo.skops` con md5 `9b58e3fead5766bcb03c85931f27a882`
(el de `dvc.lock`; la API lo reporta en `/health` como `version_modelo`), entrenado con
los datos del generador v2. Actualizada el 19 de noviembre de 2026. La versión entregada
el 11 de noviembre (datos v1) está en la etiqueta `v1.0.0`.

## Resumen

| | |
|---|---|
| **Tarea** | Clasificación binaria: ¿desertará el aprendiz después de la fecha de corte? |
| **Modelo** | Regresión logística (`class_weight="balanced"`, `C = 1.0`) |
| **Momento de la predicción** | Alrededor de la semana 4 de la ficha (`fecha_corte`) |
| **Salida** | Probabilidad de deserción y alerta (sí/no) con umbral **0.491** |
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

- **Origen:** `data/raw/aprendices.parquet`, que produce la etapa `generate` con la
  semilla y las fórmulas de `config/data/generador.yaml` (generador v2). Limpieza,
  generación y roles de cada variable en [datos.md](datos.md).
- **Exclusiones:** traslados y cancelaciones (525 registros), porque no son la
  deserción que se predice.
- **Variables:** 39 (perfil, historial en el SENA, conectividad, seguimiento hasta la
  fecha de corte y 3 derivadas en `features.py`). `id_aprendiz` e `id_cohorte` no
  entran al modelo. `sexo` se excluye y solo se usa para auditar equidad.
- **División temporal por cohorte:**

| Conjunto | Cohortes | Filas | Desertores |
|---|---|---|---|
| Train | 2023-T1 a 2025-T2 | 7892 | 2506 (31.8 %) |
| Test | 2025-T3 y 2025-T4 | 1583 | 496 (31.3 %) |

## Umbral de decisión

El umbral se eligió **sin mirar test**, con predicciones out-of-fold en train
(`GroupKFold`, 5 folds por cohorte): entre los umbrales que alcanzan recall ≥ 0.70, el
de mayor precisión. Resultado: **0.491** (recall out-of-fold 0.700, precisión 0.500).

Se prioriza el recall porque en alerta temprana perder a un desertor cuesta más que
una alerta de más. El objetivo se cambia en `config/model/logistic.yaml`
(`recall_objetivo`).

## Rendimiento en test

| Métrica | Valor |
|---|---|
| Recall | 0.694 |
| Precisión | 0.480 |
| F1 | 0.568 |
| ROC-AUC | 0.742 |
| PR-AUC | 0.564 (azar: 0.313) |

En la práctica, de los 496 desertores de test el modelo alerta a **344** y deja sin
alertar a 152, con 716 alertas. De cada 100 alertas, unas 48 corresponden a desertores
reales (al azar serían 31).

**Techo conocido.** Con el generador v2 se sabe la probabilidad real de cada aprendiz
(`data/raw/verdad.parquet`). Un modelo que la conociera llegaría a ROC-AUC 0.762 y PR-AUC
0.613, y con las mismas 716 alertas tendría recall 0.692. El modelo ya rinde cerca de ese
techo: lo que falta no se puede saber en la semana 4.

## Calibración: la probabilidad está inflada

> Etapa `calibrate`: `reports/calibracion.json` y `reports/figures/calibracion.png`.

El modelo se entrena con `class_weight="balanced"`, que le da a cada desertor más peso
del que le corresponde para subir el recall. El efecto secundario es que **la
probabilidad que muestra la API no se puede leer como un porcentaje**: su media es 0.46
cuando deserta el 31 % de los aprendices, y entre los que reciben alerta promete 0.68
cuando desertan 48 de cada 100.

| | Modelo | Corregida |
|---|---|---|
| Probabilidad media (tasa real: 0.313) | 0.463 | 0.319 |
| ECE (0 = calibración perfecta) | 0.150 | 0.028 |
| Brier (constante: 0.2152) | 0.2064 | 0.1820 |
| Alertas y recall | 716 · 0.694 | 716 · 0.694 |

La corrección resta a todos el mismo valor en log-odds
(`log(no desertores / desertores)` en train = 0.765). Como es un desplazamiento igual
para todos, **no cambia el orden de los aprendices**: con el umbral equivalente (0.310)
alerta exactamente a los mismos 716 y el recall sigue en 0.694. Lo único que cambia es
el número que se muestra.

Con el generador v2 se puede comparar contra la probabilidad real de cada aprendiz, que
ningún modelo puede conocer: el error absoluto medio baja de 0.162 a 0.075, y la
correlación de orden con la probabilidad real es 0.889.

**Qué significa para quien usa la API.** Hoy la probabilidad sirve para **ordenar** a
los aprendices, no para decir "este tiene 70 % de riesgo". Mientras no se decida
corregirla, es mejor leerla como una posición en la lista. Aplicar la corrección, o
mostrar bandas de prioridad en vez del número, queda como decisión pendiente del
equipo; también falta probar una calibración ajustada (Platt o isotónica) con
predicciones out-of-fold, que podría corregir lo que el desplazamiento no alcanza.

### Por modalidad

| Modalidad | Aprendices | Desertores | Alertas | Recall | Precisión |
|---|---|---|---|---|---|
| Mixta | 335 | 105 | 149 | 0.657 | 0.463 |
| Virtual | 794 | 307 | 456 | 0.743 | 0.500 |
| Presencial | 454 | 84 | 111 | **0.560** | 0.423 |

## Equidad

Criterio: **igualdad de oportunidades**. Un aprendiz que va a desertar debe tener la
misma probabilidad de recibir una alerta, sea cual sea su grupo. Se mide con la regla de
los cuatro quintos (recall mínimo / recall máximo ≥ 0.80) y con intervalos de confianza
por bootstrap, porque los grupos son pequeños. Detalle en el cuaderno 04 y en el
[ADR 0003](decisiones/0003-umbral-y-equidad.md).

| Grupo | Razón de recall (IC 95 %) | ¿Cumple? |
|---|---|---|
| Sexo (solo se audita, no entra al modelo) | 0.955 (0.85–1.00) | Sí: 99.8 % de las réplicas |
| Modalidad, umbral global (en uso) | 0.753 (0.60–0.89) | No: presencial queda atrás (25 % de las réplicas) |
| Modalidad, modelo perfecto con umbral global (solo v2) | 0.810 (0.64–0.94) | En el límite: 50 % de las réplicas |
| Modalidad, umbral por modalidad (evaluado) | 0.890 (0.74–0.97) | Casi: 85 % de las réplicas |

Recall por modalidad con el umbral global: mixta 0.657, presencial 0.560 y virtual 0.743.
La brecha en presencial es sistemática (también aparece en train con out-of-fold) y
**viene sobre todo de los datos**: un modelo perfecto que conociera la probabilidad real
de cada aprendiz (`verdad.parquet`), con las mismas 716 alertas, también deja atrás a
presencial (0.583 frente a 0.720 en virtual). Allí se deserta menos y los desertores se
parecen más a quienes no desertan, así que un modelo mejor no cierra la brecha.

Un umbral por modalidad (mixta 0.491, presencial 0.354, virtual 0.539) casi la cierra,
pero redistribuye la detección: presencial gana 15 desertores detectados y virtual pierde
21, y las alertas en presencial pasan de 111 a 181. Por eso se mantiene un umbral global
y el umbral por modalidad queda propuesto como decisión institucional
([ADR 0003](decisiones/0003-umbral-y-equidad.md), revisión con los datos v2).

## Explicación de cada alerta y medidas

Para cada aprendiz, el aporte de una variable es su coeficiente por la diferencia entre
su valor y el promedio de train (`models/referencia.json`), sumado por variable original.
En un modelo lineal es exactamente el valor SHAP, y los aportes suman la predicción
menos la del aprendiz promedio (`explain.py` lo verifica).

La API muestra hasta 3 factores con aporte de al menos 0.05 log-odds y, solo si hay
alerta, hasta 3 medidas de `config/recomendaciones/medidas.yaml`. Las 10 medidas cubren
24 variables; las otras 15 explican el riesgo, pero no generan medida. Nunca se sugiere
una medida por características personales (edad, nivel educativo, modalidad), ni por
una variable a la que la colinealidad le invierte el signo. Un factor cuyo dato está
vacío se muestra, pero tampoco genera medida (la asistencia de los virtuales).

Cada medida trae además un plan de acción para el instructor (qué hacer, en qué plazo,
qué decir, qué evitar, qué hacer si el aprendiz no responde, qué registrar y cuándo volver
a revisar; [ADR 0006](decisiones/0006-plan-de-accion.md)). Los guiones son una propuesta
del equipo: no están validados con instructores ni con Bienestar al Aprendiz, y nada
demuestra que reduzcan la deserción.

Importancia media en test (|aporte| en log-odds): actividades entregadas 0.46,
asistencia 0.28, promedio de calificaciones 0.24, modalidad 0.18, tasa de aprobación de
lo entregado 0.16, tasa de finalización previa 0.14. La importancia de la asistencia
viene sobre todo de su indicador de dato faltante, que es lo mismo que ser virtual.

## Limitaciones y riesgos

- **Techo de rendimiento:** el modelo está cerca de lo que permite la información de la
  semana 4 (ROC-AUC 0.742 frente a 0.762 de un modelo que conociera la probabilidad
  real; ver «Rendimiento en test»). Mejorar exige variables nuevas, no otro modelo.
- **Brecha en presencial:** el modelo detecta menos desertores en presencial (0.560)
  que en virtual (0.743). Con el generador v2 se conoce la causa: la modalidad
  presencial resta 0.30 en la fórmula de deserción, así que allí desertan menos y son
  más difíciles de distinguir. En esa modalidad conviene complementar la
  alerta con el criterio del instructor.
- **Cambio en el tiempo:** el generador v2 no cambia entre cohortes, y el monitoreo no
  encuentra drift en ninguna variable (PSI máximo 0.023). El recall por cohorte es 0.691
  en 2025-T3 (IC 0.63–0.75) y 0.696 en 2025-T4 (IC 0.64–0.76): debajo de 0.70, pero con
  intervalos que lo incluyen y al nivel del techo (0.692). Es variación de muestreo, no
  un cambio en los datos. Con datos reales hay que seguir midiendo cada cohorte nueva y
  reentrenar si el recall cae de forma sostenida.
- **Colinealidad en los factores:** `porcentaje_actividades_entregadas`,
  `porcentaje_actividades_aprobadas` y `tasa_aprobacion_entregas` dicen casi lo mismo, y
  a una le sale el signo invertido: con la v2, aprobar más termina sumando riesgo (277
  aprendices de test la tienen como factor, con 92 % aprobado frente a 64 %). Pasa lo
  mismo con `tasa_finalizacion_previa` y `abandono_ultimo_curso` frente a
  `n_cursos_abandonados`. Esas variables se muestran como factor, pero no generan
  medida, y una prueba verifica la dirección de cada variable que sí la genera. La
  solución de fondo es agrupar los factores por tema.
- **Datos de encuesta incompletos:** `horas_disponibles_estudio`,
  `dificultad_economica_reportada` y `calidad_internet` faltan en 581 aprendices, y no
  al azar (sin encuesta deserta el 36,5 %, con encuesta el 31,4 %); el modelo trata la
  ausencia como información.
- **Falsos positivos:** algo más de la mitad de las alertas son aprendices que no
  desertan (372 de 716 en test). Las acciones de apoyo deben ser útiles también para
  ellos.
- **Datos sintéticos:** las cifras describen al generador de datos, no al SENA real. Que
  el generador sea propio permite saber qué es señal y qué es ruido (por ejemplo, el
  techo de rendimiento), pero también que el modelo encuentra justo lo que se puso en las
  fórmulas. Lo que se transfiere es el método; con datos reales hay que reentrenar y volver a auditar
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
