# ADR 002: Elección del modelo final

- **Estado:** Aceptada
- **Fecha:** 2026-11-04
- **Decisores:** Andrés Medina, David Montilla

## Contexto

El modelo base (regresión logística, `config/model/logistic.yaml`) obtuvo en test una
ROC-AUC de 0.735 y una PR-AUC de 0.576. El plan preveía sustituirlo por LightGBM con
hiperparámetros optimizados, bajo el supuesto de que un modelo más flexible capturaría
relaciones no lineales e interacciones entre variables.

Había una señal de alerta previa: la mejor variable por sí sola alcanzaba una ROC-AUC de
0.72, de modo que las demás variables aportaban poca información adicional.

La decisión debía tomarse **sin usar test**: elegir entre varios modelos según su
resultado en test equivale a ajustar el modelo a test y vuelve optimista la evaluación
final.

## Método de comparación

- **Datos:** solo `data/features/train.parquet` (7974 filas, cohortes 2023-T1 a 2025-T2).
- **Validación cruzada:** `GroupKFold` con 5 folds y grupo `id_cohorte`. Cada fold valida
  sobre cohortes completas que el modelo no vio, como ocurre con test.
- **Comprobaciones previas:**
  - Sin fuga por aprendiz: 9518 filas = 9518 aprendices; ninguno aparece en más de una
    cohorte.
  - Folds sin estratificar: la tasa de deserción por cohorte es estable (0.27 a 0.33),
    así que ningún fold queda desbalanceado.
- **Métrica:** PR-AUC (`average_precision`). La clase de interés es minoritaria (~31 %)
  y la métrica no depende del umbral, que se elige por separado.
- **Comparación pareada:** todos los modelos se evalúan en los mismos folds, lo que
  permite comparar fold a fold y no solo las medias.

## Opciones evaluadas

| Modelo | PR-AUC media | Por fold |
|---|---|---|
| LightGBM (tuneado con Optuna, 50 trials) | 0.581 | 0.578 · 0.590 · 0.519 · 0.598 · 0.620 |
| Regresión logística (sin tunear, C = 1.0) | 0.574 | 0.563 · 0.587 · 0.504 · 0.607 · 0.610 |
| Red neuronal (`MLPClassifier`, capas 32-16) | 0.557 | 0.542 · 0.584 · 0.509 · 0.577 · 0.571 |
| Referencia: azar (proporción de desertores) | 0.31 | — |

- **LightGBM** supera a la logística en 4 de 5 folds, por entre 0.003 y 0.015, y pierde
  en uno (−0.008). La ventaja es menor que la variación entre folds (0.50 a 0.61) y
  además está sesgada a su favor: sus hiperparámetros se eligieron en esos mismos folds,
  y 0.581 es el mejor de 50 intentos. Detalle en `reports/tuning_lightgbm.json`.
- **Red neuronal:** queda por debajo de ambos. Con unas 8000 filas tiende a memorizar.
- **CatBoost:** considerado y descartado sin probar. Es de la misma familia que LightGBM,
  y su ventaja principal (categóricas con muchos valores) no aplica: las categóricas de
  este conjunto tienen pocos valores. Además, cada modelo adicional evaluado en los
  mismos folds aumenta el riesgo de elegir un ganador por azar.

## Decisión

**Regresión logística como modelo final**, con el umbral de decisión elegido mediante
predicciones out-of-fold en train: entre los umbrales con recall ≥ 0.70, el de mayor
precisión.

Razones:

1. **Rendimiento equivalente:** la diferencia con LightGBM está dentro del ruido.
2. **Simplicidad:** ante igual rendimiento, el modelo más simple es más fácil de
   mantener, depurar y desplegar.
3. **Interpretabilidad:** los coeficientes indican qué variables aumentan el riesgo de
   deserción y alimentan directamente las recomendaciones de apoyo
   (`config/recomendaciones/`). LightGBM exigiría herramientas adicionales, como SHAP.

## Consecuencias

- Tres modelos de naturaleza distinta (lineal, árboles, red) llegan al mismo techo de
  PR-AUC (~0.56–0.58). **El límite está en la información de los datos, no en la
  capacidad del modelo.** Mejorar la predicción exige variables nuevas, no modelos más
  complejos. Queda como recomendación de trabajo futuro.
- `src/desercion/tune.py` permanece en el repositorio como experimento reproducible,
  fuera del pipeline DVC, porque su resultado no alimenta al modelo final. Por eso
  `lightgbm` y `optuna` siguen entre las dependencias.
- `train_model.py` ya no usa un umbral fijo de 0.5. Resultados en test con el umbral
  out-of-fold (0.461):

  | Métrica | Umbral 0.5 | Umbral 0.461 |
  |---|---|---|
  | Recall | 0.623 | 0.686 |
  | Precisión | 0.512 | 0.481 |
  | F1 | 0.562 | 0.566 |
  | ROC-AUC / PR-AUC | 0.735 / 0.576 | sin cambio |

- En test el recall queda en 0.686, por debajo del objetivo de 0.70 alcanzado en
  validación cruzada (0.701). Test contiene cohortes posteriores (2025-T3 y 2025-T4) con
  algo más de deserción (31.9 %). Es una primera señal de cambio en los datos con el
  tiempo y justifica el monitoreo del modelo en producción.
- La modalidad **presencial** sigue con el menor recall (0.525, frente a 0.756 en mixta
  y 0.719 en virtual). Se analiza en el cuaderno de equidad.

## Cómo reproducir

```powershell
uv sync
uv run python -m desercion.tune               # experimento LightGBM, unos minutos
uv run python -m desercion.comparar_modelos   # tabla de comparación, en MLflow
uv run dvc repro train                        # modelo final con umbral out-of-fold
```
