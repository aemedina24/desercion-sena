"""Equidad del modelo: métricas por grupo, intervalos bootstrap y umbrales por grupo.

Criterio (acordado el 6 nov, ver cuaderno 04 y ADR 0003):

- Principal: igualdad de oportunidades. Todo desertor debe tener la misma
  probabilidad de ser detectado, sea del grupo que sea. Se mide con la regla de
  los cuatro quintos sobre el recall: recall mínimo / recall máximo >= 0.80,
  siempre con intervalo de confianza por bootstrap (los grupos son pequeños).
- Secundarias: precisión y tasa de alerta. Con tasas de deserción distintas
  (22 / 29 / 39 %) no se pueden igualar a la vez que el recall.

`sexo` es un atributo protegido: solo se usa para auditar, nunca en el modelo
ni en los umbrales. `modalidad` es un atributo operativo que el modelo ya usa.

Las funciones son puras: el cuaderno 04 las importa y muestra el resultado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from desercion.evaluate import resumen_alertas
from desercion.modelado import elegir_umbral

# --- Parámetros ------------------------------------------------------------
RATIO_MINIMO = 0.80  # regla de los cuatro quintos
N_BOOTSTRAP = 2000
NIVEL_CONFIANZA = 0.95
SEMILLA = 42


def metricas_por_grupo(
    y_real: pd.Series | np.ndarray, alerta: np.ndarray, grupos: pd.Series | np.ndarray
) -> pd.DataFrame:
    """Una fila por grupo con el resumen de evaluate.resumen_alertas."""
    y_real, alerta, grupos = (np.asarray(v) for v in (y_real, alerta, grupos))
    filas = {
        str(grupo): resumen_alertas(
            pd.Series(y_real[grupos == grupo]), alerta[grupos == grupo]
        )
        for grupo in sorted(pd.unique(grupos))
    }
    return pd.DataFrame(filas).T.rename_axis("grupo")


def ratio_cuatro_quintos(recall_por_grupo: pd.Series) -> float:
    """recall mínimo / recall máximo. 1 = todos iguales; < 0.80 = brecha."""
    maximo = recall_por_grupo.max()
    return float(recall_por_grupo.min() / maximo) if maximo > 0 else float("nan")


def _recall(y: np.ndarray, alerta: np.ndarray) -> float:
    desertores = y == 1
    return float(alerta[desertores].mean()) if desertores.any() else float("nan")


def bootstrap_recall(
    y_real: pd.Series | np.ndarray,
    alerta: np.ndarray,
    grupos: pd.Series | np.ndarray,
    n: int = N_BOOTSTRAP,
    semilla: int = SEMILLA,
    nivel: float = NIVEL_CONFIANZA,
) -> tuple[pd.DataFrame, dict]:
    """Intervalo de confianza del recall de cada grupo y del ratio min/max.

    Se remuestrea dentro de cada grupo (bootstrap estratificado): cada réplica
    conserva el tamaño de los grupos, como en los datos reales.
    Devuelve (tabla por grupo, resumen del ratio).
    """
    y_real, alerta, grupos = (np.asarray(v) for v in (y_real, alerta, grupos))
    generador = np.random.default_rng(semilla)
    nombres = [str(g) for g in sorted(pd.unique(grupos))]
    indices = {g: np.flatnonzero(grupos.astype(str) == g) for g in nombres}

    replicas = np.empty((n, len(nombres)))
    for i in range(n):
        for j, g in enumerate(nombres):
            muestra = generador.choice(indices[g], size=len(indices[g]), replace=True)
            replicas[i, j] = _recall(y_real[muestra], alerta[muestra])

    cola = (1 - nivel) / 2
    tabla = pd.DataFrame(
        {
            "recall": [
                _recall(y_real[indices[g]], alerta[indices[g]]) for g in nombres
            ],
            "ic_inferior": np.nanquantile(replicas, cola, axis=0),
            "ic_superior": np.nanquantile(replicas, 1 - cola, axis=0),
            "desertores": [int((y_real[indices[g]] == 1).sum()) for g in nombres],
        },
        index=pd.Index(nombres, name="grupo"),
    )
    ratios = np.nanmin(replicas, axis=1) / np.nanmax(replicas, axis=1)
    ratio = {
        "ratio": ratio_cuatro_quintos(tabla["recall"]),
        "ic_inferior": float(np.nanquantile(ratios, cola)),
        "ic_superior": float(np.nanquantile(ratios, 1 - cola)),
        # Fracción de réplicas que cumplen la regla: qué tan seguro es el veredicto.
        "prob_cumple": float(np.mean(ratios >= RATIO_MINIMO)),
    }
    return tabla, ratio


def umbrales_por_grupo(
    y_real: pd.Series | np.ndarray,
    probabilidad: np.ndarray,
    grupos: pd.Series | np.ndarray,
    recall_objetivo: float,
) -> dict[str, float]:
    """Un umbral por grupo con modelado.elegir_umbral.

    Debe recibir predicciones out-of-fold de train, nunca de test: elegir el
    umbral mirando test haría optimista la evaluación.
    """
    y_real, probabilidad, grupos = (
        np.asarray(v) for v in (y_real, probabilidad, grupos)
    )
    return {
        str(g): elegir_umbral(
            pd.Series(y_real[grupos == g]), probabilidad[grupos == g], recall_objetivo
        )["umbral"]
        for g in sorted(pd.unique(grupos))
    }


def alertar(
    probabilidad: np.ndarray,
    grupos: pd.Series | np.ndarray,
    umbral: float | dict[str, float],
) -> np.ndarray:
    """1 si la probabilidad supera el umbral (único o del grupo de cada aprendiz)."""
    probabilidad = np.asarray(probabilidad)
    if isinstance(umbral, dict):
        limites = np.array([umbral[str(g)] for g in np.asarray(grupos)])
    else:
        limites = np.full(len(probabilidad), umbral)
    return (probabilidad >= limites).astype(int)
