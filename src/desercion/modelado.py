"""Piezas compartidas por las etapas train y tune.

Viven aquí para que train_model.py y tune.py usen exactamente las mismas
columnas y el mismo criterio de umbral, sin importarse entre sí.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

# --- Rutas -----------------------------------------------------------------
RUTA_TRAIN = Path("data/features/train.parquet")
RUTA_TEST = Path("data/features/test.parquet")

# --- Parámetros ------------------------------------------------------------
ETIQUETA = "deserto"
# Se conservan en features.py para trazabilidad y para el split; no son
# información del aprendiz y el modelo los memorizaría.
COLUMNAS_NO_MODELO = ["id_aprendiz", "id_cohorte"]
# Grupo de la validación cruzada: cada fold valida sobre cohortes completas.
COLUMNA_GRUPO = "id_cohorte"


def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa las variables del modelo (X) de la etiqueta (y)."""
    x = df.drop(columns=[ETIQUETA, *COLUMNAS_NO_MODELO])
    # Booleanos (incluido el "boolean" con nulos) como 0/1: así se imputan y
    # escalan como cualquier número. pd.NA pasa a NaN.
    booleanas = x.select_dtypes(include=["bool", "boolean"]).columns
    x[booleanas] = x[booleanas].astype("float64")
    return x, df[ETIQUETA]


def tipos_de_columnas(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(numéricas, categóricas) según el tipo de dato de cada columna."""
    numericas = x.select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in x.columns if c not in numericas]
    return numericas, categoricas


def elegir_umbral(
    y: pd.Series, probabilidad: np.ndarray, recall_objetivo: float
) -> dict:
    """Entre los umbrales con recall >= objetivo, el de mayor precisión."""
    precision, recall, umbrales = precision_recall_curve(y, probabilidad)
    # precision y recall tienen un elemento más que umbrales: se descarta el último.
    cumplen = recall[:-1] >= recall_objetivo
    if not cumplen.any():
        raise ValueError(f"Ningún umbral alcanza recall {recall_objetivo}")
    candidatos = np.where(cumplen)[0]
    mejor = candidatos[np.argmax(precision[:-1][candidatos])]
    return {
        "umbral": float(umbrales[mejor]),
        "precision_oof": float(precision[mejor]),
        "recall_oof": float(recall[mejor]),
    }
