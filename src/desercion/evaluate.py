"""Evaluación del modelo en test: global, por modalidad y frente al azar.

Entradas:
    models/modelo.skops                 (el Pipeline que escribió train)
    data/features/test.parquet          (cohortes recientes, nunca vistas)
    reports/metricas_modelo_base.json   (de aquí sale el umbral; no se recalcula)
Salida:
    reports/evaluacion.json   (métricas que lee DVC: `uv run dvc metrics show`)

Qué agrega frente a train: train reporta el recall por modalidad. Aquí está lo
que necesita quien recibe las alertas: cuántas alertas genera el modelo, cuántas
son correctas y cuánto mejora sobre alertar al azar a la misma cantidad de
aprendices. El cuaderno 04 lee este JSON en lugar de recalcular.

Uso (desde la raíz del repositorio):
    uv run python -m desercion.evaluate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from desercion.modelado import (
    RUTA_METRICAS,
    RUTA_TEST,
    cargar_modelo,
    separar_x_y,
)

# --- Rutas -----------------------------------------------------------------
RUTA_EVALUACION = Path("reports/evaluacion.json")

# --- Parámetros ------------------------------------------------------------
COLUMNA_SUBGRUPO = "modalidad"

logger = logging.getLogger(__name__)


def resumen_alertas(y_real: pd.Series, alerta: np.ndarray) -> dict:
    """Matriz de confusión y lo que significa para quien recibe las alertas.

    El azar se calcula alertando a la misma proporción de aprendices: detecta,
    en promedio, tasa_alerta × desertores. "mejora_sobre_azar" = 1.5 quiere
    decir que el modelo encuentra 1,5 veces más desertores con el mismo esfuerzo.
    """
    vn, fp, fn, vp = confusion_matrix(y_real, alerta, labels=[0, 1]).ravel()
    n = len(y_real)
    desertores = vp + fn
    alertas = vp + fp
    tasa_alerta = alertas / n
    detectados_azar = tasa_alerta * desertores
    return {
        "aprendices": int(n),
        "desertores": int(desertores),
        "alertas": int(alertas),
        "verdaderos_positivos": int(vp),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_negativos": int(vn),
        "tasa_desercion": desertores / n,
        "tasa_alerta": tasa_alerta,
        "recall": vp / desertores if desertores else float("nan"),
        "precision": vp / alertas if alertas else float("nan"),
        # Cuántas alertas revisa el instructor por cada desertor que encuentra.
        "alertas_por_acierto": alertas / vp if vp else float("nan"),
        "detectados_azar": detectados_azar,
        "mejora_sobre_azar": vp / detectados_azar if detectados_azar else float("nan"),
    }


def evaluar(
    y_real: pd.Series, probabilidad: np.ndarray, grupos: pd.Series, umbral: float
) -> dict:
    """Resumen global y uno por cada valor de `grupos`."""
    alerta = (probabilidad >= umbral).astype(int)
    por_grupo = {}
    for grupo in sorted(grupos.unique()):
        mascara = (grupos == grupo).to_numpy()
        por_grupo[str(grupo)] = resumen_alertas(y_real[mascara], alerta[mascara])
    return {
        "umbral": umbral,
        "global": resumen_alertas(y_real, alerta),
        COLUMNA_SUBGRUPO: por_grupo,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    umbral = json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))["umbral"]
    modelo = cargar_modelo()
    x_test, y_test = separar_x_y(pd.read_parquet(RUTA_TEST))

    probabilidad = modelo.predict_proba(x_test)[:, 1]
    evaluacion = evaluar(y_test, probabilidad, x_test[COLUMNA_SUBGRUPO], umbral)

    RUTA_EVALUACION.parent.mkdir(parents=True, exist_ok=True)
    RUTA_EVALUACION.write_text(
        json.dumps(evaluacion, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    logger.info("Umbral: %.3f", umbral)
    filas = {"global": evaluacion["global"], **evaluacion[COLUMNA_SUBGRUPO]}
    logger.info(
        "%-11s %6s %7s %6s %6s %6s %7s",
        "grupo", "n", "alertas", "recall", "prec.", "x azar", "al/acierto",
    )  # fmt: skip
    for nombre, r in filas.items():
        logger.info(
            "%-11s %6d %7d %6.3f %6.3f %6.2f %7.2f",
            nombre,
            r["aprendices"],
            r["alertas"],
            r["recall"],
            r["precision"],
            r["mejora_sobre_azar"],
            r["alertas_por_acierto"],
        )


if __name__ == "__main__":
    main()
