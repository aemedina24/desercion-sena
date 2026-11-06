"""Explicación del modelo con sus coeficientes, agrupados por variable original.

Por qué no SHAP (decisión del 5 nov): el modelo es una regresión logística. Su
log-odds es una suma, intercepto + Σ coef_j · z_j, así que el aporte de cada
variable se lee directo:

    aporte_j = coef_j · (z_j − media de z_j en train)

Eso es exactamente lo que calcularía SHAP para un modelo lineal (LinearExplainer
con variables independientes), sin añadir una dependencia pesada. Los aportes
de un aprendiz suman su log-odds menos el log-odds del "aprendiz promedio".

El one-hot convierte una variable en varias columnas (modalidad → modalidad_mixta,
modalidad_presencial…) y el imputador añade indicadores de faltante. Para que la
explicación hable de las variables que conoce el instructor, los aportes se
agrupan por la variable original.

Entradas:
    models/modelo.skops, data/features/train.parquet, data/features/test.parquet
Salida:
    reports/importancia_variables.json   (importancia global en test)

Uso (desde la raíz del repositorio):
    uv run python -m desercion.explain
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from desercion.modelado import RUTA_TEST, RUTA_TRAIN, cargar_modelo, separar_x_y

# --- Rutas -----------------------------------------------------------------
RUTA_IMPORTANCIA = Path("reports/importancia_variables.json")

# --- Parámetros ------------------------------------------------------------
# Prefijos que ColumnTransformer y SimpleImputer ponen a los nombres de columna.
PREFIJOS = ("numericas__missingindicator_", "numericas__", "categoricas__")

logger = logging.getLogger(__name__)


def variable_original(columna: str, originales: list[str]) -> str:
    """'categoricas__modalidad_virtual' → 'modalidad'.

    Se busca la variable original más larga que encaje, para no confundir
    'jornada' con una hipotética 'jornada_extendida'.
    """
    sin_prefijo = next(
        columna.removeprefix(p) for p in PREFIJOS if columna.startswith(p)
    )
    candidatas = [
        o for o in originales if sin_prefijo == o or sin_prefijo.startswith(o + "_")
    ]
    if not candidatas:
        raise ValueError(f"No se encontró la variable original de {columna}")
    return max(candidatas, key=len)


def matriz_transformada(pipeline: Pipeline, x: pd.DataFrame) -> pd.DataFrame:
    """X después de imputar, escalar y hacer one-hot, con nombres de columna."""
    preprocesamiento = pipeline.named_steps["preprocesamiento"]
    z = preprocesamiento.transform(x)
    if hasattr(z, "toarray"):  # el one-hot puede devolver una matriz dispersa
        z = z.toarray()
    return pd.DataFrame(
        z, columns=preprocesamiento.get_feature_names_out(), index=x.index
    )


def aportes(pipeline: Pipeline, x: pd.DataFrame, referencia: pd.Series) -> pd.DataFrame:
    """Aporte de cada variable original al log-odds de cada aprendiz.

    `referencia` es la media de la matriz transformada en train (ver
    `referencia_train`). Positivo = sube el riesgo frente al aprendiz promedio.
    """
    z = matriz_transformada(pipeline, x)
    coeficientes = pd.Series(pipeline.named_steps["modelo"].coef_[0], index=z.columns)
    por_columna = (z - referencia[z.columns]) * coeficientes
    grupos = [variable_original(c, list(x.columns)) for c in z.columns]
    return por_columna.T.groupby(grupos, sort=False).sum().T


def referencia_train(pipeline: Pipeline, x_train: pd.DataFrame) -> pd.Series:
    """El "aprendiz promedio" contra el que se comparan los aportes."""
    return matriz_transformada(pipeline, x_train).mean()


def log_odds_base(pipeline: Pipeline, referencia: pd.Series) -> float:
    """Log-odds del aprendiz promedio: base + Σ aportes = log-odds del aprendiz."""
    modelo = pipeline.named_steps["modelo"]
    return float(modelo.intercept_[0] + modelo.coef_[0] @ referencia.to_numpy())


def importancia_global(tabla_aportes: pd.DataFrame) -> pd.DataFrame:
    """Media del |aporte| en un conjunto de aprendices, de mayor a menor.

    'sube_riesgo' es la fracción de aprendices en quienes esa variable empuja
    hacia la deserción (frente al aprendiz promedio).
    """
    return (
        pd.DataFrame(
            {
                "importancia": tabla_aportes.abs().mean(),
                "sube_riesgo": (tabla_aportes > 0).mean(),
            }
        )
        .sort_values("importancia", ascending=False)
        .rename_axis("variable")
    )


def principales_factores(fila_aportes: pd.Series, n: int = 3) -> list[dict]:
    """Los n factores que más suben el riesgo de un aprendiz (para la API)."""
    suben = fila_aportes[fila_aportes > 0].sort_values(ascending=False).head(n)
    return [{"variable": v, "aporte": float(a)} for v, a in suben.items()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    modelo = cargar_modelo()
    x_train, _ = separar_x_y(pd.read_parquet(RUTA_TRAIN))
    x_test, _ = separar_x_y(pd.read_parquet(RUTA_TEST))

    referencia = referencia_train(modelo, x_train)
    tabla = aportes(modelo, x_test, referencia)

    # Comprobación: base + Σ aportes debe reproducir el log-odds del modelo.
    reconstruido = log_odds_base(modelo, referencia) + tabla.sum(axis=1)
    if not np.allclose(reconstruido, modelo.decision_function(x_test)):
        raise RuntimeError("Los aportes no reconstruyen la predicción del modelo")

    importancia = importancia_global(tabla)
    RUTA_IMPORTANCIA.parent.mkdir(parents=True, exist_ok=True)
    RUTA_IMPORTANCIA.write_text(
        json.dumps(importancia.round(4).to_dict(orient="index"), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    logger.info("%-36s %11s %11s", "variable", "importancia", "sube_riesgo")
    for variable, fila in importancia.head(10).iterrows():
        logger.info(
            "%-36s %11.3f %11.2f", variable, fila["importancia"], fila["sube_riesgo"]
        )


if __name__ == "__main__":
    main()
