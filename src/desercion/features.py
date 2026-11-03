"""Construcción de variables y separación train/test.

Entrada:
    data/processed/aprendices.parquet
Salidas:
    data/features/train.parquet    (cohortes de entrenamiento)
    data/features/test.parquet     (cohortes más recientes)

El encoding, el escalado y la imputación NO van aquí: aprenden de los datos,
así que viven en el Pipeline de la etapa train para evitar fuga de datos.

Uso (desde la raíz del repositorio):
    uv run python -m desercion.features
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# --- Rutas -----------------------------------------------------------------
RUTA_PROCESADOS = Path("data/processed/aprendices.parquet")
RUTA_TRAIN = Path("data/features/train.parquet")
RUTA_TEST = Path("data/features/test.parquet")

# --- Parámetros ------------------------------------------------------------
DIAS_POR_MES = 30.4

# Las cohortes más recientes simulan aprendices nuevos que el modelo no ha visto.
COHORTES_TEST = ["2025-T3", "2025-T4"]

# No entran al modelo. id_aprendiz e id_cohorte se conservan para trazabilidad
# y para el split; la etapa train los excluye.
COLUMNAS_NO_MODELO = [
    "id_programa",  # demasiadas categorías: el modelo memorizaría programas
    "municipio_residencia",  # demasiadas categorías
    "fecha_inicio",  # resumida en dias_desde_inicio
    "fecha_corte",  # resumida en dias_desde_inicio (fecha_corte - fecha_inicio)
]

logger = logging.getLogger(__name__)


def cargar_datos(ruta: Path = RUTA_PROCESADOS) -> pd.DataFrame:
    """Lee los datos procesados desde Parquet."""
    return pd.read_parquet(ruta)


def agregar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables de negocio a partir de las columnas existentes."""
    df = df.copy()

    # No es lo mismo llevar 60 días en un curso de 3 meses que en uno de 24.
    duracion_dias = df["duracion_programa_meses"] * DIAS_POR_MES
    df["progreso_programa"] = (df["dias_desde_inicio"] / duracion_dias).clip(0, 1)

    # Distingue "entrega pero reprueba" de "no entrega".
    entregadas = df["porcentaje_actividades_entregadas"]
    df["tasa_aprobacion_entregas"] = (
        (df["porcentaje_actividades_aprobadas"] / entregadas)
        .where(entregadas > 0, 0.0)
        .clip(0, 1)
    )

    # Señal de alerta temprana: no entró a la plataforma en la última semana.
    df["sin_ingreso_reciente"] = df["ingresos_plataforma_ultimos_7d"].eq(0)

    return df


def quitar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las columnas que no deben llegar al modelo."""
    return df.drop(columns=COLUMNAS_NO_MODELO)


def separar_por_cohorte(
    df: pd.DataFrame, cohortes_test: list[str] = COHORTES_TEST
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa train y test por cohorte: el pasado entrena, lo reciente evalúa."""
    es_test = df["id_cohorte"].isin(cohortes_test)
    train = df[~es_test].reset_index(drop=True)
    test = df[es_test].reset_index(drop=True)
    return train, test


def construir(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica todas las transformaciones y devuelve (train, test)."""
    df = agregar_variables(df)
    df = quitar_columnas(df)
    return separar_por_cohorte(df)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    datos = cargar_datos()
    train, test = construir(datos)

    RUTA_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(RUTA_TRAIN, index=False)
    test.to_parquet(RUTA_TEST, index=False)

    logger.info("Entrada: %s filas x %s columnas", *datos.shape)
    logger.info(
        "Train: %s filas, deserción %.1f%% -> %s",
        len(train),
        100 * train["deserto"].mean(),
        RUTA_TRAIN,
    )
    logger.info(
        "Test:  %s filas, deserción %.1f%% -> %s (cohortes %s)",
        len(test),
        100 * test["deserto"].mean(),
        RUTA_TEST,
        COHORTES_TEST,
    )


if __name__ == "__main__":
    main()
