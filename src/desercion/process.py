"""Limpieza de los datos crudos de aprendices.

Entrada:
    data/raw/aprendices.parquet          (50 columnas)
Salidas:
    data/processed/aprendices.parquet    (datos listos para construir variables)
    data/processed/auditoria.parquet     (id_aprendiz, sexo, modalidad: solo para equidad)

Uso (desde la raíz del repositorio):
    uv run python -m desercion.process
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# --- Rutas -----------------------------------------------------------------
RUTA_RAW = Path("data/raw/aprendices.parquet")
RUTA_PROCESADOS = Path("data/processed/aprendices.parquet")
RUTA_AUDITORIA = Path("data/processed/auditoria.parquet")

# --- Reglas de limpieza (documentadas en docs/datos.md) --------------------
# Un traslado no es abandono: se excluye del estudio.
ESTADOS_EXCLUIDOS = ["traslado"]

# Tabla aparte para auditar la equidad; el modelo nunca la ve.
COLUMNAS_AUDITORIA = ["id_aprendiz", "sexo", "modalidad"]
COLUMNAS_SOLO_AUDITORIA = ["sexo"]

# Describen el resultado: si entran al modelo hay fuga de datos.
COLUMNAS_FUGA = ["estado_final", "fecha_desercion"]

# Redundantes o con riesgo de reidentificación.
COLUMNAS_DESCARTADAS = [
    "nombre_programa",  # repite id_programa
    "barrio_residencia",  # muchas categorías y riesgo de reidentificar
    "municipio_nacimiento",  # puede revelar desplazamiento
    "tasa_abandono_previa",  # es 1 - tasa_finalizacion_previa
]

logger = logging.getLogger(__name__)


def cargar_datos(ruta: Path = RUTA_RAW) -> pd.DataFrame:
    """Lee los datos crudos desde Parquet."""
    return pd.read_parquet(ruta)


def limpiar_texto(df: pd.DataFrame) -> pd.DataFrame:
    """Quita espacios al inicio y al final de todas las columnas de texto."""
    df = df.copy()
    columnas_texto = df.select_dtypes(include=["object", "string"]).columns
    for columna in columnas_texto:
        df[columna] = df[columna].str.strip()
    return df


def excluir_estados(
    df: pd.DataFrame, estados: list[str] = ESTADOS_EXCLUIDOS
) -> pd.DataFrame:
    """Elimina los registros cuyo estado final no corresponde al estudio."""
    return df[~df["estado_final"].isin(estados)].reset_index(drop=True)


def separar_auditoria(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa las columnas de auditoría y las quita de los datos del modelo."""
    auditoria = df[COLUMNAS_AUDITORIA].copy()
    datos = df.drop(columns=COLUMNAS_SOLO_AUDITORIA)
    return datos, auditoria


def procesar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica todas las reglas de limpieza y devuelve (datos, auditoria)."""
    df = limpiar_texto(df)
    df = excluir_estados(df)  # va antes de quitar estado_final, porque lo usa
    datos, auditoria = separar_auditoria(df)
    datos = datos.drop(columns=COLUMNAS_FUGA + COLUMNAS_DESCARTADAS)
    return datos, auditoria


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    crudos = cargar_datos()
    datos, auditoria = procesar(crudos)

    RUTA_PROCESADOS.parent.mkdir(parents=True, exist_ok=True)
    datos.to_parquet(RUTA_PROCESADOS, index=False)
    auditoria.to_parquet(RUTA_AUDITORIA, index=False)

    logger.info("Crudos: %s filas x %s columnas", *crudos.shape)
    logger.info(
        "Excluidos por estado %s: %s", ESTADOS_EXCLUIDOS, len(crudos) - len(datos)
    )
    logger.info(
        "Procesados: %s filas x %s columnas -> %s", *datos.shape, RUTA_PROCESADOS
    )
    logger.info("Auditoría: %s filas -> %s", len(auditoria), RUTA_AUDITORIA)
    logger.info("Tasa de deserción: %.1f%%", 100 * datos["deserto"].mean())


if __name__ == "__main__":
    main()
