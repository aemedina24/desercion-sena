"""Fixtures compartidas por las pruebas.

Los datos de prueba se construyen aquí, en memoria. Las pruebas nunca leen
data/raw ni data/processed: no están en Git y el CI no los tendrá.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def crudos() -> pd.DataFrame:
    """Muestra mínima con la forma de data/raw/aprendices.parquet.

    Trae las columnas que usa process.py y dos variables del modelo. Cada fila
    cubre un caso que las pruebas necesitan:

    - A-000001: completó el programa.
    - A-000002: desertó (tiene fecha_desercion) y su texto trae espacios.
    - A-000003: traslado escrito con espacios; debe excluirse igual.
    - A-000004: cancelado; no es traslado, así que se conserva.
    """
    return pd.DataFrame(
        {
            "id_aprendiz": ["A-000001", "A-000002", "A-000003", "A-000004"],
            "modalidad": ["virtual", " presencial ", "mixta", "virtual"],
            "municipio_residencia": ["Medellín", "  Bello  ", "Cali", "Turbo"],
            "sexo": ["F", "M", "F", "M"],
            "estado_final": ["completado", "deserto", " traslado ", "cancelado"],
            "fecha_desercion": pd.to_datetime([None, "2024-03-01", None, None]),
            "nombre_programa": ["Contabilidad", "Cocina", "Sistemas", "Enfermería"],
            "barrio_residencia": ["Laureles", "Niquía", "San Fernando", "Gaitán"],
            "municipio_nacimiento": ["Medellín", "Bello", "Cali", "Turbo"],
            "tasa_finalizacion_previa": [1.0, 0.5, None, 0.0],
            "tasa_abandono_previa": [0.0, 0.5, None, 1.0],
            "deserto": [0, 1, 0, 0],
        }
    )
