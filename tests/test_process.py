"""Pruebas de las reglas de limpieza de src/desercion/process.py.

Los nombres de columna van escritos a mano a propósito, en lugar de importar
COLUMNAS_FUGA o COLUMNAS_DESCARTADAS: si alguien borrara una columna de esas
listas, una prueba basada en la misma lista seguiría pasando.
"""

from __future__ import annotations

import pandas as pd
import pytest

from desercion.process import procesar


@pytest.fixture
def resultado(crudos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return procesar(crudos)


def test_excluye_traslados_aunque_tengan_espacios(resultado):
    datos, auditoria = resultado
    # A-000003 es " traslado ": la limpieza de texto va antes de la exclusión.
    assert "A-000003" not in set(datos["id_aprendiz"])
    assert "A-000003" not in set(auditoria["id_aprendiz"])
    assert len(datos) == 3


def test_conserva_estados_que_no_son_traslado(resultado):
    datos, _ = resultado
    assert list(datos["id_aprendiz"]) == ["A-000001", "A-000002", "A-000004"]


@pytest.mark.parametrize("columna", ["estado_final", "fecha_desercion"])
def test_no_quedan_columnas_de_fuga(resultado, columna):
    datos, auditoria = resultado
    assert columna not in datos.columns
    assert columna not in auditoria.columns


def test_sexo_va_solo_a_auditoria(resultado):
    datos, auditoria = resultado
    assert "sexo" not in datos.columns
    assert list(auditoria.columns) == ["id_aprendiz", "sexo", "modalidad"]
    assert list(auditoria["sexo"]) == ["F", "M", "M"]


def test_auditoria_se_alinea_con_los_datos(resultado):
    datos, auditoria = resultado
    # La auditoría se une con las predicciones por id_aprendiz.
    assert list(auditoria["id_aprendiz"]) == list(datos["id_aprendiz"])


@pytest.mark.parametrize(
    "columna",
    [
        "nombre_programa",
        "barrio_residencia",
        "municipio_nacimiento",
        "tasa_abandono_previa",
    ],
)
def test_quita_columnas_descartadas(resultado, columna):
    datos, _ = resultado
    assert columna not in datos.columns


def test_conserva_las_columnas_del_modelo(resultado):
    datos, _ = resultado
    assert list(datos.columns) == [
        "id_aprendiz",
        "modalidad",
        "municipio_residencia",
        "tasa_finalizacion_previa",
        "deserto",
    ]


def test_limpia_espacios_del_texto(resultado):
    datos, auditoria = resultado
    assert list(datos["modalidad"]) == ["virtual", "presencial", "virtual"]
    assert list(datos["municipio_residencia"]) == ["Medellín", "Bello", "Turbo"]
    assert list(auditoria["modalidad"]) == ["virtual", "presencial", "virtual"]


def test_no_modifica_la_entrada(crudos):
    original = crudos.copy()
    procesar(crudos)
    pd.testing.assert_frame_equal(crudos, original)
