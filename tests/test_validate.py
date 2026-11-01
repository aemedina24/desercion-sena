"""Pruebas del contrato de datos de src/desercion/validate.py.

Patrón: se parte de la muestra válida `procesados` (tests/conftest.py), se
rompe una sola regla y se comprueba que el esquema la rechaza por esa regla.
Así cada prueba demuestra que la regla existe y que funciona.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pandera.errors
import pytest

from desercion.validate import ESQUEMA, resumir_fallos, validar

Mutacion = Callable[[pd.DataFrame], None]


def _poner(columna: str, fila: int, valor) -> Mutacion:
    """Mutación que cambia una sola celda."""

    def mutar(df: pd.DataFrame) -> None:
        df.loc[fila, columna] = valor

    return mutar


def _fallos(df: pd.DataFrame) -> pd.DataFrame:
    """Valida df esperando que falle y devuelve la tabla de incumplimientos."""
    with pytest.raises(pandera.errors.SchemaErrors) as error:
        validar(df)
    return error.value.failure_cases


def _rechaza(df: pd.DataFrame, check: str, columna: str | None = None) -> None:
    """Comprueba que la validación falla por `check` (y en `columna`, si se da)."""
    fallos = _fallos(df)
    coincide = fallos["check"].astype(str).str.contains(check, regex=False)
    if columna is not None:
        coincide &= fallos["column"] == columna
    assert coincide.any(), f"No falló por {check!r} en {columna!r}:\n{fallos}"


# --- Datos buenos ------------------------------------------------------------
def test_acepta_datos_validos(procesados):
    pd.testing.assert_frame_equal(validar(procesados), procesados)


def test_no_modifica_la_entrada(procesados):
    original = procesados.copy()
    validar(procesados)
    pd.testing.assert_frame_equal(procesados, original)


def test_el_contrato_tiene_las_43_columnas():
    assert len(ESQUEMA.columns) == 43


# --- Columnas ----------------------------------------------------------------
@pytest.mark.parametrize(
    "columna",
    [
        "estado_final",
        "fecha_desercion",
        "sexo",
        "nombre_programa",
        "barrio_residencia",
        "municipio_nacimiento",
        "tasa_abandono_previa",
    ],
)
def test_rechaza_columnas_prohibidas(procesados, columna):
    procesados[columna] = "x"
    _rechaza(procesados, "sin_columnas_prohibidas")


def test_rechaza_columnas_extra(procesados):
    procesados["columna_nueva"] = 1
    _rechaza(procesados, "column_in_schema")


def test_rechaza_columnas_faltantes(procesados):
    fallos = _fallos(procesados.drop(columns="edad"))
    # Para una columna ausente, Pandera pone su nombre en failure_case.
    faltantes = fallos[fallos["check"] == "column_in_dataframe"]
    assert list(faltantes["failure_case"]) == ["edad"]


# --- Identificadores ---------------------------------------------------------
def test_rechaza_id_aprendiz_duplicado(procesados):
    procesados.loc[1, "id_aprendiz"] = "A-000001"
    _rechaza(procesados, "field_uniqueness", "id_aprendiz")


@pytest.mark.parametrize(
    ("columna", "valor"),
    [
        ("id_aprendiz", "A-1"),
        ("id_programa", "P101"),
        ("id_cohorte", "2025-T5"),
    ],
)
def test_rechaza_codigos_con_formato_invalido(procesados, columna, valor):
    procesados.loc[0, columna] = valor
    _rechaza(procesados, "str_matches", columna)


# --- Valores y rangos --------------------------------------------------------
def test_rechaza_etiqueta_fuera_de_0_1(procesados):
    procesados.loc[0, "deserto"] = 2
    _rechaza(procesados, "isin", "deserto")


@pytest.mark.parametrize(
    "columna",
    [
        "tasa_finalizacion_previa",
        "porcentaje_actividades_entregadas",
        "porcentaje_actividades_aprobadas",
        "asistencia_porcentaje",
    ],
)
def test_rechaza_porcentajes_en_escala_0_100(procesados, columna):
    procesados.loc[1, columna] = 85.0
    _rechaza(procesados, "in_range", columna)


@pytest.mark.parametrize(
    ("columna", "valor"),
    [
        ("edad", 13),
        ("edad", 101),
        ("promedio_calificaciones", 5.5),
        ("horas_trabajo_semana", 169),
        ("dias_desde_inicio", 40),
    ],
)
def test_rechaza_valores_fuera_de_rango(procesados, columna, valor):
    procesados.loc[1, columna] = valor
    _rechaza(procesados, "in_range", columna)


@pytest.mark.parametrize(
    ("columna", "valor"),
    [
        ("area_formacion", "deportes"),
        ("municipio_residencia", "Montería"),
        ("nivel_formacion", "técnico"),  # el contrato va en mayúsculas
        ("duracion_programa_meses", 18),
    ],
)
def test_rechaza_categorias_desconocidas(procesados, columna, valor):
    procesados.loc[1, columna] = valor
    _rechaza(procesados, "isin", columna)


# --- Nulos y tipos -----------------------------------------------------------
@pytest.mark.parametrize("columna", ["id_programa", "modalidad", "tipo_conexion"])
def test_rechaza_nulos_no_permitidos(procesados, columna):
    procesados.loc[1, columna] = None
    _rechaza(procesados, "not_nullable", columna)


def test_rechaza_entero_guardado_como_decimal(procesados):
    procesados["deserto"] = procesados["deserto"].astype("float64")
    _rechaza(procesados, "dtype", "deserto")


def test_rechaza_abandono_como_bool_en_lugar_de_boolean(procesados):
    procesados["abandono_ultimo_curso"] = [False, True, False]
    _rechaza(procesados, "dtype", "abandono_ultimo_curso")


# --- Reglas del negocio entre columnas ---------------------------------------
@pytest.mark.parametrize(
    ("mutacion", "regla"),
    [
        (_poner("asistencia_porcentaje", 1, None), "asistencia_nula_si_virtual"),
        (_poner("asistencia_porcentaje", 0, 0.5), "asistencia_nula_si_virtual"),
        (_poner("jornada", 1, "otra"), "jornada_otra_si_virtual"),
        (_poner("jornada", 0, "diurna"), "jornada_otra_si_virtual"),
        (_poner("meses_desde_ultimo_curso", 1, None), "historial_nulo_si_primer_curso"),
        (_poner("abandono_ultimo_curso", 0, True), "historial_nulo_si_primer_curso"),
        (_poner("calidad_internet", 1, None), "encuesta_nula_mismas_filas"),
        (_poner("dias_desde_inicio", 1, 29), "dias_desde_inicio"),
        (_poner("dias_sin_ingreso", 2, 22), "dias_sin_ingreso"),
        (
            _poner("porcentaje_actividades_aprobadas", 1, 0.95),
            "aprobadas_le_entregadas",
        ),
    ],
    ids=[
        "presencial_sin_asistencia",
        "virtual_con_asistencia",
        "jornada_otra_en_presencial",
        "virtual_con_jornada",
        "sin_historial_sin_ser_primer_curso",
        "primer_curso_con_historial",
        "encuesta_incompleta",
        "dias_no_cuadran_con_fechas",
        "dias_sin_ingreso_mayor_al_periodo",
        "aprobadas_mayor_que_entregadas",
    ],
)
def test_rechaza_reglas_del_negocio_incumplidas(procesados, mutacion, regla):
    mutacion(procesados)
    _rechaza(procesados, regla)


# --- Reporte de errores ------------------------------------------------------
def test_reporta_todos_los_errores_a_la_vez(procesados):
    procesados.loc[0, "deserto"] = 2
    procesados.loc[1, "edad"] = 13
    fallos = _fallos(procesados)
    assert {"deserto", "edad"} <= set(fallos["column"])


def test_resumen_cuenta_filas_en_las_reglas_entre_columnas(procesados):
    procesados.loc[1, "jornada"] = "otra"  # una fila incumple una regla
    procesados.loc[0, "edad"] = 13  # y otra fila un rango
    resumen = resumir_fallos(_fallos(procesados)).set_index("regla")
    assert resumen.loc["jornada_otra_si_virtual", "casos"] == 1
    assert resumen.loc["jornada_otra_si_virtual", "ejemplo"] == "A-000002"
    assert resumen.loc["in_range(14, 100)", "columna"] == "edad"
    assert len(resumen) == 2
