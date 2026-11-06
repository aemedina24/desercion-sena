"""Pruebas de evaluate.py, explain.py y del guardado del modelo.

No necesitan los datos reales: usan un Pipeline entrenado con datos sintéticos
construidos a partir de la fixture del contrato, así que corren también en el CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desercion.evaluate import evaluar, resumen_alertas
from desercion.explain import (
    aportes,
    importancia_global,
    log_odds_base,
    principales_factores,
    referencia_train,
    variable_original,
)
from desercion.features import agregar_variables, quitar_columnas
from desercion.modelado import (
    cargar_modelo,
    guardar_modelo,
    separar_x_y,
    tipos_de_columnas,
)
from desercion.train_model import cargar_config, construir_pipeline


@pytest.fixture
def x_y_sinteticos(procesados) -> tuple[pd.DataFrame, pd.Series]:
    """60 aprendices; desertan quienes aprueban menos."""
    base = quitar_columnas(agregar_variables(procesados))
    df = pd.concat([base] * 20, ignore_index=True)
    df["id_aprendiz"] = [f"A-{i:06d}" for i in range(len(df))]
    df["porcentaje_actividades_aprobadas"] = np.linspace(0, 1, len(df))
    df["deserto"] = (df["porcentaje_actividades_aprobadas"] < 0.4).astype("int64")
    return separar_x_y(df)


@pytest.fixture
def pipeline(x_y_sinteticos):
    x, y = x_y_sinteticos
    numericas, categoricas = tipos_de_columnas(x)
    config = cargar_config()
    return construir_pipeline(numericas, categoricas, config).fit(x, y)


# --- evaluate.py ------------------------------------------------------------
def test_resumen_alertas_con_un_caso_calculado_a_mano():
    #            desertores        no desertores
    y = pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    alerta = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])

    r = resumen_alertas(y, alerta)

    assert (r["verdaderos_positivos"], r["falsos_negativos"]) == (3, 1)
    assert (r["falsos_positivos"], r["verdaderos_negativos"]) == (2, 4)
    assert r["recall"] == pytest.approx(0.75)
    assert r["precision"] == pytest.approx(0.6)
    assert r["alertas_por_acierto"] == pytest.approx(5 / 3)
    # Al azar, alertar al 50 % encuentra en promedio 2 de los 4 desertores.
    assert r["detectados_azar"] == pytest.approx(2.0)
    assert r["mejora_sobre_azar"] == pytest.approx(1.5)


def test_resumen_alertas_sin_alertas_no_divide_por_cero():
    r = resumen_alertas(pd.Series([1, 0, 0]), np.array([0, 0, 0]))
    assert r["recall"] == 0
    assert np.isnan(r["precision"])


def test_evaluar_los_grupos_suman_el_total():
    y = pd.Series([1, 0, 1, 0, 1, 0])
    probabilidad = np.array([0.9, 0.2, 0.4, 0.6, 0.8, 0.1])
    grupos = pd.Series(["virtual", "virtual", "mixta", "mixta", "virtual", "mixta"])

    resultado = evaluar(y, probabilidad, grupos, umbral=0.5)

    assert set(resultado["modalidad"]) == {"mixta", "virtual"}
    for clave in ("aprendices", "desertores", "alertas", "verdaderos_positivos"):
        suma = sum(g[clave] for g in resultado["modalidad"].values())
        assert suma == resultado["global"][clave]


# --- Guardado del modelo ----------------------------------------------------
def test_el_modelo_guardado_predice_igual(pipeline, x_y_sinteticos, tmp_path):
    x, _ = x_y_sinteticos
    ruta = tmp_path / "modelo.skops"

    guardar_modelo(pipeline, ruta)
    recargado = cargar_modelo(ruta)

    np.testing.assert_allclose(recargado.predict_proba(x), pipeline.predict_proba(x))


# --- explain.py -------------------------------------------------------------
@pytest.mark.parametrize(
    ("columna", "esperada"),
    [
        ("numericas__edad", "edad"),
        ("numericas__missingindicator_asistencia_porcentaje", "asistencia_porcentaje"),
        ("categoricas__modalidad_virtual", "modalidad"),
        ("categoricas__area_formacion_turismo y gastronomia", "area_formacion"),
    ],
)
def test_variable_original(columna, esperada):
    originales = ["edad", "asistencia_porcentaje", "modalidad", "area_formacion"]
    assert variable_original(columna, originales) == esperada


def test_variable_original_elige_la_coincidencia_mas_larga():
    originales = ["jornada", "jornada_extendida"]
    assert (
        variable_original("categoricas__jornada_extendida_si", originales)
        == "jornada_extendida"
    )


def test_aportes_reconstruyen_la_prediccion(pipeline, x_y_sinteticos):
    x, _ = x_y_sinteticos
    referencia = referencia_train(pipeline, x)

    tabla = aportes(pipeline, x, referencia)

    # Una columna por variable original, ni más ni menos.
    assert set(tabla.columns) == set(x.columns)
    np.testing.assert_allclose(
        log_odds_base(pipeline, referencia) + tabla.sum(axis=1),
        pipeline.decision_function(x),
    )


def test_la_variable_que_define_la_etiqueta_es_la_mas_importante(
    pipeline, x_y_sinteticos
):
    x, _ = x_y_sinteticos
    tabla = aportes(pipeline, x, referencia_train(pipeline, x))

    importancia = importancia_global(tabla)

    assert importancia.index[0] == "porcentaje_actividades_aprobadas"


def test_principales_factores_solo_devuelve_los_que_suben_el_riesgo():
    fila = pd.Series({"a": 0.5, "b": -0.9, "c": 0.1, "d": 0.3, "e": 0.0})

    factores = principales_factores(fila, n=3)

    assert [f["variable"] for f in factores] == ["a", "d", "c"]
