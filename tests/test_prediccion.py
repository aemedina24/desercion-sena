"""Pruebas de prediccion.py, recomendaciones.py y la referencia de explain.

Como en test_evaluate_explain.py, usan un Pipeline entrenado con datos
sintéticos, así que corren en el CI. Solo la última prueba usa el modelo y los
datos reales (se omite si no se ha hecho `uv run dvc pull`).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from desercion.api.schemas import RiesgoAprendiz
from desercion.explain import (
    RUTA_REFERENCIA,
    cargar_referencia,
    guardar_referencia,
    referencia_train,
)
from desercion.features import agregar_variables, quitar_columnas
from desercion.modelado import (
    RUTA_METRICAS,
    RUTA_MODELO,
    RUTA_TEST,
    cargar_modelo,
    separar_x_y,
    tipos_de_columnas,
)
from desercion.prediccion import APORTE_MINIMO, COLUMNAS_PREDICCION, predecir_riesgo
from desercion.recomendaciones import cargar_medidas, medidas_para
from desercion.train_model import cargar_config, construir_pipeline

UMBRAL = 0.4606
RUTA_EVALUACION = "reports/evaluacion.json"
RUTA_IMPORTANCIA = "reports/importancia_variables.json"


# --- Datos y modelo sintéticos ---------------------------------------------------
@pytest.fixture
def aprendices(procesados) -> pd.DataFrame:
    """60 aprendices con la forma de test.parquet; desertan quienes aprueban menos."""
    base = quitar_columnas(agregar_variables(procesados))
    df = pd.concat([base] * 20, ignore_index=True)
    df["id_aprendiz"] = [f"A-{i:06d}" for i in range(1, len(df) + 1)]
    df["porcentaje_actividades_aprobadas"] = np.linspace(0, 1, len(df))
    df["deserto"] = (df["porcentaje_actividades_aprobadas"] < 0.4).astype("int64")
    return df


@pytest.fixture
def modelo(aprendices):
    x, y = separar_x_y(aprendices)
    numericas, categoricas = tipos_de_columnas(x)
    return construir_pipeline(numericas, categoricas, cargar_config()).fit(x, y)


@pytest.fixture
def referencia(modelo, aprendices) -> pd.Series:
    x, _ = separar_x_y(aprendices)
    return referencia_train(modelo, x)


@pytest.fixture
def riesgo(aprendices, modelo, referencia) -> pd.DataFrame:
    return predecir_riesgo(
        aprendices, modelo=modelo, referencia=referencia, umbral=UMBRAL
    )


# --- Contrato con la API ----------------------------------------------------------
def test_devuelve_las_columnas_del_contrato_y_una_fila_por_aprendiz(riesgo, aprendices):
    assert list(riesgo.columns) == COLUMNAS_PREDICCION
    assert riesgo["id_aprendiz"].tolist() == aprendices["id_aprendiz"].tolist()


def test_la_etiqueta_nunca_sale(riesgo):
    assert "deserto" not in riesgo.columns


def test_cada_fila_cumple_el_esquema_de_la_api(riesgo):
    for fila in riesgo.to_dict(orient="records"):
        RiesgoAprendiz(**fila, umbral=UMBRAL)


def test_funciona_sin_la_columna_deserto(aprendices, modelo, referencia, riesgo):
    # En producción los aprendices activos no tienen etiqueta.
    sin_etiqueta = aprendices.drop(columns="deserto")
    resultado = predecir_riesgo(
        sin_etiqueta, modelo=modelo, referencia=referencia, umbral=UMBRAL
    )
    pd.testing.assert_frame_equal(resultado, riesgo)


# --- Probabilidad y alerta --------------------------------------------------------
def test_probabilidad_es_la_del_modelo(riesgo, aprendices, modelo):
    x, _ = separar_x_y(aprendices)
    np.testing.assert_allclose(riesgo["probabilidad"], modelo.predict_proba(x)[:, 1])


def test_alerta_es_probabilidad_mayor_o_igual_al_umbral(riesgo):
    assert (riesgo["alerta"] == (riesgo["probabilidad"] >= UMBRAL)).all()
    # El caso sintético tiene aprendices a ambos lados del umbral.
    assert riesgo["alerta"].any()
    assert not riesgo["alerta"].all()


# --- Factores ---------------------------------------------------------------------
def test_factores_solo_suben_el_riesgo_y_van_de_mayor_a_menor(riesgo, aprendices):
    variables = set(separar_x_y(aprendices)[0].columns)
    for factores in riesgo["factores"]:
        assert len(factores) <= 3
        aportes = [f["aporte"] for f in factores]
        assert all(a >= APORTE_MINIMO for a in aportes)
        assert aportes == sorted(aportes, reverse=True)
        assert {f["variable"] for f in factores} <= variables


def test_quien_menos_aprueba_tiene_ese_factor_primero(riesgo):
    # En los datos sintéticos la deserción depende de las actividades aprobadas.
    primero = riesgo.iloc[0]  # A-000001: aprobadas = 0
    assert primero["factores"][0]["variable"] == "porcentaje_actividades_aprobadas"


@pytest.mark.parametrize("n", [0, 6])
def test_n_factores_fuera_de_rango(aprendices, modelo, referencia, n):
    with pytest.raises(ValueError, match="n_factores"):
        predecir_riesgo(
            aprendices,
            modelo=modelo,
            referencia=referencia,
            umbral=UMBRAL,
            n_factores=n,
        )


# --- Medidas ----------------------------------------------------------------------
def test_medidas_solo_para_quienes_tienen_alerta(riesgo):
    for alerta, medidas in zip(riesgo["alerta"], riesgo["medidas"], strict=True):
        if alerta:
            assert 1 <= len(medidas) <= cargar_medidas()["max_medidas"]
        else:
            assert medidas == []


def test_medidas_sin_repetir_y_en_orden_de_aporte():
    config = cargar_medidas()
    factores = [
        {"variable": "dias_sin_ingreso", "aporte": 0.5},
        {"variable": "ingresos_plataforma_ultimos_7d", "aporte": 0.4},  # misma medida
        {"variable": "porcentaje_actividades_aprobadas", "aporte": 0.3},
    ]
    medidas = medidas_para(factores, alerta=True)
    assert medidas == [
        config["por_variable"]["dias_sin_ingreso"],
        config["por_variable"]["porcentaje_actividades_aprobadas"],
    ]


def test_medida_general_si_ningun_factor_tiene_medida():
    factores = [{"variable": "modalidad", "aporte": 0.4}]
    assert medidas_para(factores, alerta=True) == [cargar_medidas()["general"]]
    assert medidas_para([], alerta=True) == [cargar_medidas()["general"]]


def test_sin_alerta_no_hay_medidas():
    factores = [{"variable": "dias_sin_ingreso", "aporte": 0.5}]
    assert medidas_para(factores, alerta=False) == []


def test_medidas_yaml_cubre_cada_variable_del_modelo_una_vez():
    # importancia_variables.json está en Git y lista las variables del modelo.
    with open(RUTA_IMPORTANCIA, encoding="utf-8") as archivo:
        del_modelo = set(json.load(archivo))
    config = cargar_medidas()
    con_medida = set(config["por_variable"])
    sin_medida = set(config["sin_medida"])

    assert not con_medida & sin_medida
    assert con_medida | sin_medida == del_modelo, (
        "Variables sin decidir: "
        f"{sorted(del_modelo - con_medida - sin_medida)}; "
        f"variables que no existen: {sorted((con_medida | sin_medida) - del_modelo)}"
    )


def test_ninguna_medida_por_caracteristicas_personales():
    personales = {"edad", "nivel_educativo", "modalidad", "area_formacion"}
    assert not personales & set(cargar_medidas()["por_variable"])


# --- Referencia (la escribe explain, la lee la API) ---------------------------------
def test_la_referencia_se_guarda_y_se_lee_igual(referencia, tmp_path):
    ruta = tmp_path / "referencia.json"
    guardar_referencia(referencia, ruta)
    pd.testing.assert_series_equal(
        cargar_referencia(ruta), referencia, check_names=False
    )
    assert b"\r\n" not in ruta.read_bytes()  # LF también en Windows


def test_referencia_de_otro_modelo_falla_con_mensaje_claro(
    aprendices, modelo, referencia
):
    incompleta = referencia.drop(referencia.index[:2])
    with pytest.raises(ValueError, match="dvc repro explain"):
        predecir_riesgo(aprendices, modelo=modelo, referencia=incompleta, umbral=UMBRAL)


# --- Con el modelo y los datos reales ------------------------------------------------
requiere_artefactos = pytest.mark.skipif(
    not (RUTA_MODELO.exists() and RUTA_TEST.exists() and RUTA_REFERENCIA.exists()),
    reason="Faltan el modelo, test.parquet o referencia.json: ejecuta "
    "`uv run dvc pull` y `uv run dvc repro explain`",
)


@requiere_artefactos
def test_reproduce_las_alertas_de_evaluate():
    umbral = json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))["umbral"]
    with open(RUTA_EVALUACION, encoding="utf-8") as archivo:
        esperadas = json.load(archivo)["global"]["alertas"]

    riesgo = predecir_riesgo(
        pd.read_parquet(RUTA_TEST),
        modelo=cargar_modelo(),
        referencia=cargar_referencia(),
        umbral=umbral,
    )
    assert int(riesgo["alerta"].sum()) == esperadas
    for fila in riesgo.to_dict(orient="records"):
        RiesgoAprendiz(**fila, umbral=umbral)
