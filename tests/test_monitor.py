"""Pruebas de monitor.py: PSI, drift por cohorte, recall por cohorte y reporte.

Usan datos sintéticos (el CI no tiene data/features). Solo la última prueba usa
los datos reales y se omite si no se ha hecho `uv run dvc pull`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from desercion.features import agregar_variables, quitar_columnas
from desercion.modelado import RUTA_TEST, RUTA_TRAIN
from desercion.monitor import (
    EPSILON,
    NULO,
    PSI_DRIFT,
    RECALL_OBJETIVO,
    RUTA_DRIFT_JSON,
    a_html,
    a_json,
    bins_numericos,
    bordes_cuantiles,
    construir_reporte,
    drift_cohorte,
    drift_variable,
    guardar,
    nivel_psi,
    psi,
    recall_por_cohorte,
    tipo_variable,
    variables_monitoreadas,
)

GENERADOR = np.random.default_rng(42)


# --- PSI ---------------------------------------------------------------------------
def test_psi_es_cero_si_las_distribuciones_son_iguales():
    p = np.array([0.2, 0.3, 0.5])
    assert psi(p, p) == pytest.approx(0.0)


def test_psi_con_un_bin_vacio_no_es_infinito():
    # Sin el mínimo EPSILON, ln(0) daría infinito.
    valor = psi(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
    assert np.isfinite(valor)
    assert valor > PSI_DRIFT


def test_psi_conocido():
    # (0.6-0.5)·ln(0.6/0.5) + (0.4-0.5)·ln(0.4/0.5)
    esperado = 0.1 * np.log(1.2) - 0.1 * np.log(0.8)
    assert psi(np.array([0.5, 0.5]), np.array([0.6, 0.4])) == pytest.approx(esperado)


@pytest.mark.parametrize(
    ("valor", "nivel"),
    [(0.0, "estable"), (0.10, "estable"), (0.15, "moderado"), (0.25, "drift")],
)
def test_nivel_psi(valor, nivel):
    assert nivel_psi(valor) == nivel


# --- Bins --------------------------------------------------------------------------
def test_bordes_son_cuantiles_de_train_sin_repetir():
    referencia = pd.Series(np.arange(100, dtype=float))
    bordes = bordes_cuantiles(referencia)
    assert len(bordes) == 9  # deciles: 9 bordes interiores
    assert np.all(np.diff(bordes) > 0)


def test_cada_bin_de_train_tiene_cerca_del_diez_por_ciento():
    referencia = pd.Series(GENERADOR.normal(size=5000))
    bins = bins_numericos(referencia, bordes_cuantiles(referencia))
    assert bins.value_counts(normalize=True).between(0.09, 0.11).all()


def test_los_nulos_van_a_su_propio_bin():
    referencia = pd.Series([1.0, 2.0, np.nan, 4.0])
    bins = bins_numericos(referencia, np.array([2.5]))
    assert bins.tolist() == ["bin_00", "bin_00", NULO, "bin_01"]


@pytest.mark.parametrize(
    ("serie", "tipo"),
    [
        (pd.Series(np.arange(100, dtype=float)), "numerica"),
        (pd.Series([0.0, 1.0, 1.0, np.nan]), "discreta"),
        (pd.Series(["virtual", "mixta"]), "categorica"),
        (pd.Series([True, False]), "categorica"),
    ],
)
def test_tipo_variable(serie, tipo):
    assert tipo_variable(serie) == tipo


# --- Drift de una variable ---------------------------------------------------------
def test_sin_cambio_no_hay_drift():
    referencia = pd.Series(GENERADOR.normal(size=8000))
    actual = pd.Series(GENERADOR.normal(size=800))
    resultado = drift_variable(referencia, actual)
    assert resultado["nivel"] == "estable"
    assert resultado["prueba"] == "ks"


def test_un_desplazamiento_marca_drift():
    referencia = pd.Series(GENERADOR.normal(size=8000))
    actual = pd.Series(GENERADOR.normal(loc=1.0, size=800))
    assert drift_variable(referencia, actual)["nivel"] == "drift"


def test_mas_nulos_tambien_es_drift():
    # El bin de nulos cuenta: si la plataforma deja de reportar asistencia, se nota.
    referencia = pd.Series(GENERADOR.normal(size=8000))
    actual = pd.Series(GENERADOR.normal(size=800))
    actual[:400] = np.nan
    assert drift_variable(referencia, actual)["nivel"] == "drift"


def test_valores_fuera_del_rango_de_train_caen_en_el_ultimo_bin():
    referencia = pd.Series(np.arange(1000, dtype=float))
    actual = pd.Series(np.full(100, 5000.0))
    resultado = drift_variable(referencia, actual)
    assert resultado["nivel"] == "drift"


def test_categoria_nueva_se_marca():
    referencia = pd.Series(["virtual"] * 50 + ["mixta"] * 50)
    actual = pd.Series(["virtual"] * 40 + ["mixta"] * 40 + ["hibrida"] * 20)
    resultado = drift_variable(referencia, actual)
    assert resultado["tipo"] == "categorica"
    assert resultado["categorias_nuevas"] == ["hibrida"]
    assert resultado["prueba"] == "chi_cuadrado"


def test_una_discreta_no_reporta_categorias_nuevas():
    referencia = pd.Series([0.0, 1.0, 2.0] * 100)
    actual = pd.Series([0.0, 1.0, 2.0, 11.0] * 25)
    assert drift_variable(referencia, actual)["categorias_nuevas"] == []


def test_categorica_sin_cambio_es_estable():
    valores = ["virtual", "mixta", "presencial"]
    referencia = pd.Series(GENERADOR.choice(valores, size=8000))
    actual = pd.Series(GENERADOR.choice(valores, size=800))
    assert drift_variable(referencia, actual)["nivel"] == "estable"


# --- Drift de una cohorte ----------------------------------------------------------
@pytest.fixture
def features(procesados) -> pd.DataFrame:
    """200 aprendices con la forma de data/features (incluye deserto)."""
    base = quitar_columnas(agregar_variables(procesados))
    df = pd.concat([base] * (200 // len(base)), ignore_index=True)
    df["id_aprendiz"] = [f"A-{i:06d}" for i in range(1, len(df) + 1)]
    df["id_cohorte"] = np.where(np.arange(len(df)) % 2 == 0, "2025-T3", "2025-T4")
    df["porcentaje_actividades_aprobadas"] = GENERADOR.uniform(size=len(df))
    df["deserto"] = (df["porcentaje_actividades_aprobadas"] < 0.4).astype("int64")
    return df


def test_nunca_se_monitorean_sexo_ni_la_etiqueta(features):
    con_sexo = features.assign(sexo="F")
    columnas = variables_monitoreadas(con_sexo).columns
    assert "sexo" not in columnas
    assert "deserto" not in columnas
    assert "id_aprendiz" not in columnas
    assert "id_cohorte" not in columnas


def test_drift_cohorte_ordena_de_mayor_a_menor_psi(features):
    actual = features.copy()
    actual["porcentaje_actividades_aprobadas"] += 2.0  # fuera del rango de train
    resultado = drift_cohorte(features, actual)

    psis = [v["psi"] for v in resultado["variables"].values()]
    assert psis == sorted(psis, reverse=True)
    assert resultado["estado"] == "drift"
    assert resultado["variables_con_drift"][0] == "porcentaje_actividades_aprobadas"


def test_drift_cohorte_contra_si_misma_es_estable(features):
    resultado = drift_cohorte(features, features)
    assert resultado["estado"] == "estable"
    assert resultado["variables_con_drift"] == []
    assert all(v["psi"] == pytest.approx(0.0) for v in resultado["variables"].values())


# --- Recall por cohorte ------------------------------------------------------------
def test_recall_por_cohorte_y_estado():
    y = pd.Series([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    cohortes = pd.Series(["T3"] * 5 + ["T4"] * 5)
    # T3 detecta 4 de 4 desertores; T4, 1 de 4.
    alerta = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0], dtype=bool)
    resultado = recall_por_cohorte(y, alerta, cohortes)

    assert resultado["T3"]["recall"] == pytest.approx(1.0)
    assert resultado["T3"]["estado"] == "ok"
    assert resultado["T4"]["recall"] == pytest.approx(0.25)
    assert resultado["T4"]["estado"] == "revisar"
    assert resultado["T4"]["alertas"] == 1
    for cohorte in resultado.values():
        assert cohorte["ic_inferior"] <= cohorte["recall"] <= cohorte["ic_superior"]


# --- Reporte -----------------------------------------------------------------------
@pytest.fixture
def reporte(features) -> dict:
    alerta = features["porcentaje_actividades_aprobadas"].to_numpy() < 0.5
    return construir_reporte(features, features, alerta)


def test_reporte_tiene_una_entrada_por_cohorte(reporte):
    assert list(reporte["cohortes"]) == ["2025-T3", "2025-T4"]
    for cohorte in reporte["cohortes"].values():
        assert {"drift", "rendimiento"} <= set(cohorte)
        assert cohorte["rendimiento"]["estado"] in {"ok", "revisar"}
    assert reporte["criterios"]["recall_objetivo"] == RECALL_OBJETIVO


def test_reporte_json_es_estable_y_valido(reporte):
    texto = a_json(reporte)
    assert texto == a_json(reporte)  # mismo reporte, mismo texto (mismo hash)
    assert json.loads(texto)["cohortes"].keys() == reporte["cohortes"].keys()
    assert (
        "deserto" not in json.loads(texto)["cohortes"]["2025-T3"]["drift"]["variables"]
    )


def test_reporte_html_muestra_cada_cohorte(reporte):
    html = a_html(reporte)
    assert "Cohorte 2025-T3" in html
    assert "Cohorte 2025-T4" in html
    assert "porcentaje_actividades_aprobadas" in html


def test_guardar_escribe_con_lf(tmp_path):
    ruta = tmp_path / "drift" / "drift.json"
    guardar("{\n}\n", ruta)
    assert b"\r\n" not in ruta.read_bytes()


def test_epsilon_es_pequeno():
    # Un bin vacío no debe pesar más que un bin con 1 aprendiz de ~770.
    assert EPSILON < 1 / 770


# --- Con los datos reales ----------------------------------------------------------
@pytest.mark.skipif(
    not (RUTA_TRAIN.exists() and RUTA_TEST.exists() and RUTA_DRIFT_JSON.exists()),
    reason="Faltan los datos o el reporte: ejecuta `uv run dvc pull` y "
    "`uv run dvc repro monitor`",
)
def test_reporte_real_cuadra_con_evaluate():
    reporte = json.loads(RUTA_DRIFT_JSON.read_text(encoding="utf-8"))
    evaluacion = json.loads(Path("reports/evaluacion.json").read_text(encoding="utf-8"))
    cohortes = reporte["cohortes"].values()

    # Las alertas y los desertores por cohorte suman lo mismo que evaluate.
    assert (
        sum(c["rendimiento"]["alertas"] for c in cohortes)
        == (evaluacion["global"]["alertas"])
    )
    assert (
        sum(c["rendimiento"]["desertores"] for c in cohortes)
        == (evaluacion["global"]["desertores"])
    )
    for cohorte in cohortes:
        assert len(cohorte["drift"]["variables"]) == 39
