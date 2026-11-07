"""Pruebas de src/desercion/equidad.py con datos sintéticos (corren en el CI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desercion.equidad import (
    RATIO_MINIMO,
    alertar,
    bootstrap_recall,
    metricas_por_grupo,
    ratio_cuatro_quintos,
    umbrales_por_grupo,
)


@pytest.fixture
def dos_grupos() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grupo A: el modelo detecta 8 de 10 desertores. Grupo B: 4 de 10."""
    y = np.array([1] * 10 + [0] * 10 + [1] * 10 + [0] * 10)
    alerta = np.array(
        [1] * 8 + [0] * 2 + [1] * 3 + [0] * 7 + [1] * 4 + [0] * 6 + [1] * 1 + [0] * 9
    )
    grupos = np.array(["A"] * 20 + ["B"] * 20)
    return y, alerta, grupos


def test_metricas_por_grupo(dos_grupos):
    y, alerta, grupos = dos_grupos
    tabla = metricas_por_grupo(y, alerta, grupos)

    assert list(tabla.index) == ["A", "B"]
    assert tabla.loc["A", "recall"] == pytest.approx(0.8)
    assert tabla.loc["B", "recall"] == pytest.approx(0.4)
    assert tabla.loc["A", "precision"] == pytest.approx(8 / 11)


@pytest.mark.parametrize(
    ("recalls", "esperado"),
    [([0.8, 0.4], 0.5), ([0.7, 0.7, 0.7], 1.0), ([0.75, 0.6], 0.8)],
)
def test_ratio_cuatro_quintos(recalls, esperado):
    assert ratio_cuatro_quintos(pd.Series(recalls)) == pytest.approx(esperado)


def test_bootstrap_contiene_el_valor_observado(dos_grupos):
    y, alerta, grupos = dos_grupos
    tabla, ratio = bootstrap_recall(y, alerta, grupos, n=500)

    for _, fila in tabla.iterrows():
        assert fila["ic_inferior"] <= fila["recall"] <= fila["ic_superior"]
    assert ratio["ic_inferior"] <= ratio["ratio"] <= ratio["ic_superior"]
    assert ratio["ratio"] == pytest.approx(0.5)
    assert 0 <= ratio["prob_cumple"] <= 1


def test_bootstrap_es_reproducible(dos_grupos):
    y, alerta, grupos = dos_grupos
    a = bootstrap_recall(y, alerta, grupos, n=200, semilla=7)
    b = bootstrap_recall(y, alerta, grupos, n=200, semilla=7)
    pd.testing.assert_frame_equal(a[0], b[0])
    assert a[1] == b[1]


def test_intervalo_mas_angosto_con_mas_datos(dos_grupos):
    y, alerta, grupos = dos_grupos
    chico, _ = bootstrap_recall(y, alerta, grupos, n=500)
    grande, _ = bootstrap_recall(
        np.tile(y, 10), np.tile(alerta, 10), np.tile(grupos, 10), n=500
    )
    ancho_grande = (grande["ic_superior"] - grande["ic_inferior"]).to_numpy()
    ancho_chico = (chico["ic_superior"] - chico["ic_inferior"]).to_numpy()
    assert (ancho_grande < ancho_chico).all()


def test_grupos_iguales_cumplen_la_regla():
    generador = np.random.default_rng(0)
    y = generador.integers(0, 2, 2000)
    alerta = y.copy()  # recall 1 en ambos grupos
    grupos = np.array(["A", "B"] * 1000)
    _, ratio = bootstrap_recall(y, alerta, grupos, n=200)
    assert ratio["ratio"] == pytest.approx(1.0)
    assert ratio["prob_cumple"] == 1.0
    assert ratio["ic_inferior"] >= RATIO_MINIMO


def test_umbrales_por_grupo_alcanzan_el_recall_objetivo():
    generador = np.random.default_rng(1)
    grupos = np.array(["A"] * 300 + ["B"] * 300)
    y = generador.integers(0, 2, 600)
    # En B las probabilidades son más bajas: con un umbral único, B tendría
    # menos recall. Con su propio umbral, ambos llegan al objetivo.
    desplazamiento = np.where(grupos == "B", -0.2, 0.0)
    probabilidad = np.clip(
        0.3 * y + generador.uniform(0.2, 0.7, 600) + desplazamiento, 0, 1
    )

    umbrales = umbrales_por_grupo(y, probabilidad, grupos, 0.7)
    assert set(umbrales) == {"A", "B"}
    assert umbrales["B"] < umbrales["A"]

    alerta = alertar(probabilidad, grupos, umbrales)
    tabla = metricas_por_grupo(y, alerta, grupos)
    assert (tabla["recall"] >= 0.7).all()


def test_alertar_con_umbral_unico_o_por_grupo():
    probabilidad = np.array([0.3, 0.5, 0.3, 0.5])
    grupos = np.array(["A", "A", "B", "B"])
    np.testing.assert_array_equal(alertar(probabilidad, grupos, 0.4), [0, 1, 0, 1])
    np.testing.assert_array_equal(
        alertar(probabilidad, grupos, {"A": 0.6, "B": 0.2}), [0, 0, 1, 1]
    )
