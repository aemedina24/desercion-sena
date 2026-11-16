"""Pruebas de calibracion.py.

Usan probabilidades construidas a mano, así que corren en el CI sin datos.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from desercion import calibracion
from desercion.calibracion import (
    brier,
    comparar_con_la_verdad,
    corregir_por_balanceo,
    curva_calibracion,
    desplazamiento_por_balanceo,
    ece,
    figura,
    resumen,
)

TASA_TRAIN = 0.32


@pytest.fixture
def calibrado() -> tuple[np.ndarray, np.ndarray]:
    """1.000 aprendices cuya probabilidad es exactamente la real."""
    rng = np.random.default_rng(7)
    probabilidad = rng.uniform(0.02, 0.95, size=1000)
    y_real = (rng.random(1000) < probabilidad).astype(int)
    return y_real, probabilidad


# --- Desplazamiento y corrección ---------------------------------------------
def test_sin_desbalance_no_hay_nada_que_corregir():
    assert desplazamiento_por_balanceo(0.5) == pytest.approx(0)


def test_el_desplazamiento_crece_cuando_la_clase_es_mas_rara():
    assert desplazamiento_por_balanceo(0.2) == pytest.approx(np.log(4))
    assert desplazamiento_por_balanceo(0.1) > desplazamiento_por_balanceo(0.2)


@pytest.mark.parametrize("tasa", [0, 1, -0.1, 1.5])
def test_tasa_imposible_falla(tasa):
    with pytest.raises(ValueError, match="entre 0 y 1"):
        desplazamiento_por_balanceo(tasa)


def test_corregir_baja_la_probabilidad_sin_cambiar_el_orden(calibrado):
    _, probabilidad = calibrado
    corregida = corregir_por_balanceo(probabilidad, TASA_TRAIN)

    assert (corregida < probabilidad).all()
    assert (0 < corregida).all() and (corregida < 1).all()
    # Lo que decide las alertas es el orden, y el orden no se toca.
    np.testing.assert_array_equal(np.argsort(corregida), np.argsort(probabilidad))


# --- Curva, ECE y Brier --------------------------------------------------------
def test_la_curva_reparte_a_todos_en_grupos_parejos(calibrado):
    y_real, probabilidad = calibrado
    curva = curva_calibracion(y_real, probabilidad, n_grupos=10)

    assert len(curva) == 10
    assert sum(g["aprendices"] for g in curva) == len(y_real)
    assert [g["grupo"] for g in curva] == list(range(1, 11))
    # Los grupos van de menor a mayor probabilidad y no se solapan.
    assert [g["predicho"] for g in curva] == sorted(g["predicho"] for g in curva)
    for anterior, siguiente in pairwise(curva):
        assert anterior["hasta"] <= siguiente["desde"]


def test_con_muchos_empates_salen_menos_grupos():
    y_real = np.array([0, 1, 0, 1, 1, 0])
    probabilidad = np.array([0.3, 0.3, 0.3, 0.9, 0.9, 0.9])
    curva = curva_calibracion(y_real, probabilidad, n_grupos=10)

    assert len(curva) == 2
    assert sum(g["aprendices"] for g in curva) == len(y_real)


def test_tamanos_distintos_fallan():
    with pytest.raises(ValueError, match="mismo tamaño"):
        curva_calibracion(np.array([0, 1]), np.array([0.5]))


def test_un_modelo_bien_calibrado_tiene_ece_y_brier_bajos(calibrado):
    y_real, probabilidad = calibrado
    assert ece(curva_calibracion(y_real, probabilidad)) < 0.05
    # Mejor que predecirle a todos la tasa de deserción.
    constante = np.full_like(probabilidad, y_real.mean())
    assert brier(y_real, probabilidad) < brier(y_real, constante)


def test_brier_de_una_prediccion_perfecta_es_cero():
    y_real = np.array([0, 1, 1])
    assert brier(y_real, y_real.astype(float)) == 0
    assert brier(y_real, 1 - y_real.astype(float)) == 1


def test_ece_pesa_cada_grupo_por_su_tamano():
    curva = [
        {"aprendices": 90, "predicho": 0.5, "observado": 0.5},
        {"aprendices": 10, "predicho": 0.5, "observado": 1.0},
    ]
    assert ece(curva) == pytest.approx(0.05)


# --- Resumen -------------------------------------------------------------------
def test_corregir_no_cambia_las_alertas_ni_el_recall(calibrado):
    """La corrección solo desplaza: con el umbral equivalente alerta a los mismos."""
    y_real, probabilidad = calibrado
    umbral = 0.6
    corregida = corregir_por_balanceo(probabilidad, TASA_TRAIN)
    umbral_corregido = float(corregir_por_balanceo(np.array([umbral]), TASA_TRAIN)[0])

    antes = resumen(y_real, probabilidad, umbral)
    despues = resumen(y_real, corregida, umbral_corregido)

    assert antes["alertas"] == despues["alertas"]
    assert antes["recall"] == pytest.approx(despues["recall"])
    assert antes["tasa_real_en_alertas"] == pytest.approx(
        despues["tasa_real_en_alertas"]
    )
    # Lo que sí cambia: el número que ve el instructor.
    assert (
        despues["probabilidad_media_en_alertas"]
        < antes["probabilidad_media_en_alertas"]
    )


def test_el_resumen_reporta_la_tasa_real(calibrado):
    y_real, probabilidad = calibrado
    r = resumen(y_real, probabilidad, 0.5)
    assert r["tasa_real"] == pytest.approx(y_real.mean())
    assert r["umbral"] == 0.5


# --- Figura y verdad del generador ---------------------------------------------
def test_la_figura_se_escribe(tmp_path, calibrado):
    y_real, probabilidad = calibrado
    ruta = tmp_path / "figuras" / "calibracion.png"
    figura(y_real, probabilidad, corregir_por_balanceo(probabilidad, TASA_TRAIN), ruta)
    assert ruta.stat().st_size > 0


def test_sin_verdad_no_se_compara(monkeypatch, tmp_path, calibrado):
    # Con los datos de la v1 no existe verdad.parquet: el reporte lo deja en null.
    monkeypatch.setattr(calibracion, "RUTA_VERDAD", tmp_path / "no_existe.parquet")
    _, probabilidad = calibrado
    assert (
        comparar_con_la_verdad(pd.Series(["A-1"]), probabilidad, probabilidad) is None
    )


def test_la_correccion_acerca_la_probabilidad_a_la_real(monkeypatch, tmp_path):
    """Con el generador v2 se puede medir contra la probabilidad que usó el sorteo."""
    rng = np.random.default_rng(3)
    real = rng.uniform(0.05, 0.6, size=200)
    ids = pd.Series([f"A-{i:06d}" for i in range(200)])
    inflada = corregir_por_balanceo(real, 1 - TASA_TRAIN)  # el efecto contrario

    ruta = tmp_path / "verdad.parquet"
    pd.DataFrame({"id_aprendiz": ids, "probabilidad_desercion": real}).to_parquet(ruta)
    monkeypatch.setattr(calibracion, "RUTA_VERDAD", ruta)

    comparacion = comparar_con_la_verdad(
        ids, inflada, corregir_por_balanceo(inflada, TASA_TRAIN)
    )
    assert comparacion["probabilidad_real_media"] == pytest.approx(real.mean())
    assert (
        comparacion["error_absoluto_medio_corregida"]
        < comparacion["error_absoluto_medio"]
    )
    assert comparacion["correlacion_de_orden"] == pytest.approx(1)
