"""Pruebas del generador de datos sintéticos (generar_datos.py).

Usan la configuración real (config/data/generador.yaml) con menos aprendices,
así que corren en el CI sin necesitar data/raw.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from desercion.generar_datos import COLUMNAS, cargar_config, generar
from desercion.process import procesar
from desercion.validate import validar

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CONFIG = RAIZ / "config" / "data" / "generador.yaml"
N_APRENDICES = 1500


@pytest.fixture(scope="module")
def config() -> dict:
    return cargar_config(RUTA_CONFIG) | {"n_aprendices": N_APRENDICES}


@pytest.fixture(scope="module")
def generados(config) -> tuple[pd.DataFrame, pd.DataFrame]:
    return generar(config)


@pytest.fixture(scope="module")
def aprendices(generados) -> pd.DataFrame:
    return generados[0]


@pytest.fixture(scope="module")
def verdad(generados) -> pd.DataFrame:
    return generados[1]


# --- Reproducibilidad -------------------------------------------------------
def test_la_misma_semilla_da_los_mismos_datos(config, aprendices, verdad):
    otros_aprendices, otra_verdad = generar(config)
    pd.testing.assert_frame_equal(aprendices, otros_aprendices)
    pd.testing.assert_frame_equal(verdad, otra_verdad)


def test_otra_semilla_da_otros_datos(config, aprendices):
    otros, _ = generar(config | {"semilla": config["semilla"] + 1})
    assert not aprendices["deserto"].equals(otros["deserto"])


# --- Forma del archivo ------------------------------------------------------
def test_tiene_las_50_columnas_de_la_v1_en_el_mismo_orden(aprendices):
    assert len(COLUMNAS) == 50
    assert list(aprendices.columns) == COLUMNAS
    assert len(aprendices) == N_APRENDICES


def test_los_codigos_de_aprendiz_son_unicos(aprendices):
    assert aprendices["id_aprendiz"].is_unique


def test_cumple_el_contrato_de_datos_despues_de_process(aprendices):
    """Lo que sale del generador pasa por process.py y cumple validate.py."""
    procesados, auditoria = procesar(aprendices)
    validar(procesados)
    assert len(auditoria) == len(procesados)
    assert len(procesados) < len(aprendices)  # hay traslados y cancelaciones


# --- Etiqueta, estado final y fechas ---------------------------------------
def test_deserto_coincide_con_el_estado_final(aprendices):
    assert (
        (aprendices["deserto"] == 1) == (aprendices["estado_final"] == "deserto")
    ).all()
    assert set(aprendices["estado_final"]) == {
        "completado",
        "deserto",
        "traslado",
        "cancelado",
    }


def test_solo_quien_deserta_tiene_fecha_y_es_posterior_al_corte(aprendices):
    deserto = aprendices["deserto"] == 1
    assert aprendices.loc[~deserto, "fecha_desercion"].isna().all()
    desertores = aprendices[deserto]
    assert (desertores["fecha_desercion"] > desertores["fecha_corte"]).all()


# --- Coherencia que la v1 no cumplía ---------------------------------------
def test_sin_ingresos_recientes_si_lleva_una_semana_sin_entrar(aprendices):
    semana_sin_entrar = aprendices["dias_sin_ingreso"] >= 7
    ingresos = aprendices["ingresos_plataforma_ultimos_7d"]
    assert (ingresos[semana_sin_entrar] == 0).all()
    assert (ingresos[~semana_sin_entrar] >= 1).all()


def test_abandono_del_ultimo_curso_exige_un_curso_abandonado(aprendices):
    abandono_ultimo = aprendices["abandono_ultimo_curso"].fillna(False).astype(bool)
    assert (aprendices.loc[abandono_ultimo, "n_cursos_abandonados"] >= 1).all()


def test_los_cursos_cerrados_no_superan_los_previos(aprendices):
    cerrados = aprendices["n_cursos_completados"] + aprendices["n_cursos_abandonados"]
    assert (cerrados <= aprendices["n_cursos_previos"]).all()


# --- La verdad del generador -----------------------------------------------
def test_la_verdad_esta_alineada_con_los_aprendices(aprendices, verdad):
    assert list(verdad.columns) == [
        "id_aprendiz",
        "constancia",
        "compromiso",
        "probabilidad_desercion",
    ]
    assert verdad["id_aprendiz"].equals(aprendices["id_aprendiz"])
    assert verdad["probabilidad_desercion"].between(0, 1, inclusive="neither").all()


def test_deserta_mas_quien_tiene_mayor_probabilidad_real(aprendices, verdad):
    """La etiqueta sale de la probabilidad: la mitad de mayor riesgo deserta más."""
    probabilidad = verdad["probabilidad_desercion"]
    riesgo_alto = probabilidad > probabilidad.median()
    tasa_alta = aprendices.loc[riesgo_alto, "deserto"].mean()
    tasa_baja = aprendices.loc[~riesgo_alto, "deserto"].mean()
    assert tasa_alta > tasa_baja + 0.15
    # Y en promedio desertan tantos como dice la probabilidad.
    assert aprendices["deserto"].mean() == pytest.approx(probabilidad.mean(), abs=0.04)


def test_menos_compromiso_es_peor_seguimiento(aprendices, verdad):
    """Las variables de la semana 4 salen del compromiso oculto."""
    compromiso = verdad["compromiso"]
    assert compromiso.corr(aprendices["porcentaje_actividades_entregadas"]) > 0.6
    assert compromiso.corr(aprendices["asistencia_porcentaje"]) > 0.5
    assert compromiso.corr(aprendices["actividades_pendientes"]) < -0.5
