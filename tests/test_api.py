"""Pruebas de la API (src/desercion/api).

No usan models/modelo.skops (lo versiona DVC y el CI no lo tiene): cada prueba
entrena un modelo pequeño con datos sintéticos y lo guarda en una carpeta
temporal. Así se prueba el mismo camino de carga que usa la API real.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from desercion.api.main import crear_app, md5_archivo
from desercion.api.schemas import Factor, RiesgoAprendiz, Salud
from desercion.features import agregar_variables, quitar_columnas
from desercion.modelado import guardar_modelo, separar_x_y, tipos_de_columnas
from desercion.train_model import cargar_config, construir_pipeline

UMBRAL = 0.4606


@pytest.fixture
def archivos_modelo(procesados, tmp_path):
    """(ruta del modelo, ruta de las métricas) con un modelo sintético."""
    base = quitar_columnas(agregar_variables(procesados))
    df = pd.concat([base] * 20, ignore_index=True)
    df["porcentaje_actividades_aprobadas"] = np.linspace(0, 1, len(df))
    df["deserto"] = (df["porcentaje_actividades_aprobadas"] < 0.4).astype("int64")
    x, y = separar_x_y(df)
    numericas, categoricas = tipos_de_columnas(x)
    pipeline = construir_pipeline(numericas, categoricas, cargar_config())

    ruta_modelo = tmp_path / "modelo.skops"
    ruta_metricas = tmp_path / "metricas.json"
    guardar_modelo(pipeline.fit(x, y), ruta_modelo)
    ruta_metricas.write_text(json.dumps({"umbral": UMBRAL}), encoding="utf-8")
    return ruta_modelo, ruta_metricas


# --- /health ------------------------------------------------------------------
def test_health_ok_con_modelo(archivos_modelo):
    ruta_modelo, ruta_metricas = archivos_modelo
    with TestClient(crear_app(ruta_modelo, ruta_metricas)) as cliente:
        respuesta = cliente.get("/health")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "ok"
    assert cuerpo["modelo_cargado"] is True
    assert cuerpo["umbral"] == UMBRAL
    # La versión es el md5 del archivo: el mismo dato que guarda dvc.lock.
    assert cuerpo["version_modelo"] == md5_archivo(ruta_modelo)


def test_health_503_si_falta_el_modelo(tmp_path):
    app = crear_app(tmp_path / "no_existe.skops", tmp_path / "no_existe.json")
    with TestClient(app) as cliente:
        respuesta = cliente.get("/health")

    assert respuesta.status_code == 503
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "sin_modelo"
    assert cuerpo["modelo_cargado"] is False
    assert "FileNotFoundError" in cuerpo["detalle"]


def test_health_503_si_el_archivo_no_es_un_modelo(archivos_modelo, tmp_path):
    # Un archivo cualquiera con extensión .skops no debe cargarse.
    _, ruta_metricas = archivos_modelo
    falso = tmp_path / "falso.skops"
    falso.write_bytes(b"esto no es un modelo")
    with TestClient(crear_app(falso, ruta_metricas)) as cliente:
        assert cliente.get("/health").status_code == 503


def test_health_503_si_falta_el_umbral(archivos_modelo, tmp_path):
    ruta_modelo, _ = archivos_modelo
    sin_umbral = tmp_path / "metricas.json"
    sin_umbral.write_text(json.dumps({"recall": 0.7}), encoding="utf-8")
    with TestClient(crear_app(ruta_modelo, sin_umbral)) as cliente:
        respuesta = cliente.get("/health")

    assert respuesta.status_code == 503
    assert "umbral" in respuesta.json()["detalle"]


def test_documentacion_disponible(tmp_path):
    with TestClient(crear_app(tmp_path / "x", tmp_path / "y")) as cliente:
        esquema = cliente.get("/openapi.json").json()
    assert "/health" in esquema["paths"]


# --- Esquemas -------------------------------------------------------------------
def test_riesgo_aprendiz_valido():
    riesgo = RiesgoAprendiz(
        id_aprendiz="A-000123",
        probabilidad=0.72,
        alerta=True,
        umbral=UMBRAL,
        factores=[Factor(variable="porcentaje_actividades_aprobadas", aporte=0.41)],
        medidas=["Contactar al aprendiz esta semana"],
    )
    assert riesgo.factores[0].variable == "porcentaje_actividades_aprobadas"


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("probabilidad", 1.3),
        ("probabilidad", -0.1),
        ("umbral", 0.0),
        ("id_aprendiz", "123"),
    ],
)
def test_riesgo_aprendiz_rechaza_valores_invalidos(campo, valor):
    datos = {
        "id_aprendiz": "A-000123",
        "probabilidad": 0.72,
        "alerta": True,
        "umbral": UMBRAL,
        campo: valor,
    }
    with pytest.raises(ValidationError):
        RiesgoAprendiz(**datos)


def test_factor_solo_admite_aportes_que_suben_el_riesgo():
    with pytest.raises(ValidationError):
        Factor(variable="edad", aporte=-0.2)


def test_salud_rechaza_un_estado_desconocido():
    with pytest.raises(ValidationError):
        Salud(estado="tal_vez", modelo_cargado=True)
