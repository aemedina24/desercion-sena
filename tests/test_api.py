"""Pruebas de la API (src/desercion/api).

No usan models/modelo.skops ni data/features (los versiona DVC y el CI no los
tiene). Cada prueba crea en una carpeta temporal:

- un modelo pequeño entrenado con datos sintéticos (mismo camino de carga que la
  API real) y su referencia (como la escribe `explain`),
- 60 aprendices activos en dos cohortes, con modalidad, área y nivel conocidos,
- y usa un predictor falso con probabilidades conocidas, para comprobar filtros
  y orden sin depender del modelo. El predictor real (`predecir_riesgo`) se
  prueba aparte, al final.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from desercion.api.main import (
    LIMITE_MAXIMO,
    LIMITE_POR_DEFECTO,
    crear_app,
    md5_archivo,
)
from desercion.api.schemas import Accion, Factor, RiesgoAprendiz, RiesgoCohorte, Salud
from desercion.explain import guardar_referencia, referencia_train
from desercion.features import agregar_variables, quitar_columnas
from desercion.modelado import guardar_modelo, separar_x_y, tipos_de_columnas
from desercion.recomendaciones import MEDIDA_GENERAL, cargar_medidas
from desercion.train_model import cargar_config, construir_pipeline
from desercion.validate import AREAS_FORMACION, MODALIDADES, NIVELES_FORMACION

UMBRAL = 0.4606
N_APRENDICES = 60


# --- Datos y modelo sintéticos ---------------------------------------------------
@pytest.fixture
def aprendices(procesados) -> pd.DataFrame:
    """60 aprendices con la forma de data/features/test.parquet (incluye deserto)."""
    base = quitar_columnas(agregar_variables(procesados))
    df = pd.concat([base] * (N_APRENDICES // len(base)), ignore_index=True)
    i = np.arange(len(df))
    df["id_aprendiz"] = [f"A-{n:06d}" for n in i + 1]
    df["id_cohorte"] = np.where(i % 2 == 0, "2025-T3", "2025-T4")
    df["modalidad"] = [MODALIDADES[n % 3] for n in i]
    df["area_formacion"] = [AREAS_FORMACION[n % 2] for n in i]
    df["nivel_formacion"] = [NIVELES_FORMACION[n % 2] for n in i]
    df["porcentaje_actividades_aprobadas"] = np.linspace(0, 1, len(df))
    df["deserto"] = (df["porcentaje_actividades_aprobadas"] < 0.4).astype("int64")
    return df


@pytest.fixture
def archivos(aprendices, tmp_path) -> dict:
    """Modelo, métricas, aprendices y referencia guardados como los usaría la API."""
    x, y = separar_x_y(aprendices)
    numericas, categoricas = tipos_de_columnas(x)
    pipeline = construir_pipeline(numericas, categoricas, cargar_config())

    rutas = {
        "ruta_modelo": tmp_path / "modelo.skops",
        "ruta_metricas": tmp_path / "metricas.json",
        "ruta_aprendices": tmp_path / "aprendices.parquet",
        "ruta_referencia": tmp_path / "referencia.json",
    }
    pipeline.fit(x, y)
    guardar_modelo(pipeline, rutas["ruta_modelo"])
    guardar_referencia(referencia_train(pipeline, x), rutas["ruta_referencia"])
    rutas["ruta_metricas"].write_text(json.dumps({"umbral": UMBRAL}), encoding="utf-8")
    aprendices.to_parquet(rutas["ruta_aprendices"], index=False)
    return rutas


def probabilidad_conocida(id_aprendiz: str) -> float:
    """Probabilidad del predictor falso: depende solo del número del id."""
    return (int(id_aprendiz[2:]) % 20) / 20 + 0.01


# Plan de acción del predictor falso: los campos del esquema Accion.
PLAN_FALSO = {
    "medida": "general",
    "accion": "Contactar al aprendiz",
    "plazo_dias": 5,
    "decir": ["Quiero saber cómo te has sentido en la formación."],
    "evitar": "Suponer una causa",
    "si_no_responde": "Segundo intento por otro canal",
    "registrar": "Situación que contó el aprendiz",
    "seguimiento_dias": 14,
}


def predictor_falso(aprendices: pd.DataFrame) -> pd.DataFrame:
    probabilidad = aprendices["id_aprendiz"].map(probabilidad_conocida).to_numpy()
    alerta = probabilidad >= UMBRAL
    return pd.DataFrame(
        {
            "id_aprendiz": aprendices["id_aprendiz"].to_numpy(),
            "probabilidad": probabilidad,
            "alerta": alerta,
            "factores": [
                [{"variable": "porcentaje_actividades_aprobadas", "aporte": 0.3}]
                if a
                else []
                for a in alerta
            ],
            "medidas": [
                ["Contactar al aprendiz esta semana"] if a else [] for a in alerta
            ],
            "plan_de_accion": [[dict(PLAN_FALSO)] if a else [] for a in alerta],
        }
    )


@pytest.fixture
def cliente(archivos):
    with TestClient(crear_app(**archivos, predecir=predictor_falso)) as c:
        yield c


# --- /health --------------------------------------------------------------------
def test_health_ok(cliente, archivos):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "ok"
    assert cuerpo["modelo_cargado"] is True
    assert cuerpo["umbral"] == UMBRAL
    assert cuerpo["aprendices"] == N_APRENDICES
    # La versión es el md5 del archivo: el mismo dato que guarda dvc.lock.
    assert cuerpo["version_modelo"] == md5_archivo(archivos["ruta_modelo"])


def _salud(archivos: dict, predecir=predictor_falso):
    with TestClient(crear_app(**archivos, predecir=predecir)) as c:
        return c.get("/health")


def test_health_503_si_falta_el_modelo(archivos, tmp_path):
    respuesta = _salud({**archivos, "ruta_modelo": tmp_path / "no_existe.skops"})
    assert respuesta.status_code == 503
    assert respuesta.json()["estado"] == "sin_modelo"
    assert "FileNotFoundError" in respuesta.json()["detalle"]


def test_health_503_si_el_archivo_no_es_un_modelo(archivos, tmp_path):
    falso = tmp_path / "falso.skops"
    falso.write_bytes(b"esto no es un modelo")
    assert _salud({**archivos, "ruta_modelo": falso}).status_code == 503


def test_health_503_si_falta_el_umbral(archivos):
    archivos["ruta_metricas"].write_text(json.dumps({"recall": 0.7}), encoding="utf-8")
    respuesta = _salud(archivos)
    assert respuesta.status_code == 503
    assert respuesta.json()["estado"] == "sin_modelo"
    assert "umbral" in respuesta.json()["detalle"]


def test_health_503_si_faltan_los_aprendices(archivos, tmp_path):
    respuesta = _salud({**archivos, "ruta_aprendices": tmp_path / "no.parquet"})
    assert respuesta.status_code == 503
    # El modelo sí cargó: el estado distingue qué falló.
    assert respuesta.json()["estado"] == "sin_datos"
    assert respuesta.json()["modelo_cargado"] is True
    assert "FileNotFoundError" in respuesta.json()["detalle"]


def test_health_503_si_el_predictor_no_cumple_el_contrato(archivos):
    def sin_medidas(df):
        return predictor_falso(df).drop(columns="medidas")

    respuesta = _salud(archivos, predecir=sin_medidas)
    assert respuesta.status_code == 503
    # Mensaje claro para quien opera la API, no un KeyError de pandas.
    assert "no devolvió las columnas ['medidas']" in respuesta.json()["detalle"]


def test_health_503_si_el_predictor_pierde_aprendices(archivos):
    def incompleto(df):
        return predictor_falso(df).head(10)

    respuesta = _salud(archivos, predecir=incompleto)
    assert respuesta.status_code == 503
    assert "devolvió 10 filas para 60 aprendices" in respuesta.json()["detalle"]


def test_documentacion_incluye_los_endpoints(cliente):
    rutas = cliente.get("/openapi.json").json()["paths"]
    assert {
        "/health",
        "/aprendices/{id_aprendiz}/riesgo",
        "/cohortes/{id_cohorte}/riesgo",
        "/cohortes",
        "/instructor",
    } <= set(rutas)


@pytest.mark.parametrize(
    "ruta",
    ["/health", "/aprendices/{id_aprendiz}/riesgo", "/cohortes/{id_cohorte}/riesgo"],
)
def test_documentacion_incluye_el_503(cliente, ruta):
    # Quien integre la API debe saber que puede recibir un 503 y qué significa.
    respuestas = cliente.get("/openapi.json").json()["paths"][ruta]["get"]["responses"]
    assert "503" in respuestas
    assert respuestas["503"]["description"].startswith("Modelo o datos no cargados")


def test_el_503_de_health_documenta_el_esquema_salud(cliente):
    esquema = cliente.get("/openapi.json").json()["paths"]["/health"]["get"][
        "responses"
    ]
    referencia = esquema["503"]["content"]["application/json"]["schema"]["$ref"]
    assert referencia.endswith("/Salud")


# --- /aprendices/{id}/riesgo ----------------------------------------------------
def test_riesgo_de_un_aprendiz(cliente):
    respuesta = cliente.get("/aprendices/A-000019/riesgo")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["id_aprendiz"] == "A-000019"
    assert cuerpo["probabilidad"] == pytest.approx(probabilidad_conocida("A-000019"))
    assert cuerpo["alerta"] is (cuerpo["probabilidad"] >= UMBRAL)
    assert cuerpo["umbral"] == UMBRAL
    assert cuerpo["factores"][0]["variable"] == "porcentaje_actividades_aprobadas"
    assert cuerpo["medidas"]
    assert cuerpo["plan_de_accion"] == [PLAN_FALSO]


def test_cada_factor_trae_valor_y_promedio(archivos):
    def con_valores(df):
        riesgo = predictor_falso(df)
        riesgo["factores"] = [
            [{**f, "valor": 0.25, "promedio": 0.73} for f in factores]
            for factores in riesgo["factores"]
        ]
        return riesgo

    app = crear_app(**archivos, predecir=con_valores)
    with TestClient(app) as cliente:
        factor = cliente.get("/aprendices/A-000019/riesgo").json()["factores"][0]
    assert factor == {
        "variable": "porcentaje_actividades_aprobadas",
        "aporte": 0.3,
        "valor": 0.25,
        "promedio": 0.73,
    }


def test_la_documentacion_describe_valor_y_promedio(cliente):
    factor = cliente.get("/openapi.json").json()["components"]["schemas"]["Factor"]
    assert {"valor", "promedio"} <= set(factor["properties"])


def test_sin_alerta_no_hay_plan_de_accion(cliente):
    cuerpo = cliente.get("/aprendices/A-000020/riesgo").json()  # probabilidad 0.01
    assert cuerpo["alerta"] is False
    assert cuerpo["medidas"] == []
    assert cuerpo["plan_de_accion"] == []


def test_health_503_si_el_predictor_no_entrega_el_plan_de_accion(archivos):
    def sin_plan(df):
        return predictor_falso(df).drop(columns="plan_de_accion")

    respuesta = _salud(archivos, predecir=sin_plan)
    assert respuesta.status_code == 503
    assert "['plan_de_accion']" in respuesta.json()["detalle"]


def test_la_documentacion_describe_el_plan_de_accion(cliente):
    esquemas = cliente.get("/openapi.json").json()["components"]["schemas"]
    assert set(esquemas["Accion"]["properties"]) == set(PLAN_FALSO)
    plan = esquemas["RiesgoAprendiz"]["properties"]["plan_de_accion"]
    assert plan["items"]["$ref"].endswith("/Accion")


def test_aprendiz_inexistente_404(cliente):
    respuesta = cliente.get("/aprendices/A-999999/riesgo")
    assert respuesta.status_code == 404


def test_id_de_aprendiz_mal_formado_422(cliente):
    assert cliente.get("/aprendices/123/riesgo").status_code == 422


def test_endpoints_503_si_la_api_no_esta_lista(archivos, tmp_path):
    rotos = {**archivos, "ruta_modelo": tmp_path / "no_existe.skops"}
    with TestClient(crear_app(**rotos, predecir=predictor_falso)) as c:
        assert c.get("/aprendices/A-000001/riesgo").status_code == 503
        assert c.get("/cohortes/2025-T3/riesgo").status_code == 503
        assert c.get("/cohortes").status_code == 503


# --- /cohortes y vista del instructor ---------------------------------------------
def test_lista_las_cohortes_con_aprendices_y_alertas(cliente, aprendices):
    esperado = []
    alerta = aprendices["id_aprendiz"].map(probabilidad_conocida) >= UMBRAL
    for cohorte, grupo in aprendices.groupby("id_cohorte"):
        esperado.append(
            {
                "id_cohorte": cohorte,
                "aprendices": len(grupo),
                "alertas": int(alerta[grupo.index].sum()),
            }
        )
    assert cliente.get("/cohortes").json() == esperado


def test_la_vista_del_instructor_es_una_pagina_autonoma(cliente):
    respuesta = cliente.get("/instructor")
    assert respuesta.status_code == 200
    assert respuesta.headers["content-type"].startswith("text/html")
    html = respuesta.text
    assert "vista del instructor" in html
    # Solo usa los endpoints de esta API y no carga nada de internet.
    assert "fetch(url)" in html and "/cohortes" in html
    assert "http://" not in html and "https://" not in html
    # Nunca muestra la etiqueta ni la probabilidad inflada como porcentaje.
    assert "deserto" not in html and ".probabilidad" not in html


def _claves_js(html: str, constante: str) -> set[str]:
    bloque = html.split(f"const {constante} = {{", 1)[1].split("};", 1)[0]
    return set(re.findall(r"^\s*([a-z0-9_]+):", bloque, flags=re.MULTILINE))


def test_la_vista_nombra_cada_variable_con_medida_y_cada_medida(cliente):
    html = cliente.get("/instructor").text
    medidas = cargar_medidas()
    sin_nombre = set(medidas["por_variable"]) - _claves_js(html, "ETIQUETAS")
    assert not sin_nombre, (
        f"Variables con medida sin nombre en la vista: {sorted(sin_nombre)}"
    )
    ids = {plan["medida"] for plan in medidas["planes"].values()} | {MEDIDA_GENERAL}
    sin_titulo = ids - _claves_js(html, "MEDIDAS")
    assert not sin_titulo, f"Medidas sin título en la vista: {sorted(sin_titulo)}"


# --- /cohortes/{id}/riesgo ------------------------------------------------------
def _ids(respuesta) -> list[str]:
    return [a["id_aprendiz"] for a in respuesta.json()["aprendices"]]


def test_cohorte_ordenada_de_mayor_a_menor_riesgo(cliente, aprendices):
    respuesta = cliente.get("/cohortes/2025-T3/riesgo")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()

    en_cohorte = aprendices[aprendices["id_cohorte"] == "2025-T3"]
    assert cuerpo["id_cohorte"] == "2025-T3"
    assert cuerpo["total_aprendices"] == len(en_cohorte)
    assert set(_ids(respuesta)) == set(en_cohorte["id_aprendiz"])
    probabilidades = [a["probabilidad"] for a in cuerpo["aprendices"]]
    assert probabilidades == sorted(probabilidades, reverse=True)


def test_limite_corta_la_lista_pero_no_los_totales(cliente):
    respuesta = cliente.get("/cohortes/2025-T3/riesgo", params={"limite": 5})
    cuerpo = respuesta.json()
    assert len(cuerpo["aprendices"]) == 5
    assert cuerpo["total_aprendices"] == N_APRENDICES // 2
    # Los 5 devueltos son los de mayor riesgo de la cohorte.
    completa = cliente.get("/cohortes/2025-T3/riesgo").json()["aprendices"]
    assert _ids(respuesta) == [a["id_aprendiz"] for a in completa[:5]]


@pytest.mark.parametrize(
    ("filtro", "valor"),
    [
        ("modalidad", "presencial"),
        ("area_formacion", AREAS_FORMACION[1]),
        ("nivel_formacion", NIVELES_FORMACION[0]),
    ],
)
def test_filtros_por_columna(cliente, aprendices, filtro, valor):
    respuesta = cliente.get("/cohortes/2025-T4/riesgo", params={filtro: valor})
    esperados = aprendices[
        (aprendices["id_cohorte"] == "2025-T4") & (aprendices[filtro] == valor)
    ]
    assert respuesta.status_code == 200
    assert respuesta.json()["total_aprendices"] == len(esperados)
    assert set(_ids(respuesta)) == set(esperados["id_aprendiz"])


def test_solo_alertas(cliente):
    cuerpo = cliente.get(
        "/cohortes/2025-T3/riesgo", params={"solo_alertas": True}
    ).json()
    assert cuerpo["aprendices"]
    assert all(a["alerta"] for a in cuerpo["aprendices"])
    assert cuerpo["total_aprendices"] == cuerpo["total_alertas"]


def test_filtros_combinados_sin_resultados(cliente):
    # Existe la cohorte, pero ningún aprendiz cumple todo: lista vacía, no 404.
    respuesta = cliente.get(
        "/cohortes/2025-T3/riesgo",
        params={"modalidad": "virtual", "area_formacion": AREAS_FORMACION[5]},
    )
    assert respuesta.status_code == 200
    assert respuesta.json()["total_aprendices"] == 0
    assert respuesta.json()["aprendices"] == []


def test_cohorte_sin_aprendices_activos_404(cliente):
    assert cliente.get("/cohortes/2024-T1/riesgo").status_code == 404


@pytest.mark.parametrize(
    "params",
    [
        {"modalidad": "hibrida"},
        {"nivel_formacion": "técnico"},  # el contrato va en mayúsculas
        {"limite": 0},
        {"limite": LIMITE_MAXIMO + 1},
    ],
)
def test_parametros_invalidos_422(cliente, params):
    assert cliente.get("/cohortes/2025-T3/riesgo", params=params).status_code == 422


def test_cohorte_mal_formada_422(cliente):
    assert cliente.get("/cohortes/2025-T9/riesgo").status_code == 422


def test_limite_por_defecto(cliente, archivos, aprendices, tmp_path):
    muchos = pd.concat([aprendices] * 2, ignore_index=True)
    muchos["id_aprendiz"] = [f"A-{n:06d}" for n in range(1, len(muchos) + 1)]
    muchos["id_cohorte"] = "2025-T3"
    ruta = tmp_path / "muchos.parquet"
    muchos.to_parquet(ruta, index=False)
    with TestClient(
        crear_app(**{**archivos, "ruta_aprendices": ruta}, predecir=predictor_falso)
    ) as c:
        cuerpo = c.get("/cohortes/2025-T3/riesgo").json()
    assert len(cuerpo["aprendices"]) == LIMITE_POR_DEFECTO
    assert cuerpo["total_aprendices"] == len(muchos)


# --- Privacidad: la etiqueta nunca sale de la API ---------------------------------
def test_ninguna_respuesta_expone_la_etiqueta(cliente):
    respuestas = [
        cliente.get("/health"),
        cliente.get("/aprendices/A-000001/riesgo"),
        cliente.get("/cohortes/2025-T3/riesgo"),
        cliente.get("/openapi.json"),
    ]
    for respuesta in respuestas:
        assert "deserto" not in respuesta.text


# --- Predictor real: predecir_riesgo (factores y medidas) ---------------------------
def test_predictor_real_entrega_factores_y_medidas(archivos):
    with TestClient(crear_app(**archivos)) as c:
        assert c.get("/health").json()["estado"] == "ok"
        cuerpo = c.get("/cohortes/2025-T3/riesgo", params={"limite": 100}).json()

    assert cuerpo["total_aprendices"] == N_APRENDICES // 2
    con_alerta = [a for a in cuerpo["aprendices"] if a["alerta"]]
    sin_alerta = [a for a in cuerpo["aprendices"] if not a["alerta"]]
    assert con_alerta and sin_alerta  # el caso sintético tiene de los dos

    for aprendiz in cuerpo["aprendices"]:
        assert aprendiz["alerta"] is (aprendiz["probabilidad"] >= UMBRAL)
    # Quien tiene alerta llega con sus factores, al menos una medida y un plan de
    # acción por medida; sin alerta, no hay medidas ni plan (recomendaciones.py).
    for aprendiz in con_alerta:
        assert aprendiz["factores"]
        assert aprendiz["medidas"]
        assert len(aprendiz["plan_de_accion"]) == len(aprendiz["medidas"])
        for accion in aprendiz["plan_de_accion"]:
            assert accion["accion"] and accion["decir"]
    assert all(a["medidas"] == [] for a in sin_alerta)
    assert all(a["plan_de_accion"] == [] for a in sin_alerta)


def test_quien_menos_aprueba_tiene_ese_factor_primero(archivos):
    # En los datos sintéticos la deserción depende de las actividades aprobadas, y
    # A-000001 es quien menos aprueba.
    with TestClient(crear_app(**archivos)) as c:
        cuerpo = c.get("/aprendices/A-000001/riesgo").json()
    assert cuerpo["alerta"] is True
    assert cuerpo["factores"][0]["variable"] == "porcentaje_actividades_aprobadas"


def test_health_503_sin_datos_si_falta_la_referencia(archivos, tmp_path):
    respuesta = _salud(
        {**archivos, "ruta_referencia": tmp_path / "no_existe.json"}, None
    )
    assert respuesta.status_code == 503
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "sin_datos"
    assert cuerpo["modelo_cargado"] is True
    assert "FileNotFoundError" in cuerpo["detalle"]


def test_health_503_si_la_referencia_es_de_otro_modelo(archivos):
    ruta = archivos["ruta_referencia"]
    referencia = json.loads(ruta.read_text(encoding="utf-8"))
    referencia.pop(next(iter(referencia)))
    ruta.write_text(json.dumps(referencia), encoding="utf-8")

    respuesta = _salud(archivos, None)
    assert respuesta.status_code == 503
    assert respuesta.json()["estado"] == "sin_datos"
    assert "dvc repro explain" in respuesta.json()["detalle"]


# --- Esquemas -------------------------------------------------------------------
def test_riesgo_aprendiz_valido():
    riesgo = RiesgoAprendiz(
        id_aprendiz="A-000123",
        probabilidad=0.72,
        alerta=True,
        umbral=UMBRAL,
        factores=[Factor(variable="porcentaje_actividades_aprobadas", aporte=0.41)],
        medidas=["Contactar al aprendiz esta semana"],
        plan_de_accion=[Accion(**PLAN_FALSO)],
    )
    assert riesgo.factores[0].variable == "porcentaje_actividades_aprobadas"
    assert riesgo.plan_de_accion[0].medida == "general"


def test_riesgo_aprendiz_sin_plan_de_accion_queda_vacio():
    # El campo es opcional: quien no tiene alerta no trae plan.
    riesgo = RiesgoAprendiz(
        id_aprendiz="A-000123", probabilidad=0.1, alerta=False, umbral=UMBRAL
    )
    assert riesgo.plan_de_accion == []


@pytest.mark.parametrize(
    ("campo", "valor"),
    [("plazo_dias", 0), ("seguimiento_dias", 0), ("decir", [])],
)
def test_accion_rechaza_valores_invalidos(campo, valor):
    with pytest.raises(ValidationError):
        Accion(**{**PLAN_FALSO, campo: valor})


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


def test_riesgo_cohorte_rechaza_un_id_mal_formado():
    with pytest.raises(ValidationError):
        RiesgoCohorte(
            id_cohorte="T3", total_aprendices=0, total_alertas=0, aprendices=[]
        )


def test_factor_solo_admite_aportes_que_suben_el_riesgo():
    with pytest.raises(ValidationError):
        Factor(variable="edad", aporte=-0.2)


def test_salud_rechaza_un_estado_desconocido():
    with pytest.raises(ValidationError):
        Salud(estado="tal_vez", modelo_cargado=True)
