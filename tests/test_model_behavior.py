"""Pruebas de comportamiento del modelo final (regresión logística, ADR 0002).

Dos niveles:

1. Sin datos (corren siempre, también en el CI): el umbral registrado en
   reports/metricas_modelo_base.json, la elección del umbral y la robustez del
   Pipeline con datos sintéticos construidos a partir de la fixture del contrato.
2. Con el modelo real: necesitan data/features/*.parquet (`uv run dvc pull` y
   `uv run dvc repro features`). Si no están, como en el CI, se saltan con aviso.
   Cubren rendimiento mínimo, dirección esperada de las variables, invariancia y
   fuga.

El umbral no es 0.5: se lee de reports/metricas_modelo_base.json (clave "umbral"),
que lo elige la etapa train con predicciones out-of-fold en train.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from desercion.features import agregar_variables, quitar_columnas
from desercion.modelado import (
    COLUMNA_GRUPO,
    elegir_umbral,
    separar_x_y,
    tipos_de_columnas,
)
from desercion.train_model import calcular_metricas, cargar_config, construir_pipeline

RAIZ = Path(__file__).resolve().parents[1]
RUTA_METRICAS = RAIZ / "reports" / "metricas_modelo_base.json"
RUTA_CONFIG = RAIZ / "config" / "model" / "logistic.yaml"
RUTA_TRAIN = RAIZ / "data" / "features" / "train.parquet"
RUTA_TEST = RAIZ / "data" / "features" / "test.parquet"

# Pisos de rendimiento en test: barandas para detectar un modelo roto, no metas.
# Valores actuales (ADR 0002): ROC-AUC 0.735, PR-AUC 0.576, recall 0.686 y
# recall presencial 0.525. Si un cambio los baja de aquí, hay que explicarlo.
ROC_AUC_MINIMO = 0.70
PR_AUC_MINIMO = 0.50
RECALL_MINIMO = 0.60
RECALL_MINIMO_POR_MODALIDAD = 0.45

# Columnas que nunca deben llegar al modelo (CLAUDE.md).
COLUMNAS_PROHIBIDAS = {
    "deserto",
    "estado_final",
    "fecha_desercion",
    "sexo",
    "id_aprendiz",
    "id_cohorte",
}

requiere_datos = pytest.mark.skipif(
    not (RUTA_TRAIN.exists() and RUTA_TEST.exists()),
    reason="Faltan data/features/*.parquet: ejecuta `uv run dvc pull` y "
    "`uv run dvc repro features`",
)


@pytest.fixture(scope="module")
def metricas_registradas() -> dict:
    return json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def config() -> dict:
    return cargar_config(RUTA_CONFIG)


# --- 1. Sin datos: umbral y elección del umbral -----------------------------
def test_umbral_registrado_no_es_el_por_defecto(metricas_registradas):
    umbral = metricas_registradas["umbral"]
    assert 0 < umbral < 1
    assert not np.isclose(umbral, 0.5), "El umbral debe venir de out-of-fold, no 0.5"


def test_umbral_registrado_cumple_el_recall_objetivo(metricas_registradas, config):
    assert metricas_registradas["recall_oof"] >= config["recall_objetivo"]


@pytest.fixture
def puntajes_sinteticos() -> tuple[pd.Series, np.ndarray]:
    """200 aprendices con puntajes que separan las clases solo en parte."""
    generador = np.random.default_rng(0)
    y = pd.Series(generador.integers(0, 2, 200))
    probabilidad = np.clip(0.3 * y + generador.uniform(0, 0.7, 200), 0, 1)
    return y, probabilidad


@pytest.mark.parametrize("recall_objetivo", [0.5, 0.7, 0.9])
def test_elegir_umbral_alcanza_el_recall_objetivo(puntajes_sinteticos, recall_objetivo):
    y, probabilidad = puntajes_sinteticos
    eleccion = elegir_umbral(y, probabilidad, recall_objetivo)

    alerta = (probabilidad >= eleccion["umbral"]).astype(int)
    assert recall_score(y, alerta) >= recall_objetivo
    assert recall_score(y, alerta) == pytest.approx(eleccion["recall_oof"])


def test_elegir_umbral_toma_la_mayor_precision_posible(puntajes_sinteticos):
    y, probabilidad = puntajes_sinteticos
    eleccion = elegir_umbral(y, probabilidad, 0.7)

    # Ningún otro umbral que también cumpla el recall tiene mejor precisión.
    for umbral in np.unique(probabilidad):
        alerta = probabilidad >= umbral
        if recall_score(y, alerta) >= 0.7:
            assert y[alerta].mean() <= eleccion["precision_oof"] + 1e-12


def test_elegir_umbral_rechaza_un_objetivo_imposible(puntajes_sinteticos):
    y, probabilidad = puntajes_sinteticos
    with pytest.raises(ValueError):
        elegir_umbral(y, probabilidad, 1.01)


# --- 1. Sin datos: robustez del Pipeline con datos sintéticos ---------------
@pytest.fixture
def datos_sinteticos(procesados) -> pd.DataFrame:
    """60 filas con la forma de data/features, a partir de la fixture del contrato.

    Desertan quienes aprueban menos: el modelo tiene una señal que aprender.
    """
    base = quitar_columnas(agregar_variables(procesados))
    df = pd.concat([base] * 20, ignore_index=True)
    df["id_aprendiz"] = [f"A-{i:06d}" for i in range(len(df))]
    df["porcentaje_actividades_aprobadas"] = np.linspace(0, 1, len(df))
    df["deserto"] = (df["porcentaje_actividades_aprobadas"] < 0.4).astype("int64")
    return df


@pytest.fixture
def pipeline_sintetico(datos_sinteticos, config):
    x, y = separar_x_y(datos_sinteticos)
    numericas, categoricas = tipos_de_columnas(x)
    return construir_pipeline(numericas, categoricas, config).fit(x, y)


def test_separar_x_y_deja_fuera_etiqueta_e_identificadores(datos_sinteticos):
    x, y = separar_x_y(datos_sinteticos)
    assert not COLUMNAS_PROHIBIDAS & set(x.columns)
    assert y.name == "deserto"
    assert x.select_dtypes(include=["bool", "boolean"]).empty


def test_pipeline_tolera_categorias_nuevas_y_nulos(
    datos_sinteticos, pipeline_sintetico
):
    # En la API puede llegar un valor que train nunca vio, o un dato faltante.
    x, _ = separar_x_y(datos_sinteticos.head(1))
    x["tipo_conexion"] = "satelital"
    x["calidad_internet"] = np.nan
    x["promedio_calificaciones"] = np.nan

    probabilidad = pipeline_sintetico.predict_proba(x)[:, 1]
    assert np.isfinite(probabilidad).all()
    assert ((probabilidad >= 0) & (probabilidad <= 1)).all()


# --- 2. Con el modelo real ----------------------------------------------------
@pytest.fixture(scope="module")
def modelo_real(config):
    """Entrena el modelo final igual que la etapa train y predice sobre test."""
    train = pd.read_parquet(RUTA_TRAIN)
    test = pd.read_parquet(RUTA_TEST)
    x_train, y_train = separar_x_y(train)
    x_test, y_test = separar_x_y(test)
    numericas, categoricas = tipos_de_columnas(x_train)
    pipeline = construir_pipeline(numericas, categoricas, config)
    pipeline.fit(x_train, y_train)
    return {
        "pipeline": pipeline,
        "train": train,
        "x_train": x_train,
        "y_train": y_train,
        "test": test,
        "x_test": x_test,
        "y_test": y_test,
        "probabilidad": pipeline.predict_proba(x_test)[:, 1],
    }


@requiere_datos
def test_reproduce_las_metricas_registradas(modelo_real, metricas_registradas):
    metricas = calcular_metricas(
        modelo_real["y_test"],
        modelo_real["probabilidad"],
        metricas_registradas["umbral"],
    )
    for nombre, valor in metricas.items():
        assert valor == pytest.approx(metricas_registradas[nombre]), nombre


@requiere_datos
def test_reproduce_el_umbral_out_of_fold(modelo_real, metricas_registradas, config):
    probabilidad_oof = cross_val_predict(
        modelo_real["pipeline"],
        modelo_real["x_train"],
        modelo_real["y_train"],
        groups=modelo_real["train"][COLUMNA_GRUPO],
        cv=GroupKFold(n_splits=config["cv_n_splits"]),
        method="predict_proba",
    )[:, 1]
    eleccion = elegir_umbral(
        modelo_real["y_train"], probabilidad_oof, config["recall_objetivo"]
    )
    assert eleccion["umbral"] == pytest.approx(metricas_registradas["umbral"])


@requiere_datos
def test_rendimiento_minimo_en_test(modelo_real, metricas_registradas):
    y, probabilidad = modelo_real["y_test"], modelo_real["probabilidad"]
    alerta = (probabilidad >= metricas_registradas["umbral"]).astype(int)

    assert roc_auc_score(y, probabilidad) >= ROC_AUC_MINIMO
    pr_auc = average_precision_score(y, probabilidad)
    assert pr_auc >= PR_AUC_MINIMO
    assert pr_auc >= 1.5 * y.mean(), "Apenas mejor que alertar al azar"
    assert recall_score(y, alerta) >= RECALL_MINIMO


@requiere_datos
@pytest.mark.parametrize("modalidad", ["mixta", "presencial", "virtual"])
def test_recall_minimo_por_modalidad(modelo_real, metricas_registradas, modalidad):
    es_grupo = modelo_real["x_test"]["modalidad"] == modalidad
    alerta = modelo_real["probabilidad"] >= metricas_registradas["umbral"]
    recall = recall_score(modelo_real["y_test"][es_grupo], alerta[es_grupo])
    assert recall >= RECALL_MINIMO_POR_MODALIDAD


# Dirección esperada (+1 = sube el riesgo). No se incluye
# porcentaje_actividades_entregadas: por su correlación de 0.95 con aprobadas su
# coeficiente sale positivo (cuaderno 02), así que su efecto aislado no se interpreta.
DIRECCIONES = [
    ("porcentaje_actividades_aprobadas", 0.2, -1),
    ("promedio_calificaciones", 0.5, -1),
    ("ingresos_plataforma_ultimos_7d", 3, -1),
    ("evaluaciones_no_presentadas", 1, +1),
    ("dias_sin_ingreso", 5, +1),
]


@requiere_datos
@pytest.mark.parametrize(("columna", "aumento", "signo"), DIRECCIONES)
def test_direccion_esperada_de_las_variables(modelo_real, columna, aumento, signo):
    x = modelo_real["x_test"]
    modificado = x.copy()
    modificado[columna] = modificado[columna] + aumento

    antes = modelo_real["probabilidad"]
    despues = modelo_real["pipeline"].predict_proba(modificado)[:, 1]
    cambiaron = x[columna].notna().to_numpy()

    assert cambiaron.any()
    assert (np.sign(despues - antes)[cambiaron] == signo).all()


@requiere_datos
def test_identificadores_no_cambian_la_prediccion(modelo_real):
    test = modelo_real["test"].copy()
    test["id_aprendiz"] = test["id_aprendiz"].iloc[::-1].to_numpy()
    test["id_cohorte"] = "2023-T1"
    x, _ = separar_x_y(test)

    despues = modelo_real["pipeline"].predict_proba(x)[:, 1]
    np.testing.assert_array_equal(despues, modelo_real["probabilidad"])


@requiere_datos
def test_prediccion_individual_igual_a_la_del_lote(modelo_real):
    # La API predice de a un aprendiz: debe dar lo mismo que en lote.
    x = modelo_real["x_test"]
    for fila in [0, len(x) // 2, len(x) - 1]:
        individual = modelo_real["pipeline"].predict_proba(x.iloc[[fila]])[0, 1]
        assert individual == pytest.approx(modelo_real["probabilidad"][fila])


@requiere_datos
def test_ninguna_variable_predice_sola_la_etiqueta(modelo_real):
    # Una variable que casi separa las clases por sí sola suele ser una fuga.
    # La más fuerte hoy es porcentaje_actividades_aprobadas (AUC 0.72).
    x, y = modelo_real["x_train"], modelo_real["y_train"]
    numericas, _ = tipos_de_columnas(x)
    for columna in numericas:
        valores = x[columna].fillna(x[columna].median())
        auc = roc_auc_score(y, valores)
        assert 0.1 < auc < 0.9, f"{columna}: AUC univariada {auc:.3f}"
