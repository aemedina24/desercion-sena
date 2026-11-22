"""Pruebas de prediccion.py, recomendaciones.py y la referencia de explain.

Como en test_evaluate_explain.py, usan un Pipeline entrenado con datos
sintéticos, así que corren en el CI. Solo las cuatro últimas pruebas usan el modelo
y los datos reales (se omiten si no se ha hecho `uv run dvc pull`).
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
    matriz_transformada,
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
from desercion.prediccion import (
    APORTE_MINIMO,
    COLUMNAS_PREDICCION,
    agregar_valores,
    factores_con_dato,
    predecir_riesgo,
    promedios_train,
)
from desercion.recomendaciones import (
    CAMPOS_PLAN,
    MEDIDA_GENERAL,
    cargar_medidas,
    medidas_para,
    plan_para,
)
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
        {"variable": "promedio_calificaciones", "aporte": 0.3},
    ]
    medidas = medidas_para(factores, alerta=True)
    assert medidas == [
        config["por_variable"]["dias_sin_ingreso"],
        config["por_variable"]["promedio_calificaciones"],
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


# Dirección en que cada variable con medida sube el riesgo: +1 = más es peor,
# -1 = menos es peor (los booleanos cuentan como 0/1). El texto de la medida
# supone esa dirección: "no está entrando", "abandonó cursos anteriores". Por la
# colinealidad, el modelo puede darle a una variable el signo contrario, y
# entonces la marcaría como factor en quien está MEJOR que el promedio (pasó con
# porcentaje_actividades_aprobadas y tasa_finalizacion_previa con los datos v2).
DIRECCION_CON_MEDIDA = {
    "porcentaje_actividades_entregadas": -1,
    "actividades_pendientes": +1,
    "evaluaciones_no_presentadas": +1,
    "tasa_aprobacion_entregas": -1,
    "promedio_calificaciones": -1,
    "ingresos_plataforma_ultimos_7d": -1,
    "dias_sin_ingreso": +1,
    "sin_ingreso_reciente": +1,
    "asistencia_porcentaje": -1,
    "disponibilidad_equipo": -1,
    "comparte_dispositivo": +1,
    "horas_disponibles_estudio": -1,
    "horas_trabajo_semana": +1,
    "responsabilidades_cuidado": +1,
    "experiencia_virtual_previa": -1,
    "es_primer_curso": +1,
    "n_cursos_abandonados": +1,
    "solicito_apoyo": +1,
}
# En las categóricas el factor es una categoría, no una dirección: se revisan a
# mano (con los datos v2 solo salen dificultad alta, plan "otro" y conexión "otra").
CATEGORICAS_CON_MEDIDA = {
    "calidad_internet",
    "tipo_conexion",
    "tipo_plan_internet",
    "dispositivo_principal",
    "dificultad_economica_reportada",
    "situacion_laboral",
}


# --- Valor del aprendiz junto a cada factor -----------------------------------------
def test_agregar_valores_segun_el_tipo_de_dato():
    x = pd.DataFrame(
        {
            "promedio_calificaciones": [2.5, 4.0],
            "asistencia_porcentaje": [np.nan, 0.9],
            "modalidad": ["presencial", "virtual"],
            "responsabilidades_cuidado": [True, False],
        }
    )
    promedios = {
        "promedio_calificaciones": 3.8,
        "asistencia_porcentaje": 0.85,
        "responsabilidades_cuidado": 0.3,
    }
    factores = [
        [
            {"variable": "promedio_calificaciones", "aporte": 0.9},
            {"variable": "asistencia_porcentaje", "aporte": 0.2},
            {"variable": "modalidad", "aporte": 0.1},
            {"variable": "responsabilidades_cuidado", "aporte": 0.06},
        ],
        [],
    ]

    con_valores = agregar_valores(factores, x, promedios)

    nota, asistencia, modalidad, cuidado = con_valores[0]
    assert (nota["valor"], nota["promedio"]) == (2.5, 3.8)
    assert (asistencia["valor"], asistencia["promedio"]) == (None, 0.85)
    assert (modalidad["valor"], modalidad["promedio"]) == ("presencial", None)
    assert (cuidado["valor"], cuidado["promedio"]) == (1.0, 0.3)
    assert con_valores[1] == []
    assert "valor" not in factores[0][0], "No debe modificar la entrada"


def test_el_promedio_es_el_de_train(modelo, aprendices):
    x, _ = separar_x_y(aprendices)
    promedios = promedios_train(modelo)
    # Sin vacíos, la media del escalador es la media de train.
    assert promedios["porcentaje_actividades_aprobadas"] == pytest.approx(
        x["porcentaje_actividades_aprobadas"].mean()
    )
    assert not any(nombre.startswith("missingindicator_") for nombre in promedios)
    assert "modalidad" not in promedios  # categórica: sin promedio


def test_cada_factor_trae_su_valor_y_su_promedio(riesgo, aprendices, modelo):
    x, _ = separar_x_y(aprendices)
    promedios = promedios_train(modelo)
    for posicion, fila in enumerate(riesgo.itertuples()):
        for factor in fila.factores:
            crudo = x[factor["variable"]].iloc[posicion]
            if pd.isna(crudo):
                assert factor["valor"] is None
            elif isinstance(crudo, str):
                assert factor["valor"] == crudo
            else:
                assert factor["valor"] == pytest.approx(float(crudo))
            assert factor["promedio"] == promedios.get(factor["variable"])


def test_factor_con_dato_vacio_no_genera_medida():
    # Un virtual no tiene asistencia: el factor se muestra, pero sin medida.
    x = pd.DataFrame(
        {"asistencia_porcentaje": [np.nan, 0.5], "dias_sin_ingreso": [9.0, 9.0]}
    )
    asistencia = {"variable": "asistencia_porcentaje", "aporte": 0.3}
    inactividad = {"variable": "dias_sin_ingreso", "aporte": 0.1}
    factores = [[asistencia, inactividad], [asistencia, inactividad]]

    con_dato = factores_con_dato(factores, x)

    assert con_dato == [[inactividad], [asistencia, inactividad]]
    assert factores[0] == [asistencia, inactividad], "No debe modificar la entrada"
    assert medidas_para(con_dato[0], alerta=True) == medidas_para(
        [inactividad], alerta=True
    )


def test_si_el_unico_factor_esta_vacio_va_la_medida_general():
    x = pd.DataFrame({"asistencia_porcentaje": [np.nan]})
    factores = [[{"variable": "asistencia_porcentaje", "aporte": 0.3}]]
    plan = plan_para(factores_con_dato(factores, x)[0], alerta=True)
    assert [p["medida"] for p in plan] == [MEDIDA_GENERAL]


def test_cada_variable_con_medida_declara_su_direccion():
    # Una variable nueva con medida obliga a decir hacia dónde sube el riesgo.
    con_medida = set(cargar_medidas()["por_variable"])
    declaradas = set(DIRECCION_CON_MEDIDA) | CATEGORICAS_CON_MEDIDA
    assert not set(DIRECCION_CON_MEDIDA) & CATEGORICAS_CON_MEDIDA
    assert declaradas == con_medida, (
        f"Sin dirección: {sorted(con_medida - declaradas)}; "
        f"ya no tienen medida: {sorted(declaradas - con_medida)}"
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


# --- Plan de acción -----------------------------------------------------------------
# Lo que el guion nunca le dice al aprendiz: que hay una alerta, un modelo o una
# predicción, ni que podría desertar o ser sancionado (ADR 0006).
PALABRAS_PROHIBIDAS = [
    "riesgo",
    "alerta",
    "desert",
    "abandon",
    "modelo",
    "predic",
    "sanci",
]


def test_cada_medida_tiene_un_plan_completo():
    config = cargar_medidas()
    assert set(config["planes"]) == set(config["textos"])
    assert MEDIDA_GENERAL in config["planes"]
    for nombre, plan in config["planes"].items():
        assert set(plan) == {"medida", *CAMPOS_PLAN}, nombre
        assert plan["medida"] == nombre
        # Guion corto: apertura, pregunta abierta y propuesta.
        assert len(plan["decir"]) == 3, nombre
        assert any("¿" in frase for frase in plan["decir"]), nombre
        assert 1 <= plan["plazo_dias"] <= 7, nombre
        assert plan["plazo_dias"] < plan["seguimiento_dias"] <= 30, nombre


def test_el_guion_no_menciona_la_alerta_ni_la_desercion():
    for nombre, plan in cargar_medidas()["planes"].items():
        for frase in plan["decir"]:
            usadas = [p for p in PALABRAS_PROHIBIDAS if p in frase.lower()]
            assert not usadas, f"{nombre}: '{frase}' usa {usadas}"


def test_plan_va_en_el_mismo_orden_que_las_medidas():
    config = cargar_medidas()
    factores = [
        {"variable": "dias_sin_ingreso", "aporte": 0.5},
        {"variable": "ingresos_plataforma_ultimos_7d", "aporte": 0.4},  # misma medida
        {"variable": "promedio_calificaciones", "aporte": 0.3},
    ]
    plan = plan_para(factores, alerta=True)
    assert [p["medida"] for p in plan] == [
        "contacto_por_inactividad",
        "plan_de_mejoramiento",
    ]
    assert [config["textos"][p["medida"]] for p in plan] == medidas_para(
        factores, alerta=True
    )


def test_plan_general_si_ningun_factor_tiene_medida():
    plan = plan_para([{"variable": "modalidad", "aporte": 0.4}], alerta=True)
    assert [p["medida"] for p in plan] == [MEDIDA_GENERAL]
    assert plan_para([], alerta=True) == plan


def test_sin_alerta_no_hay_plan():
    factores = [{"variable": "dias_sin_ingreso", "aporte": 0.5}]
    assert plan_para(factores, alerta=False) == []


def test_modificar_un_plan_no_cambia_la_configuracion():
    factores = [{"variable": "dias_sin_ingreso", "aporte": 0.5}]
    plan = plan_para(factores, alerta=True)
    plan[0]["decir"].append("otra frase")
    plan[0]["accion"] = "otra acción"
    assert plan_para(factores, alerta=True) != plan


def test_medidas_yaml_sin_plan_falla_con_mensaje_claro(tmp_path):
    ruta = tmp_path / "medidas.yaml"
    ruta.write_text(
        "max_medidas: 3\n"
        "general: Contactar al aprendiz\n"
        "plan_general:\n"
        "  accion: Contactar al aprendiz\n"
        "medidas: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="faltan los campos"):
        cargar_medidas(ruta)


def test_el_riesgo_trae_un_plan_por_medida(riesgo):
    assert riesgo["alerta"].any()
    for _, fila in riesgo.iterrows():
        assert len(fila["plan_de_accion"]) == len(fila["medidas"])
        assert bool(fila["plan_de_accion"]) == bool(fila["alerta"])


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
        respuesta = RiesgoAprendiz(**fila, umbral=umbral)
        # Toda alerta llega con entre 1 y 3 acciones para el instructor.
        assert (1 <= len(respuesta.plan_de_accion) <= 3) == respuesta.alerta


@requiere_artefactos
def test_factores_con_medida_van_en_la_direccion_esperada():
    # Con el modelo real: si una variable con medida sale como factor, el valor
    # del aprendiz debe estar del lado malo del promedio de train. Si no, el
    # modelo le dio el signo invertido y hay que pasarla a `sin_medida`.
    # Los nulos se saltan: su factor viene del indicador de dato faltante y no
    # genera medida (prueba siguiente).
    umbral = json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))["umbral"]
    test = pd.read_parquet(RUTA_TEST)
    modelo = cargar_modelo()
    referencia = cargar_referencia()
    riesgo = predecir_riesgo(test, modelo=modelo, referencia=referencia, umbral=umbral)

    x, _ = separar_x_y(test)
    frente_al_promedio = matriz_transformada(modelo, x) - referencia
    invertidos: dict[str, int] = {}
    for posicion, fila in enumerate(riesgo.itertuples()):
        if not fila.alerta:
            continue
        for factor in fila.factores:
            variable = factor["variable"]
            if variable not in DIRECCION_CON_MEDIDA or pd.isna(
                x[variable].iloc[posicion]
            ):
                continue
            lado = frente_al_promedio[f"numericas__{variable}"].iloc[posicion]
            if np.sign(lado) != DIRECCION_CON_MEDIDA[variable]:
                invertidos[variable] = invertidos.get(variable, 0) + 1

    assert not invertidos, (
        "Factores con medida en aprendices que están mejor que el promedio "
        f"(signo invertido en el modelo): {invertidos}"
    )


@requiere_artefactos
def test_ninguna_medida_sale_de_un_dato_vacio():
    # Con el modelo real: cada medida debe venir de un factor con dato. Antes,
    # 397 virtuales con alerta recibían el guion de inasistencia porque su
    # asistencia es nula (no tienen sesiones).
    umbral = json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))["umbral"]
    test = pd.read_parquet(RUTA_TEST)
    riesgo = predecir_riesgo(
        test, modelo=cargar_modelo(), referencia=cargar_referencia(), umbral=umbral
    )
    medida_por_variable = cargar_medidas()["medida_por_variable"]
    x, _ = separar_x_y(test)

    for posicion, fila in enumerate(riesgo.itertuples()):
        activadas = {
            medida_por_variable.get(f["variable"])
            for f in fila.factores
            if pd.notna(x[f["variable"]].iloc[posicion])
        }
        for plan in fila.plan_de_accion:
            assert plan["medida"] == MEDIDA_GENERAL or plan["medida"] in activadas
    virtuales = riesgo[test["modalidad"].to_numpy() == "virtual"]
    assert not any(
        p["medida"] == "inasistencia" for ps in virtuales["plan_de_accion"] for p in ps
    )


@requiere_artefactos
def test_el_valor_de_cada_factor_esta_del_lado_malo_de_su_promedio():
    # Con el modelo real: lo que ve el instructor ("entregó 25 %; promedio
    # 73 %") debe contar lo mismo que el aporte. En las variables con dirección
    # declarada, el valor queda del lado que sube el riesgo.
    umbral = json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))["umbral"]
    riesgo = predecir_riesgo(
        pd.read_parquet(RUTA_TEST),
        modelo=cargar_modelo(),
        referencia=cargar_referencia(),
        umbral=umbral,
    )
    contrarios = {}
    revisados = 0
    for factores in riesgo["factores"]:
        for f in factores:
            if f["variable"] not in DIRECCION_CON_MEDIDA or f["valor"] is None:
                continue
            revisados += 1
            if (
                np.sign(f["valor"] - f["promedio"])
                != DIRECCION_CON_MEDIDA[f["variable"]]
            ):
                contrarios[f["variable"]] = contrarios.get(f["variable"], 0) + 1
    assert revisados > 0
    assert not contrarios, f"Valor del lado bueno del promedio: {contrarios}"
