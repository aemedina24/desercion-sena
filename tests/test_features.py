"""Pruebas de src/desercion/features.py.

Además de los casos borde de cada variable, aquí se prueba que no haya fuga:

- Entre filas: la variable de un aprendiz solo usa sus propios datos. Un cálculo
  con todo el DataFrame (una media, un ranking) mezclaría cohortes de train y
  test; eso debe ir en el Pipeline de la etapa train, ajustado solo con train.
- De la etiqueta: ninguna variable cambia si cambia `deserto`.
- Temporal: el test son las cohortes más recientes y train es todo lo anterior.
"""

from __future__ import annotations

import pandas as pd
import pytest

from desercion.features import (
    COHORTES_TEST,
    COLUMNAS_NO_MODELO,
    agregar_variables,
    construir,
    separar_por_cohorte,
)

# "AAAA-Tn" ordena igual como texto que en el tiempo.
TODAS_LAS_COHORTES = [
    f"{anio}-T{t}" for anio in (2023, 2024, 2025) for t in range(1, 5)
]


@pytest.fixture
def datos() -> pd.DataFrame:
    """Cuatro aprendices mínimos: dos en cohortes de train y dos en test."""
    return pd.DataFrame(
        {
            "id_aprendiz": ["a1", "a2", "a3", "a4"],
            "id_cohorte": ["2024-T1", "2025-T2", "2025-T3", "2025-T4"],
            "id_programa": ["p1", "p2", "p1", "p2"],
            "municipio_residencia": ["Medellín", "Bello", "Itagüí", "Envigado"],
            "fecha_inicio": pd.to_datetime(["2024-01-01"] * 4),
            "fecha_corte": pd.to_datetime(["2024-03-01"] * 4),
            "duracion_programa_meses": [3, 12, 3, 6],
            "dias_desde_inicio": [45, 30, 500, 0],
            "porcentaje_actividades_entregadas": [0.8, 0.0, 0.5, 1.0],
            "porcentaje_actividades_aprobadas": [0.4, 0.0, 0.5, 0.9],
            "ingresos_plataforma_ultimos_7d": [3, 0, 1, 0],
            "deserto": [0, 1, 0, 1],
        }
    )


def test_progreso_queda_entre_0_y_1(datos):
    resultado = agregar_variables(datos)
    assert resultado["progreso_programa"].between(0, 1).all()
    # a3 lleva 500 días en un curso de 3 meses: debe topar en 1
    assert resultado.loc[2, "progreso_programa"] == 1


def test_tasa_aprobacion_sin_entregas_es_cero(datos):
    resultado = agregar_variables(datos)
    # a2 no entregó nada: no debe dar error ni infinito
    assert resultado.loc[1, "tasa_aprobacion_entregas"] == 0
    assert resultado.loc[0, "tasa_aprobacion_entregas"] == pytest.approx(0.5)


def test_construir_quita_columnas_no_modelo(datos):
    train, test = construir(datos)
    for columna in COLUMNAS_NO_MODELO:
        assert columna not in train.columns
        assert columna not in test.columns


def test_ninguna_cohorte_de_test_queda_en_train(datos):
    train, test = separar_por_cohorte(datos)

    # 1. Ninguna cohorte de test se coló en train
    assert not train["id_cohorte"].isin(COHORTES_TEST).any()

    # 2. Todo lo que está en test pertenece a las cohortes de test
    assert test["id_cohorte"].isin(COHORTES_TEST).all()

    # 3. No se perdió ni se duplicó ningún aprendiz
    assert len(train) + len(test) == len(datos)


# --- Fuga entre filas y de la etiqueta -----------------------------------------
def _variables_nuevas(df: pd.DataFrame) -> list[str]:
    return [c for c in agregar_variables(df).columns if c not in df.columns]


def test_hay_variables_nuevas(procesados):
    # Si agregar_variables dejara de crear columnas, las pruebas de abajo no
    # probarían nada.
    assert _variables_nuevas(procesados)


def test_variables_de_un_aprendiz_no_dependen_de_los_demas(procesados):
    completo = agregar_variables(procesados)
    for fila in range(len(procesados)):
        sola = agregar_variables(procesados.iloc[[fila]])
        pd.testing.assert_frame_equal(
            sola.reset_index(drop=True),
            completo.iloc[[fila]].reset_index(drop=True),
        )


@pytest.mark.parametrize(
    ("columna", "otros_valores"),
    [
        ("deserto", [0, 1, 1]),  # la fixture trae [1, 0, 0]
        ("id_aprendiz", ["A-999997", "A-999998", "A-999999"]),
        ("id_cohorte", ["2023-T1", "2023-T1", "2023-T1"]),  # la fixture: 2025-T3
    ],
)
def test_variables_no_usan_etiqueta_ni_identificadores(
    procesados, columna, otros_valores
):
    nuevas = _variables_nuevas(procesados)
    base = agregar_variables(procesados)[nuevas]

    alterado = procesados.copy()
    alterado[columna] = pd.Series(otros_valores, dtype=procesados[columna].dtype)
    assert not alterado[columna].equals(procesados[columna])

    pd.testing.assert_frame_equal(agregar_variables(alterado)[nuevas], base)


def test_agregar_variables_no_cambia_las_columnas_existentes(procesados):
    resultado = agregar_variables(procesados)
    pd.testing.assert_frame_equal(resultado[procesados.columns], procesados)


# --- Fuga temporal ------------------------------------------------------------
@pytest.fixture
def una_por_cohorte() -> pd.DataFrame:
    """Un aprendiz por cada una de las 12 cohortes, de 2023-T1 a 2025-T4."""
    return pd.DataFrame(
        {
            "id_aprendiz": [f"A-{i:06d}" for i in range(len(TODAS_LAS_COHORTES))],
            "id_cohorte": TODAS_LAS_COHORTES,
        }
    )


def test_test_son_las_cohortes_mas_recientes(una_por_cohorte):
    _, test = separar_por_cohorte(una_por_cohorte)
    assert sorted(test["id_cohorte"]) == TODAS_LAS_COHORTES[-len(COHORTES_TEST) :]


def test_train_es_anterior_a_test(una_por_cohorte):
    train, test = separar_por_cohorte(una_por_cohorte)
    assert train["id_cohorte"].max() < test["id_cohorte"].min()
    assert set(train["id_aprendiz"]).isdisjoint(test["id_aprendiz"])


# --- De punta a punta con el contrato de los datos procesados -------------------
def test_construir_respeta_el_contrato(procesados):
    train, test = construir(procesados)
    salida = pd.concat([train, test])

    esperadas = (set(procesados.columns) - set(COLUMNAS_NO_MODELO)) | set(
        _variables_nuevas(procesados)
    )
    assert set(salida.columns) == esperadas
    assert len(salida) == len(procesados)


def test_construir_no_modifica_la_entrada(procesados):
    original = procesados.copy()
    construir(procesados)
    pd.testing.assert_frame_equal(procesados, original)
