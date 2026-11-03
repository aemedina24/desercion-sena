import pandas as pd
import pytest

from desercion.features import (
    COHORTES_TEST,
    COLUMNAS_NO_MODELO,
    agregar_variables,
    construir,
    separar_por_cohorte,
)


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
            "porcentaje_actividades_entregadas": [80.0, 0.0, 50.0, 100.0],
            "porcentaje_actividades_aprobadas": [40.0, 0.0, 50.0, 90.0],
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
