"""Pruebas de comparar_modelos.py con datos sintéticos (corren en el CI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desercion.comparar_modelos import N_FOLDS, construir_modelos, evaluar_en_folds
from desercion.features import agregar_variables, quitar_columnas
from desercion.modelado import separar_x_y, tipos_de_columnas


@pytest.fixture
def datos(procesados):
    """100 aprendices en 5 cohortes; desertan quienes aprueban menos."""
    base = quitar_columnas(agregar_variables(procesados))
    df = pd.concat([base] * 34, ignore_index=True).head(100)
    df["id_aprendiz"] = [f"A-{i:06d}" for i in range(len(df))]
    df["id_cohorte"] = [f"C{i % 5}" for i in range(len(df))]
    df["porcentaje_actividades_aprobadas"] = np.linspace(0, 1, len(df))
    df["deserto"] = (df["porcentaje_actividades_aprobadas"] < 0.4).astype("int64")
    grupos = df["id_cohorte"]
    x, y = separar_x_y(df)
    return x, y, grupos


def test_todos_los_modelos_se_evaluan_en_los_mismos_folds(datos):
    x, y, grupos = datos
    numericas, categoricas = tipos_de_columnas(x)
    params_lgbm = {"n_estimators": 20, "verbose": -1, "random_state": 0}

    modelos = construir_modelos(numericas, categoricas, params_lgbm)
    resultados = {n: evaluar_en_folds(p, x, y, grupos) for n, p in modelos.items()}

    assert set(resultados) == {"logistica", "lightgbm", "red_neuronal", "azar"}
    for tabla in resultados.values():
        assert list(tabla.columns) == ["pr_auc", "roc_auc"]
        assert len(tabla) == N_FOLDS


def test_el_azar_es_el_piso(datos):
    x, y, grupos = datos
    numericas, categoricas = tipos_de_columnas(x)
    modelos = construir_modelos(numericas, categoricas, {"verbose": -1})

    azar = evaluar_en_folds(modelos["azar"], x, y, grupos)
    logistica = evaluar_en_folds(modelos["logistica"], x, y, grupos)

    # Un modelo constante tiene ROC-AUC 0.5 exacta.
    np.testing.assert_allclose(azar["roc_auc"], 0.5)
    assert logistica["pr_auc"].mean() > azar["pr_auc"].mean()
