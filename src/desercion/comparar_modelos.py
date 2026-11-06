"""Comparación de modelos en los mismos folds, registrada en MLflow (ADR 0002).

Reproduce la tabla del ADR 0002: regresión logística, LightGBM tuneado, red
neuronal y una referencia al azar, evaluados en los MISMOS 5 folds de
GroupKFold por cohorte. Como los folds son iguales para todos, la comparación
es pareada: se puede ver fold a fold quién gana, no solo las medias.

Cada modelo queda como un run del experimento "desercion-comparacion", con la
PR-AUC y la ROC-AUC de cada fold (step = número de fold) y sus medias. El
cuaderno 03 lee esos runs con mlflow.search_runs; no reentrena nada.

Solo usa train: elegir modelo mirando test sería ajustar el modelo a test.
Está fuera del pipeline DVC, igual que tune.py: su resultado justifica una
decisión, pero no alimenta al modelo final.

Entradas:
    data/features/train.parquet
    config/model/logistic.yaml, config/model/lightgbm.yaml
    reports/tuning_lightgbm.json   (mejores hiperparámetros de LightGBM)

Uso (desde la raíz del repositorio, unos minutos):
    uv run python -m desercion.comparar_modelos

Requiere MLFLOW_TRACKING_URI y credenciales en .env, igual que train_model.py.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from desercion.modelado import (
    COLUMNA_GRUPO,
    RUTA_TRAIN,
    separar_x_y,
    tipos_de_columnas,
)
from desercion.train_model import (
    cargar_config,
    construir_pipeline,
    md5_archivo,
    version_del_codigo,
)
from desercion.tune import RUTA_LIGHTGBM, cargar_yaml, construir_pipeline_lgbm

# --- Rutas -----------------------------------------------------------------
RUTA_TUNING_RESULTADO = Path("reports/tuning_lightgbm.json")

# --- Parámetros ------------------------------------------------------------
EXPERIMENTO = "desercion-comparacion"
N_FOLDS = 5
METRICAS = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}
# Red neuronal pequeña: con ~8000 filas una más grande memoriza.
# early_stopping reserva el 10 % de cada fold de entrenamiento para parar a tiempo.
PARAMS_MLP = {
    "hidden_layer_sizes": (32, 16),
    "early_stopping": True,
    "max_iter": 500,
    "random_state": 42,
}

logger = logging.getLogger(__name__)


def construir_modelos(
    numericas: list[str], categoricas: list[str], params_lgbm: dict
) -> dict[str, Pipeline]:
    """Los candidatos, cada uno con su preprocesamiento dentro del Pipeline."""
    logistica = construir_pipeline(numericas, categoricas, cargar_config())
    # Misma imputación, escalado y one-hot que la logística; solo cambia el
    # modelo. Así la diferencia se debe al modelo y no al preprocesamiento.
    red = clone(logistica).set_params(modelo=MLPClassifier(**PARAMS_MLP))
    # "prior" predice siempre la proporción de desertores: su PR-AUC es esa
    # proporción. Es el piso: un modelo que no lo supera no sirve.
    azar = clone(logistica).set_params(modelo=DummyClassifier(strategy="prior"))
    return {
        "logistica": logistica,
        "lightgbm": construir_pipeline_lgbm(numericas, categoricas, params_lgbm),
        "red_neuronal": red,
        "azar": azar,
    }


def evaluar_en_folds(
    pipeline: Pipeline, x: pd.DataFrame, y: pd.Series, grupos: pd.Series
) -> pd.DataFrame:
    """Una fila por fold con cada métrica de METRICAS."""
    resultado = cross_validate(
        pipeline,
        x,
        y,
        groups=grupos,
        cv=GroupKFold(n_splits=N_FOLDS),  # sin azar: los mismos folds para todos
        scoring=METRICAS,
    )
    return pd.DataFrame({nombre: resultado[f"test_{nombre}"] for nombre in METRICAS})


def registrar_run(
    nombre: str, por_fold: pd.DataFrame, params: dict, etiquetas: dict
) -> None:
    """Un run de MLflow: métricas por fold (step) y su resumen."""
    with mlflow.start_run(run_name=nombre):
        mlflow.set_tags({**etiquetas, "modelo": nombre})
        mlflow.log_params(params)
        for fold, fila in por_fold.iterrows():
            for metrica, valor in fila.items():
                mlflow.log_metric(f"{metrica}_fold", valor, step=int(fold))
        for metrica in por_fold.columns:
            mlflow.log_metric(f"{metrica}_media", por_fold[metrica].mean())
            mlflow.log_metric(f"{metrica}_std", por_fold[metrica].std())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    load_dotenv()
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise SystemExit(
            "Falta MLFLOW_TRACKING_URI: crea .env a partir de .env.example"
        )

    df = pd.read_parquet(RUTA_TRAIN)
    # Los grupos se toman antes de separar: separar_x_y elimina id_cohorte de X.
    grupos = df[COLUMNA_GRUPO]
    x, y = separar_x_y(df)
    numericas, categoricas = tipos_de_columnas(x)

    tuning = json.loads(RUTA_TUNING_RESULTADO.read_text(encoding="utf-8"))
    params_lgbm = {**cargar_yaml(RUTA_LIGHTGBM), **tuning["mejores_params"]}
    modelos = construir_modelos(numericas, categoricas, params_lgbm)

    etiquetas = {
        **version_del_codigo(),
        "datos_train_md5": md5_archivo(RUTA_TRAIN),
        # Identifica esta tanda: el cuaderno 03 compara runs de la misma tanda.
        "comparacion_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
    }
    params_por_modelo = {
        "logistica": cargar_config(),
        "lightgbm": params_lgbm,
        "red_neuronal": PARAMS_MLP,
        "azar": {"strategy": "prior"},
    }

    mlflow.set_experiment(EXPERIMENTO)
    resumen = {}
    for nombre, pipeline in modelos.items():
        logger.info("Evaluando %s...", nombre)
        por_fold = evaluar_en_folds(pipeline, x, y, grupos)
        registrar_run(
            nombre,
            por_fold,
            {**params_por_modelo[nombre], "n_folds": N_FOLDS, "filas": len(x)},
            etiquetas,
        )
        resumen[nombre] = por_fold["pr_auc"]

    tabla = pd.DataFrame(resumen).T
    tabla.columns = [f"fold_{i}" for i in tabla.columns]
    tabla.insert(0, "media", tabla.mean(axis=1))
    logger.info(
        "PR-AUC por fold (comparacion_id=%s):\n%s",
        etiquetas["comparacion_id"],
        tabla.sort_values("media", ascending=False).round(3).to_string(),
    )


if __name__ == "__main__":
    main()
