"""Entrenamiento del modelo base: regresión logística registrada en MLflow.

Entradas:
    data/features/train.parquet   (cohortes antiguas: ajustan el modelo)
    data/features/test.parquet    (cohortes recientes: solo se evalúan)
    config/model/logistic.yaml    (hiperparámetros)
Salidas:
    reports/metricas_modelo_base.json   (métricas en test, las lee DVC)
    Un run en MLflow con parámetros, métricas, el modelo y la versión de los
    datos y del código que lo produjeron.

La imputación, el escalado y el one-hot viven dentro del Pipeline: se ajustan
solo con train, así que el test no filtra información al entrenamiento, y el
mismo objeto sirve para predecir en la API.

Uso (desde la raíz del repositorio):
    uv run python -m desercion.train_model

Requiere MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME y MLFLOW_TRACKING_PASSWORD
en .env (no se versiona): los runs quedan en el MLflow de DagsHub, compartido
por el equipo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from desercion.modelado import (
    COLUMNA_GRUPO,
    RUTA_TEST,
    RUTA_TRAIN,
    elegir_umbral,
    separar_x_y,
    tipos_de_columnas,
)

# --- Rutas -----------------------------------------------------------------
RUTA_CONFIG = Path("config/model/logistic.yaml")
RUTA_METRICAS = Path("reports/metricas_modelo_base.json")

# --- Parámetros ------------------------------------------------------------
# Las métricas se reportan también por modalidad (decisión en CLAUDE.md).
COLUMNA_SUBGRUPO = "modalidad"

EXPERIMENTO = "desercion-seguimiento"
NOMBRE_RUN = "modelo-base-logistica"

logger = logging.getLogger(__name__)


def cargar_config(ruta: Path = RUTA_CONFIG) -> dict:
    """Lee los hiperparámetros del modelo."""
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


def construir_pipeline(
    numericas: list[str], categoricas: list[str], config: dict
) -> Pipeline:
    """Preprocesamiento + regresión logística en un solo objeto."""
    preprocesamiento = ColumnTransformer(
        [
            (
                "numericas",
                Pipeline(
                    [
                        # add_indicator: que falte un dato también informa
                        # (asistencia nula = virtual, encuesta nula = sin encuesta).
                        (
                            "imputar",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("escalar", StandardScaler()),
                    ]
                ),
                numericas,
            ),
            (
                "categoricas",
                Pipeline(
                    [
                        (
                            "imputar",
                            SimpleImputer(strategy="constant", fill_value="faltante"),
                        ),
                        # ignore: una categoría que train no vio no rompe la predicción.
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categoricas,
            ),
        ]
    )
    modelo = LogisticRegression(
        C=config["C"],
        max_iter=config["max_iter"],
        class_weight=config["class_weight"],
    )
    return Pipeline([("preprocesamiento", preprocesamiento), ("modelo", modelo)])


def calcular_metricas(y_real: pd.Series, probabilidad, umbral: float) -> dict:
    """Métricas de la clase "deserta". No se usa accuracy: con 69/31 engaña."""
    prediccion = (probabilidad >= umbral).astype(int)
    return {
        "recall": recall_score(y_real, prediccion),
        "precision": precision_score(y_real, prediccion, zero_division=0),
        "f1": f1_score(y_real, prediccion),
        "roc_auc": roc_auc_score(y_real, probabilidad),
        "pr_auc": average_precision_score(y_real, probabilidad),
    }


def metricas_por_subgrupo(
    y_real: pd.Series, probabilidad, grupos: pd.Series, umbral: float
) -> dict:
    """Recall por modalidad: un buen promedio puede esconder un grupo olvidado."""
    prediccion = pd.Series((probabilidad >= umbral).astype(int), index=y_real.index)
    return {
        f"recall_{grupo}": recall_score(
            y_real[grupos == grupo], prediccion[grupos == grupo]
        )
        for grupo in sorted(grupos.unique())
    }


def md5_archivo(ruta: Path) -> str:
    """El mismo md5 que DVC registra en dvc.lock para ese archivo."""
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def version_del_codigo() -> dict:
    """Commit actual y si hay cambios sin commit (entonces no es reproducible)."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_sucio": str(bool(git("status", "--porcelain"))),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    load_dotenv()  # credenciales de DagsHub desde .env
    if not os.getenv("MLFLOW_TRACKING_URI"):
        # Sin esto MLflow guardaría el run en una carpeta local que nadie más ve.
        raise SystemExit(
            "Falta MLFLOW_TRACKING_URI: crea .env a partir de .env.example"
        )
    logger.info("MLflow: %s", mlflow.get_tracking_uri())

    config = cargar_config()
    train = pd.read_parquet(RUTA_TRAIN)
    test = pd.read_parquet(RUTA_TEST)
    x_train, y_train = separar_x_y(train)
    x_test, y_test = separar_x_y(test)
    numericas, categoricas = tipos_de_columnas(x_train)
    logger.info(
        "Variables: %s numéricas, %s categóricas", len(numericas), len(categoricas)
    )

    pipeline = construir_pipeline(numericas, categoricas, config)

    # Umbral con predicciones out-of-fold en train: cada fila la predice un
    # modelo que no la vio. Test no participa en la elección.
    probabilidad_oof = cross_val_predict(
        pipeline,
        x_train,
        y_train,
        groups=train[COLUMNA_GRUPO],
        cv=GroupKFold(n_splits=config["cv_n_splits"]),
        method="predict_proba",
    )[:, 1]
    eleccion = elegir_umbral(y_train, probabilidad_oof, config["recall_objetivo"])
    umbral = eleccion["umbral"]
    logger.info(
        "Umbral out-of-fold: %.3f (recall %.3f, precisión %.3f)",
        umbral,
        eleccion["recall_oof"],
        eleccion["precision_oof"],
    )

    pipeline.fit(x_train, y_train)

    probabilidad = pipeline.predict_proba(x_test)[:, 1]
    metricas = eleccion | calcular_metricas(y_test, probabilidad, umbral)
    metricas |= metricas_por_subgrupo(
        y_test, probabilidad, x_test[COLUMNA_SUBGRUPO], umbral
    )

    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name=NOMBRE_RUN):
        mlflow.set_tags(
            {
                **version_del_codigo(),
                # Responde "¿qué versión de los datos produjo este modelo?".
                "datos_train_md5": md5_archivo(RUTA_TRAIN),
                "datos_test_md5": md5_archivo(RUTA_TEST),
            }
        )
        mlflow.log_params(
            {**config, "filas_train": len(x_train), "filas_test": len(x_test)}
        )
        mlflow.log_metrics(metricas)
        mlflow.sklearn.log_model(
            pipeline,
            name="modelo",
            input_example=x_train.head(5),
            # skops guarda el modelo sin pickle y exige declarar los tipos
            # externos que contiene; numpy.dtype lo usan imputador y one-hot.
            skops_trusted_types=["numpy.dtype"],
        )

    RUTA_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_METRICAS.write_text(
        json.dumps(metricas, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for nombre, valor in metricas.items():
        logger.info("%-20s %.3f", nombre, valor)


if __name__ == "__main__":
    main()
