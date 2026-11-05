"""Etapa tune: busca hiperparámetros de LightGBM con Optuna.

Entradas: data/features/train.parquet, config/tuning/optuna.yaml,
          config/model/lightgbm.yaml
Salida:   reports/tuning_lightgbm.json (mejores hiperparámetros, umbral y
          métricas de validación cruzada)

Test NO se lee en esta etapa: todo se decide con validación cruzada en train.
"""

import json
import logging
from pathlib import Path

import optuna
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from desercion.modelado import RUTA_TRAIN, elegir_umbral, separar_x_y, tipos_de_columnas

# --- Rutas -----------------------------------------------------------------
RUTA_TUNING = Path("config/tuning/optuna.yaml")
RUTA_LIGHTGBM = Path("config/model/lightgbm.yaml")
RUTA_SALIDA = Path("reports/tuning_lightgbm.json")

logger = logging.getLogger(__name__)


def cargar_yaml(ruta: Path) -> dict:
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


def construir_pipeline_lgbm(
    numericas: list[str], categoricas: list[str], params: dict
) -> Pipeline:
    """Preprocesamiento mínimo + LightGBM en un solo objeto."""
    preprocesamiento = ColumnTransformer(
        [
            # Sin imputar ni escalar: LightGBM maneja NaN y no depende de la escala.
            ("numericas", "passthrough", numericas),
            (
                "categoricas",
                Pipeline(
                    [
                        (
                            "imputar",
                            SimpleImputer(strategy="constant", fill_value="faltante"),
                        ),
                        (
                            "codificar",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categoricas,
            ),
        ]
    )
    return Pipeline(
        [("preprocesamiento", preprocesamiento), ("modelo", LGBMClassifier(**params))]
    )


def sugerir_params(trial: optuna.Trial, espacio: dict) -> dict:
    """Traduce el espacio de búsqueda del YAML a sugerencias de Optuna."""
    params = {}
    for nombre, rango in espacio.items():
        log = rango.get("log", False)
        if isinstance(rango["low"], int) and isinstance(rango["high"], int):
            params[nombre] = trial.suggest_int(
                nombre, rango["low"], rango["high"], log=log
            )
        else:
            params[nombre] = trial.suggest_float(
                nombre, rango["low"], rango["high"], log=log
            )
    return params


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    tuning = cargar_yaml(RUTA_TUNING)
    base = cargar_yaml(RUTA_LIGHTGBM)

    df = pd.read_parquet(RUTA_TRAIN)
    # Los grupos se toman antes de separar: separar_x_y elimina id_cohorte de X.
    grupos = df[tuning["cv"]["group_col"]]
    x, y = separar_x_y(df)
    numericas, categoricas = tipos_de_columnas(x)
    cv = GroupKFold(n_splits=tuning["cv"]["n_splits"])  # sin azar: folds reproducibles

    def objetivo(trial: optuna.Trial) -> float:
        params = {**base, **sugerir_params(trial, tuning["espacio_busqueda"])}
        pipeline = construir_pipeline_lgbm(numericas, categoricas, params)
        puntajes = cross_val_score(
            pipeline, x, y, groups=grupos, cv=cv, scoring=tuning["metrica"]
        )
        return float(puntajes.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=tuning["seed"])
    )
    estudio.optimize(objetivo, n_trials=tuning["n_trials"])
    logger.info(
        "Mejor PR-AUC en CV: %.4f (trial %d)",
        estudio.best_value,
        estudio.best_trial.number,
    )

    # Predicciones out-of-fold con los mejores parámetros: cada fila la predice
    # un modelo que no la vio. Sobre ellas se elige el umbral.
    mejores = {**base, **estudio.best_params}
    pipeline = construir_pipeline_lgbm(numericas, categoricas, mejores)
    probabilidad_oof = cross_val_predict(
        pipeline, x, y, groups=grupos, cv=cv, method="predict_proba"
    )[:, 1]
    umbral = elegir_umbral(y, probabilidad_oof, tuning["umbral"]["recall_objetivo"])

    resultado = {
        "mejores_params": estudio.best_params,
        "pr_auc_cv_media": float(estudio.best_value),
        "pr_auc_oof": float(average_precision_score(y, probabilidad_oof)),
        **umbral,
        "n_trials": tuning["n_trials"],
    }
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": en Windows evita CRLF y que el hash de dvc.lock no coincida con Git.
    RUTA_SALIDA.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Umbral elegido: %.4f | %s", umbral["umbral"], RUTA_SALIDA)


if __name__ == "__main__":
    main()
