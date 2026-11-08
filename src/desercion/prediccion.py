"""Riesgo de deserción de los aprendices activos, listo para la API.

`predecir_riesgo` junta el modelo, la explicación y las medidas, y devuelve una
fila por aprendiz con las columnas que exige la API (ADR 0004):

    id_aprendiz, probabilidad, alerta, factores, medidas

No carga nada por su cuenta: recibe el modelo, la referencia y el umbral, que la
API carga una sola vez al arrancar. Así `/health` refleja lo que realmente está
cargado y las pruebas pueden pasarle un modelo pequeño.

Conexión con la API (en crear_app):

    funcion = partial(
        predecir_riesgo, modelo=modelo, referencia=referencia, umbral=umbral
    )
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from desercion.explain import aportes, principales_factores
from desercion.modelado import ETIQUETA, separar_x_y
from desercion.recomendaciones import medidas_para

COLUMNAS_PREDICCION = ["id_aprendiz", "probabilidad", "alerta", "factores", "medidas"]
# Tope del esquema RiesgoAprendiz (factores: max_length=5).
MAX_FACTORES = 5
# Aporte mínimo (en log-odds) para mostrar un factor y sugerir su medida. Por
# debajo es ruido: 0.05 mueve la probabilidad alrededor de 1 punto cerca del
# umbral. Es el orden de las variables menos importantes en test
# (reports/importancia_variables.json).
APORTE_MINIMO = 0.05


def predecir_riesgo(
    aprendices: pd.DataFrame,
    *,
    modelo: Pipeline,
    referencia: pd.Series,
    umbral: float,
    n_factores: int = 3,
) -> pd.DataFrame:
    """Probabilidad, alerta, factores y medidas de cada aprendiz.

    - `aprendices`: con `id_aprendiz` y las variables del modelo. Puede traer o
      no `deserto`: se ignora y nunca sale en el resultado.
    - `alerta`: probabilidad >= umbral, el mismo criterio de evaluate.py.
    - `factores`: los `n_factores` que más suben el riesgo frente al aprendiz
      promedio de train, con aporte de al menos APORTE_MINIMO.
    - `medidas`: solo para quienes tienen alerta (recomendaciones.py).
    """
    if not 1 <= n_factores <= MAX_FACTORES:
        raise ValueError(f"n_factores debe estar entre 1 y {MAX_FACTORES}")

    validar_referencia(modelo, referencia)
    x = variables_modelo(aprendices)

    probabilidad = modelo.predict_proba(x)[:, 1]
    alerta = probabilidad >= umbral
    tabla_aportes = aportes(modelo, x, referencia)
    factores = [
        principales_factores(fila[fila >= APORTE_MINIMO], n=n_factores)
        for _, fila in tabla_aportes.iterrows()
    ]
    medidas = [medidas_para(f, bool(a)) for f, a in zip(factores, alerta, strict=True)]

    return pd.DataFrame(
        {
            "id_aprendiz": aprendices["id_aprendiz"].to_numpy(),
            "probabilidad": probabilidad,
            "alerta": alerta,
            "factores": factores,
            "medidas": medidas,
        },
        columns=COLUMNAS_PREDICCION,
    )


def validar_referencia(modelo: Pipeline, referencia: pd.Series) -> None:
    """Falla con un mensaje claro si la referencia no es la de este modelo.

    Pasa si se reentrena el modelo sin volver a correr explain: las columnas
    transformadas cambian y los factores saldrían mal (o con un KeyError).
    """
    columnas = modelo.named_steps["preprocesamiento"].get_feature_names_out()
    faltantes = set(columnas) - set(referencia.index)
    if faltantes:
        raise ValueError(
            f"La referencia no corresponde al modelo: le faltan {len(faltantes)} "
            f"columnas (por ejemplo {min(faltantes)}). Ejecuta `uv run dvc repro explain`."
        )


def variables_modelo(aprendices: pd.DataFrame) -> pd.DataFrame:
    """X del modelo, haya o no etiqueta.

    En producción los aprendices activos no tienen `deserto`. Se reutiliza
    `separar_x_y` tal cual (mismas columnas y conversión de booleanos) con una
    etiqueta vacía que se descarta. No se cambia modelado.py porque es
    dependencia de `train` en DVC: tocarlo obligaría a reentrenar el modelo.
    """
    if ETIQUETA not in aprendices.columns:
        aprendices = aprendices.assign(**{ETIQUETA: pd.NA})
    x, _ = separar_x_y(aprendices)
    return x
