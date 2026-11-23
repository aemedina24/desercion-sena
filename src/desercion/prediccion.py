"""Riesgo de deserción de los aprendices activos, listo para la API.

`predecir_riesgo` junta el modelo, la explicación y las medidas, y devuelve una
fila por aprendiz con las columnas que exige la API (ADR 0004 y ADR 0006):

    id_aprendiz, probabilidad, alerta, factores, medidas, plan_de_accion

No carga nada por su cuenta: recibe el modelo, la referencia y el umbral, que la
API carga una sola vez al arrancar. Así `/health` refleja lo que realmente está
cargado y las pruebas pueden pasarle un modelo pequeño.

Conexión con la API (en crear_app):

    funcion = partial(
        predecir_riesgo, modelo=modelo, referencia=referencia, umbral=umbral
    )
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from desercion.explain import aportes, principales_factores
from desercion.modelado import ETIQUETA, separar_x_y
from desercion.recomendaciones import medidas_para, plan_para

# Dirección en que cada variable sube el riesgo según el sentido común (y el
# texto de su medida): +1 = más es peor, -1 = menos es peor; las booleanas
# cuentan como 0/1. Por la colinealidad, el modelo puede darle a una variable el
# signo contrario, y entonces la marcaría como factor en quien está MEJOR que el
# promedio: con los datos v2, "Cursos previos terminados: 100 % (promedio 79 %)"
# en 40 alertas de test. Esos factores no se muestran (quitar_contrarios). Están
# todas las variables con medida (una prueba lo exige) y las tres que el modelo
# invierte con los datos v1 o v2, que están en `sin_medida`.
DIRECCION_RIESGO = {
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
    # Sin medida: el modelo les invierte el signo por la colinealidad.
    "porcentaje_actividades_aprobadas": -1,
    "tasa_finalizacion_previa": -1,
    "abandono_ultimo_curso": +1,
}

COLUMNAS_PREDICCION = [
    "id_aprendiz",
    "probabilidad",
    "alerta",
    "factores",
    "medidas",
    "plan_de_accion",
]
# Tope del esquema RiesgoAprendiz (factores: max_length=5).
MAX_FACTORES = 5
# Aporte mínimo (en log-odds) para mostrar un factor y sugerir su medida. Por
# debajo es ruido: 0.05 mueve la probabilidad alrededor de 1 punto cerca del
# umbral. Es el orden de las variables menos importantes en test
# (reports/importancia_variables.json).
APORTE_MINIMO = 0.05
# Probabilidades que nunca llegan a 0 ni a 1: el log-odds sería infinito.
EPSILON = 1e-9
# El umbral de reports/calibracion.json y el de metricas_modelo_base.json salen
# del mismo modelo; entre Windows y Linux difieren desde el decimal 13.
TOLERANCIA_UMBRAL = 1e-6


def predecir_riesgo(
    aprendices: pd.DataFrame,
    *,
    modelo: Pipeline,
    referencia: pd.Series,
    umbral: float,
    n_factores: int = 3,
) -> pd.DataFrame:
    """Probabilidad, alerta, factores, medidas y plan de acción de cada aprendiz.

    - `aprendices`: con `id_aprendiz` y las variables del modelo. Puede traer o
      no `deserto`: se ignora y nunca sale en el resultado.
    - `alerta`: probabilidad >= umbral, el mismo criterio de evaluate.py.
    - `factores`: los `n_factores` que más suben el riesgo frente al aprendiz
      promedio de train, con aporte de al menos APORTE_MINIMO. Cada uno trae el
      valor del aprendiz y el promedio de train contra el que se compara
      (agregar_valores). No sale un factor cuyo valor está del lado bueno del
      promedio (quitar_contrarios): su lugar lo toma el siguiente.
    - `medidas`: solo para quienes tienen alerta (recomendaciones.py). Un factor
      cuyo dato está vacío se muestra, pero no genera medida (factores_con_dato).
    - `plan_de_accion`: qué hacer y qué decir por cada medida, en el mismo orden.
    """
    if not 1 <= n_factores <= MAX_FACTORES:
        raise ValueError(f"n_factores debe estar entre 1 y {MAX_FACTORES}")

    validar_referencia(modelo, referencia)
    x = variables_modelo(aprendices)

    probabilidad = modelo.predict_proba(x)[:, 1]
    alerta = probabilidad >= umbral
    tabla_aportes = aportes(modelo, x, referencia)
    # Se piden de más para que, si alguno se quita por contrario, entre el siguiente.
    candidatos = [
        principales_factores(fila[fila >= APORTE_MINIMO], n=MAX_FACTORES)
        for _, fila in tabla_aportes.iterrows()
    ]
    candidatos = agregar_valores(candidatos, x, promedios_train(modelo))
    factores = [fila[:n_factores] for fila in quitar_contrarios(candidatos)]
    con_dato = factores_con_dato(factores, x)
    medidas = [medidas_para(f, bool(a)) for f, a in zip(con_dato, alerta, strict=True)]
    planes = [plan_para(f, bool(a)) for f, a in zip(con_dato, alerta, strict=True)]

    return pd.DataFrame(
        {
            "id_aprendiz": aprendices["id_aprendiz"].to_numpy(),
            "probabilidad": probabilidad,
            "alerta": alerta,
            "factores": factores,
            "medidas": medidas,
            "plan_de_accion": planes,
        },
        columns=COLUMNAS_PREDICCION,
    )


def promedios_train(modelo: Pipeline) -> dict[str, float]:
    """Promedio de train de cada variable numérica, el que usa el modelo.

    Es la media que guarda el StandardScaler: la misma referencia contra la que
    se calcula el aporte de cada factor. Las booleanas quedan como proporción (0.35 = 35 % de los
    aprendices). Los vacíos se rellenan con la mediana antes de escalar, así que
    en variables con vacíos (asistencia, encuesta, historial) es el promedio con
    esos vacíos rellenados. Sale del modelo: no hay que leer train ni tocar DVC.
    """
    numericas = modelo.named_steps["preprocesamiento"].named_transformers_["numericas"]
    nombres = numericas.get_feature_names_out()
    medias = numericas.named_steps["escalar"].mean_
    # Las columnas del indicador de faltante (missingindicator_*) no son variables.
    return {
        str(n): float(m)
        for n, m in zip(nombres, medias, strict=True)
        if not str(n).startswith("missingindicator_")
    }


def agregar_valores(
    factores: list[list[dict]], x: pd.DataFrame, promedios: dict[str, float]
) -> list[list[dict]]:
    """Agrega a cada factor el valor del aprendiz y el promedio de train.

    Así el instructor ve el porqué con números: "entregó el 45 % de las
    actividades; el promedio es 72 %". `valor` es None si el dato está vacío, y
    un texto en las categóricas (la modalidad, la situación laboral), que no
    tienen promedio.
    """
    resultado = []
    for i, fila in enumerate(factores):
        nueva = []
        for f in fila:
            crudo = x[f["variable"]].iloc[i]
            if pd.isna(crudo):
                valor = None
            elif isinstance(crudo, str):
                valor = crudo
            else:
                valor = float(crudo)
                valor = None if math.isnan(valor) else valor
            nueva.append(
                {**f, "valor": valor, "promedio": promedios.get(f["variable"])}
            )
        resultado.append(nueva)
    return resultado


def es_contrario(factor: dict) -> bool:
    """True si el valor del aprendiz está del lado bueno de su promedio.

    Solo para las variables de DIRECCION_RIESGO con valor y promedio numéricos:
    un dato vacío o una categoría no tienen lado.
    """
    direccion = DIRECCION_RIESGO.get(factor["variable"])
    valor, promedio = factor.get("valor"), factor.get("promedio")
    if direccion is None or not isinstance(valor, float) or promedio is None:
        return False
    return (valor - promedio) * direccion < 0


def quitar_contrarios(factores: list[list[dict]]) -> list[list[dict]]:
    """Quita los factores que dirían lo contrario de lo que muestran.

    El aporte es real para el modelo, pero al instructor le llegaría "Cursos
    previos terminados: 100 % (promedio 79 %)" como razón de la alerta: un
    número que contradice el texto. Pasa con las variables a las que la
    colinealidad les invierte el signo. Con el modelo real, una prueba verifica
    que los que se quitan son justo esos.
    """
    return [[f for f in fila if not es_contrario(f)] for fila in factores]


def factores_con_dato(factores: list[list[dict]], x: pd.DataFrame) -> list[list[dict]]:
    """Quita de cada aprendiz los factores cuyo valor está vacío.

    Un dato vacío no es un problema del aprendiz: su aporte viene del indicador
    de faltante que agrega el imputador. Por ejemplo, `asistencia_porcentaje` es
    nula en todos los virtuales (no tienen sesiones) y ese indicador sube el
    riesgo, así que la medida de inasistencia les llegaba con un guion que dice
    "te he echado de menos en las sesiones". El factor se sigue mostrando, porque
    explica la probabilidad; solo deja de activar medidas y planes.
    """
    vacios = x.isna().to_numpy()
    posicion = {columna: i for i, columna in enumerate(x.columns)}
    return [
        [f for f in fila if not vacios[i, posicion[f["variable"]]]]
        for i, fila in enumerate(factores)
    ]


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


def cargar_desplazamiento(ruta: Path, umbral: float) -> float:
    """Corrimiento de la calibración (reports/calibracion.json) para este modelo.

    Lo escribe la etapa `calibrate` de DVC: `log(no desertores / desertores)` en
    train, lo que `class_weight="balanced"` le sumó al log-odds de todos. Falla
    si el reporte es de otro modelo (su umbral no coincide con el que cargó la
    API): pasa si se reentrena sin volver a correr `calibrate`. La tolerancia
    deja pasar la diferencia de decimales entre Windows y Linux.
    """
    calibracion = json.loads(ruta.read_text(encoding="utf-8"))
    umbral_reporte = float(calibracion["modelo"]["umbral"])
    if not math.isclose(umbral_reporte, umbral, abs_tol=TOLERANCIA_UMBRAL):
        raise ValueError(
            f"La calibración es de otro modelo: su umbral es {umbral_reporte:.6f} y el "
            f"de la API {umbral:.6f}. Ejecuta `uv run dvc repro calibrate`."
        )
    desplazamiento = float(calibracion["desplazamiento_por_balanceo"])
    if not math.isfinite(desplazamiento):
        raise ValueError(f"Desplazamiento de calibración inválido: {desplazamiento}")
    return desplazamiento


def corregir_probabilidad(probabilidad: pd.Series, desplazamiento: float) -> pd.Series:
    """Probabilidad sin el empujón del balanceo: la que se puede leer como porcentaje.

    Resta el mismo valor al log-odds de todos (la corrección de calibracion.py,
    que no se importa aquí porque usa matplotlib, ausente en la imagen Docker).
    No cambia el orden de los aprendices, así que tampoco las alertas.
    """
    p = probabilidad.clip(EPSILON, 1 - EPSILON)
    log_odds = np.log(p / (1 - p)) - desplazamiento
    return 1 / (1 + np.exp(-log_odds))


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
