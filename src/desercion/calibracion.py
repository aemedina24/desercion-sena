"""Calibración del modelo: ¿la probabilidad que ve el instructor es creíble?

Entradas:
    models/modelo.skops           (el Pipeline que escribió train)
    data/features/train.parquet   (solo la tasa de deserción, para la corrección)
    data/features/test.parquet
    data/raw/verdad.parquet       (opcional: solo con los datos del generador v2)
Salidas:
    reports/calibracion.json
    reports/figures/calibracion.png

El modelo se entrena con `class_weight="balanced"`, que le da a cada desertor
más peso del que le corresponde: el modelo aprende como si desertara la mitad
de los aprendices. Eso mejora el recall, que es lo que buscamos, pero infla la
probabilidad. Un 0,70 de la API no quiere decir que 7 de cada 10 aprendices
así desertan.

Aquí se mide ese desajuste de tres formas:

1. **Brier:** error cuadrático medio entre la probabilidad y lo que pasó. Más
   bajo es mejor; se compara contra predecir siempre la tasa de deserción.
2. **ECE** (error de calibración esperado): se parte la población en 10 grupos
   del mismo tamaño por probabilidad y se promedia, pesando por tamaño, la
   diferencia entre lo que el modelo dijo y lo que pasó. 0 es calibración
   perfecta.
3. **Curva de calibración:** esos mismos 10 grupos, para ver dónde falla.

Y se prueba una corrección que no obliga a reentrenar: restarle al log-odds el
peso que metió el balanceo (`log(no desertores / desertores)` en train). Es un
desplazamiento, así que **no cambia el orden de los aprendices**: el umbral, las
alertas, el recall y la precisión siguen siendo exactamente los mismos. Solo
cambia el número que se muestra.

Con los datos del generador v2 se compara además contra la probabilidad real de
cada aprendiz (`data/raw/verdad.parquet`), que ningún modelo puede conocer: es
el techo de lo que se podría lograr calibrando.

Este módulo no decide nada: no toca el modelo, la API ni el umbral. Mide, y la
decisión de qué mostrarle al instructor queda para el equipo.

Uso (desde la raíz del repositorio):
    uv run python -m desercion.calibracion
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

# Sin ventana: la etapa corre en CI y en Docker.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from desercion.modelado import (
    RUTA_METRICAS,
    RUTA_TEST,
    RUTA_TRAIN,
    cargar_modelo,
    separar_x_y,
)

# --- Rutas -----------------------------------------------------------------
RUTA_CALIBRACION = Path("reports/calibracion.json")
RUTA_FIGURA = Path("reports/figures/calibracion.png")
RUTA_VERDAD = Path("data/raw/verdad.parquet")

# --- Parámetros ------------------------------------------------------------
# Grupos del mismo tamaño para la curva y el ECE.
N_GRUPOS = 10
# Probabilidades que nunca llegan a 0 ni a 1: el log-odds sería infinito.
EPSILON = 1e-9

logger = logging.getLogger(__name__)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPSILON, 1 - EPSILON)
    return np.log(p / (1 - p))


def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def desplazamiento_por_balanceo(tasa_train: float) -> float:
    """Cuánto infló `class_weight="balanced"` el log-odds de cada aprendiz.

    Con pesos balanceados cada clase pesa lo mismo en total, así que el modelo
    aprende con una tasa de deserción del 50 % en vez de la real. Eso suma
    `log(no desertores / desertores)` al log-odds de todos por igual.
    """
    if not 0 < tasa_train < 1:
        raise ValueError(
            f"La tasa de deserción de train debe estar entre 0 y 1: {tasa_train}"
        )
    return float(np.log((1 - tasa_train) / tasa_train))


def corregir_por_balanceo(probabilidad: np.ndarray, tasa_train: float) -> np.ndarray:
    """Probabilidad sin el empujón del balanceo.

    Resta el mismo valor al log-odds de todos, así que el orden de los
    aprendices no cambia: las alertas son las mismas con el umbral equivalente.
    """
    return _sigmoide(_logit(probabilidad) - desplazamiento_por_balanceo(tasa_train))


def curva_calibracion(
    y_real: np.ndarray, probabilidad: np.ndarray, n_grupos: int = N_GRUPOS
) -> list[dict]:
    """Grupos del mismo tamaño por probabilidad: qué dijo el modelo y qué pasó.

    Los cortes son cuantiles. Si hay empates, dos cortes pueden coincidir y
    salen menos grupos de los pedidos; es correcto y se reporta tal cual.
    """
    if len(y_real) != len(probabilidad):
        raise ValueError("y_real y probabilidad deben tener el mismo tamaño")
    cortes = np.unique(np.quantile(probabilidad, np.linspace(0, 1, n_grupos + 1)))
    # `digitize` con los cortes interiores: el primer y el último grupo quedan abiertos.
    grupo = np.digitize(probabilidad, cortes[1:-1], right=False)

    curva = []
    for indice in range(len(cortes) - 1):
        en_el_grupo = grupo == indice
        if not en_el_grupo.any():
            continue
        curva.append(
            {
                "grupo": indice + 1,
                "desde": float(probabilidad[en_el_grupo].min()),
                "hasta": float(probabilidad[en_el_grupo].max()),
                "aprendices": int(en_el_grupo.sum()),
                "predicho": float(probabilidad[en_el_grupo].mean()),
                "observado": float(y_real[en_el_grupo].mean()),
            }
        )
    return curva


def ece(curva: list[dict]) -> float:
    """Error de calibración esperado: diferencia media, pesada por tamaño."""
    total = sum(g["aprendices"] for g in curva)
    return float(
        sum(g["aprendices"] * abs(g["predicho"] - g["observado"]) for g in curva)
        / total
    )


def brier(y_real: np.ndarray, probabilidad: np.ndarray) -> float:
    """Error cuadrático medio de la probabilidad."""
    return float(np.mean((probabilidad - y_real) ** 2))


def resumen(y_real: np.ndarray, probabilidad: np.ndarray, umbral: float) -> dict:
    """Las tres medidas de calibración de una serie de probabilidades.

    `alertas` y `recall` se incluyen para dejar ver que una corrección que solo
    desplaza el log-odds no cambia a quién se alerta (con su umbral equivalente).
    """
    curva = curva_calibracion(y_real, probabilidad)
    alerta = probabilidad >= umbral
    desertores = int(y_real.sum())
    return {
        "umbral": float(umbral),
        "probabilidad_media": float(probabilidad.mean()),
        "tasa_real": float(y_real.mean()),
        "brier": brier(y_real, probabilidad),
        "ece": ece(curva),
        "alertas": int(alerta.sum()),
        "recall": float(y_real[alerta].sum() / desertores)
        if desertores
        else float("nan"),
        "probabilidad_media_en_alertas": float(probabilidad[alerta].mean())
        if alerta.any()
        else float("nan"),
        "tasa_real_en_alertas": float(y_real[alerta].mean())
        if alerta.any()
        else float("nan"),
        "curva": curva,
    }


def figura(
    y_real: np.ndarray,
    probabilidad: np.ndarray,
    corregida: np.ndarray,
    ruta: Path = RUTA_FIGURA,
) -> None:
    """Curva de calibración antes y después de corregir, con el reparto abajo."""
    fig, (ejes_curva, ejes_hist) = plt.subplots(
        2, 1, figsize=(7, 7), height_ratios=[3, 1], constrained_layout=True
    )

    ejes_curva.plot(
        [0, 1], [0, 1], color="gray", linestyle="--", label="Calibración perfecta"
    )
    for datos, etiqueta, color in [
        (probabilidad, "Modelo (class_weight balanced)", "tab:red"),
        (corregida, "Corregida por el balanceo", "tab:blue"),
    ]:
        curva = curva_calibracion(y_real, datos)
        ejes_curva.plot(
            [g["predicho"] for g in curva],
            [g["observado"] for g in curva],
            marker="o",
            color=color,
            label=f"{etiqueta} (ECE {ece(curva):.3f})",
        )
    ejes_curva.set_xlabel("Probabilidad que muestra la API")
    ejes_curva.set_ylabel("Desertaron en realidad")
    ejes_curva.set_title("Calibración en test: 10 grupos del mismo tamaño")
    ejes_curva.legend(loc="upper left")
    ejes_curva.grid(alpha=0.3)

    ejes_hist.hist(probabilidad, bins=30, color="tab:red", alpha=0.6, label="Modelo")
    ejes_hist.hist(corregida, bins=30, color="tab:blue", alpha=0.6, label="Corregida")
    ejes_hist.set_xlabel("Probabilidad")
    ejes_hist.set_ylabel("Aprendices")
    ejes_hist.legend()
    ejes_hist.grid(alpha=0.3)

    for ejes in (ejes_curva, ejes_hist):
        ejes.set_xlim(0, 1)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def comparar_con_la_verdad(
    ids: pd.Series, probabilidad: np.ndarray, corregida: np.ndarray
) -> dict | None:
    """Qué tan cerca está la probabilidad de la real (solo con el generador v2).

    `verdad.parquet` trae la probabilidad con que el generador sorteó la
    deserción de cada aprendiz. Ningún modelo puede conocerla: sirve de techo.
    Devuelve None si el archivo no está (datos de la v1 o sin `dvc pull`).
    """
    if not RUTA_VERDAD.exists():
        return None
    verdad = pd.read_parquet(RUTA_VERDAD).set_index("id_aprendiz")
    real = verdad.loc[ids, "probabilidad_desercion"].to_numpy()
    return {
        "probabilidad_real_media": float(real.mean()),
        "error_absoluto_medio": float(np.abs(probabilidad - real).mean()),
        "error_absoluto_medio_corregida": float(np.abs(corregida - real).mean()),
        # El orden es lo que decide las alertas, y una corrección no lo cambia.
        "correlacion_de_orden": float(
            pd.Series(probabilidad).corr(pd.Series(real), method="spearman")
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    umbral = json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))["umbral"]
    modelo = cargar_modelo()
    test = pd.read_parquet(RUTA_TEST)
    x_test, y_test = separar_x_y(test)
    tasa_train = float(
        pd.read_parquet(RUTA_TRAIN, columns=["deserto"])["deserto"].mean()
    )

    y_real = y_test.to_numpy()
    probabilidad = modelo.predict_proba(x_test)[:, 1]
    desplazamiento = desplazamiento_por_balanceo(tasa_train)
    corregida = corregir_por_balanceo(probabilidad, tasa_train)
    # Umbral equivalente: alerta exactamente a los mismos aprendices.
    umbral_corregido = float(_sigmoide(_logit(np.array([umbral]))[0] - desplazamiento))

    calibracion = {
        "tasa_desercion_train": tasa_train,
        "desplazamiento_por_balanceo": desplazamiento,
        # Brier de predecirle a todos la tasa de deserción: la referencia a batir.
        "brier_tasa_constante": brier(
            y_real, np.full_like(y_real, y_real.mean(), dtype=float)
        ),
        "modelo": resumen(y_real, probabilidad, umbral),
        "corregida": resumen(y_real, corregida, umbral_corregido),
        "frente_a_la_verdad": comparar_con_la_verdad(
            test["id_aprendiz"], probabilidad, corregida
        ),
    }

    RUTA_CALIBRACION.parent.mkdir(parents=True, exist_ok=True)
    RUTA_CALIBRACION.write_text(
        json.dumps(calibracion, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    figura(y_real, probabilidad, corregida)

    logger.info("Tasa de deserción en train: %.3f", tasa_train)
    logger.info(
        "%-10s %10s %8s %6s %8s %7s",
        "serie",
        "prob.media",
        "umbral",
        "ECE",
        "Brier",
        "alertas",
    )
    for nombre in ("modelo", "corregida"):
        r = calibracion[nombre]
        logger.info(
            "%-10s %10.3f %8.3f %6.3f %8.4f %7d",
            nombre,
            r["probabilidad_media"],
            r["umbral"],
            r["ece"],
            r["brier"],
            r["alertas"],
        )
    logger.info("Tasa real en test: %.3f", y_real.mean())
    logger.info("Reporte: %s | Figura: %s", RUTA_CALIBRACION, RUTA_FIGURA)


if __name__ == "__main__":
    main()
