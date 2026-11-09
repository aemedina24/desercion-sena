"""Monitoreo del modelo: drift de datos y recall por cohorte (ADR 0005).

Compara cada cohorte nueva con los datos de entrenamiento:

1. **Drift de datos (criterio de alerta): PSI por variable.** El PSI
   (Population Stability Index) mide cuánto cambió la distribución de una
   variable. Para las numéricas, los bins salen de los cuantiles de train; las
   categóricas y las discretas (numéricas o booleanas con pocos valores) usan
   cada valor como bin. Los nulos son un bin propio. Así todas las variables
   quedan en la misma escala. Las categorías que no existían en train se marcan:

       PSI < 0.10 estable · 0.10–0.20 moderado · > 0.20 drift

2. **Pruebas KS y chi-cuadrado (solo informativas).** Con 7.974 aprendices de
   train contra unos 770 por cohorte, cambios triviales dan p-valores minúsculos;
   por eso no deciden la alerta.
3. **Recall por cohorte con IC bootstrap**, como en el cuaderno 04. En la demo
   se conoce el resultado real; en producción llega cuando termina la cohorte.
   Si el recall queda por debajo de 0.70 el reporte marca "revisar", pero el
   pipeline no falla: es una señal para el equipo, no un error.

`sexo` (atributo protegido, solo se audita) y `deserto` (la etiqueta) nunca
entran al drift: se usan solo las variables del modelo.

Entradas:
    data/features/train.parquet, data/features/test.parquet,
    models/modelo.skops, reports/metricas_modelo_base.json (umbral)
Salidas:
    reports/drift/drift.json   (resumen por cohorte; lo versiona Git)
    reports/drift/drift.html   (el mismo resumen en tablas, para la demo)

Uso (desde la raíz del repositorio):
    uv run python -m desercion.monitor
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from desercion.equidad import bootstrap_recall
from desercion.modelado import (
    RUTA_METRICAS,
    RUTA_TEST,
    RUTA_TRAIN,
    cargar_modelo,
    separar_x_y,
)

logger = logging.getLogger(__name__)

# --- Rutas -----------------------------------------------------------------
RUTA_DRIFT_JSON = Path("reports/drift/drift.json")
RUTA_DRIFT_HTML = Path("reports/drift/drift.html")

# --- Parámetros ------------------------------------------------------------
N_BINS = 10  # bins por cuantiles de train (deciles)
EPSILON = 1e-4  # proporción mínima de un bin vacío: evita log(0) en el PSI
PSI_MODERADO = 0.10
PSI_DRIFT = 0.20  # criterio de alerta
RECALL_OBJETIVO = 0.70  # el mismo objetivo con que se eligió el umbral
COLUMNA_COHORTE = "id_cohorte"
# Nunca entran al drift (ver docstring). `sexo` no está en features, pero la
# exclusión explícita protege si algún día se agrega.
EXCLUIDAS = {"sexo", "deserto"}
NULO = "(nulo)"


# --- PSI -------------------------------------------------------------------
def psi(proporcion_ref: np.ndarray, proporcion_act: np.ndarray) -> float:
    """PSI = Σ (actual − ref) · ln(actual / ref), con un mínimo para bins vacíos."""
    ref = np.clip(proporcion_ref, EPSILON, None)
    act = np.clip(proporcion_act, EPSILON, None)
    return float(np.sum((act - ref) * np.log(act / ref)))


def bordes_cuantiles(referencia: pd.Series, n_bins: int = N_BINS) -> np.ndarray:
    """Bordes interiores de los bins: cuantiles de train, sin repetir."""
    cuantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.nanquantile(referencia.to_numpy(dtype=float), cuantiles))


def bins_numericos(valores: pd.Series, bordes: np.ndarray) -> pd.Series:
    """Asigna cada valor a su bin (bin_00 … bin_NN); los nulos, a su propio bin.

    Los valores fuera del rango de train caen en el primer o el último bin.
    """
    numeros = valores.to_numpy(dtype=float)
    # Etiquetas de texto ("bin_00" …) para mezclarlas con el bin de nulos.
    bins = np.array(
        [f"bin_{i:02d}" for i in np.searchsorted(bordes, numeros, side="right")],
        dtype=object,
    )
    bins[np.isnan(numeros)] = NULO
    return pd.Series(bins, index=valores.index)


def como_categoria(valores: pd.Series) -> pd.Series:
    """Cada valor es un bin; los nulos, uno propio."""
    return valores.astype(object).where(valores.notna(), NULO).astype(str)


def proporciones(ref: pd.Series, act: pd.Series) -> pd.DataFrame:
    """Proporción de cada bin en train y en la cohorte (mismas filas en ambas)."""
    tabla = pd.DataFrame(
        {
            "referencia": ref.value_counts(normalize=True),
            "actual": act.value_counts(normalize=True),
        }
    )
    return tabla.fillna(0.0).sort_index()


def tipo_variable(referencia: pd.Series, n_bins: int = N_BINS) -> str:
    """numerica (bins por cuantiles), discreta (≤ n_bins valores) o categorica."""
    if not pd.api.types.is_numeric_dtype(referencia) or pd.api.types.is_bool_dtype(
        referencia
    ):
        return "categorica"
    return "discreta" if referencia.nunique(dropna=True) <= n_bins else "numerica"


def drift_variable(referencia: pd.Series, actual: pd.Series) -> dict:
    """PSI, nivel, prueba informativa y categorías nuevas de una variable."""
    tipo = tipo_variable(referencia)
    if tipo == "numerica":
        bordes = bordes_cuantiles(referencia)
        tabla = proporciones(
            bins_numericos(referencia, bordes), bins_numericos(actual, bordes)
        )
        nuevas = []
        prueba = "ks"
        p_valor = (
            float(stats.ks_2samp(referencia.dropna(), actual.dropna()).pvalue)
            if actual.notna().any()
            else 1.0
        )
    else:
        ref_bins, act_bins = como_categoria(referencia), como_categoria(actual)
        tabla = proporciones(ref_bins, act_bins)
        # Solo en las de texto: un valor nuevo en una discreta (por ejemplo, 11
        # actividades pendientes) ya se refleja en el PSI.
        nuevas = sorted(set(act_bins) - set(ref_bins)) if tipo == "categorica" else []
        prueba = "chi_cuadrado"
        conteos = pd.DataFrame(
            {"ref": ref_bins.value_counts(), "act": act_bins.value_counts()}
        ).fillna(0)
        # Sin variación (un solo valor en ambas) no hay prueba posible.
        p_valor = (
            float(stats.chi2_contingency(conteos.to_numpy().T)[1])
            if len(conteos) > 1
            else 1.0
        )

    valor = psi(tabla["referencia"].to_numpy(), tabla["actual"].to_numpy())
    return {
        "tipo": tipo,
        "psi": valor,
        "nivel": nivel_psi(valor),
        "prueba": prueba,
        "p_valor": p_valor,
        "categorias_nuevas": nuevas,
    }


def nivel_psi(valor: float) -> str:
    if valor > PSI_DRIFT:
        return "drift"
    if valor > PSI_MODERADO:
        return "moderado"
    return "estable"


def variables_monitoreadas(df: pd.DataFrame) -> pd.DataFrame:
    """Las variables del modelo (sin ids ni etiqueta) y nunca las EXCLUIDAS."""
    x, _ = separar_x_y(df)
    return x.drop(columns=[c for c in x.columns if c in EXCLUIDAS])


def drift_cohorte(referencia: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Drift de todas las variables de una cohorte, de mayor a menor PSI."""
    x_ref = variables_monitoreadas(referencia)
    x_act = variables_monitoreadas(actual)
    # Los booleanos llegan como 0/1 desde separar_x_y: quedan como discretas.
    variables = {
        columna: drift_variable(x_ref[columna], x_act[columna])
        for columna in x_ref.columns
    }
    variables = dict(sorted(variables.items(), key=lambda kv: -kv[1]["psi"]))
    con_drift = [v for v, r in variables.items() if r["nivel"] == "drift"]
    return {
        "aprendices": len(actual),
        "estado": "drift" if con_drift else "estable",
        "variables_con_drift": con_drift,
        "variables_moderadas": [
            v for v, r in variables.items() if r["nivel"] == "moderado"
        ],
        "categorias_nuevas": {
            v: r["categorias_nuevas"]
            for v, r in variables.items()
            if r["categorias_nuevas"]
        },
        "variables": variables,
    }


# --- Rendimiento -----------------------------------------------------------
def recall_por_cohorte(
    y_real: pd.Series, alerta: np.ndarray, cohortes: pd.Series
) -> dict[str, dict]:
    """Recall de cada cohorte con IC bootstrap (equidad.bootstrap_recall)."""
    tabla, _ = bootstrap_recall(y_real, alerta, cohortes)
    alertas = pd.Series(np.asarray(alerta)).groupby(np.asarray(cohortes)).sum()
    return {
        str(cohorte): {
            "recall": float(fila["recall"]),
            "ic_inferior": float(fila["ic_inferior"]),
            "ic_superior": float(fila["ic_superior"]),
            "desertores": int(fila["desertores"]),
            "alertas": int(alertas[cohorte]),
            # "revisar" no detiene el pipeline: avisa al equipo (ADR 0005).
            "estado": "ok" if fila["recall"] >= RECALL_OBJETIVO else "revisar",
        }
        for cohorte, fila in tabla.iterrows()
    }


# --- Reporte ---------------------------------------------------------------
def construir_reporte(
    train: pd.DataFrame, test: pd.DataFrame, alerta: np.ndarray
) -> dict:
    """Drift y rendimiento de cada cohorte de `test` frente a `train`."""
    rendimiento = recall_por_cohorte(
        test["deserto"], alerta, test[COLUMNA_COHORTE].astype(str)
    )
    cohortes = {}
    for cohorte in sorted(test[COLUMNA_COHORTE].astype(str).unique()):
        actual = test[test[COLUMNA_COHORTE].astype(str) == cohorte]
        cohortes[cohorte] = {
            "drift": drift_cohorte(train, actual),
            "rendimiento": rendimiento[cohorte],
        }
    return {
        "criterios": {
            "n_bins": N_BINS,
            "psi_moderado": PSI_MODERADO,
            "psi_drift": PSI_DRIFT,
            "recall_objetivo": RECALL_OBJETIVO,
        },
        "referencia": {
            "datos": str(RUTA_TRAIN.as_posix()),
            "aprendices": len(train),
        },
        "cohortes": cohortes,
    }


def a_json(reporte: dict) -> str:
    """JSON estable: mismas cifras → mismo texto → mismo hash en dvc.lock."""

    def redondear(valor):
        if isinstance(valor, float):
            return round(valor, 6)
        if isinstance(valor, dict):
            return {k: redondear(v) for k, v in valor.items()}
        if isinstance(valor, list):
            return [redondear(v) for v in valor]
        return valor

    return json.dumps(redondear(reporte), indent=2, ensure_ascii=False) + "\n"


def a_html(reporte: dict) -> str:
    """El reporte en tablas HTML, sin dependencias ni fechas (es reproducible)."""
    partes = [
        "<!doctype html>",
        '<html lang="es"><head><meta charset="utf-8">',
        "<title>Monitoreo del modelo</title>",
        (
            "<style>body{font-family:sans-serif;margin:2rem;max-width:960px}"
            "table{border-collapse:collapse;margin:.5rem 0 1.5rem}"
            "td,th{border:1px solid #ccc;padding:.25rem .6rem;text-align:right}"
            "th{background:#f2f2f2}.drift{background:#f8d7da}"
            ".moderado{background:#fff3cd}.revisar{background:#fff3cd}</style>"
        ),
        "</head><body>",
        "<h1>Monitoreo del modelo: drift y recall por cohorte</h1>",
        (
            f"<p>Referencia: {reporte['referencia']['datos']} "
            f"({reporte['referencia']['aprendices']} aprendices). PSI &gt; "
            f"{PSI_DRIFT} = drift; {PSI_MODERADO}–{PSI_DRIFT} = moderado. KS y "
            "chi-cuadrado son informativos. Recall objetivo: "
            f"{RECALL_OBJETIVO}.</p>"
        ),
    ]
    for cohorte, datos in reporte["cohortes"].items():
        drift, rend = datos["drift"], datos["rendimiento"]
        partes += [
            f"<h2>Cohorte {cohorte} ({drift['aprendices']} aprendices)</h2>",
            (
                f'<p class="{rend["estado"]}">Recall {rend["recall"]:.3f} '
                f"(IC 95 %: {rend['ic_inferior']:.3f}–{rend['ic_superior']:.3f}), "
                f"{rend['desertores']} desertores, {rend['alertas']} alertas. "
                f"Estado: <b>{rend['estado']}</b>.</p>"
            ),
            (
                f"<p>Drift: <b>{drift['estado']}</b>. Con drift: "
                f"{', '.join(drift['variables_con_drift']) or 'ninguna'}. "
                f"Moderadas: {', '.join(drift['variables_moderadas']) or 'ninguna'}. "
                "Categorías nuevas: "
                f"{json.dumps(drift['categorias_nuevas'], ensure_ascii=False)}.</p>"
            ),
            (
                "<table><tr><th>variable</th><th>tipo</th><th>PSI</th><th>nivel</th>"
                "<th>prueba</th><th>p-valor</th></tr>"
            ),
        ]
        for variable, r in drift["variables"].items():
            partes.append(
                f'<tr class="{r["nivel"]}"><td style="text-align:left">{variable}'
                f"</td><td>{r['tipo']}</td><td>{r['psi']:.4f}</td>"
                f"<td>{r['nivel']}</td><td>{r['prueba']}</td>"
                f"<td>{r['p_valor']:.3g}</td></tr>"
            )
        partes.append("</table>")
    partes.append("</body></html>")
    return "\n".join(partes) + "\n"


def guardar(texto: str, ruta: Path) -> None:
    """Con LF también en Windows, para que el hash de dvc.lock sea el mismo."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8", newline="\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    train = pd.read_parquet(RUTA_TRAIN)
    test = pd.read_parquet(RUTA_TEST)
    umbral = json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))["umbral"]
    x_test, _ = separar_x_y(test)
    alerta = cargar_modelo().predict_proba(x_test)[:, 1] >= umbral

    reporte = construir_reporte(train, test, alerta)
    guardar(a_json(reporte), RUTA_DRIFT_JSON)
    guardar(a_html(reporte), RUTA_DRIFT_HTML)

    for cohorte, datos in reporte["cohortes"].items():
        drift, rend = datos["drift"], datos["rendimiento"]
        logger.info(
            "%s · drift: %s (%d con drift, %d moderadas) · recall %.3f "
            "[%.3f–%.3f] · %s",
            cohorte,
            drift["estado"],
            len(drift["variables_con_drift"]),
            len(drift["variables_moderadas"]),
            rend["recall"],
            rend["ic_inferior"],
            rend["ic_superior"],
            rend["estado"],
        )
    logger.info("Reporte en %s y %s", RUTA_DRIFT_JSON, RUTA_DRIFT_HTML)


if __name__ == "__main__":
    main()
