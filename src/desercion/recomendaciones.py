"""Medidas sugeridas al instructor a partir de los factores de riesgo.

Las medidas viven en config/recomendaciones/medidas.yaml para que se puedan
ajustar sin tocar código. Aquí solo está la regla:

1. Sin alerta, no hay medidas: el instructor concentra su tiempo en quienes la
   tienen.
2. Con alerta, cada factor (de mayor a menor aporte) activa la medida de su
   variable, sin repetir, hasta `max_medidas`.
3. Si ningún factor tiene medida propia, se sugiere la medida `general`.

Las medidas son sugerencias para el instructor, no decisiones automáticas.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

RUTA_MEDIDAS = Path("config/recomendaciones/medidas.yaml")


@cache
def cargar_medidas(ruta: Path = RUTA_MEDIDAS) -> dict:
    """Lee medidas.yaml y valida su estructura (una sola vez por proceso).

    Devuelve {"general", "max_medidas", "por_variable", "sin_medida"}, con
    `por_variable` = {variable: texto de su medida}.
    """
    config = yaml.safe_load(ruta.read_text(encoding="utf-8"))

    por_variable: dict[str, str] = {}
    for nombre, medida in config["medidas"].items():
        for variable in medida["variables"]:
            if variable in por_variable:
                raise ValueError(
                    f"'{variable}' aparece en más de una medida ({nombre})"
                )
            por_variable[variable] = medida["texto"].strip()

    sin_medida = list(config.get("sin_medida", []))
    repetidas = set(por_variable) & set(sin_medida)
    if repetidas:
        raise ValueError(f"Variables con medida y en sin_medida: {sorted(repetidas)}")

    return {
        "general": config["general"].strip(),
        "max_medidas": int(config["max_medidas"]),
        "por_variable": por_variable,
        "sin_medida": sin_medida,
    }


def medidas_para(
    factores: list[dict], alerta: bool, config: dict | None = None
) -> list[str]:
    """Medidas sugeridas para un aprendiz.

    `factores` viene de `explain.principales_factores`: [{"variable", "aporte"}]
    ordenados de mayor a menor aporte.
    """
    if not alerta:
        return []
    config = config or cargar_medidas()

    medidas: list[str] = []
    for factor in factores:
        texto = config["por_variable"].get(factor["variable"])
        if texto and texto not in medidas:
            medidas.append(texto)
    return medidas[: config["max_medidas"]] or [config["general"]]
