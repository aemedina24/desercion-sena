"""Medidas y plan de acción sugeridos al instructor a partir de los factores de riesgo.

Las medidas viven en config/recomendaciones/medidas.yaml para que se puedan
ajustar sin tocar código. Aquí solo está la regla:

1. Sin alerta, no hay medidas: el instructor concentra su tiempo en quienes la
   tienen.
2. Con alerta, cada factor (de mayor a menor aporte) activa la medida de su
   variable, sin repetir, hasta `max_medidas`.
3. Si ningún factor tiene medida propia, se sugiere la medida `general`.

Cada medida se entrega de dos formas, siempre en el mismo orden:

- `medidas_para`: el texto corto (una frase).
- `plan_para`: el plan de acción, que le dice al instructor qué hacer, en qué
  plazo, qué decir, qué evitar, qué hacer si el aprendiz no responde, qué
  registrar y cuándo volver a revisar (ADR 0006).

Las medidas y los guiones son sugerencias para el instructor, no decisiones
automáticas.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

RUTA_MEDIDAS = Path("config/recomendaciones/medidas.yaml")

# Nombre con que se identifica la medida general en el plan de acción.
MEDIDA_GENERAL = "general"
# Campos que debe traer el plan de cada medida, con su tipo.
CAMPOS_PLAN = {
    "accion": str,
    "plazo_dias": int,
    "decir": list,
    "evitar": str,
    "si_no_responde": str,
    "registrar": str,
    "seguimiento_dias": int,
}


def _validar_plan(nombre: str, plan: dict | None) -> dict:
    """Comprueba que el plan de una medida trae todos los campos y los limpia."""
    # Una medida sin plan se reporta igual que una a la que le faltan campos.
    plan = plan if isinstance(plan, dict) else {}
    if set(plan) != set(CAMPOS_PLAN):
        faltan = sorted(set(CAMPOS_PLAN) - set(plan))
        sobran = sorted(set(plan) - set(CAMPOS_PLAN))
        raise ValueError(
            f"Plan de '{nombre}': faltan los campos {faltan} y sobran {sobran}"
        )
    for campo, tipo in CAMPOS_PLAN.items():
        if not isinstance(plan[campo], tipo) or not plan[campo]:
            raise ValueError(
                f"Plan de '{nombre}': '{campo}' debe ser {tipo.__name__} y no estar vacío"
            )
    if not all(isinstance(frase, str) and frase.strip() for frase in plan["decir"]):
        raise ValueError(f"Plan de '{nombre}': 'decir' debe ser una lista de frases")
    if plan["plazo_dias"] < 1 or plan["seguimiento_dias"] < 1:
        raise ValueError(f"Plan de '{nombre}': los días deben ser 1 o más")

    return {
        "medida": nombre,
        **{
            campo: valor.strip() if isinstance(valor, str) else valor
            for campo, valor in plan.items()
        },
        "decir": [frase.strip() for frase in plan["decir"]],
    }


@cache
def cargar_medidas(ruta: Path = RUTA_MEDIDAS) -> dict:
    """Lee medidas.yaml y valida su estructura (una sola vez por proceso).

    Devuelve:
    - `general` y `max_medidas`.
    - `por_variable`: {variable: texto de su medida}.
    - `medida_por_variable`: {variable: nombre de su medida}.
    - `textos`: {nombre de la medida: texto}, incluida la general.
    - `planes`: {nombre de la medida: plan de acción}, incluida la general.
    - `sin_medida`: variables que no generan medida.
    """
    config = yaml.safe_load(ruta.read_text(encoding="utf-8"))

    por_variable: dict[str, str] = {}
    medida_por_variable: dict[str, str] = {}
    textos = {MEDIDA_GENERAL: config["general"].strip()}
    planes = {MEDIDA_GENERAL: _validar_plan(MEDIDA_GENERAL, config.get("plan_general"))}
    for nombre, medida in config["medidas"].items():
        if nombre == MEDIDA_GENERAL:
            raise ValueError(
                f"'{MEDIDA_GENERAL}' está reservado para la medida general"
            )
        textos[nombre] = medida["texto"].strip()
        planes[nombre] = _validar_plan(nombre, medida.get("plan"))
        for variable in medida["variables"]:
            if variable in por_variable:
                raise ValueError(
                    f"'{variable}' aparece en más de una medida ({nombre})"
                )
            por_variable[variable] = textos[nombre]
            medida_por_variable[variable] = nombre

    sin_medida = list(config.get("sin_medida", []))
    repetidas = set(por_variable) & set(sin_medida)
    if repetidas:
        raise ValueError(f"Variables con medida y en sin_medida: {sorted(repetidas)}")

    return {
        "general": textos[MEDIDA_GENERAL],
        "max_medidas": int(config["max_medidas"]),
        "por_variable": por_variable,
        "medida_por_variable": medida_por_variable,
        "textos": textos,
        "planes": planes,
        "sin_medida": sin_medida,
    }


def _medidas_activas(factores: list[dict], config: dict) -> list[str]:
    """Nombres de las medidas que activan los factores, en orden y sin repetir."""
    nombres: list[str] = []
    for factor in factores:
        nombre = config["medida_por_variable"].get(factor["variable"])
        if nombre and nombre not in nombres:
            nombres.append(nombre)
    return nombres[: config["max_medidas"]] or [MEDIDA_GENERAL]


def medidas_para(
    factores: list[dict], alerta: bool, config: dict | None = None
) -> list[str]:
    """Medidas sugeridas para un aprendiz (texto corto de cada una).

    `factores` viene de `explain.principales_factores`: [{"variable", "aporte"}]
    ordenados de mayor a menor aporte.
    """
    if not alerta:
        return []
    config = config or cargar_medidas()
    return [config["textos"][nombre] for nombre in _medidas_activas(factores, config)]


def plan_para(
    factores: list[dict], alerta: bool, config: dict | None = None
) -> list[dict]:
    """Plan de acción para un aprendiz: uno por cada medida de `medidas_para`.

    Cada plan trae `medida` (su nombre en medidas.yaml) y los campos de
    CAMPOS_PLAN. Va en el mismo orden que `medidas_para`: el primero responde al
    factor que más sube el riesgo.
    """
    if not alerta:
        return []
    config = config or cargar_medidas()
    return [
        # Copia: el resultado de cargar_medidas se comparte entre llamadas.
        {**config["planes"][nombre], "decir": list(config["planes"][nombre]["decir"])}
        for nombre in _medidas_activas(factores, config)
    ]
