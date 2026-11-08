"""Contrato de la API: qué devuelve cada endpoint.

Los esquemas de Pydantic validan las respuestas y FastAPI los usa para generar
la documentación en /docs. Si un endpoint intenta devolver algo que no cumple el
esquema (una probabilidad de 1.3, un campo faltante), la API falla en lugar de
entregar un dato equivocado al instructor.

Los usan /health, /aprendices/{id}/riesgo y /cohortes/{id}/riesgo (ADR 0004).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Salud(BaseModel):
    """Respuesta de /health: si la API puede predecir y con qué modelo."""

    # sin_modelo: no cargó el modelo o el umbral. sin_datos: el modelo cargó, pero
    # no los aprendices activos o su riesgo.
    estado: Literal["ok", "sin_modelo", "sin_datos"]
    modelo_cargado: bool
    # md5 de models/modelo.skops: se compara con dvc.lock para saber qué modelo
    # está sirviendo la API.
    version_modelo: str | None = None
    umbral: float | None = Field(default=None, gt=0, lt=1)
    # Aprendices activos con el riesgo ya calculado al arrancar.
    aprendices: int | None = Field(default=None, ge=0)
    detalle: str | None = None


class Factor(BaseModel):
    """Una variable que sube el riesgo del aprendiz frente al aprendiz promedio."""

    variable: str
    # Aporte al log-odds (explain.aportes). Siempre positivo: solo se muestran
    # los factores que suben el riesgo.
    aporte: float = Field(gt=0)


class RiesgoAprendiz(BaseModel):
    """Riesgo de deserción de un aprendiz en la fecha de corte."""

    id_aprendiz: str = Field(pattern=r"^A-\d{6}$")
    probabilidad: float = Field(ge=0, le=1)
    # True si probabilidad >= umbral (elegido con out-of-fold en train).
    alerta: bool
    umbral: float = Field(gt=0, lt=1)
    factores: list[Factor] = Field(default_factory=list, max_length=5)
    medidas: list[str] = Field(default_factory=list)


class RiesgoCohorte(BaseModel):
    """Aprendices de una cohorte, filtrados y ordenados de mayor a menor riesgo.

    No hay id de ficha en los datos (ADR 0004): el instructor acota la lista con
    filtros sobre columnas reales (modalidad, área, nivel, solo alertas).
    """

    id_cohorte: str = Field(pattern=r"^\d{4}-T[1-4]$")
    # Conteos después de aplicar los filtros y antes de aplicar el límite.
    total_aprendices: int = Field(ge=0)
    total_alertas: int = Field(ge=0)
    aprendices: list[RiesgoAprendiz]
