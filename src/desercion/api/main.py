"""API del sistema de alerta temprana (ADR 0004).

Al arrancar (lifespan) se carga todo una sola vez:

1. El modelo (models/modelo.skops) y el umbral (reports/metricas_modelo_base.json).
2. Los aprendices activos (data/features/test.parquet en la demo; en producción
   vendrían de la base del SENA).
3. El riesgo de todos, calculado con `predecir`: probabilidad, alerta, factores
   y medidas.

Los endpoints solo consultan esa tabla. Si algo falla al arrancar, la API arranca
igual: /health responde 503 con el motivo y los endpoints de riesgo también 503.

Uso (desde la raíz del repositorio):
    uv run uvicorn desercion.api.main:app --reload
    Documentación interactiva: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi import Path as RutaParam

from desercion.api.schemas import RiesgoAprendiz, RiesgoCohorte, Salud
from desercion.modelado import (
    RUTA_METRICAS,
    RUTA_MODELO,
    RUTA_TEST,
    cargar_modelo,
    separar_x_y,
)
from desercion.validate import AREAS_FORMACION, MODALIDADES, NIVELES_FORMACION

logger = logging.getLogger(__name__)

# Una función que recibe los aprendices activos y devuelve una fila por aprendiz.
Predictor = Callable[[pd.DataFrame], pd.DataFrame]

# Columnas que debe devolver el predictor (contrato con predecir_riesgo).
COLUMNAS_PREDICCION = ["id_aprendiz", "probabilidad", "alerta", "factores", "medidas"]
# Columnas de los aprendices que se usan para filtrar. Nunca se expone `deserto`.
COLUMNAS_FILTRO = [
    "id_aprendiz",
    "id_cohorte",
    "modalidad",
    "area_formacion",
    "nivel_formacion",
]
LIMITE_POR_DEFECTO = 50
LIMITE_MAXIMO = 500

# Valores permitidos en los filtros: los mismos del contrato de datos (validate.py).
# Un valor fuera de la lista responde 422 y /docs los muestra como desplegable.
Modalidad = Literal[tuple(MODALIDADES)]  # type: ignore[valid-type]
Area = Literal[tuple(AREAS_FORMACION)]  # type: ignore[valid-type]
Nivel = Literal[tuple(NIVELES_FORMACION)]  # type: ignore[valid-type]


def md5_archivo(ruta: Path) -> str:
    """El mismo hash que registra dvc.lock para ese archivo."""
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def predictor_sin_explicacion(modelo, umbral: float) -> Predictor:
    """Predictor provisional: probabilidad y alerta, sin factores ni medidas.

    Se reemplaza por `predecir_riesgo` (Andrés) cuando estén `referencia.json`,
    `medidas.yaml` y `recomendaciones.py`.
    """

    def predecir(aprendices: pd.DataFrame) -> pd.DataFrame:
        x, _ = separar_x_y(aprendices)
        probabilidad = modelo.predict_proba(x)[:, 1]
        return pd.DataFrame(
            {
                "id_aprendiz": aprendices["id_aprendiz"].to_numpy(),
                "probabilidad": probabilidad,
                "alerta": probabilidad >= umbral,
                "factores": [[] for _ in range(len(aprendices))],
                "medidas": [[] for _ in range(len(aprendices))],
            }
        )

    return predecir


def construir_tabla_riesgo(
    aprendices: pd.DataFrame, prediccion: pd.DataFrame
) -> pd.DataFrame:
    """Une el riesgo con las columnas de filtro y ordena de mayor a menor riesgo.

    Falla si el predictor no cumple el contrato: mejor un 503 al arrancar que una
    respuesta incompleta.
    """
    faltantes = set(COLUMNAS_PREDICCION) - set(prediccion.columns)
    if faltantes:
        raise ValueError(f"El predictor no devolvió las columnas {sorted(faltantes)}")
    if len(prediccion) != len(aprendices):
        raise ValueError(
            f"El predictor devolvió {len(prediccion)} filas para {len(aprendices)} aprendices"
        )
    tabla = aprendices[COLUMNAS_FILTRO].merge(
        prediccion[COLUMNAS_PREDICCION], on="id_aprendiz", how="inner", validate="1:1"
    )
    return tabla.sort_values("probabilidad", ascending=False, ignore_index=True)


def a_respuesta(fila: pd.Series, umbral: float) -> RiesgoAprendiz:
    return RiesgoAprendiz(
        id_aprendiz=fila["id_aprendiz"],
        probabilidad=float(fila["probabilidad"]),
        alerta=bool(fila["alerta"]),
        umbral=umbral,
        factores=list(fila["factores"]),
        medidas=list(fila["medidas"]),
    )


def crear_app(
    ruta_modelo: Path = RUTA_MODELO,
    ruta_metricas: Path = RUTA_METRICAS,
    ruta_aprendices: Path = RUTA_TEST,
    predecir: Predictor | None = None,
) -> FastAPI:
    """Construye la API.

    Las rutas y el predictor son parámetros para poder probarla: las pruebas usan
    archivos temporales y un predictor falso con valores conocidos. Si `predecir`
    es None se usa el predictor provisional con el modelo cargado.
    """

    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
        app.state.modelo = None
        app.state.umbral = None
        app.state.version_modelo = None
        app.state.riesgo = None
        app.state.error_carga = None
        try:
            modelo = cargar_modelo(ruta_modelo)
            metricas = json.loads(ruta_metricas.read_text(encoding="utf-8"))
            # El modelo solo cuenta como cargado si también hay umbral.
            app.state.umbral = float(metricas["umbral"])
            app.state.modelo = modelo
            app.state.version_modelo = md5_archivo(ruta_modelo)

            aprendices = pd.read_parquet(ruta_aprendices)
            funcion = predecir or predictor_sin_explicacion(
                app.state.modelo, app.state.umbral
            )
            app.state.riesgo = construir_tabla_riesgo(aprendices, funcion(aprendices))
            logger.info(
                "Modelo %s · %s aprendices activos",
                app.state.version_modelo,
                len(app.state.riesgo),
            )
        # Cualquier fallo (archivo ausente o corrupto, tipo no confiable, JSON sin
        # umbral, predictor fuera de contrato) se reporta en /health en vez de
        # impedir que la API arranque.
        except Exception as error:  # noqa: BLE001
            app.state.riesgo = None
            app.state.error_carga = f"{type(error).__name__}: {error}"
            logger.error("No se pudo cargar la API: %s", app.state.error_carga)
        yield

    app = FastAPI(
        title="Alerta temprana de deserción SENA",
        description=(
            "Riesgo de deserción por aprendiz y por cohorte en la semana 4. "
            "Umbral global (ADR 0003); consulta por cohorte con filtros (ADR 0004)."
        ),
        version="0.2.0",
        lifespan=ciclo_de_vida,
    )

    def tabla_riesgo() -> pd.DataFrame:
        if app.state.riesgo is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"La API no está lista: {app.state.error_carga}",
            )
        return app.state.riesgo

    @app.get("/health", response_model=Salud, tags=["operación"])
    def salud(response: Response) -> Salud:
        """Estado del servicio: 200 si puede responder, 503 si no."""
        if app.state.riesgo is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            modelo_cargado = app.state.modelo is not None
            return Salud(
                estado="sin_datos" if modelo_cargado else "sin_modelo",
                modelo_cargado=modelo_cargado,
                version_modelo=app.state.version_modelo,
                umbral=app.state.umbral,
                detalle=app.state.error_carga,
            )
        return Salud(
            estado="ok",
            modelo_cargado=True,
            version_modelo=app.state.version_modelo,
            umbral=app.state.umbral,
            aprendices=len(app.state.riesgo),
        )

    @app.get(
        "/aprendices/{id_aprendiz}/riesgo",
        response_model=RiesgoAprendiz,
        tags=["riesgo"],
        responses={404: {"description": "El aprendiz no está entre los activos"}},
    )
    def riesgo_aprendiz(
        id_aprendiz: Annotated[str, RutaParam(pattern=r"^A-\d{6}$")],
    ) -> RiesgoAprendiz:
        """Riesgo de un aprendiz: probabilidad, alerta, factores y medidas."""
        tabla = tabla_riesgo()
        fila = tabla[tabla["id_aprendiz"] == id_aprendiz]
        if fila.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No hay un aprendiz activo con id {id_aprendiz}",
            )
        return a_respuesta(fila.iloc[0], app.state.umbral)

    @app.get(
        "/cohortes/{id_cohorte}/riesgo",
        response_model=RiesgoCohorte,
        tags=["riesgo"],
        responses={404: {"description": "La cohorte no tiene aprendices activos"}},
    )
    def riesgo_cohorte(
        id_cohorte: Annotated[str, RutaParam(pattern=r"^\d{4}-T[1-4]$")],
        modalidad: Modalidad | None = None,
        area_formacion: Area | None = None,
        nivel_formacion: Nivel | None = None,
        solo_alertas: bool = False,
        limite: Annotated[int, Query(ge=1, le=LIMITE_MAXIMO)] = LIMITE_POR_DEFECTO,
    ) -> RiesgoCohorte:
        """Aprendices de una cohorte de mayor a menor riesgo, con filtros opcionales."""
        tabla = tabla_riesgo()
        cohorte = tabla[tabla["id_cohorte"] == id_cohorte]
        if cohorte.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No hay aprendices activos en la cohorte {id_cohorte}",
            )
        filtros = {
            "modalidad": modalidad,
            "area_formacion": area_formacion,
            "nivel_formacion": nivel_formacion,
        }
        for columna, valor in filtros.items():
            if valor is not None:
                cohorte = cohorte[cohorte[columna] == valor]
        if solo_alertas:
            cohorte = cohorte[cohorte["alerta"]]

        return RiesgoCohorte(
            id_cohorte=id_cohorte,
            total_aprendices=len(cohorte),
            total_alertas=int(cohorte["alerta"].sum()),
            aprendices=[
                a_respuesta(fila, app.state.umbral)
                for _, fila in cohorte.head(limite).iterrows()
            ],
        )

    return app


app = crear_app()
