"""API del sistema de alerta temprana.

El modelo se carga una sola vez, al arrancar (lifespan), y queda en app.state.
Si no se puede cargar, la API arranca igual pero /health responde 503: así un
orquestador (Docker, un balanceador) sabe que no debe enviarle tráfico, y quien
la opera ve el motivo en "detalle".

Uso (desde la raíz del repositorio):
    uv run uvicorn desercion.api.main:app --reload
    Documentación interactiva: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response, status

from desercion.api.schemas import Salud
from desercion.modelado import RUTA_METRICAS, RUTA_MODELO, cargar_modelo

logger = logging.getLogger(__name__)


def md5_archivo(ruta: Path) -> str:
    """El mismo hash que registra dvc.lock para ese archivo."""
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def crear_app(
    ruta_modelo: Path = RUTA_MODELO, ruta_metricas: Path = RUTA_METRICAS
) -> FastAPI:
    """Construye la API. Las rutas son parámetros para poder probarla."""

    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
        app.state.modelo = None
        app.state.umbral = None
        app.state.version_modelo = None
        app.state.error_carga = None
        try:
            app.state.modelo = cargar_modelo(ruta_modelo)
            metricas = json.loads(ruta_metricas.read_text(encoding="utf-8"))
            app.state.umbral = float(metricas["umbral"])
            app.state.version_modelo = md5_archivo(ruta_modelo)
            logger.info("Modelo cargado: %s", app.state.version_modelo)
        # Cualquier fallo (archivo ausente, corrupto, tipo no confiable, JSON sin
        # umbral) se reporta en /health en vez de impedir que la API arranque.
        except Exception as error:  # noqa: BLE001
            app.state.modelo = None
            app.state.error_carga = f"{type(error).__name__}: {error}"
            logger.error("No se pudo cargar el modelo: %s", app.state.error_carga)
        yield

    app = FastAPI(
        title="Alerta temprana de deserción SENA",
        description="Riesgo de deserción por aprendiz y por ficha (semana 4).",
        version="0.1.0",
        lifespan=ciclo_de_vida,
    )

    @app.get("/health", response_model=Salud, tags=["operación"])
    def salud(response: Response) -> Salud:
        """Estado del servicio: 200 si puede predecir, 503 si no."""
        if app.state.modelo is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return Salud(
                estado="sin_modelo",
                modelo_cargado=False,
                detalle=app.state.error_carga,
            )
        return Salud(
            estado="ok",
            modelo_cargado=True,
            version_modelo=app.state.version_modelo,
            umbral=app.state.umbral,
        )

    return app


app = crear_app()
