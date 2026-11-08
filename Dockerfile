# Imagen de la API de alerta temprana (ADR 0004).
#
# Antes de construir hay que tener los artefactos que versiona DVC:
#     uv run dvc pull
#     docker build -t desercion-api .
#     docker run --rm -p 8000:8000 desercion-api
#     → http://127.0.0.1:8000/health y http://127.0.0.1:8000/docs
#
# Dos etapas: la primera instala las dependencias con uv; la segunda solo lleva
# Python, el entorno ya instalado y los archivos que la API lee al arrancar.

# --- Etapa 1: dependencias ----------------------------------------------------
FROM python:3.11-slim AS dependencias

# uv en la misma versión que usa el equipo, copiado de su imagen oficial.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Primero solo las dependencias (cambian poco): Docker reutiliza esta capa
# mientras pyproject.toml y uv.lock no cambien, aunque cambie el código.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# Después el paquete. --no-editable lo instala dentro de .venv, así la imagen
# final no necesita la carpeta src/.
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# --- Etapa 2: imagen final ----------------------------------------------------
FROM python:3.11-slim

# Sin root: si alguien compromete la API, no controla el contenedor.
RUN useradd --create-home --uid 1000 api

WORKDIR /app

COPY --from=dependencias /app/.venv /app/.venv

# Lo que la API lee al arrancar, con las mismas rutas relativas del repositorio
# (modelado.py, explain.py, recomendaciones.py).
COPY config/recomendaciones/medidas.yaml config/recomendaciones/
COPY models/modelo.skops models/referencia.json models/
COPY reports/metricas_modelo_base.json reports/
COPY data/features/test.parquet data/features/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER api
EXPOSE 8000

# Docker marca el contenedor como "unhealthy" si /health no responde 200
# (por ejemplo, si no cargó el modelo y la API responde 503).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "desercion.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
