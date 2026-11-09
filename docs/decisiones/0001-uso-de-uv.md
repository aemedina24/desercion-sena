# ADR 001: uv para el entorno y las dependencias

- **Estado:** Aceptada
- **Fecha:** 2026-10-28
- **Decisores:** Andrés Medina, David Montilla

## Contexto

Dos personas trabajan en Windows, y el mismo código corre además en GitHub Actions
(Linux) y en la imagen Docker de la API. DVC identifica cada etapa del pipeline por el
hash de su código y de sus datos: si dos entornos instalan versiones distintas de una
librería, el mismo pipeline puede dar resultados distintos y nadie sabe cuál es el
correcto. Hacía falta que todos tuvieran exactamente el mismo entorno, sin pasos
manuales.

Opciones evaluadas:

| Opción | Problema |
|---|---|
| `pip` + `requirements.txt` | Fija solo lo que uno escribe; las dependencias de las dependencias quedan libres salvo que se congele el entorno a mano. No separa las herramientas de desarrollo. |
| Conda | Entornos pesados y lentos de resolver; mezcla paquetes de dos fuentes (conda y pip) y en Windows interfiere con otros entornos (el `(base)` activo). |
| Poetry | Cumple con el lock, pero es más lento y no instala Python. |
| **uv** | Un solo archivo de lock para todos los sistemas, instala Python y es muy rápido. |

## Decisión

Usar **uv** con:

- `pyproject.toml` como única lista de dependencias, y `uv.lock` con la versión exacta
  de cada paquete, incluidas las dependencias indirectas. Ambos van en Git.
- Python 3.11, fijado en `.python-version`.
- Dos grupos: las dependencias principales (pipeline y API) y el grupo `dev` (DVC,
  pytest, Ruff, Jupyter, gráficas, cliente de pruebas de la API).
- Todo se ejecuta con `uv run ...` (incluido DVC: `uv run dvc repro`), para no depender
  de qué entorno tenga activo cada uno.
- Una dependencia nueva se agrega con `uv add` y entra en un PR junto con el `uv.lock`
  actualizado.

## Consecuencias

- **Mismo entorno en todas partes:** la CI instala con `uv sync --locked`, que falla si
  `uv.lock` no corresponde a `pyproject.toml`; la imagen Docker instala con
  `uv sync --frozen --no-dev`, sin las herramientas de desarrollo.
- **Reproducibilidad comprobada:** la comparación de modelos dio los mismos resultados
  en Windows y en Linux, y `dvc.lock` tiene los mismos hashes en los dos sistemas (junto
  con `.gitattributes` para los saltos de línea).
- **En Windows**, `uv sync` falla con `os error 32` si un kernel de Jupyter o un
  `uv run python` tiene abierto un archivo del entorno: hay que cerrarlos antes.
- **Las versiones quedan fijas** hasta que alguien las actualice a propósito. Por eso la
  incompatibilidad entre Evidently y pandas 3 se detectó al agregar Evidently, sin
  romper el entorno de nadie ([ADR 005](0005-monitoreo-sin-evidently.md)).
