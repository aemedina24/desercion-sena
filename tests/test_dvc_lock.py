"""El dvc.lock debe corresponder al código que está en Git.

Si alguien ejecuta `dvc repro` en Windows con un archivo en CRLF y luego Git lo
normaliza a LF, dvc.lock queda con un hash que nadie más puede reproducir: en
cualquier copia limpia DVC ve la etapa como modificada (pasó en los PR #7 y #13).

La prueba solo revisa las dependencias de código (src/ y config/), que sí están en
Git y en el CI. Los datos no se revisan aquí: no están en Git.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
CARPETAS_DE_CODIGO = ("src/", "config/")


def _dependencias_de_codigo() -> list[tuple[str, str, str]]:
    """(etapa, ruta, md5 registrado) de cada dependencia de código en dvc.lock."""
    lock = yaml.safe_load((RAIZ / "dvc.lock").read_text(encoding="utf-8"))
    return [
        (etapa, dep["path"], dep["md5"])
        for etapa, datos in lock["stages"].items()
        for dep in datos.get("deps", [])
        if dep["path"].startswith(CARPETAS_DE_CODIGO)
    ]


@pytest.mark.parametrize(("etapa", "ruta", "md5_lock"), _dependencias_de_codigo())
def test_hash_de_codigo_coincide_con_dvc_lock(etapa, ruta, md5_lock):
    md5_archivo = hashlib.md5((RAIZ / ruta).read_bytes()).hexdigest()
    assert md5_archivo == md5_lock, (
        f"dvc.lock (etapa {etapa!r}) registra {ruta} con md5 {md5_lock}, pero el "
        f"archivo en Git da {md5_archivo}. Si el código no cambió, casi seguro el "
        "lock se generó con saltos CRLF: ejecuta `uv run dvc repro` desde una copia "
        "en LF y sube el dvc.lock resultante."
    )
