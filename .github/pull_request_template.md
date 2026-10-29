## Qué hace este PR

<!-- Describe el cambio en 2 o 3 líneas. -->

Cierra #<!-- número del Issue -->

## Cómo probarlo

```powershell
git fetch
git checkout NOMBRE-DE-LA-RAMA
uv sync
uv run dvc repro
uv run pytest
```

## Checklist del autor

- [ ] `uv run ruff check .` sin errores
- [ ] `uv run pytest` pasa
- [ ] Si cambió el pipeline, ejecuté `uv run dvc repro` y subí `dvc.lock`
- [ ] Si es un cuaderno, lo ejecuté con Restart → Run All
- [ ] Actualicé `docs/plan_trabajo.md`

## Checklist del revisor

- [ ] Leí el código y lo entiendo
- [ ] Lo ejecuté en mi computador con los mismos resultados
- [ ] No se suben datos, modelos ni secretos
