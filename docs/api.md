# API de riesgo de deserción

La API entrega al instructor el riesgo de deserción de sus aprendices, los factores que
lo explican y medidas sugeridas. Está hecha con FastAPI (`src/desercion/api/`) y su
documentación interactiva se genera sola en `/docs`. Las decisiones de diseño están en
el [ADR 004](decisiones/0004-api-consulta-y-datos.md).

## Cómo levantarla

Primero hay que descargar el modelo y los datos que versiona DVC:

```powershell
uv run dvc pull
```

**Con Docker** (como se despliega):

```powershell
docker build -t desercion-api .
docker run --rm -d -p 8000:8000 --name api desercion-api
```

**Sin Docker** (para desarrollo):

```powershell
uv run uvicorn desercion.api.main:app --reload
```

La API tarda unos segundos en arrancar, porque calcula el riesgo de todos los
aprendices antes de aceptar consultas. Luego responde en <http://127.0.0.1:8000>:
`/health` para ver el estado y `/docs` para probar cada endpoint desde el navegador.

## Qué carga al arrancar

| Archivo | Para qué | Lo genera |
|---|---|---|
| `models/modelo.skops` | El modelo (regresión logística) | etapa `train` |
| `reports/metricas_modelo_base.json` | El umbral de alerta (0.461) | etapa `train` |
| `models/referencia.json` | El aprendiz promedio de train, para calcular los factores | etapa `explain` |
| `data/features/test.parquet` | Los aprendices activos (en la demo, las cohortes 2025-T3 y 2025-T4) | etapa `features` |
| `config/recomendaciones/medidas.yaml` | Las medidas sugeridas | se edita a mano |

Con eso calcula una sola vez la probabilidad, la alerta, los factores y las medidas de
cada aprendiz (`prediccion.predecir_riesgo`), y después solo consulta esa tabla. Si
algo falla al cargar, la API arranca igual y `/health` informa el motivo con un 503.
Para tomar datos o un modelo nuevos hay que reiniciarla.

## Endpoints

| Método y ruta | Qué devuelve | Códigos |
|---|---|---|
| `GET /health` | Si la API puede responder y con qué modelo | 200, 503 |
| `GET /aprendices/{id_aprendiz}/riesgo` | El riesgo de un aprendiz | 200, 404, 422, 503 |
| `GET /cohortes/{id_cohorte}/riesgo` | Los aprendices de una cohorte, de mayor a menor riesgo | 200, 404, 422, 503 |

### `GET /health`

```json
{
  "estado": "ok",
  "modelo_cargado": true,
  "version_modelo": "f0dd9683c0396c196eb6cd409b605c29",
  "umbral": 0.4606367905142532,
  "aprendices": 1544,
  "detalle": null
}
```

| Campo | Significado |
|---|---|
| `estado` | `ok`; `sin_modelo` si no cargó el modelo o el umbral; `sin_datos` si el modelo cargó pero faltan los aprendices o la referencia, o la referencia es de otro modelo |
| `version_modelo` | md5 de `models/modelo.skops`. Debe coincidir con el de `dvc.lock`: así se sabe qué modelo está sirviendo la API |
| `aprendices` | Cuántos aprendices activos tiene cargados |
| `detalle` | Si el estado no es `ok`, qué falló y cómo resolverlo (por ejemplo, `uv run dvc repro explain`) |

Si el estado no es `ok`, responde con código 503 y el mismo formato.

### `GET /aprendices/{id_aprendiz}/riesgo`

`id_aprendiz` tiene la forma `A-` y seis dígitos. Ejemplo:
`GET /aprendices/A-007925/riesgo`

```json
{
  "id_aprendiz": "A-007925",
  "probabilidad": 0.868141321837692,
  "alerta": true,
  "umbral": 0.4606367905142532,
  "factores": [
    {"variable": "porcentaje_actividades_aprobadas", "aporte": 0.852154256192562},
    {"variable": "asistencia_porcentaje", "aporte": 0.526702669891089},
    {"variable": "promedio_calificaciones", "aporte": 0.32414265930411534}
  ],
  "medidas": [
    "Revisar con el aprendiz las actividades reprobadas y ofrecer un plan de mejoramiento o una tutoría.",
    "Hablar con el aprendiz sobre sus inasistencias y registrar la causa en el seguimiento."
  ]
}
```

Responde 404 si el aprendiz no está entre los activos, y 422 si el código no tiene el
formato correcto.

### `GET /cohortes/{id_cohorte}/riesgo`

`id_cohorte` tiene la forma `AAAA-T1` a `AAAA-T4` (por ejemplo, `2025-T3`). Los datos
no traen un número de ficha: el instructor consulta su cohorte y la acota con filtros.

| Parámetro | Valores | Por defecto |
|---|---|---|
| `modalidad` | `virtual`, `presencial`, `mixta` | todas |
| `area_formacion` | `administracion`, `agropecuaria`, `industria`, `salud`, `tecnologia`, `turismo y gastronomia` | todas |
| `nivel_formacion` | `AUXILIAR`, `OPERARIO`, `TÉCNICO`, `TECNÓLOGO`, `ESPECIALIZACIÓN TECNOLÓGICA` | todos |
| `solo_alertas` | `true` o `false` | `false` |
| `limite` | de 1 a 500 | 50 |

Los valores permitidos son los mismos del contrato de datos (`validate.py`); otro valor
responde 422. Ejemplo:
`GET /cohortes/2025-T3/riesgo?modalidad=presencial&solo_alertas=true&limite=1`

```json
{
  "id_cohorte": "2025-T3",
  "total_aprendices": 66,
  "total_alertas": 66,
  "aprendices": [
    {"id_aprendiz": "A-007925", "probabilidad": 0.868141321837692, "alerta": true, "...": "..."}
  ]
}
```

`total_aprendices` y `total_alertas` cuentan a todos los que cumplen los filtros, antes
de aplicar `limite`. La lista va de mayor a menor probabilidad. Responde 404 si la
cohorte no tiene aprendices activos.

## Cómo leer una respuesta

- **`probabilidad`:** riesgo estimado de desertar, entre 0 y 1.
- **`alerta`:** `true` si la probabilidad es mayor o igual al umbral (0.461). El umbral
  se eligió para detectar al menos el 70 % de los desertores
  ([ADR 003](decisiones/0003-umbral-y-equidad.md)).
- **`factores`:** hasta 3 variables que más suben el riesgo de ese aprendiz frente al
  aprendiz promedio de entrenamiento. El `aporte` está en log-odds y solo se muestran
  los que aportan al menos 0.05; por debajo es ruido.
- **`medidas`:** solo cuando hay alerta. Hasta 3, en orden de aporte y sin repetir (dos
  factores pueden compartir medida). Se definen en `config/recomendaciones/medidas.yaml`
  y nunca se sugieren por características personales como la edad o el nivel
  educativo. `porcentaje_actividades_entregadas` tampoco genera medida: por
  colinealidad sale como factor en aprendices que entregan más que el promedio (ver la
  [model card](model_card.md)).
- **Lo que nunca sale:** la etiqueta `deserto` (si el aprendiz desertó) no aparece en
  ninguna respuesta; una prueba automática lo verifica.

## Errores

| Código | Cuándo | Cuerpo |
|---|---|---|
| 404 | El aprendiz o la cohorte no tienen datos activos | `{"detail": "..."}` |
| 422 | Un parámetro con formato o valor no permitido | el detalle de validación de FastAPI |
| 503 | La API arrancó pero no cargó el modelo, el umbral, los aprendices o la referencia | en `/health`, el formato de `Salud`; en los demás, `{"detail": "..."}` |

## Limitaciones

- **Sin autenticación:** cualquiera puede consultar cualquier cohorte. Es aceptable en
  la demo solo porque los datos son sintéticos; en producción habría que autenticar a
  cada instructor, limitarlo a sus grupos y servir la API por HTTPS.
- **Datos fijos al arrancar:** para tomar una cohorte nueva o un modelo reentrenado hay
  que reiniciar la API (o reconstruir la imagen). Para todo el SENA, el cálculo se haría
  en lote y se guardaría en una base de datos que la API consulte.
- **Las medidas son sugerencias** para el instructor, no decisiones automáticas.
