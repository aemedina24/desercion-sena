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
| `reports/metricas_modelo_base.json` | El umbral de alerta (0.491) | etapa `train` |
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
| `GET /cohortes` | Las cohortes con aprendices activos, con su número de aprendices y de alertas | 200, 503 |
| `GET /instructor` | La vista del instructor: una página web que usa estos mismos endpoints | 200 |

### `GET /health`

```json
{
  "estado": "ok",
  "modelo_cargado": true,
  "version_modelo": "9b58e3fead5766bcb03c85931f27a882",
  "umbral": 0.4910441072721096,
  "aprendices": 1583,
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
`GET /aprendices/A-005693/riesgo`

```json
{
  "id_aprendiz": "A-005693",
  "probabilidad": 0.8758127287801957,
  "alerta": true,
  "umbral": 0.4910441072721096,
  "factores": [
    {"variable": "porcentaje_actividades_entregadas", "aporte": 1.4711291358811265, "valor": 0.254, "promedio": 0.7334016725798277},
    {"variable": "promedio_calificaciones", "aporte": 0.889264713040094, "valor": 2.5, "promedio": 3.814749113025849},
    {"variable": "tasa_aprobacion_entregas", "aporte": 0.1856946887722884, "valor": 0.7637795275590551, "promedio": 0.8573115365426366}
  ],
  "medidas": [
    "Acordar con el aprendiz un calendario para entregar las actividades y evaluaciones pendientes.",
    "Revisar con el aprendiz las actividades reprobadas y ofrecer un plan de mejoramiento o una tutoría."
  ],
  "plan_de_accion": [
    {"medida": "ponerse_al_dia", "...": "..."},
    {"medida": "plan_de_mejoramiento", "...": "..."}
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
  "total_aprendices": 51,
  "total_alertas": 51,
  "aprendices": [
    {"id_aprendiz": "A-005693", "probabilidad": 0.8758127287801957, "alerta": true, "...": "..."}
  ]
}
```

`total_aprendices` y `total_alertas` cuentan a todos los que cumplen los filtros, antes
de aplicar `limite`. La lista va de mayor a menor probabilidad. Responde 404 si la
cohorte no tiene aprendices activos.

### `GET /cohortes`

```json
[
  {"id_cohorte": "2025-T3", "aprendices": 795, "alertas": 347},
  {"id_cohorte": "2025-T4", "aprendices": 788, "alertas": 369}
]
```

### `GET /instructor`

Una página para el instructor, servida por la misma API, en
<http://127.0.0.1:8000/instructor>. Se elige la cohorte y, si se quiere, la modalidad.
A la izquierda salen los aprendices con alerta, de mayor a menor riesgo, cada uno con su
factor principal en palabras («Actividades entregadas: 14 % (promedio 73 %)»): el
primero con dato, porque la asistencia de los virtuales está vacía (en el detalle dice
«No aplica (sin sesiones presenciales)»). Al
elegir uno, a la derecha salen el porqué (cada factor con el valor del aprendiz y el
promedio) y el plan: qué hacer y en cuántos días, qué decirle, qué evitar, qué hacer
si no responde, qué registrar y cuándo volver a revisar.

- No muestra la probabilidad como porcentaje, porque está inflada por
  `class_weight="balanced"` (ver la calibración en la [model card](model_card.md)):
  muestra el puesto en la lista.
- Solo usa `GET /cohortes` y `GET /cohortes/{id_cohorte}/riesgo`, con `solo_alertas` y
  hasta 500 aprendices. No carga nada de internet: funciona igual dentro de Docker.
- Los nombres para el instructor de cada variable y de cada medida están en la misma
  página; una prueba falla si una variable o una medida de `medidas.yaml` no tiene
  nombre.

## Cómo leer una respuesta

- **`probabilidad`:** riesgo estimado de desertar, entre 0 y 1. Sirve para **ordenar**
  a los aprendices, no como porcentaje: el balanceo de clases la infla (una alerta de
  0.68 corresponde a unos 48 desertores de cada 100). Ver
  [calibración](model_card.md#calibración-la-probabilidad-está-inflada).
- **`alerta`:** `true` si la probabilidad es mayor o igual al umbral (0.491). El umbral
  se eligió para detectar al menos el 70 % de los desertores
  ([ADR 003](decisiones/0003-umbral-y-equidad.md)).
- **`factores`:** hasta 3 variables que más suben el riesgo de ese aprendiz frente al
  aprendiz promedio de entrenamiento. El `aporte` está en log-odds y solo se muestran
  los que aportan al menos 0.05; por debajo es ruido. Cada factor trae el `valor` del
  aprendiz y el `promedio` de entrenamiento contra el que se compara, para que el
  instructor vea el porqué con números: en el ejemplo, entregó el 25 % de las
  actividades frente a un promedio del 73 %. Los porcentajes van de 0 a 1 y las
  booleanas son 0 o 1 (su promedio es la proporción de aprendices). En las categóricas
  el `valor` es un texto y no hay `promedio`; si el dato está vacío (la asistencia de un
  virtual), `valor` es `null`. El promedio sale del modelo (la media que guarda su
  escalador), así que en variables con vacíos incluye los vacíos rellenados con la
  mediana. No sale un factor cuyo valor está del lado bueno del promedio: por la
  colinealidad, el modelo marcaba «Cursos previos terminados: 100 % (promedio 79 %)»
  como razón de 40 alertas. Su lugar lo toma el siguiente factor.
- **`medidas`:** solo cuando hay alerta. Hasta 3, en orden de aporte y sin repetir (dos
  factores pueden compartir medida). Se definen en `config/recomendaciones/medidas.yaml`
  y nunca se sugieren por características personales como la edad o el nivel
  educativo. Tampoco generan medida las variables a las que la colinealidad les invierte
  el signo (con los datos v2: `porcentaje_actividades_aprobadas`,
  `tasa_finalizacion_previa` y `abandono_ultimo_curso`), porque salen como factor en
  aprendices que están mejor que el promedio (ver la [model card](model_card.md)). Un
  factor cuyo dato está vacío se muestra en `factores`, pero no genera medida: la
  asistencia de los virtuales es nula porque no tienen sesiones, y sin ese filtro les
  llegaba el guion de inasistencia.
- **`plan_de_accion`:** un plan por cada medida, en el mismo orden; vacío si no hay
  alerta ([ADR 006](decisiones/0006-plan-de-accion.md)). Cada plan trae:

  | Campo | Significado |
  |---|---|
  | `medida` | Nombre de la medida en `medidas.yaml` |
  | `accion` | Qué hacer y por qué canal |
  | `plazo_dias` | En cuántos días hacerlo |
  | `decir` | Guion de tres frases: apertura, pregunta abierta y propuesta |
  | `evitar` | Lo que no se debe decir o hacer |
  | `si_no_responde` | Segundo intento y a quién acudir |
  | `registrar` | Qué anotar en el seguimiento |
  | `seguimiento_dias` | En cuántos días volver a revisar |

  Ejemplo de un plan:

  ```json
  {
    "medida": "contacto_por_inactividad",
    "accion": "Llamar al aprendiz; si no contesta, escribirle por mensaje o correo.",
    "plazo_dias": 2,
    "decir": [
      "Te llamo porque noté que no has podido entrar a la plataforma estos días y quería saber cómo estás.",
      "¿Qué te ha impedido conectarte?",
      "¿Qué te parece si acordamos qué actividad retomas primero y hablamos de nuevo en unos días?"
    ],
    "evitar": "Empezar por las actividades atrasadas o por las consecuencias de no ingresar.",
    "si_no_responde": "Segundo intento a los 3 días por otro canal; si sigue sin responder, remitir a Bienestar al Aprendiz.",
    "registrar": "Fecha, canal, causa que dio el aprendiz y compromiso acordado.",
    "seguimiento_dias": 7
  }
  ```

  El guion nunca menciona la alerta ni la deserción, y es una guía que el instructor
  adapta. Los textos son una propuesta del equipo que falta validar con instructores y
  con Bienestar al Aprendiz.
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
- **Las medidas y el plan de acción son sugerencias** para el instructor, no decisiones
  automáticas. Un factor indica qué explica la alerta del modelo, no la causa de la
  deserción.
