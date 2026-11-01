# Diccionario de datos

Describe los datos que produce la etapa `process` y el rol de cada variable en el modelo. Es el contrato que valida `src/desercion/validate.py`: si cambia una columna, se actualiza este archivo en el mismo PR.

## Origen y limpieza

| | Archivo | Filas | Columnas |
|---|---|---|---|
| Entrada | `data/raw/aprendices.parquet` (versionado con DVC) | 10.000 | 50 |
| Salida | `data/processed/aprendices.parquet` | 9.518 | 43 |
| Salida | `data/processed/auditoria.parquet` | 9.518 | 3 |

Reglas que aplica `src/desercion/process.py`:

1. **Texto:** se quitan los espacios al inicio y al final de todas las columnas de texto.
2. **Traslados y cancelaciones:** se excluyen los registros con `estado_final` igual a `"traslado"` (211 filas) o `"cancelado"` (271 filas), 482 en total. Un traslado no es abandono, y una cancelación es una decisión administrativa; ninguno es la deserción que el modelo predice. Si se conservaran con `deserto = 0`, el modelo aprendería que salir del programa por cancelación equivale a terminarlo.
3. **Auditoría:** `id_aprendiz`, `sexo` y `modalidad` se copian a `auditoria.parquet`. `sexo` sale de los datos del modelo y solo se usa para auditar equidad.
4. **Fuga de datos:** se eliminan `estado_final` y `fecha_desercion`, porque describen el resultado que el modelo debe predecir.
5. **Descartadas:**

| Columna | Motivo |
|---|---|
| `nombre_programa` | Repite `id_programa` |
| `barrio_residencia` | Muchas categorías y riesgo de reidentificar al aprendiz |
| `municipio_nacimiento` | Puede revelar desplazamiento (dato sensible) |
| `tasa_abandono_previa` | Es `1 - tasa_finalizacion_previa` (redundante) |

## Roles

| Rol | Significado |
|---|---|
| **Identificador** | Identifica el registro. No entra al modelo. |
| **Agrupación** | Sirve para dividir los datos (validación temporal, `GroupKFold`) o agrupar resultados. No entra al modelo. |
| **Etiqueta** | Lo que el modelo predice. |
| **Variable** | Entra al modelo (directamente o a través de `features.py`). |

## Variables

### Identificadores, agrupación y etiqueta

| Columna | Tipo | Nulos | Valores / rango | Rol | Descripción |
|---|---|---|---|---|---|
| `id_aprendiz` | texto | 0 | `A-000001`–`A-010000`, único | Identificador | Código del aprendiz |
| `id_cohorte` | texto | 0 | 12 cohortes, `2023-T1`–`2025-T4` | Agrupación | Trimestre de inicio. La cohorte más reciente se usa para evaluar |
| `fecha_inicio` | fecha | 0 | 2023-01-23 – 2025-10-06 (12 fechas) | Agrupación | Inicio de la ficha |
| `fecha_corte` | fecha | 0 | 2023-02-13 – 2025-11-10 | Agrupación | Fecha en que se toma la foto del aprendiz (alrededor de la semana 4) |
| `deserto` | entero | 0 | {0, 1} | Etiqueta | 1 = desertó después de la fecha de corte |

### Programa de formación

| Columna | Tipo | Nulos | Valores / rango | Rol | Descripción |
|---|---|---|---|---|---|
| `id_programa` | texto | 0 | 16 programas, `P-101`–`P-502` | Variable | Programa de formación |
| `nivel_formacion` | texto | 0 | `AUXILIAR`, `OPERARIO`, `TÉCNICO`, `TECNÓLOGO`, `ESPECIALIZACIÓN TECNOLÓGICA` | Variable | Nivel del programa |
| `area_formacion` | texto | 0 | 6 áreas (ej. `administracion`, `turismo y gastronomia`) | Variable | Área del programa |
| `modalidad` | texto | 0 | `virtual`, `presencial`, `mixta` | Variable | Modalidad. También se usa para reportar métricas por modalidad |
| `duracion_programa_meses` | entero | 0 | 3 valores, 6–24 | Variable | Duración del programa |
| `jornada` | texto | 0 | 4 valores (ej. `diurna`, `otra`) | Variable | Jornada de estudio |
| `dias_desde_inicio` | entero | 0 | 21–35 | Variable | Días entre `fecha_inicio` y `fecha_corte` |

### Perfil del aprendiz

| Columna | Tipo | Nulos | Valores / rango | Rol | Descripción |
|---|---|---|---|---|---|
| `edad` | entero | 0 | 16–60 | Variable | Edad al iniciar (rango válido 14–100: el SENA admite desde los 14 años, sin edad máxima) |
| `municipio_residencia` | texto | 0 | 16 municipios (ej. `Apartadó`, `Turbo`) | Variable | Municipio donde vive |
| `nivel_educativo` | texto | 0 | 6 niveles (ej. `bachillerato`, `tecnologo`) | Variable | Máximo nivel educativo alcanzado |
| `situacion_laboral` | texto | 0 | 4 valores (ej. `desempleado`, `otra`) | Variable | Situación laboral |
| `horas_trabajo_semana` | entero | 0 | 0–60 | Variable | Horas de trabajo a la semana |
| `responsabilidades_cuidado` | booleano | 0 | True/False | Variable | Tiene personas a su cargo |
| `horas_disponibles_estudio` | decimal | 679 | 2–33 | Variable | Horas a la semana que dice tener para estudiar |
| `dificultad_economica_reportada` | texto | 679 | `alta`, `media`, `baja` | Variable | Dificultad económica que reporta |

### Historial en el SENA

| Columna | Tipo | Nulos | Valores / rango | Rol | Descripción |
|---|---|---|---|---|---|
| `n_cursos_previos` | entero | 0 | 0–7 | Variable | Cursos anteriores en el SENA |
| `n_cursos_completados` | entero | 0 | 0–7 | Variable | Cursos anteriores terminados |
| `n_cursos_abandonados` | entero | 0 | 0–4 | Variable | Cursos anteriores abandonados |
| `tasa_finalizacion_previa` | decimal | 3.022 | 0–1 | Variable | Proporción de cursos previos terminados |
| `es_primer_curso` | booleano | 0 | True/False | Variable | Es su primer curso en el SENA |
| `meses_desde_ultimo_curso` | decimal | 2.647 | 0,1–58,5 | Variable | Meses desde que terminó o dejó su último curso |
| `abandono_ultimo_curso` | booleano (admite nulos) | 2.647 | True/False | Variable | Abandonó su último curso |
| `experiencia_virtual_previa` | booleano | 0 | True/False | Variable | Ya estudió antes en modalidad virtual |

### Conectividad y equipo

| Columna | Tipo | Nulos | Valores / rango | Rol | Descripción |
|---|---|---|---|---|---|
| `tipo_conexion` | texto | 0 | 4 valores (ej. `cable`, `otra`) | Variable | Cómo se conecta a internet |
| `tipo_plan_internet` | texto | 0 | 3 valores (ej. `prepago`, `otro`) | Variable | Tipo de plan de internet |
| `calidad_internet` | texto | 679 | `alta`, `media`, `baja` | Variable | Calidad de conexión que reporta |
| `dispositivo_principal` | texto | 0 | 4 valores (ej. `celular`, `tableta`) | Variable | Dispositivo con el que estudia |
| `comparte_dispositivo` | booleano | 0 | True/False | Variable | Comparte el dispositivo con otras personas |
| `disponibilidad_equipo` | booleano | 0 | True/False | Variable | Tiene equipo disponible para estudiar |

### Seguimiento hasta la fecha de corte

Estas variables se miden **hasta `fecha_corte`**. Si alguna usara información posterior habría fuga temporal (lo prueba `tests/test_features.py`).

| Columna | Tipo | Nulos | Valores / rango | Rol | Descripción |
|---|---|---|---|---|---|
| `porcentaje_actividades_entregadas` | decimal | 0 | 0,036–1 | Variable | Proporción de actividades entregadas |
| `porcentaje_actividades_aprobadas` | decimal | 0 | 0–1 | Variable | Proporción de actividades aprobadas |
| `promedio_calificaciones` | decimal | 1 | 1,8–5,0 | Variable | Promedio de notas (escala 0–5) |
| `dias_sin_ingreso` | entero | 0 | 0–35 | Variable | Días desde el último ingreso a la plataforma |
| `ingresos_plataforma_ultimos_7d` | entero | 0 | 0–18 | Variable | Ingresos a la plataforma en los últimos 7 días |
| `actividades_pendientes` | entero | 0 | 0–9 | Variable | Actividades sin entregar |
| `evaluaciones_no_presentadas` | entero | 0 | 0–2 | Variable | Evaluaciones que no presentó |
| `asistencia_porcentaje` | decimal | 4.712 | 0,41–1 | Variable | Proporción de asistencia a sesiones |
| `solicito_apoyo` | booleano | 0 | True/False | Variable | Pidió apoyo (bienestar, instructor) |

## Valores faltantes

Los nulos no son errores de carga: siguen reglas del negocio. El esquema de validación debe permitirlos **solo** en estas columnas.

| Columnas | Nulos | Motivo | Estado |
|---|---|---|---|
| `meses_desde_ultimo_curso`, `abandono_ultimo_curso` | 2.647 | No hay curso anterior: nulo si y solo si `es_primer_curso = True` | ✅ Confirmado |
| `tasa_finalizacion_previa` | 3.022 | No hay cursos previos cerrados (`n_cursos_completados + n_cursos_abandonados = 0`), así que la tasa sería una división por cero: 2.647 por primer curso y 375 con cursos previos que siguen abiertos | ✅ Confirmado |
| `calidad_internet`, `horas_disponibles_estudio`, `dificultad_economica_reportada` | 679 | Son las mismas 679 filas: no respondieron la encuesta de caracterización | ✅ Confirmado (mismas filas) |
| `asistencia_porcentaje` | 4.712 | Nulo si y solo si `modalidad = virtual` (no hay sesiones con asistencia) | ✅ Confirmado |
| `promedio_calificaciones` | 1 | Sin calificaciones registradas a la fecha de corte (ninguna actividad aprobada). Se conserva la fila: es un caso real que el modelo verá en producción | ✅ Decidido |

Implicación para el modelo: estos nulos tienen significado, así que en `features.py` no se rellenan a ciegas con la media. Se decide por columna (por ejemplo, indicador "sin encuesta" o dejar que LightGBM los maneje).

## Reglas de validación

Acordadas entre Andrés y David; se implementan en `src/desercion/validate.py` con Pandera sobre `data/processed/aprendices.parquet`. Si una regla cambia, se actualizan este documento y el esquema en el mismo PR.

**Tipos:** se valida sin convertir (`coerce=False`): un tipo incorrecto hace fallar la validación. Ojo: `abandono_ultimo_curso` es `boolean` (admite nulos), no `bool`.

**Categorías:** lista cerrada con los valores observados en todas las columnas categóricas, incluido `municipio_residencia` (un valor nuevo detiene el pipeline para decidir qué hacer). Los códigos se validan por formato: `id_programa` con `P-\d{3}` e `id_cohorte` con `\d{4}-T[1-4]`.

**Rangos:**

| Columnas | Regla |
|---|---|
| Porcentajes y tasas | 0–1 |
| `promedio_calificaciones` | 0–5 (admite nulo) |
| `edad` | 14–100 |
| Conteos y horas | ≥ 0 |
| `horas_trabajo_semana`, `horas_disponibles_estudio` | ≤ 168 |
| `dias_desde_inicio` | 21–35 (ventana de la semana 4, decisión del proyecto) |
| `duracion_programa_meses` | 6, 12 o 24 |

**Relaciones entre columnas:**

- `dias_sin_ingreso ≤ dias_desde_inicio`
- `dias_desde_inicio = fecha_corte − fecha_inicio`
- `porcentaje_actividades_aprobadas ≤ porcentaje_actividades_entregadas` (no se aprueba lo que no se entregó)
- `jornada = "otra"` si y solo si `modalidad = "virtual"` (en formación virtual no hay jornada)
- `asistencia_porcentaje` nulo si y solo si `modalidad = "virtual"`
- `meses_desde_ultimo_curso` y `abandono_ultimo_curso` nulos si y solo si `es_primer_curso = True`
- `calidad_internet`, `horas_disponibles_estudio` y `dificultad_economica_reportada` nulos en las mismas filas (encuesta no respondida)

**Etapa DVC `validate`:** depende de `data/processed/aprendices.parquet` y de `src/desercion/validate.py`, y escribe `reports/validacion.json` como métrica (`cache: false`, se sube a Git). La etapa `features` depende de ese JSON, así que no se construyen variables con datos que no pasaron la validación.

**Alcance:** este esquema es para los datos de entrenamiento. En la API, un valor de categoría nuevo (por ejemplo, un municipio) no debe tumbar el servicio: el modelo lo trata como desconocido.

## Puntos abiertos

- [x] Nulo de `promedio_calificaciones`: se permite (`nullable`) y no se elimina la fila.
- [x] `nivel_formacion` se deja en mayúsculas, tal como viene: los valores son consistentes y no hay duplicados. Si algún día se normaliza, se hace en `process.py` y se actualiza el esquema en el mismo PR.
- [ ] Revisar las descripciones con la fuente de los datos.

## Tabla de auditoría

`data/processed/auditoria.parquet`: `id_aprendiz`, `sexo`, `modalidad`. Se une con las predicciones por `id_aprendiz` solo para medir equidad por subgrupo. Nunca entra al modelo.

## Datos que no se usan

Por la Ley 1581 de 2012 no se usan datos sensibles: etnia, religión, orientación sexual ni datos biométricos.
