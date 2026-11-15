"""Generador de los datos sintéticos de aprendices (v2).

Entrada:
    config/data/generador.yaml      (semilla y fórmulas)
Salidas:
    data/raw/aprendices.parquet     (50 columnas: lo que lee process.py)
    data/raw/verdad.parquet         (id_aprendiz, constancia, compromiso y
                                     probabilidad real de desertar)

Los datos de la v1 se recibieron ya generados y sin código. Este generador
produce un archivo con las mismas columnas y las mismas reglas, pero con la
fórmula a la vista: se sabe exactamente por qué deserta cada aprendiz.

Cómo se generan (en este orden):

1. Programa, cohorte y modalidad.
2. Perfil: edad, municipio, nivel educativo, trabajo, cuidado.
3. `constancia`: rasgo oculto del aprendiz. Decide su historial en el SENA.
4. Conectividad, equipo y encuesta de caracterización.
5. `compromiso`: variable oculta que depende del perfil (fórmula 1 del YAML).
6. Seguimiento de la semana 4: cada variable es el compromiso más ruido.
7. Deserción: probabilidad con la fórmula 2 del YAML y un sorteo por aprendiz.
8. Estado final y fecha de deserción.

`sexo` no interviene en ninguna fórmula: la auditoría de equidad por sexo debe
dar una razón cercana a 1.

`verdad.parquet` nunca entra al pipeline. Sirve para dos comprobaciones:
cuánto rinde un modelo que conociera la probabilidad real (el techo) y si los
factores que explica el modelo coinciden con los de la fórmula.

Uso (desde la raíz del repositorio):
    uv run python -m desercion.generar_datos
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# --- Rutas -----------------------------------------------------------------
RUTA_CONFIG = Path("config/data/generador.yaml")
RUTA_RAW = Path("data/raw/aprendices.parquet")
RUTA_VERDAD = Path("data/raw/verdad.parquet")

# --- Programas: (id, nombre, nivel, área, meses, peso) ----------------------
# Los 15 primeros salen de la muestra de la v1; P-201 no aparecía en ella y se
# completó para llegar a los 16 del diccionario de datos.
PROGRAMAS = [
    ("P-101", "Auxiliar en Servicios de Apoyo al Cliente", "AUXILIAR", "administracion", 6, 0.04),
    ("P-102", "Auxiliar en Cocina", "AUXILIAR", "turismo y gastronomia", 6, 0.03),
    ("P-201", "Operario en Confección Industrial", "OPERARIO", "industria", 6, 0.02),
    ("P-202", "Operario en Producción Agrícola", "OPERARIO", "agropecuaria", 6, 0.04),
    ("P-301", "Técnico en Sistemas", "TÉCNICO", "tecnologia", 12, 0.09),
    ("P-302", "Técnico en Contabilización de Operaciones", "TÉCNICO", "administracion", 12, 0.08),
    ("P-303", "Técnico en Asistencia Administrativa", "TÉCNICO", "administracion", 12, 0.10),
    ("P-304", "Técnico en Atención Integral a la Primera Infancia", "TÉCNICO", "salud", 12, 0.09),
    ("P-305", "Técnico en Instalaciones Eléctricas", "TÉCNICO", "industria", 12, 0.08),
    ("P-401", "Tecnólogo en Análisis y Desarrollo de Software", "TECNÓLOGO", "tecnologia", 24, 0.11),
    ("P-402", "Tecnólogo en Gestión Administrativa", "TECNÓLOGO", "administracion", 24, 0.09),
    ("P-403", "Tecnólogo en Gestión Logística", "TECNÓLOGO", "administracion", 24, 0.06),
    ("P-404", "Tecnólogo en Producción Agropecuaria Ecológica", "TECNÓLOGO", "agropecuaria", 24, 0.07),
    ("P-405", "Tecnólogo en Regencia de Farmacia", "TECNÓLOGO", "salud", 24, 0.05),
    ("P-501", "Especialización Tecnológica en Seguridad Informática", "ESPECIALIZACIÓN TECNOLÓGICA", "tecnologia", 12, 0.03),
    ("P-502", "Especialización Tecnológica en Gestión de Proyectos", "ESPECIALIZACIÓN TECNOLÓGICA", "administracion", 12, 0.02),
]  # fmt: skip

# Una ficha por trimestre: (cohorte, fecha de inicio).
COHORTES = [
    ("2023-T1", "2023-01-23"),
    ("2023-T2", "2023-04-17"),
    ("2023-T3", "2023-07-17"),
    ("2023-T4", "2023-10-09"),
    ("2024-T1", "2024-01-22"),
    ("2024-T2", "2024-04-15"),
    ("2024-T3", "2024-07-15"),
    ("2024-T4", "2024-10-07"),
    ("2025-T1", "2025-01-20"),
    ("2025-T2", "2025-04-21"),
    ("2025-T3", "2025-07-14"),
    ("2025-T4", "2025-10-06"),
]

# --- Distribuciones del perfil: {valor: probabilidad} -----------------------
MODALIDADES = {"virtual": 0.50, "presencial": 0.29, "mixta": 0.21}
JORNADAS = {"diurna": 0.50, "nocturna": 0.30, "mixta": 0.20}  # virtual: "otra"
MUNICIPIOS = {
    "Bogotá": 0.20,
    "Medellín": 0.16,
    "Cali": 0.11,
    "Barranquilla": 0.08,
    "Bello": 0.07,
    "Bucaramanga": 0.06,
    "Pereira": 0.05,
    "Soacha": 0.05,
    "Itagüí": 0.05,
    "Pasto": 0.04,
    "Envigado": 0.03,
    "Rionegro": 0.03,
    "Apartadó": 0.03,
    "Caucasia": 0.02,
    "Quibdó": 0.01,
    "Turbo": 0.01,
}
# El nivel educativo depende del nivel del programa (requisitos de ingreso).
NIVEL_EDUCATIVO = {
    "AUXILIAR": {"primaria": 0.45, "basica_secundaria": 0.35, "bachillerato": 0.20},
    "OPERARIO": {"primaria": 0.40, "basica_secundaria": 0.40, "bachillerato": 0.20},
    "TÉCNICO": {"basica_secundaria": 0.30, "bachillerato": 0.50, "tecnico": 0.20},
    "TECNÓLOGO": {"bachillerato": 0.45, "tecnico": 0.35, "tecnologo": 0.20},
    "ESPECIALIZACIÓN TECNOLÓGICA": {"tecnologo": 0.25, "profesional": 0.75},
}
SITUACION_LABORAL = {
    "empleado": 0.38,
    "desempleado": 0.31,
    "independiente": 0.26,
    "otra": 0.05,
}
TIPO_CONEXION = {"cable": 0.31, "datos_moviles": 0.30, "fibra": 0.29, "otra": 0.10}
# La calidad que reporta depende del tipo de conexión.
CALIDAD_INTERNET = {
    "fibra": {"alta": 0.50, "media": 0.38, "baja": 0.12},
    "cable": {"alta": 0.38, "media": 0.42, "baja": 0.20},
    "datos_moviles": {"alta": 0.15, "media": 0.42, "baja": 0.43},
    "otra": {"alta": 0.10, "media": 0.40, "baja": 0.50},
}
PLAN_INTERNET = {
    "datos_moviles": {"prepago": 0.80, "pospago": 0.20},
    "resto": {"pospago": 0.78, "prepago": 0.14, "otro": 0.08},
}
DISPOSITIVO = {"computador": 0.50, "celular": 0.36, "tableta": 0.10, "otro": 0.04}
# Quien no tiene trabajo reporta más dificultad económica.
DIFICULTAD_ECONOMICA = {
    "sin_trabajo": {"alta": 0.36, "media": 0.34, "baja": 0.30},
    "con_trabajo": {"alta": 0.22, "media": 0.33, "baja": 0.45},
}

MAX_CURSOS_PREVIOS = 7
MAX_CURSOS_ABANDONADOS = 4
DIAS_POR_MES = 30.4

# Orden de las columnas de la v1.
COLUMNAS = [
    "id_aprendiz",
    "id_programa",
    "nivel_formacion",
    "nombre_programa",
    "area_formacion",
    "modalidad",
    "duracion_programa_meses",
    "jornada",
    "fecha_inicio",
    "dias_desde_inicio",
    "edad",
    "sexo",
    "municipio_residencia",
    "barrio_residencia",
    "municipio_nacimiento",
    "nivel_educativo",
    "n_cursos_previos",
    "n_cursos_completados",
    "n_cursos_abandonados",
    "tasa_finalizacion_previa",
    "tasa_abandono_previa",
    "experiencia_virtual_previa",
    "meses_desde_ultimo_curso",
    "es_primer_curso",
    "abandono_ultimo_curso",
    "tipo_conexion",
    "tipo_plan_internet",
    "calidad_internet",
    "dispositivo_principal",
    "comparte_dispositivo",
    "disponibilidad_equipo",
    "situacion_laboral",
    "horas_trabajo_semana",
    "responsabilidades_cuidado",
    "horas_disponibles_estudio",
    "dificultad_economica_reportada",
    "porcentaje_actividades_entregadas",
    "porcentaje_actividades_aprobadas",
    "promedio_calificaciones",
    "dias_sin_ingreso",
    "ingresos_plataforma_ultimos_7d",
    "actividades_pendientes",
    "evaluaciones_no_presentadas",
    "asistencia_porcentaje",
    "solicito_apoyo",
    "fecha_corte",
    "deserto",
    "fecha_desercion",
    "estado_final",
    "id_cohorte",
]

logger = logging.getLogger(__name__)


def cargar_config(ruta: Path = RUTA_CONFIG) -> dict:
    """Lee la semilla y las fórmulas del generador."""
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


# --- Utilidades ------------------------------------------------------------
def _sortear(
    rng: np.random.Generator, opciones: dict[str, float], n: int
) -> np.ndarray:
    """n valores de una distribución categórica {valor: probabilidad}."""
    valores = list(opciones)
    return rng.choice(valores, size=n, p=list(opciones.values()))


def _sortear_segun(
    rng: np.random.Generator, grupo: np.ndarray, tablas: dict[str, dict[str, float]]
) -> np.ndarray:
    """Como `_sortear`, pero cada fila usa la tabla de su grupo."""
    resultado = np.empty(len(grupo), dtype=object)
    for nombre, opciones in tablas.items():
        filas = grupo == nombre
        resultado[filas] = _sortear(rng, opciones, int(filas.sum()))
    return resultado


def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# --- Pasos de la generación ------------------------------------------------
def _programa_y_cohorte(rng: np.random.Generator, n: int) -> pd.DataFrame:
    programas = pd.DataFrame(
        PROGRAMAS,
        columns=[
            "id_programa",
            "nombre_programa",
            "nivel_formacion",
            "area_formacion",
            "duracion_programa_meses",
            "peso",
        ],
    )
    elegido = rng.choice(len(programas), size=n, p=programas["peso"])
    df = programas.drop(columns="peso").iloc[elegido].reset_index(drop=True)

    cohortes = pd.DataFrame(COHORTES, columns=["id_cohorte", "fecha_inicio"])
    cohorte = cohortes.iloc[rng.integers(len(cohortes), size=n)].reset_index(drop=True)
    df["id_cohorte"] = cohorte["id_cohorte"]
    df["fecha_inicio"] = pd.to_datetime(cohorte["fecha_inicio"]).dt.as_unit("us")
    # La foto del aprendiz se toma alrededor de la semana 4.
    df["dias_desde_inicio"] = rng.integers(21, 36, size=n)
    df["fecha_corte"] = df["fecha_inicio"] + pd.to_timedelta(
        df["dias_desde_inicio"], unit="D"
    )

    df["modalidad"] = _sortear(rng, MODALIDADES, n)
    # En formación virtual no hay jornada.
    df["jornada"] = np.where(
        df["modalidad"] == "virtual", "otra", _sortear(rng, JORNADAS, n)
    )
    return df


def _perfil(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    # Mayoría entre 18 y 30 años, con cola hacia edades mayores.
    df["edad"] = np.clip(16 + rng.gamma(2.2, 5.2, size=n), 16, 60).astype("int64")
    df["sexo"] = np.where(rng.random(n) < 0.55, "F", "M")

    df["municipio_residencia"] = _sortear(rng, MUNICIPIOS, n)
    barrio = pd.Series(
        [f"Barrio {numero:02d}" for numero in rng.integers(1, 41, size=n)]
    )
    df["barrio_residencia"] = barrio.mask(rng.random(n) < 0.09)
    # Dos de cada tres viven donde nacieron.
    df["municipio_nacimiento"] = np.where(
        rng.random(n) < 0.68, df["municipio_residencia"], _sortear(rng, MUNICIPIOS, n)
    )
    df["nivel_educativo"] = _sortear_segun(
        rng, df["nivel_formacion"].to_numpy(), NIVEL_EDUCATIVO
    )

    df["situacion_laboral"] = _sortear(rng, SITUACION_LABORAL, n)
    horas = np.select(
        [
            df["situacion_laboral"] == "empleado",
            df["situacion_laboral"] == "independiente",
        ],
        [rng.normal(44, 10, size=n), rng.normal(36, 14, size=n)],
        default=np.nan,
    )
    # Quien no trabaja tiene 0 horas; quien trabaja, entre 10 y 60.
    df["horas_trabajo_semana"] = (
        pd.Series(np.clip(np.round(horas), 10, 60)).fillna(0).astype("int64")
    )
    df["responsabilidades_cuidado"] = rng.random(n) < 0.33
    return df


def _historial(
    rng: np.random.Generator, df: pd.DataFrame, constancia: np.ndarray
) -> pd.DataFrame:
    """Cursos previos en el SENA: los decide la constancia del aprendiz."""
    n = len(df)
    primer_curso = rng.random(n) < 0.275
    previos = np.where(
        primer_curso,
        0,
        1 + np.minimum(rng.poisson(0.85, size=n), MAX_CURSOS_PREVIOS - 1),
    )

    # Cada curso previo: sigue abierto (10 %), o se cerró terminado o abandonado.
    es_curso = np.arange(MAX_CURSOS_PREVIOS) < previos[:, None]
    abierto = rng.random((n, MAX_CURSOS_PREVIOS)) < 0.10
    probabilidad_terminar = _sigmoide(1.0 + 0.9 * constancia)[:, None]
    termino = rng.random((n, MAX_CURSOS_PREVIOS)) < probabilidad_terminar
    completados = (es_curso & ~abierto & termino).sum(axis=1)
    abandonados = (es_curso & ~abierto & ~termino).sum(axis=1)
    # Tope de abandonos: los que sobran cuentan como cursos aún abiertos.
    abandonados = np.minimum(abandonados, MAX_CURSOS_ABANDONADOS)
    cerrados = completados + abandonados

    df["n_cursos_previos"] = previos
    df["n_cursos_completados"] = completados
    df["n_cursos_abandonados"] = abandonados
    # Sin cursos cerrados la tasa sería una división por cero: queda nula.
    tasa = pd.Series(completados / np.where(cerrados > 0, cerrados, np.nan)).round(3)
    df["tasa_finalizacion_previa"] = tasa
    df["tasa_abandono_previa"] = (1 - tasa).round(3)
    df["es_primer_curso"] = primer_curso

    df["experiencia_virtual_previa"] = ~primer_curso & (rng.random(n) < 0.56)
    meses = np.clip(0.1 + rng.exponential(11, size=n), 0.1, 58.5).round(1)
    df["meses_desde_ultimo_curso"] = pd.Series(meses).mask(primer_curso)
    # El último curso cerrado fue un abandono con probabilidad abandonados/cerrados.
    probabilidad_abandono = abandonados / np.where(cerrados > 0, cerrados, 1)
    abandono_ultimo = pd.array(rng.random(n) < probabilidad_abandono, dtype="boolean")
    abandono_ultimo[primer_curso] = pd.NA
    df["abandono_ultimo_curso"] = abandono_ultimo
    return df


def _conectividad_y_encuesta(
    rng: np.random.Generator, df: pd.DataFrame
) -> pd.DataFrame:
    n = len(df)
    df["tipo_conexion"] = _sortear(rng, TIPO_CONEXION, n)
    grupo_plan = np.where(
        df["tipo_conexion"] == "datos_moviles", "datos_moviles", "resto"
    )
    df["tipo_plan_internet"] = _sortear_segun(rng, grupo_plan, PLAN_INTERNET)
    df["calidad_internet"] = _sortear_segun(
        rng, df["tipo_conexion"].to_numpy(), CALIDAD_INTERNET
    )
    df["dispositivo_principal"] = _sortear(rng, DISPOSITIVO, n)
    df["comparte_dispositivo"] = rng.random(n) < 0.32
    # Solo quien comparte el dispositivo puede quedarse sin equipo para estudiar.
    df["disponibilidad_equipo"] = ~df["comparte_dispositivo"] | (rng.random(n) < 0.47)

    # Menos horas para estudiar si trabaja más o cuida a otras personas.
    horas = (
        rng.normal(15, 5, size=n)
        - 0.11 * df["horas_trabajo_semana"]
        - 2.0 * df["responsabilidades_cuidado"]
    )
    df["horas_disponibles_estudio"] = np.clip(np.round(horas), 2, 33).astype("float64")
    con_trabajo = df["situacion_laboral"].isin(["empleado", "independiente"])
    df["dificultad_economica_reportada"] = _sortear_segun(
        rng, np.where(con_trabajo, "con_trabajo", "sin_trabajo"), DIFICULTAD_ECONOMICA
    )
    return df


def _compromiso(
    rng: np.random.Generator, df: pd.DataFrame, constancia: np.ndarray, coef: dict
) -> np.ndarray:
    """Fórmula 1 del YAML: el compromiso depende del perfil, más ruido."""
    horas = df["horas_disponibles_estudio"]
    horas_estandar = (horas - horas.mean()) / horas.std()
    no_presencial = df["modalidad"] != "presencial"
    return (
        coef["constancia"] * constancia
        + coef["horas_disponibles_estudio"] * horas_estandar
        + coef["dificultad_economica_alta"]
        * (df["dificultad_economica_reportada"] == "alta")
        + coef["calidad_internet_baja"]
        * ((df["calidad_internet"] == "baja") & no_presencial)
        + coef["sin_equipo_disponible"] * (~df["disponibilidad_equipo"] & no_presencial)
        + coef["experiencia_virtual_previa"]
        * (df["experiencia_virtual_previa"] & (df["modalidad"] == "virtual"))
        + rng.normal(0, coef["ruido"], size=len(df))
    ).to_numpy()


def _seguimiento(
    rng: np.random.Generator, df: pd.DataFrame, compromiso: np.ndarray
) -> pd.DataFrame:
    """Variables de la semana 4: cada una es el compromiso más su propio ruido."""
    n = len(df)
    virtual = (df["modalidad"] == "virtual").to_numpy()
    presencial = (df["modalidad"] == "presencial").to_numpy()

    entregadas = np.clip(
        0.76 + 0.17 * compromiso + rng.normal(0, 0.11, size=n), 0.03, 1
    )
    entregadas = entregadas.round(3)
    # De lo entregado, qué parte aprueba. Nunca se aprueba más de lo entregado.
    tasa_aprobacion = np.clip(
        0.87 + 0.06 * compromiso + rng.normal(0, 0.09, size=n), 0, 1
    )
    aprobadas = np.minimum((entregadas * tasa_aprobacion).round(3), entregadas)
    df["porcentaje_actividades_entregadas"] = entregadas
    df["porcentaje_actividades_aprobadas"] = aprobadas

    notas = np.clip(3.85 + 0.30 * compromiso + rng.normal(0, 0.33, size=n), 1, 5).round(
        1
    )
    # Sin actividades aprobadas no hay promedio.
    df["promedio_calificaciones"] = pd.Series(notas).mask(aprobadas == 0)

    # Días desde el último ingreso: la mayoría pocos, unos cuantos muchos. En
    # virtual la plataforma es el único contacto y las ausencias son más largas.
    media_dias = np.exp(1.1 - 0.40 * compromiso + 0.40 * virtual - 0.40 * presencial)
    dias_sin_ingreso = rng.poisson(rng.gamma(1.2, media_dias / 1.2))
    dias_sin_ingreso = np.minimum(dias_sin_ingreso, df["dias_desde_inicio"])
    df["dias_sin_ingreso"] = dias_sin_ingreso
    # En presencial se usa menos la plataforma. Si el último ingreso fue hace
    # 7 días o más, no puede haber ingresos en los últimos 7 días.
    ingresos = rng.poisson(np.exp(1.75 + 0.20 * compromiso - 0.60 * presencial))
    ingresos = np.clip(ingresos, 1, 18)
    df["ingresos_plataforma_ultimos_7d"] = np.where(dias_sin_ingreso >= 7, 0, ingresos)

    # Pendientes: lo no entregado, sobre el total de actividades de la ficha.
    total_actividades = rng.integers(6, 11, size=n)
    df["actividades_pendientes"] = np.round(
        (1 - entregadas) * total_actividades
    ).astype("int64")
    df["evaluaciones_no_presentadas"] = rng.binomial(
        2, _sigmoide(-1.6 - 0.5 * compromiso)
    )

    asistencia = np.clip(0.86 + 0.09 * compromiso + rng.normal(0, 0.07, size=n), 0.3, 1)
    # En formación virtual no hay sesiones con asistencia.
    df["asistencia_porcentaje"] = pd.Series(asistencia.round(2)).mask(virtual)

    # Pide apoyo más quien reporta dificultad económica alta.
    dificultad_alta = df["dificultad_economica_reportada"] == "alta"
    df["solicito_apoyo"] = rng.random(n) < _sigmoide(-1.9 + 0.5 * dificultad_alta)
    return df


def _probabilidad_desercion(
    df: pd.DataFrame, constancia: np.ndarray, compromiso: np.ndarray, coef: dict
) -> np.ndarray:
    """Fórmula 2 del YAML: probabilidad real de desertar de cada aprendiz."""
    log_odds = (
        coef["intercepto"]
        + coef["compromiso"] * compromiso
        + coef["constancia"] * constancia
        + coef["modalidad_virtual"] * (df["modalidad"] == "virtual")
        + coef["modalidad_presencial"] * (df["modalidad"] == "presencial")
        + coef["programa_de_24_meses"] * (df["duracion_programa_meses"] == 24)
        + coef["horas_trabajo_por_10"] * df["horas_trabajo_semana"] / 10
        + coef["responsabilidades_cuidado"] * df["responsabilidades_cuidado"]
        + coef["dificultad_economica_alta"]
        * (df["dificultad_economica_reportada"] == "alta")
    )
    return _sigmoide(log_odds.to_numpy(dtype="float64"))


def _estado_y_fecha(
    rng: np.random.Generator, df: pd.DataFrame, deserto: np.ndarray, otros: dict
) -> pd.DataFrame:
    n = len(df)
    df["deserto"] = deserto.astype("int64")
    # Traslados y cancelaciones: al azar entre quienes no desertan.
    sorteo = rng.random(n)
    df["estado_final"] = np.select(
        [
            deserto,
            sorteo < otros["traslado"],
            sorteo < otros["traslado"] + otros["cancelado"],
        ],
        ["deserto", "traslado", "cancelado"],
        default="completado",
    )
    # Siempre después de la fecha de corte y antes del 80 % del programa, con
    # más deserciones al principio que al final.
    limite = np.floor(0.8 * df["duracion_programa_meses"] * DIAS_POR_MES)
    margen = limite - df["dias_desde_inicio"] - 1
    dias = df["dias_desde_inicio"] + 1 + np.floor(rng.beta(1.3, 2.2, size=n) * margen)
    fecha = df["fecha_inicio"] + pd.to_timedelta(dias, unit="D")
    df["fecha_desercion"] = fecha.where(deserto).dt.as_unit("us")
    return df


def _sin_encuesta(
    rng: np.random.Generator, df: pd.DataFrame, compromiso: np.ndarray
) -> pd.DataFrame:
    """Borra las tres respuestas de la encuesta en las mismas filas.

    Va al final: las fórmulas usan la respuesta real y lo que se borra es solo
    lo que el SENA alcanzó a registrar. Responde menos quien está menos comprometido.
    """
    sin_encuesta = rng.random(len(df)) < _sigmoide(-2.75 - 0.35 * compromiso)
    for columna in [
        "calidad_internet",
        "horas_disponibles_estudio",
        "dificultad_economica_reportada",
    ]:
        df[columna] = df[columna].mask(sin_encuesta)
    return df


def generar(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera (aprendices, verdad) a partir de la semilla y las fórmulas.

    Todos los sorteos salen de un único generador, siempre en el mismo orden:
    con la misma configuración el resultado es idéntico.
    """
    rng = np.random.default_rng(config["semilla"])
    n = config["n_aprendices"]

    df = _programa_y_cohorte(rng, n)
    df = _perfil(rng, df)
    constancia = rng.normal(0, 1, size=n)
    df = _historial(rng, df, constancia)
    df = _conectividad_y_encuesta(rng, df)
    compromiso = _compromiso(rng, df, constancia, config["compromiso"])
    df = _seguimiento(rng, df, compromiso)
    probabilidad = _probabilidad_desercion(
        df, constancia, compromiso, config["desercion"]
    )
    deserto = rng.random(n) < probabilidad
    df = _estado_y_fecha(rng, df, deserto, config["otros_estados"])
    df = _sin_encuesta(rng, df, compromiso)

    # Los códigos no siguen el orden de las cohortes.
    codigos = rng.permutation(n) + 1
    df["id_aprendiz"] = [f"A-{codigo:06d}" for codigo in codigos]

    verdad = pd.DataFrame(
        {
            "id_aprendiz": df["id_aprendiz"],
            "constancia": constancia,
            "compromiso": compromiso,
            "probabilidad_desercion": probabilidad,
        }
    )
    orden = df.sort_values(["id_cohorte", "id_aprendiz"]).index
    df = df.loc[orden, COLUMNAS].reset_index(drop=True)
    verdad = verdad.loc[orden].reset_index(drop=True)
    return df, verdad


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    config = cargar_config()
    aprendices, verdad = generar(config)

    RUTA_RAW.parent.mkdir(parents=True, exist_ok=True)
    aprendices.to_parquet(RUTA_RAW, index=False)
    verdad.to_parquet(RUTA_VERDAD, index=False)

    logger.info("Semilla %s", config["semilla"])
    logger.info("Aprendices: %s filas x %s columnas -> %s", *aprendices.shape, RUTA_RAW)
    logger.info("Verdad: %s filas -> %s", len(verdad), RUTA_VERDAD)
    logger.info(
        "Estado final: %s",
        aprendices["estado_final"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()
