"""Validación del contrato de los datos procesados.

Comprueba que data/processed/aprendices.parquet cumple el contrato acordado en
docs/datos.md: columnas exactas, tipos, nulos permitidos, rangos, categorías y
reglas del negocio entre columnas. Si algo falla, se detiene el pipeline antes
de que los datos lleguen a features.py.

Uso (desde la raíz del repositorio):
    uv run python -m desercion.validate
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column

RUTA_PROCESADOS = Path("data/processed/aprendices.parquet")

# Nunca deben llegar al modelo (fuga, auditoría o descartadas en process.py).
COLUMNAS_PROHIBIDAS = [
    "estado_final",
    "fecha_desercion",
    "sexo",
    "nombre_programa",
    "barrio_residencia",
    "municipio_nacimiento",
    "tasa_abandono_previa",
]

# --- Categorías permitidas (listas cerradas, docs/datos.md) -----------------
NIVELES_FORMACION = [
    "AUXILIAR",
    "OPERARIO",
    "TÉCNICO",
    "TECNÓLOGO",
    "ESPECIALIZACIÓN TECNOLÓGICA",
]
AREAS_FORMACION = [
    "administracion",
    "agropecuaria",
    "industria",
    "salud",
    "tecnologia",
    "turismo y gastronomia",
]
MODALIDADES = ["virtual", "presencial", "mixta"]
JORNADAS = ["diurna", "nocturna", "mixta", "otra"]
MUNICIPIOS = [
    "Apartadó",
    "Barranquilla",
    "Bello",
    "Bogotá",
    "Bucaramanga",
    "Cali",
    "Caucasia",
    "Envigado",
    "Itagüí",
    "Medellín",
    "Pasto",
    "Pereira",
    "Quibdó",
    "Rionegro",
    "Soacha",
    "Turbo",
]
NIVELES_EDUCATIVOS = [
    "primaria",
    "basica_secundaria",
    "bachillerato",
    "tecnico",
    "tecnologo",
    "profesional",
]
TIPOS_CONEXION = ["cable", "datos_moviles", "fibra", "otra"]
TIPOS_PLAN_INTERNET = ["pospago", "prepago", "otro"]
NIVELES_ALTA_MEDIA_BAJA = ["alta", "media", "baja"]
DISPOSITIVOS = ["computador", "celular", "tableta", "otro"]
SITUACIONES_LABORALES = ["empleado", "desempleado", "independiente", "otra"]

HORAS_SEMANA = 168

logger = logging.getLogger(__name__)


# --- Reglas del negocio entre columnas -------------------------------------
# Cada regla devuelve una Serie booleana por fila: si falla, Pandera reporta
# exactamente qué filas la incumplen.
def _sin_columnas_prohibidas(df: pd.DataFrame) -> bool:
    return not set(COLUMNAS_PROHIBIDAS) & set(df.columns)


def _asistencia_nula_si_y_solo_si_virtual(df: pd.DataFrame) -> pd.Series:
    return df["asistencia_porcentaje"].isna() == (df["modalidad"] == "virtual")


def _jornada_otra_si_y_solo_si_virtual(df: pd.DataFrame) -> pd.Series:
    return (df["jornada"] == "otra") == (df["modalidad"] == "virtual")


def _historial_nulo_si_y_solo_si_primer_curso(df: pd.DataFrame) -> pd.Series:
    primer_curso = df["es_primer_curso"]
    return (df["meses_desde_ultimo_curso"].isna() == primer_curso) & (
        df["abandono_ultimo_curso"].isna() == primer_curso
    )


def _encuesta_nula_en_las_mismas_filas(df: pd.DataFrame) -> pd.Series:
    nulos = df[
        [
            "calidad_internet",
            "horas_disponibles_estudio",
            "dificultad_economica_reportada",
        ]
    ].isna()
    return nulos.all(axis=1) | ~nulos.any(axis=1)


def _dias_desde_inicio_cuadra_con_las_fechas(df: pd.DataFrame) -> pd.Series:
    dias = (df["fecha_corte"] - df["fecha_inicio"]).dt.days
    return dias == df["dias_desde_inicio"]


def _dias_sin_ingreso_dentro_del_periodo(df: pd.DataFrame) -> pd.Series:
    return df["dias_sin_ingreso"] <= df["dias_desde_inicio"]


def _aprobadas_no_superan_entregadas(df: pd.DataFrame) -> pd.Series:
    return (
        df["porcentaje_actividades_aprobadas"]
        <= df["porcentaje_actividades_entregadas"]
    )


# --- Constructores de columnas ---------------------------------------------
def _texto(*checks: Check, nullable: bool = False, unique: bool = False) -> Column:
    return Column(str, list(checks), nullable=nullable, unique=unique)


def _categoria(valores: list[str], nullable: bool = False) -> Column:
    return _texto(Check.isin(valores), nullable=nullable)


def _entero(*checks: Check) -> Column:
    return Column("int64", list(checks))


def _decimal(*checks: Check, nullable: bool = False) -> Column:
    return Column("float64", list(checks), nullable=nullable)


def _proporcion(nullable: bool = False) -> Column:
    return _decimal(Check.in_range(0, 1), nullable=nullable)


def _booleano() -> Column:
    return Column(bool)


def _fecha() -> Column:
    return Column("datetime64[us]")


ESQUEMA = pa.DataFrameSchema(
    {
        # Identificadores, agrupación y etiqueta
        "id_aprendiz": _texto(Check.str_matches(r"^A-\d{6}$"), unique=True),
        "id_cohorte": _texto(Check.str_matches(r"^\d{4}-T[1-4]$")),
        "fecha_inicio": _fecha(),
        "fecha_corte": _fecha(),
        "deserto": _entero(Check.isin([0, 1])),
        # Programa de formación
        "id_programa": _texto(Check.str_matches(r"^P-\d{3}$")),
        "nivel_formacion": _categoria(NIVELES_FORMACION),
        "area_formacion": _categoria(AREAS_FORMACION),
        "modalidad": _categoria(MODALIDADES),
        "duracion_programa_meses": _entero(Check.isin([6, 12, 24])),
        "jornada": _categoria(JORNADAS),
        "dias_desde_inicio": _entero(Check.in_range(21, 35)),
        # Perfil del aprendiz
        "edad": _entero(Check.in_range(14, 100)),
        "municipio_residencia": _categoria(MUNICIPIOS),
        "nivel_educativo": _categoria(NIVELES_EDUCATIVOS),
        "situacion_laboral": _categoria(SITUACIONES_LABORALES),
        "horas_trabajo_semana": _entero(Check.in_range(0, HORAS_SEMANA)),
        "responsabilidades_cuidado": _booleano(),
        "horas_disponibles_estudio": _decimal(
            Check.in_range(0, HORAS_SEMANA), nullable=True
        ),
        "dificultad_economica_reportada": _categoria(
            NIVELES_ALTA_MEDIA_BAJA, nullable=True
        ),
        # Historial en el SENA
        "n_cursos_previos": _entero(Check.ge(0)),
        "n_cursos_completados": _entero(Check.ge(0)),
        "n_cursos_abandonados": _entero(Check.ge(0)),
        "tasa_finalizacion_previa": _proporcion(nullable=True),
        "es_primer_curso": _booleano(),
        "meses_desde_ultimo_curso": _decimal(Check.ge(0), nullable=True),
        # Booleano con nulos (dtype "boolean" de pandas), no bool.
        "abandono_ultimo_curso": Column(pd.BooleanDtype(), nullable=True),
        "experiencia_virtual_previa": _booleano(),
        # Conectividad y equipo
        "tipo_conexion": _categoria(TIPOS_CONEXION),
        "tipo_plan_internet": _categoria(TIPOS_PLAN_INTERNET),
        "calidad_internet": _categoria(NIVELES_ALTA_MEDIA_BAJA, nullable=True),
        "dispositivo_principal": _categoria(DISPOSITIVOS),
        "comparte_dispositivo": _booleano(),
        "disponibilidad_equipo": _booleano(),
        # Seguimiento hasta la fecha de corte
        "porcentaje_actividades_entregadas": _proporcion(),
        "porcentaje_actividades_aprobadas": _proporcion(),
        # Nulo cuando no tiene actividades aprobadas: caso real (docs/datos.md).
        "promedio_calificaciones": _decimal(Check.in_range(0, 5), nullable=True),
        "dias_sin_ingreso": _entero(Check.ge(0)),
        "ingresos_plataforma_ultimos_7d": _entero(Check.ge(0)),
        "actividades_pendientes": _entero(Check.ge(0)),
        "evaluaciones_no_presentadas": _entero(Check.ge(0)),
        "asistencia_porcentaje": _proporcion(nullable=True),
        "solicito_apoyo": _booleano(),
    },
    checks=[
        Check(_sin_columnas_prohibidas, name="sin_columnas_prohibidas"),
        Check(_asistencia_nula_si_y_solo_si_virtual, name="asistencia_nula_si_virtual"),
        Check(_jornada_otra_si_y_solo_si_virtual, name="jornada_otra_si_virtual"),
        Check(
            _historial_nulo_si_y_solo_si_primer_curso,
            name="historial_nulo_si_primer_curso",
        ),
        Check(_encuesta_nula_en_las_mismas_filas, name="encuesta_nula_mismas_filas"),
        Check(_dias_desde_inicio_cuadra_con_las_fechas, name="dias_desde_inicio"),
        Check(_dias_sin_ingreso_dentro_del_periodo, name="dias_sin_ingreso"),
        Check(_aprobadas_no_superan_entregadas, name="aprobadas_le_entregadas"),
    ],
    strict=True,  # una columna de más o de menos rompe la validación
    coerce=False,  # validar, no transformar: un tipo distinto es un error
    name="aprendices_procesados",
)


def validar(df: pd.DataFrame) -> pd.DataFrame:
    """Valida df contra el contrato y lo devuelve sin cambios.

    Lanza pandera.errors.SchemaErrors con todos los incumplimientos a la vez
    (lazy=True), no solo el primero.
    """
    return ESQUEMA.validate(df, lazy=True)


def resumir_fallos(fallos: pd.DataFrame) -> pd.DataFrame:
    """Resume SchemaErrors.failure_cases: una línea por regla incumplida.

    Las reglas entre columnas reportan cada columna de cada fila que falla;
    aquí se cuentan filas y el ejemplo es el id_aprendiz de la primera.
    """
    reglas_entre_columnas = {check.name for check in ESQUEMA.checks}
    resumen = []
    for (columna, regla), grupo in fallos.groupby(
        ["column", "check"], dropna=False, sort=False
    ):
        if regla in reglas_entre_columnas:
            continue
        resumen.append(
            {
                "regla": regla,
                "columna": columna,
                "casos": len(grupo),
                "ejemplo": grupo["failure_case"].iloc[0],
            }
        )
    for regla, grupo in fallos.groupby("check", sort=False):
        if regla not in reglas_entre_columnas:
            continue
        ids = grupo.loc[grupo["column"] == "id_aprendiz", "failure_case"]
        resumen.append(
            {
                "regla": regla,
                "columna": "(entre columnas)",
                "casos": len(ids) or len(grupo),
                "ejemplo": ids.iloc[0] if len(ids) else grupo["failure_case"].iloc[0],
            }
        )
    return pd.DataFrame(resumen, columns=["regla", "columna", "casos", "ejemplo"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    datos = pd.read_parquet(RUTA_PROCESADOS)
    try:
        validar(datos)
    except pa.errors.SchemaErrors as error:
        resumen = resumir_fallos(error.failure_cases)
        logger.error(
            "Los datos no cumplen el contrato (%s):\n%s",
            RUTA_PROCESADOS,
            resumen.to_string(index=False),
        )
        raise SystemExit(1) from None

    logger.info(
        "Contrato OK: %s filas x %s columnas en %s", *datos.shape, RUTA_PROCESADOS
    )


if __name__ == "__main__":
    main()
