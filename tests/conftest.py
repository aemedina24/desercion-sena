"""Fixtures compartidas por las pruebas.

Los datos de prueba se construyen aquí, en memoria. Las pruebas nunca leen
data/raw ni data/processed: no están en Git y el CI no los tendrá.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def crudos() -> pd.DataFrame:
    """Muestra mínima con la forma de data/raw/aprendices.parquet.

    Trae las columnas que usa process.py y dos variables del modelo. Cada fila
    cubre un caso que las pruebas necesitan:

    - A-000001: completó el programa.
    - A-000002: desertó (tiene fecha_desercion) y su texto trae espacios.
    - A-000003: traslado escrito con espacios; debe excluirse igual.
    - A-000004: cancelado; también se excluye (decisión del PR #8).
    """
    return pd.DataFrame(
        {
            "id_aprendiz": ["A-000001", "A-000002", "A-000003", "A-000004"],
            "modalidad": ["virtual", " presencial ", "mixta", "virtual"],
            "municipio_residencia": ["Medellín", "  Bello  ", "Cali", "Turbo"],
            "sexo": ["F", "M", "F", "M"],
            "estado_final": ["completado", "deserto", " traslado ", "cancelado"],
            "fecha_desercion": pd.to_datetime([None, "2024-03-01", None, None]),
            "nombre_programa": ["Contabilidad", "Cocina", "Sistemas", "Enfermería"],
            "barrio_residencia": ["Laureles", "Niquía", "San Fernando", "Gaitán"],
            "municipio_nacimiento": ["Medellín", "Bello", "Cali", "Turbo"],
            "tasa_finalizacion_previa": [1.0, 0.5, None, 0.0],
            "tasa_abandono_previa": [0.0, 0.5, None, 1.0],
            "deserto": [0, 1, 0, 0],
        }
    )


@pytest.fixture
def procesados() -> pd.DataFrame:
    """Muestra válida con la forma de data/processed/aprendices.parquet.

    Tiene las 43 columnas del contrato con sus tipos exactos y cumple todas las
    reglas. Cada fila cubre un caso de nulos permitidos:

    - A-000001: virtual (sin asistencia, jornada "otra"), primer curso (sin
      historial), sin encuesta y sin actividades aprobadas (sin promedio).
    - A-000002: presencial, con historial completo y encuesta.
    - A-000003: mixta, con un curso previo aún abierto (sin tasa previa).
    """
    fecha_inicio = pd.to_datetime(["2025-07-14"] * 3).as_unit("us")
    dias_desde_inicio = [28, 30, 21]
    fecha_corte = fecha_inicio + pd.to_timedelta(dias_desde_inicio, unit="D")
    return pd.DataFrame(
        {
            "id_aprendiz": ["A-000001", "A-000002", "A-000003"],
            "id_programa": ["P-101", "P-302", "P-502"],
            "nivel_formacion": ["AUXILIAR", "TÉCNICO", "ESPECIALIZACIÓN TECNOLÓGICA"],
            "area_formacion": ["administracion", "salud", "turismo y gastronomia"],
            "modalidad": ["virtual", "presencial", "mixta"],
            "duracion_programa_meses": [6, 12, 24],
            "jornada": ["otra", "diurna", "nocturna"],
            "fecha_inicio": fecha_inicio,
            "dias_desde_inicio": dias_desde_inicio,
            "edad": [14, 27, 100],
            "municipio_residencia": ["Medellín", "Itagüí", "Quibdó"],
            "nivel_educativo": ["basica_secundaria", "bachillerato", "profesional"],
            "n_cursos_previos": [0, 2, 1],
            "n_cursos_completados": [0, 1, 0],
            "n_cursos_abandonados": [0, 1, 0],
            "tasa_finalizacion_previa": [None, 0.5, None],
            "experiencia_virtual_previa": [False, True, False],
            "meses_desde_ultimo_curso": [None, 6.5, 0.1],
            "es_primer_curso": [True, False, False],
            "abandono_ultimo_curso": pd.array([None, True, False], dtype="boolean"),
            "tipo_conexion": ["datos_moviles", "fibra", "cable"],
            "tipo_plan_internet": ["prepago", "pospago", "otro"],
            "calidad_internet": [None, "alta", "baja"],
            "dispositivo_principal": ["celular", "computador", "tableta"],
            "comparte_dispositivo": [True, False, False],
            "disponibilidad_equipo": [False, True, True],
            "situacion_laboral": ["desempleado", "empleado", "independiente"],
            "horas_trabajo_semana": [0, 48, 20],
            "responsabilidades_cuidado": [True, False, True],
            "horas_disponibles_estudio": [None, 10.0, 15.0],
            "dificultad_economica_reportada": [None, "media", "alta"],
            "porcentaje_actividades_entregadas": [0.036, 0.9, 1.0],
            "porcentaje_actividades_aprobadas": [0.0, 0.8, 1.0],
            "promedio_calificaciones": [None, 4.1, 5.0],
            "dias_sin_ingreso": [28, 1, 0],
            "ingresos_plataforma_ultimos_7d": [0, 6, 18],
            "actividades_pendientes": [9, 1, 0],
            "evaluaciones_no_presentadas": [2, 0, 0],
            "asistencia_porcentaje": [None, 0.9, 1.0],
            "solicito_apoyo": [False, True, False],
            "fecha_corte": fecha_corte,
            "deserto": [1, 0, 0],
            "id_cohorte": ["2025-T3", "2025-T3", "2025-T3"],
        }
    )
