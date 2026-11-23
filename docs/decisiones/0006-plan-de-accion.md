# ADR 006: Plan de acción para el instructor

- **Estado:** Propuesta (rama `plan-de-accion`, trabajo adicional a la entrega del 11 de noviembre)
- **Fecha:** 2026-11-15
- **Decisores:** Andrés Medina, David Montilla

## Contexto

La API entrega, para cada aprendiz con alerta, hasta tres medidas: una frase por
medida (por ejemplo, "Llamar o escribir al aprendiz: no está entrando a la
plataforma"). La frase dice el tema, pero no cuándo actuar, qué decirle al aprendiz
ni qué hacer si no responde. Dos instructores pueden leer la misma medida y actuar
de formas muy distintas, y nada queda registrado para saber después qué funcionó.

## Decisión

Cada medida de `config/recomendaciones/medidas.yaml`, incluida la general, lleva un
**plan de acción** con los mismos siete campos:

| Campo | Para qué sirve |
|---|---|
| `accion` | Qué hacer y por qué canal |
| `plazo_dias` | En cuántos días hacerlo |
| `decir` | Guion de tres frases: apertura, pregunta abierta y propuesta |
| `evitar` | Lo que no se debe decir o hacer |
| `si_no_responde` | Segundo intento y a quién acudir |
| `registrar` | Qué anotar en el seguimiento |
| `seguimiento_dias` | En cuántos días volver a revisar |

1. **Campo nuevo en la API, sin romper el contrato.** `RiesgoAprendiz` gana
   `plan_de_accion` (lista de `Accion`); `medidas` no cambia. Hay un plan por cada
   medida, en el mismo orden: el primero responde al factor que más sube el riesgo.
   Sin alerta, la lista va vacía. La versión de la API pasa a 0.4.0.
2. **La regla no cambia.** `recomendaciones.plan_para` usa la misma selección que
   `medidas_para` (factores de mayor a menor aporte, sin repetir, máximo 3, medida
   general si ninguno tiene medida propia).
3. **El guion nunca le dice al aprendiz que tiene una alerta.** Las frases de `decir`
   abren con interés por el aprendiz y una pregunta abierta. No mencionan el riesgo,
   la alerta, el modelo, la deserción ni sanciones: decirle a alguien que "va a
   desertar" puede empujarlo a hacerlo. Una prueba lo verifica palabra por palabra.
4. **Los textos viven en el YAML.** Se ajustan sin tocar código. `cargar_medidas`
   falla al arrancar si a una medida le falta el plan o algún campo.
5. **No toca el pipeline.** Ni `medidas.yaml` ni `recomendaciones.py` son
   dependencias de etapas de DVC: no hay que reentrenar ni cambia `dvc.lock`.

## Consecuencias

- El instructor recibe una guía concreta y pareja entre instructores, y el campo
  `registrar` deja definido qué anotar. Ese registro es lo que permitiría, más
  adelante, medir qué acciones funcionan.
- **Los guiones son una propuesta del equipo y no están validados.** Con datos
  sintéticos no se puede demostrar que reducen la deserción. Antes de usarlos hay
  que revisarlos con instructores y con Bienestar al Aprendiz, y ajustar plazos y
  rutas de remisión a los procedimientos reales del SENA.
- **Un factor no es una causa.** El plan se activa por lo que explica la alerta del
  modelo, no por lo que hace desertar al aprendiz. Además, un factor significa "peor
  que el aprendiz promedio", no "cero": un aprendiz con pocos ingresos a la
  plataforma puede recibir el guion de inactividad aunque sí haya entrado. Por eso
  el guion empieza con una pregunta y el instructor lo adapta.
- **Un dato vacío no activa un plan.** El imputador agrega un indicador de faltante
  que puede subir el riesgo, pero eso no describe algo que le pase al aprendiz: la
  asistencia es nula en todos los virtuales porque no tienen sesiones. Con los datos
  v2, 397 virtuales con alerta recibían el guion "te he echado de menos en las
  sesiones". El factor se sigue mostrando, pero no genera medida
  (`prediccion.factores_con_dato`); si no queda ninguno, va el plan general.
- La respuesta de la API es más larga: hasta tres planes por aprendiz con alerta.

## Pendiente

- ~~Mostrar el valor del aprendiz junto a cada factor~~. Hecho el 21 de noviembre
  (API 0.5.0): cada factor trae `valor` y `promedio`; una prueba con el modelo real
  verifica que el valor queda del lado del promedio que sube el riesgo.
- Prioridad alta o media por aprendiz. La probabilidad calibrada, que ya se puede leer
  como porcentaje, está en la API desde la 0.6.0 (`probabilidad_calibrada`); falta
  decidir con los instructores dónde cortar las bandas.
- Un lugar donde registrar el seguimiento (hoy la API solo consulta).
