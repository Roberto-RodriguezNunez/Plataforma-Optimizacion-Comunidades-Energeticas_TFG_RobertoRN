Tengo todos los datos. El total actual en seed.py es 54.4 kWp (no 50 — hay inconsistencia también ahí). La escala correcta hacia 42 kWp es ×0.772. Ya tengo los valores exactos calculados para cada casa. Ahora escribo el plan completo.

● Updated plan
/plan to preview
User rejected Claude's plan:
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Plan: Tareas pendientes para completar el TFG (2026-06-18)                                                                                                   │
│                                                                                                                                                              │
│ Contexto                                                                                                                                                     │
│                                                                                                                                                              │
│ Análisis exhaustivo del codebase realizado el 2026-06-18. La integración Edge → SaaS está                                                                    │
│ 100% implementada y funcional: 182 tests en SaaS, blueprint edge con CSRF exempt, /health,                                                                   │
│ modelos ONNX exportados, edge_loop.py con POST al SaaS, generación de cierres desde                                                                          │
│ OperacionHoraria. Lo que queda son tareas operativas, de datos y de documentación.                                                                           │
│                                                                                                                                                              │
│ ---                                                                                                                                                          │
│ BLOQUE A — Código sin commitear (inmediato)                                                                                                                  │
│                                                                                                                                                              │
│ Hay 5 cambios implementados en sesiones anteriores que nunca se commitearon:                                                                                 │
│                                                                                                                                                              │
│ 1. src/saas/app/__init__.py — /health endpoint + csrf.exempt(edge_bp)                                                                                        │
│ 2. docker-compose.yml — healthcheck web, depends_on: service_healthy, START_STEP=5137                                                                        │
│ 3. src/saas/app/modules/viviendas/routes.py — n_cierres y cierres_tabla separados                                                                            │
│ 4. src/saas/app/templates/viviendas/detalle.html — usa n_cierres y cierres_tabla                                                                             │
│ 5. src/saas/tests/test_edge.py — TestHealth y TestEdgeCsrfExento                                                                                             │
│ 6. tests/test_edge_loop_saas.py — nuevo archivo (3 tests de resiliencia del edge loop)                                                                       │
│                                                                                                                                                              │
│ Acción: Un único commit que agrupe todos estos cambios.                                                                                                      │
│                                                                                                                                                              │
│ ---                                                                                                                                                          │
│ BLOQUE B — Re-ejecución Docker (datos estacionales correctos)                                                                   │
│                                                                                                                                                              │
│ La BD actual tiene datos con estaciones invertidas (verano 2021 etiquetado como enero 2025).                                                                 │
│ El fix (START_STEP=5137) está en docker-compose.yml pero no se ha apl                                                           │
│                                                                                                                                                              │
│ docker compose down -v          # borra pgdata con datos erróneos                                                                                            │
│ docker compose up --build -d    # reconstruye + START_STEP=5137 activ                                                           │
│ docker compose exec web python seed.py  # re-seed                                                                               │
│ docker compose up edge          # corre 8760 pasos (≈ 6-7 min)                                                                  │
│                                                                                                                                 │
│ Verificar tras el run:                                                                                                                                       │
│ - Enero 2025: factura_real > 0 (inverno → poca solar, compra de red normal)                                                                                  │
│ - Julio 2025: factura_real ≈ 0 o muy baja (verano → solar cubre casi                                                            │
│ - {{ n_cierres }} muestra 12 en el dashboard de vivienda                                                                                                     │
│                                                                                                                                                              │
│ ---                                                                                                                             │
│ BLOQUE C — Comunidad 2 (Brañas de Ulla) — sin datos edge                                                                        │
│                                                                                                                                 │
│ COMUNIDAD_ID=1 en docker-compose.yml → comunidad 2 nunca recibe Opera                                                           │
│ Opciones:                                                                                                                       │
│ - (Recomendada) Documentarlo como limitación del demo: edge está configurado para una sola                                                                   │
│ comunidad en este despliegue, extensible añadiendo una segunda instancia del servicio edge.                                                                  │
│ - (Alternativa) Añadir segunda instancia edge en docker-compose.yml c                                                           │
│                                                                                                                                                              │
│ Para el TFG basta con documentarlo. No requiere código adicional.                                                               │
│                                                                                                                                 │
│ ---                                                                                                                             │
│ BLOQUE D — Evaluación formal (para la memoria)                                                                                  │
│                                                                                                                                                              │
│ src/eval_escenarios.py está completo y calcula resultados en vivo. Hay que ejecutarlo                                                                        │
│ y capturar la salida para la Tabla 7.4 de la memoria.                                                                           │
│                                                                                                                                 │
│ # Desde el entorno Python con SB3 y numpy:                                                                                                                   │
│ python src/eval_escenarios.py                                                                                                                                │
│ # Salida: tabla con 7 escenarios (A, B, G, H, I, J, K) en €/sem ± std                                                           │
│                                                                                                                                                              │
│ Los resultados ya están documentados en doc/anotaciones/historial_ver                                                           │
│ las Tablas 7.3–7.5 de la memoria. Solo es necesario si los números de                                                           │
│                                                                                                                                 │
│ ---                                                                                                                                                          │
│ BLOQUE E — Figuras (YA DISPONIBLES ✅)                                                                                                                       │
│                                                                                                                                 │
│ Figuras Fig 4.1, Fig 7.4 y Fig 8.2 ya generadas por el usuario. Solo .tex.                                                      │
│                                                                                                                                                              │
│ ---                                                                                                                             │
│ BLOQUE F — Apéndices pendientes en la memoria LaTeX                                                                             │
│                                                                                                                                 │
│ ┌──────────┬──────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ Apéndice │                     Placeholder                      │                                         Acción                                         │ │
│ ├──────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤ │
│ │ A        │ [Espacio reservado para config/system.yaml completo] │ Copiar config/system.yaml literalmente                                                 │ │
│ ├──────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤ │
│ │ B        │ [Espacio reservado para árbol de directorios]        │ tree -I '__pycache__|*.pyc|pgdata' y pegar                                             │ │
│ ├──────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤ │
│ │ D        │ [Espacio reservado para estimación ROI]              │ Cste 33k€, payback ~17 años; con subvención 40% → 10 años │ │
│ └──────────┴──────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────┘ │
│                                                                                                                                 │
│ Apéndice D ya tiene números de sesiones anteriores:                                                                             │
│ - Sin subvención: payback ≈ 17 años (supera vida útil batería de 10-1                                                           │
│ - Con subvención 40%: coste neto 19.8k€ → payback ≈ 10 años → límite                                                            │
│                                                                                                                                                              │
│ ---                                                                                                                                                          │
│ BLOQUE G — Portada oficial                                                                                                      │
│                                                                                                                                 │
│ \includepdf{portada_oficial.pdf} en la memoria requiere el PDF oficia                                                           │
│ Generarlo desde el sistema de gestión de TFGs de la UDC o solicitarlo                                                           │
│                                                                                                                                 │
│ ---                                                                                                                             │
│ BLOQUE H — Qué añadir a la memoria LaTeX                                                                                        │
│                                                                                                                                                              │
│ H.1 Plataforma web SaaS — nuevo capítulo (recomendado)                                                                                                       │
│                                                                                                                                 │
│ Insertar un capítulo nuevo entre el actual Cap 7 (Motor de Decisión)                                                            │
│                                                                                                                                 │
│ Cap 7 — Motor de Decisión (ya escrito)                                                                                          │
│ Cap 8 — Plataforma Web LeaLink  ← NUEVO                                                                                         │
│ Cap 9 — Planificación (ex Cap 8, renumerar)                                                                                     │
│ Cap 10 — Aportaciones (ex Cap 9)                                                                                                │
│ Cap 11 — Conclusiones (ex Cap 10)                                                                                                                            │
│ Cap 12 — Trabajo Futuro (ex Cap 11)                                                                                                                          │
│                                                                                                                                 │
│ Estructura del nuevo Cap 8:                                                                                                     │
│ - 8.1 Arquitectura Flask (blueprints, modelos, patrón MVC)                                                                      │
│ - 8.2 Modelo de datos (Comunidad, Vivienda, Batería, CierreMensual, OperacionHoraria)                                                                        │
│ - 8.3 Casos de uso (diagrama ya disponible)                                                                                                                  │
│ - 8.4 Diagramas de secuencia: flujo edge→API→OperacionHoraria; admin→oard                                                       │
│ - 8.5 API REST edge (/api/edge/decision, autenticación Bearer, CSRF e                                                           │
│ - 8.6 Screenshots del dashboard (opcional)                                                                                      │
│                                                                                                                                                              │
│ H.2 ONNX — subsección dentro del Cap 7 (no capítulo aparte)                                                                                                  │
│                                                                                                                                 │
│ Añadir 7.X Despliegue en producción al final del Cap 7:                                                                         │
│ - 7.X.1 Exportación del actor a ONNX (sin SB3/PyTorch en runtime)                                                               │
│ - 7.X.2 Clase OnnxResidualController (inferencia edge)                                                                          │
│ - 7.X.3 Verificación de equivalencia (test_onnx_equivalence.py)                                                                                              │
│ - 7.X.4 Contenedor Docker (Dockerfile.edge, dependencias mínimas: numpy, scipy, onnxruntime)                                                                 │
│                                                                                                                                 │
│ Esta subsección es clave para justificar viabilidad de despliegue reas).                                                        │
│                                                                                                                                                              │
│ H.3 Diagramas de secuencia y casos de uso                                                                                                                    │
│                                                                                                                                 │
│ Los diagramas van en Cap 8 (plataforma web), no en Cap 7 (motor SAC).                                                           │
│ Muestran interacción entre componentes desde perspectiva del usuario                                                            │
│                                                                                                                                                              │
│ ---                                                                                                                             │
│ BLOQUE I — Redimensionado solar + batería + corrección factura_base (                                                           │
│                                                                                                                                 │
│ Justificación del cambio (para documentar en memoria)                                                                           │
│                                                                                                                                                              │
│ ▎ "La instalación inicial de 50 kWp producía ~62.500 kWh/año frente a                                                           │
│ ▎ consumo (excedente neto del 18%). Bajo el mecanismo de compensación simplificada del                                                                       │
│ ▎ RD 244/2019, el excedente no genera retorno económico directo. La instalación se redimensiona                                                              │
│ ▎ a 42 kWp (2,8 kWp/vivienda media), que produce 52.500 kWh/año con l                                                           │
│ ▎ equivalente de ~1.250 h/año en Galicia, maximizando la tasa de autoconsumo.                                                                                │
│ ▎ La batería de 100 kWh proporciona ~80 kWh de capacidad útil (SoC 10-90%) con un                                                                            │
│ ▎ inversor de 50 kW (ratio C/2), adecuada para el ciclo diario de 15                                                            │
│                                                                                                                                                              │
│ ---                                                                                                                             │
│ I.0 Inconsistencia detectada (contexto)                                                                                         │
│                                                                                                                                                              │
│ - config/system.yaml: potencia_solar_total_kwp: 50.0 (usado en entrenamiento)                                                                                │
│ - src/saas/seed.py VILARÍN: suma individual = 54.4 kWp (no 50, había                                                            │
│ - src/saas/seed.py baterías: 20 kWh (Vilarín) y 30 kWh (Brañas) — no coinciden con 100 kWh del training                                                      │
│                                                                                                                                 │
│ I.1 config/system.yaml                                                                                                          │
│                                                                                                                                 │
│ # ANTES                                                                                                                         │
│ potencia_solar_total_kwp: 50.0                                                                                                                               │
│                                                                                                                                                              │
│ # DESPUÉS                                                                                                                       │
│ potencia_solar_total_kwp: 42.0                                                                                                                               │
│                                                                                                                                 │
│ I.2 Regenerar dataset                                                                                                           │
│                                                                                                                                                              │
│ python src/utils/generar_dataset_final.py                                                                                                                    │
│ generar_dataset_final.py lee POTENCIA_SOLAR_TOTAL de system.yaml (lín                                                           │
│ la generación solar de 1 kWp a 42 kWp. No requiere más cambios en ese                                                           │
│                                                                                                                                                              │
│ I.3 Reentrenar modelo SAC                                                                                                                                    │
│                                                                                                                                 │
│ python src/main_sac.py --seed 42 --n-steps 1000000                                                                              │
│                                                                                                                                 │
│ I.4 Re-exportar ONNX                                                                                                            │
│                                                                                                                                 │
│ python src/production/export_onnx.py                                                                                            │
│                                                                                                                                 │
│ I.5 src/saas/seed.py — nuevos kWp por vivienda Vilarín (×0.772 sobre valores anteriores, suma = 42.0 kWp)                                                    │
│                                                                                                                                                              │
│ VIVIENDAS_VILARIN = [                                                                                                           │
│     # (identificador, direccion, cups_suffix, potencia, coef, paneles                                                           │
│     ('Casa 1 — Rúa da Igrexa',    ..., True, 2.4,  6, 'sur'),   # era                                                           │
│     ('Casa 2 — Rúa da Igrexa',    ..., True, 2.4,  6, 'sur'),   # era                                                           │
│     ('Casa 3 — Rúa da Igrexa',    ..., True, 2.0,  5, 'sur'),   # era 2.8                                                                                    │
│     ('Casa 4 — Rúa da Igrexa',    ..., True, 2.0,  5, 'sur'),   # era                                                           │
│     ('Casa 5 — Rúa da Igrexa',    ..., True, 3.2,  8, 'sur'),   # era 4.0                                                                                    │
│     ('Casa 6 — Rúa da Igrexa',    ..., True, 2.0,  5, 'sur'),   # era 2.8                                                                                    │
│     ('Casa 1 — Camiño do Río',    ..., True, 4.4, 11, 'sur'),   # era 5.6                                                                                    │
│     ('Casa 2 — Camiño do Río',    ..., True, 2.8,  7, 'sur'),   # era                                                           │
│     ('Casa 3 — Camiño do Río',    ..., True, 2.0,  5, 'sur'),   # era 2.4                                                                                    │
│     ('Casa 4 — Camiño do Río',    ..., True, 2.4,  6, 'sur'),   # era                                                           │
│     ('Casa 5 — Camiño do Río',    ..., True, 5.2, 13, 'sur'),   # era 6.4                                                                                    │
│     ('Casa 6 — Camiño do Río',    ..., True, 2.8,  7, 'sur'),   # era                                                           │
│     ('Casa 1 — Lugar de Outeiro', ..., True, 2.4,  6, 'sur'),   # era                                                           │
│     ('Casa 2 — Lugar de Outeiro', ..., True, 2.4,  6, 'sur'),   # era                                                           │
│     ('Casa 3 — Lugar de Outeiro', ..., True, 3.6,  9, 'sur'),   # era 4.8                                                                                    │
│ ]                                                                                                                                                            │
│ # Suma: 42.0 kWp exactos, 105 paneles de 400 Wp                                                                                 │
│                                                                                                                                 │
│ Brañas: mantener kWp individual sin cambio (comunidad 2 sin datos edg                                                           │
│                                                                                                                                 │
│ I.6 src/saas/seed.py — batería a 100 kWh (ambas comunidades)                                                                                                 │
│                                                                                                                                 │
│ # Vilarín (líneas ~340-341)                                                                                                     │
│ capacidad_nominal_kwh=100.0,     # era 20.0                                                                                     │
│ capacidad_util_actual_kwh=98.0,  # era 18.4                                                                                                                  │
│                                                                                                                                                              │
│ # Brañas (líneas ~350-351)                                                                                                      │
│ capacidad_nominal_kwh=100.0,     # era 30.0                                                                                     │
│ capacidad_util_actual_kwh=98.0,  # era 29.1                                                                                     │
│                                                                                                                                 │
│ I.7 src/saas/app/modules/cierres/routes.py — corregir factura_base                                                              │
│                                                                                                                                 │
│ En acums añadir dos acumuladores nuevos:                                                                                        │
│ acums = [dict(..., compra_base_cost=0.0, compensacion_base=0.0) for _ in viviendas]                                                                          │
│                                                                                                                                                              │
│ En el bucle por hora/vivienda, reemplazar acums[j]['coste_base'] += c                                                           │
│ compra_base  = max(0.0, c - g)                                                                                                  │
│ surplus_base = max(0.0, g - c)                                                                                                  │
│ acums[j]['compra_base_cost']  += compra_base * pc                                                                               │
│ acums[j]['compensacion_base'] += surplus_base * pe                                                                              │
│                                                                                                                                                              │
│ En el cálculo final, reemplazar factura_base = round(acum['coste_base                                                           │
│ factura_base = round(max(0.0, acum['compra_base_cost'] - acum['compensacion_base']), 2)                                                                      │
│                                                                                                                                                              │
│ I.8 doc/memoria_latex/memoria_tipo2 (3).tex — actualizar referencias                                                            │
│                                                                                                                                                              │
│ - Línea 483: \SI{50}{kWp} → \SI{42}{kWp}                                                                                        │
│ - Línea 686: \SI{50}{kWp} → \SI{42}{kWp}                                                                                                                     │
│ - Añadir párrafo de justificación del dimensionado en Cap 6 (Ingeniería de Datos) citando                                                                    │
│ RD 244/2019 y la irradiancia equivalente de Galicia                                                                             │
│                                                                                                                                 │
│ I.9 doc/anotaciones/historial_versiones.md                                                                                      │
│                                                                                                                                                              │
│ Añadir entrada al inicio:                                                                                                                                    │
│ ## Redimensionado comunidad (2026-06-18)                                                                                        │
│ - Solar: 50 kWp → 42 kWp (equilibrio consumo=generación bajo RD 244/2                                                           │
│ - Batería seed: 20/30 kWh → 100 kWh (coherente con system.yaml y trai                                                           │
│ - factura_base: corregida para incluir autoconsumo propio sin comunid                                                           │
│ - Requiere: regenerar dataset + reentrenar SAC v6 + re-exportar ONNX                                                                                         │
│                                                                                                                                 │
│ ---                                                                                                                             │
│ Orden de ejecución                                                                                                                                           │
│                                                                                                                                                              │
│ 1. config/system.yaml: cambiar 50→42                                                                                            │
│ 2. python src/utils/generar_dataset_final.py — regenerar dataset                                                                                             │
│ 3. python src/main_sac.py — reentrenar (~1-2h)                                                                                                               │
│ 4. python src/production/export_onnx.py — re-exportar ONNX                                                                                                   │
│ 5. src/saas/seed.py — nuevos kWp + baterías 100 kWh                                                                             │
│ 6. src/saas/app/modules/cierres/routes.py — fix factura_base                                                                                                 │
│ 7. doc/memoria_latex/ — actualizar 50→42 + párrafo justificación                                                                                             │
│ 8. doc/anotaciones/historial_versiones.md — entrada de cambio                                                                   │
│ 9. docker compose down -v && docker compose up --build -d && docker cy                                                          │
│                                                                                                                                                              │
│ ---                                                                                                                             │
│ I.1 Corregir factura_base en _calcular_cierre_desde_operaciones()                                                               │
│                                                                                                                                                              │
│ Archivo: src/saas/app/modules/cierres/routes.py                                                                                                              │
│                                                                                                                                 │
│ Problema: factura_base = consumo × precio_compra modela "sin ningún panel solar", que es                                                                     │
│ una comparación irreal. La casa YA tiene paneles, así que el baseline                                                           │
│ "con mis propios paneles pero sin batería comunitaria ni compartir en                                                           │
│                                                                                                                                 │
│ Cambio en el dict acums — añadir dos acumuladores:                                                                                                           │
│ acums = [dict(..., compra_base_cost=0.0, compensacion_base=0.0) for _ in viviendas]                                                                          │
│                                                                                                                                 │
│ Cambio en el bucle por vivienda — reemplazar coste_base:                                                                                                     │
│ # ANTES                                                                                                                                                      │
│ acums[j]['coste_base'] += c * pc                                                                                                                             │
│                                                                                                                                 │
│ # DESPUÉS                                                                                                                                                    │
│ compra_base  = max(0.0, c - g)   # compra si solo tienes tus paneles,                                                           │
│ surplus_base = max(0.0, g - c)   # excedente que venderías individualmente                                                                                   │
│ acums[j]['compra_base_cost']  += compra_base * pc                                                                                                            │
│ acums[j]['compensacion_base'] += surplus_base * pe                                                                              │
│                                                                                                                                 │
│ Cambio en el cálculo final — reemplazar factura_base:                                                                                                        │
│ # ANTES                                                                                                                                                      │
│ factura_base = round(acum['coste_base'], 2)                                                                                     │
│                                                                                                                                 │
│ # DESPUÉS                                                                                                                       │
│ factura_base = round(max(0.0, acum['compra_base_cost'] - acum['compen                                                           │
│                                                                                                                                                              │
│ Ahora ahorro = factura_base - factura_real mide el valor REAL de la c                                                           │
│ (batería + compartir energía entre vecinos) frente a tener paneles in                                                           │
│                                                                                                                                 │
│ I.2 Corregir capacidad de batería en seed.py                                                                                                                 │
│                                                                                                                                                              │
│ Archivo: src/saas/seed.py                                                                                                       │
│                                                                                                                                 │
│ Cambiar las dos baterías de 20/30 kWh → 100 kWh para que coincidan con system.yaml                                                                           │
│ y con el modelo ONNX entrenado:                                                                                                 │
│ # Vilarín (comunidad 1)                                                                                                                                      │
│ capacidad_nominal_kwh=100.0,     # era 20.0                                                                                     │
│ capacidad_util_actual_kwh=98.0,  # era 18.4                                                                                     │
│                                                                                                                                 │
│ # Brañas (comunidad 2)                                                                                                          │
│ capacidad_nominal_kwh=100.0,     # era 30.0                                                                                                                  │
│ capacidad_util_actual_kwh=98.0,  # era 29.1                                                                                                                  │
│                                                                                                                                 │
│ I.3 Realismo de los parámetros del sistema                                                                                                                   │
│                                                                                                                                                              │
│ 50 kWp / 15 casas = 3,3 kWp por vivienda → Realista para España (instalación residencial                                                                     │
│ típica 3-6 kWp). ✓                                                                                                                                           │
│                                                                                                                                 │
│ 100 kWh de batería comunitaria → Realista para 15 casas con 137 kWh/d                                                           │
│ Las baterías comunitarias en España suelen ser 50-200 kWh para comuni                                                           │
│                                                                                                                                                              │
│ Por qué gen > consumo: 50 kWp × 1.400 h equiv./año (Galicia) ≈ 70.000                                                           │
│ vs 52.500 kWh/año consumidos → excedente del 33%. No es irreal para una comunidad                                                                            │
│ orientada a maximizar autoconsumo y exportar excedentes. Lo que hace que factura_real ≈ 0                                                                    │
│ es que el baseline (coste_base) no restaba el autoconsumo propio — es                                                           │
│                                                                                                                                 │
│ Por qué el seed tenía 20/30 kWh: El seed.py fue escrito antes de que                                                            │
│ system.yaml. Los 100 kWh se pusieron en training pero nunca se propagaron al seed.                                                                           │
│ Es un error de mantenimiento, no de diseño.                                                                                                                  │
│                                                                                                                                 │
│ ---                                                                                                                             │
│ Orden de prioridad                                                                                                              │
│                                                                                                                                 │
│ 1. Commitear cambios pendientes (Bloque A) — 5 min                                                                                                           │
│ 2. Re-ejecutar Docker con START_STEP=5137 y verificar datos (Bloque B) — 15 min                                                                              │
│ 3. Fig 7.4 curvas TensorBoard (Bloque E) — más importante para la mem                                                           │
│ 4. Apéndices A y B (Bloque F) — copiar/pegar, triviales                                                                                                      │
│ 5. Fig 4.1 arquitectura y Fig 8.2 Gantt (Bloque E) — diseño visual                                                                                           │
│ 6. Apéndice D análisis económico (Bloque F) — calcular y escribir                                                                                            │
│ 7. Portada (Bloque G) — depende de universidad                                                                                  │
│ 8. Comunidad 2 (Bloque C) — solo documentar como limitación                                                                     │
│                                                                                                                                 │
│ ---                                                                                                                                                          │
│ Estado del código (resumen)                                                                                                                                  │
│                                                                                                                                 │
│ ┌────────────────────────────────────────────────────┬─────────────────────────────────┐                                        │
│ │                        Área                        │                                 │                                        │
│ ├────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤                                        │
│ │ SaaS Flask (12 módulos, 182 tests)                 │ ✅ Completo                                                  │                                        │
│ ├────────────────────────────────────────────────────┼─────────────────────────────────┤                                        │
│ │ Modelos ONNX exportados                            │ ✅ Listos (models/residual_sac_actor.onnx)                   │                                        │
│ ├────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤                                        │
│ │ edge_loop.py + onnx_inference.py + obs_builder.py  │ ✅ Completo                     │                                        │
│ ├────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤                                        │
│ │ eval_escenarios.py + eval_unificada.py             │ ✅ Completo                     │                                        │
│ ├────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤                                        │
│ │ main_sac.py + mpc_benchmark.py                     │ ✅ Completo                     │                                        │
│ ├────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤                                        │
│ │ controllers/ (base, heurístico, residual SAC, DQN) │ ✅ Completo                     │                                        │
│ ├────────────────────────────────────────────────────┼─────────────────────────────────┤                                        │
│ │ Memoria LaTeX (capítulos 1–11)                     │ ✅ Escrita                      │                                        │
│ ├────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤                                        │
│ │ Figuras en memoria                                 │ ⚠️ 3 placehold                  │                                        │
│ ├────────────────────────────────────────────────────┼─────────────────────────────────┤                                        │
│ │ Apéndices A, B, D                                  │ ⚠️ 3 espacios reservados                                     │                                        │
│ ├────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤                                        │
│ │ Cambios sin commitear                              │ ⚠️ 6 archivos                   │                                        │
│ ├────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤                                        │
│ │ Datos Docker (estaciones)                          │ ⚠️ Requiere re-ejecución                                     │                                        │
│ ├────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤                                        │
│ │ src/utils/generador_datos.py                       │ ⚠️ Vacío (pero no se usa — ETL usa generar_dataset_final.py) │