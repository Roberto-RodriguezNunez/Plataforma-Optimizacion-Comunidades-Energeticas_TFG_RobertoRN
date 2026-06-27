# Cambios del entorno — invalidación de resultados previos

Este documento registra **qué cambió en el entorno/simulador** respecto a las versiones
anteriores de los modelos, para justificar en la memoria por qué **todos los resultados
históricos quedan invalidados** y se reentrena la suite completa sobre el entorno definitivo.

Todas las cifras anteriores (DQN +45.23, PPO +45.69, Residual SAC +48.x €/sem, etc.) se
midieron sobre entornos distintos y **no son comparables** con los nuevos.

## Cambios (todos ya en `main`/rama de entreno)

1. **Capacidad de batería 80 kWh (desde config).**
   El simulador hardcodeaba 100 kWh mientras `config/system.yaml`, memoria y SaaS decían 80.
   Ahora `ComunidadSimulador` lee toda la batería de `config['bateria']` (80 kWh, inversor 50 kW,
   eficiencias, SoC, autodescarga, degradación). Cambia el margen de arbitraje → cambia el €/sem
   de todos. Baseline MPC realista pasó de ~+47.7 (100 kWh) a ~+43.5 (80 kWh).

2. **Generación 42 kWp** (antes referida como 50). El dataset se regeneró con 42 kWp
   (`generar_dataset_final` lee config). Menos excedente solar → menos energía que desplazar.

3. **Convención sin lookahead (ruido desde k=0).**
   Antes el agente/MPC podían ver el presente exacto; ahora **ambos** reciben ruido en la hora
   actual (k=0): ni el agente ni el MPC conocen el consumo/generación exactos de la hora en curso.
   Comparación justa por construcción y sin fuga de información. Implementado en `src/core/forecast.py`
   (ruido aplicado desde `k=0` en `aplicar_ruido_ventana`).

4. **Fuente única de pronóstico y observación.**
   Ventana + ruido viven en `forecast.py` (`ventana_observada`, AR(1) solar/consumo + 3 capas de
   precio); la observación (108 dims) en `obs_builder.build_obs`. Entreno == evaluación == producción
   ven exactamente lo mismo. Antes había lógica duplicada que podía divergir.

5. **Física de 4 flujos unificada.**
   `simulador.aplicar_fisica_4flujos` es la única física continua (la usan EnergyEnvContinuo, el
   Residual SAC y el MPC). Verificada byte-idéntica a la implementación previa (`test_fisica_4flujos`).

6. **MPC recalibrado en 80 kWh.**
   Tras pasar a 80 kWh se recalibró el MPC: `k_deg_lin=0.005` y `valor_terminal.lambda=0.0` siguen
   siendo óptimos (insensibles). El MPC realista es el baseline competitivo (G2) sobre el que se mide
   el margen del resto.

## Protocolo de evaluación (definitivo)

- Pool **held-out de 50 semanas** (`sim.semanas_eval`, muestreo aleatorio fijo, `semilla_split=42`),
  cubriendo todo el rango temporal (sin sesgo crisis/post-crisis).
- Métrica: **beneficio marginal vs IDLE** (€/semana), sin pendiente terminal.
- **3 semillas de entreno** (42, 1337, 2024) → se guarda el mejor modelo.
- **10 semillas de ruido AR(1)** en evaluación → media ± std sobre semillas, IC95% (bootstrap) y
  Wilcoxon pareado vs MPC realista. Las semillas de ruido miden robustez frente a la realización del
  pronóstico, no varianza de entrenamiento.

## Limitaciones heredadas (siguen vigentes para la memoria)

- Degradación de batería = coste en la recompensa, no reduce capacidad física (ni en simulador ni en SaaS).
- Perfiles por casa sintéticos (ruido log-normal), no medidos.
- Coeficientes de reparto fijos.
