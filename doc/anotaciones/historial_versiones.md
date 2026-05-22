# Historial de versiones DQN/PPO — SGEC

Referencia rápida de configuración y resultados de cada agente entrenado.
Métrica principal: `eval/mean_reward` = beneficio marginal vs IDLE (€/semana, 50 episodios).
MPC con ruido AR(1) = 48.46 €/sem (cota de referencia justa).

---

## PPO_1
| Parámetro | Valor |
|-----------|-------|
| Algoritmo | PPO (Proximal Policy Optimization) |
| Observación | 101 dims |
| Acciones | Discrete(9) — mismas que DQN |
| N_STEPS (rollout) | 2048 (~12 episodios completos por rollout) |
| BATCH_SIZE | 64 |
| N_EPOCHS | 10 |
| GAE_LAMBDA | 0.95 |
| GAMMA | 0.99 |
| CLIP_RANGE | 0.2 |
| ENT_COEF | 0.01 |
| LR | 3e-4 (Adam, default PPO) |
| Red actor | 101→64→64→9 |
| Red critic | 101→64→64→1 |
| Pasos totales | 3M |
| EVAL_SEED | 42 (semilla fija — mismas 50 semanas) |

**Motivación:** DQN_10–DQN_17 estancados en media ~39 €/sem (max ocasional 50-51).
El cuello de botella identificado es el crédito tardío: DQN con n-step solo propaga
crédito n pasos. PPO+GAE usa el rollout completo (2048 pasos = ~12 episodios) para
entrenar la función de valor, resolviendo el horizonte temporal de forma estructural.

**Resultados:**
- Máximo: +39.91 €/sem (paso 320k)
- Colapso posterior hasta ~36 €/sem
- D_RED: 27-45% (muy superior al 4.3% del MPC)
- Diagnóstico: ENT_COEF=0.01 demasiado alto → destruye la política al explorar

---

## PPO_2
| Parámetro | Valor |
|-----------|-------|
| Algoritmo | PPO |
| Observación | 101 dims |
| Acciones | Discrete(9) |
| N_STEPS (rollout) | 4096 (~24 episodios por rollout) |
| BATCH_SIZE | 64 |
| N_EPOCHS | 10 |
| GAE_LAMBDA | 0.95 |
| GAMMA | 0.99 |
| CLIP_RANGE | 0.2 |
| ENT_COEF | 0.003 (constante) |
| LR | 3e-4 |
| Red actor | 101→64→64→9 |
| Red critic | 101→64→64→1 |
| Pasos totales | 3M |
| EVAL_SEED | 42 |

**Cambios respecto a PPO_1:** N_STEPS 2048→4096, ENT_COEF 0.01→0.003.

**Resultados:**
- Máximo: +41.82 €/sem (paso 960k)
- Colapso posterior hasta ~37 €/sem (1.8M)
- D_RED: 20-40% (mejora respecto a PPO_1 pero sigue alto)
- Diagnóstico: ENT_COEF constante bajo retrasa el colapso pero no lo evita.
  Problema estructural: PPO on-policy olvida la buena política tras actualizaciones sucesivas.

---

## PPO_3
| Parámetro | Valor |
|-----------|-------|
| Algoritmo | PPO |
| Observación | **108 dims** |
| — Desglose | 5 actuales + 96 forecast 24h + 6 temporales (sin/cos hora, dia_sem, mes) + 1 margen_solar |
| — margen_solar | solar excedente que desbordará la batería en 24h / CAP (señal para D_RED) |
| Acciones | Discrete(9) |
| N_STEPS (rollout) | **8192** (~48 episodios por rollout) |
| BATCH_SIZE | **128** |
| N_EPOCHS | **5** |
| GAE_LAMBDA | 0.95 |
| GAMMA | 0.99 |
| CLIP_RANGE | 0.2 |
| ENT_COEF | **decay lineal 0.005 → 0.0005** (EntCoefScheduler) |
| LR | 3e-4 |
| Red actor | 108→64→64→9 |
| Red critic | 108→64→64→1 |
| Pasos totales | 3M |
| EVAL_SEED | 42 |
| Reward D_RED | **Coste oportunidad dinámico**: 0 si SoC≥0.70, escala hasta 4× spread si SoC→SOC_MIN |

**Cambios respecto a PPO_2:**
- Features temporales + margen_solar: 101→108 dims
- ENT_COEF decay: exploración alta al inicio, explotación estable al final → sin colapso
- Rollouts más largos (8192) con menos épocas (5): mejor estimación de V(s), menos sobrefit
- Reward shaping D_RED SoC-dependiente: sin corte duro, pero muy caro con batería baja

**Resultados:** *(en curso — 1.04M pasos a 19/05/2026)*
- Máximo hasta ahora: +44.72 €/sem (paso 980k) — mejor PPO hasta la fecha
- Sin colapso visible a 1M (PPO_1 colapsó en 320k, PPO_2 en ~1.1M)
- D_RED: 31% (inicio) → **3-8%** (desde 460k) — prácticamente al nivel del MPC (4.3%)
- D_CASA: 10% (inicio) → **55-65%** (desde 480k)
- Tendencia: plateau leve entre 980k-1040k (+/-0.5€), faltan 2M pasos

---

## DQN_10
| Parámetro | Valor |
|-----------|-------|
| Observación | 101 dims (sin features temporales) |
| Acciones | 13 |
| N-step | 1 (estándar) |
| Buffer | 100k |
| Red | 64×64 |
| LR | 1e-4 |
| Pasos totales | ~1M |
| Exploración | — |

**Resultados**: baseline inicial, rendimiento bajo. Sin n-step, sin ruido AR(1).

---

## DQN_11
| Parámetro | Valor |
|-----------|-------|
| Observación | 101 dims |
| Acciones | 13 |
| N-step | 1 |
| Buffer | 100k |
| Red | 64×64 |
| LR | 1e-4 |
| Pasos totales | ~2M |

**Resultados**: ~39.38 €/sem. Ruido AR(1) añadido al forecast. Sin n-step aún.

---

## DQN_12
| Parámetro | Valor |
|-----------|-------|
| Observación | 101 dims |
| Acciones | 13 → 9 |
| N-step | 8 |
| Buffer | 200k |
| Red | 64×64 |
| LR | 1e-4 |
| Pasos totales | ~2M (parado antes) |
| Exploración | 50% |

**Resultados**: entrenamiento interrumpido. Primera versión con n-step=8 y 9 acciones.
Eliminados niveles redundantes en CARGAR_SOLAR y DESCARGAR_CASA.

---

## DQN_13
| Parámetro | Valor |
|-----------|-------|
| Observación | 101 dims |
| Acciones | 9 |
| N-step | 8 |
| γ modelo | 0.99^8 ≈ 0.923 |
| Buffer | 200k |
| Red | 64×64 |
| LR | 1e-4 |
| Batch size | 64 |
| Pasos totales | ~1.88M (parado antes de 3M) |
| Exploración | 50% → 1.5M pasos |
| EVAL_FREQ | 20k pasos |
| EVAL_EPISODES | 50 |

**Resultados**:
- Máximo: +50.97 €/sem
- Media últimos 10 evals: +39.44 €/sem
- Tendencia: plateau y ligera bajada en explotación
- SoC medio explotación: ~0.39
- DESCARGAR_CASA: 33.3%, DESCARGAR_RED: 27.4%
- Problema: vaciaba batería sin respetar picos de precio (sin contexto temporal)

---

## DQN_14
| Parámetro | Valor |
|-----------|-------|
| Observación | 101 dims |
| Acciones | 9 |
| N-step | 24 |
| γ modelo | 0.99^24 ≈ 0.787 |
| Buffer | **400k** *(corregido — el código tenía 400k, no 200k como se anotó inicialmente)* |
| Red | 64×64 |
| LR | 1e-4 |
| Batch size | 64 |
| Pasos totales | 3M |
| Exploración | 50% → 1.5M pasos |

**Resultados**:
- Máximo: +46.77 €/sem (en el último paso — aún subiendo)
- Media últimos 10 evals: +39.02 €/sem
- Tendencia: **alcista hasta el final** — no convergió en 3M pasos
- Conclusión: n=24 aprende más despacio pero no convergió, necesitaría 5-6M pasos
- Mismo problema de gestión de batería que DQN_13

---

## DQN_15
| Parámetro | Valor |
|-----------|-------|
| Observación | **107 dims** (+6 features temporales) |
| Features temporales | sin/cos(hora/24), sin/cos(dia_semana/7), sin/cos(mes/12) |
| Acciones | 9 |
| N-step | 8 |
| γ modelo | 0.99^8 ≈ 0.923 |
| Buffer | 400k *(debería ser 200k — error al subir a Kaggle)* |
| Red | 107→64→64→9 |
| LR | 1e-4 |
| Batch size | 64 |
| Pasos totales | 3M |
| Exploración | 50% → 1.5M pasos |

**Resultados** *(en curso — 1.63M pasos)*:
- Máximo hasta ahora: +51.83 €/sem (paso 1.34M)
- Media últimos 10 evals: ~37 €/sem (inicio de explotación)
- DESCARGAR_CASA: 39.5% ↑ vs DQN_13, CARGAR_MIXTA: 19.4% ↑
- SoC medio: 0.385 ≈ DQN_13
- Tendencia: por determinar (explotación recién iniciada)
- Nota: buffer=400k en local y en Kaggle (no es comparación limpia con DQN_13)

---

## DQN_16
| Parámetro | Valor |
|-----------|-------|
| Observación | 107 dims (features temporales) |
| Acciones | 9 |
| N-step | 24 |
| γ modelo | 0.99^24 ≈ 0.787 |
| Buffer | 200k |
| Red | 107→64→64→9 |
| LR | 1e-4 |
| Batch size | 64 |
| Pasos totales | 5M |
| Exploración | 40% → 2M pasos, explotación 3M |
| EVAL_SEED | 42 — semilla fija, siempre las mismas 50 semanas |

**Cambios respecto a DQN_15:**
- n=8 → n=24: cubre un ciclo completo carga/descarga (24h)
- 3M → 5M pasos: n=24 converge más despacio (DQN_14 seguía subiendo al final)
- Eval semilla fija: elimina varianza inter-evaluación (~±5 €/sem en versiones anteriores)
- Valor terminal corregido: precio según balance solar/déficit en última hora
- EXPLORATION_FRAC 0.5 → 0.4: exploración hasta 2M, explotación 3M

**Resultados**: *(crash a ~4M pasos — run interrumpido, datos perdidos)*
- best_model.zip sobreescrito por el run de resume (VecNormalize incompatible)

---

## DQN_17
| Parámetro | Valor |
|-----------|-------|
| Observación | 101 dims (features temporales eliminadas) |
| N-step | 24 |
| γ modelo | 0.995^24 ≈ 0.887 |
| Buffer | 200k |
| Red | 101→64→64→9 |
| Pasos totales | 5M |
| Exploración | 40% → 2M pasos |
| EVAL_SEED | 42 (SeededEvalCallback — fix del bug de eval) |

**Resultados**: *(run corto ~24 min, abortado)*

---

## DQN_18
| Parámetro | Valor |
|-----------|-------|
| Observación | 101 dims (sin features temporales) |
| N-step | 24 |
| γ modelo | 0.995^24 ≈ 0.887 |
| Buffer | 200k |
| Red | 101→64→64→9 |
| Pasos totales | 5M |
| Exploración | 40% → 2M pasos |
| EVAL_SEED | 42 (SeededEvalCallback) |

**Resultados**: *(en curso — ~600-700k pasos a 18/05/2026 21:20)*

---

## DQN_19
| Parámetro | Valor |
|-----------|-------|
| Observación | 107 dims (features temporales restauradas) |
| N-step | 8 |
| γ modelo | 0.99^8 ≈ 0.923 |
| Buffer | 200k |
| Red | 107→64→64→9 |
| Pasos totales | 3M |
| Exploración | 50% → 1.5M pasos |
| EVAL_SEED | 42 (SeededEvalCallback) |

**Cambios respecto a DQN_18:**
- n=24 → n=8: mejor convergencia histórica (DQN_13/15 con n=8 superaron a DQN_14 con n=24)
- GAMMA_BASE: 0.995 → 0.99 (estándar para n=8)
- Features temporales restauradas: 101 → 107 dims
- 5M → 3M pasos (n=8 converge más rápido)
- EXPLORATION_FRAC: 0.4 → 0.5

**Resultados**: igual de malo que DQN_18 (~30-35 €/sem). Confirmado que gamma_base=0.995 era
el problema (no el buffer ni las features temporales). Con gamma_modelo=0.887 el agente aprende
el ciclo CARGAR_MIXTA→DESCARGAR_CASA que es break-even menos degradación.

---

## DQN_20
| Parámetro | Valor |
|-----------|-------|
| Algoritmo | DQN |
| Observación | 101 dims (sin features temporales) |
| Acciones | Discrete(9) |
| N-step | 8 |
| γ base | 0.99 |
| γ modelo | 0.99^8 ≈ 0.923 |
| Buffer | 200k |
| Red | 101→64→64→9 |
| LR | 1e-4 |
| Batch size | 64 |
| Pasos totales | 3M completos |
| Exploración | 50% → 1.5M pasos, explotación 1.5M |
| EVAL_SEED | 42 (SeededEvalCallback — mismas 50 semanas) |
| Reward shaping | Coste oportunidad dinámico D_RED: 0 si SoC≥0.70, escala hasta 4× spread si SoC→SOC_MIN |

**Motivación:** DQN_13 (misma config base) alcanzó 51 €/sem pero fue parado a 1.88M.
DQN_18/19 (gamma=0.887) se estancaron en 30-35 €/sem — causa raíz confirmada: gamma alto
hace rentable el ciclo grid→batería→casa que es break-even. n=8 + gamma=0.923 es la config
validada. Se añade reward shaping D_RED (copiado de rama PPO) para frenar el exceso de D_RED.

**Resultados**: +45.07 EUR/sem (eval_unificada, 50 semanas).

---

## PPO_4
| Parámetro | Valor |
|-----------|-------|
| Algoritmo | PPO |
| Observación | 108 dims |
| Acciones | Discrete(9) |
| N_STEPS (rollout) | 8192 |
| BATCH_SIZE | 128 |
| N_EPOCHS | 5 |
| GAMMA | 0.99 |
| ENT_COEF | decay lineal 0.005 → 0.0005 |
| LR | decay lineal 3e-4 → 3e-5 |
| Red actor | 108→64→64→9 |
| Red critic | 108→64→64→1 |
| Pasos totales | 4M |
| EVAL_SEED | 42 |

**Cambios respecto a PPO_3:** LR decay (3e-4→3e-5), 3M→4M pasos.

**Resultados**: pico +45.69 EUR/sem (paso 1.48M), estabilización ~44 EUR/sem.
Sin colapso. Plateau a partir de 1.5M — la política convergió antes de los 4M.

---

## Residual SAC v2 (aditivo, delta_max=0.10)
| Parámetro | Valor |
|-----------|-------|
| Algoritmo | SAC (Soft Actor-Critic) |
| Arquitectura | Residual **aditivo** sobre MPC: flow + delta × 0.10 × P_MAX |
| Observación | 112 dims (108 base + 4 flujos MPC normalizados) |
| Acción | Box(4) in [-1,1]: delta sobre (cs, cm, dc, dr) |
| delta_max | 0.10 → ±5.0 kW por flujo |
| DAWN warmup | 50,000 pasos (buffer pre-llenado con delta=0) |
| LR | 1e-4 |
| Buffer | 100k |
| Batch size | 256 |
| GAMMA | 0.99 |
| TAU | 0.005 |
| ent_coef | 0.01 |
| Red actor/critic | 112→256→256→4 |
| Pasos totales | 500k (OOM a 280k) |
| Entorno | EnergyEnvContinuo **sin neteo** (bug: carga+descarga simultánea posible) |

**Resultados**:
- OOM (RuntimeError alloc_cpu) a paso ~280k
- Mejor eval reward (env): +45.41 a 280k
- eval_unificada (best_model): **+48.80 EUR/sem**
- Delta L1 medio: 2.75 kW (55% del rango de 5.0 kW) — deltas agresivos
- Tendencia: subiendo lento pero estable hasta OOM

**Diagnóstico**: deltas demasiado grandes destruían la señal MPC. El env sin neteo
permitía carga+descarga simultánea (físicamente imposible). Resultado no comparable
limpiamente con MPC.

---

## Residual SAC v3 (aditivo, delta_max=0.05)
| Parámetro | Valor |
|-----------|-------|
| Algoritmo | SAC |
| Arquitectura | Residual **aditivo** sobre MPC: flow + delta × 0.05 × P_MAX |
| Observación | 112 dims |
| Acción | Box(4) in [-1,1] |
| delta_max | 0.05 → ±2.5 kW por flujo |
| DAWN warmup | 50,000 |
| ent_coef | 0.005 |
| Red actor/critic | 112→256→256→4 |
| Pasos totales | 500k (matado a ~100k, estancado) |
| Entorno | EnergyEnvContinuo **con neteo** (corrección carga/descarga simultánea) |

**Cambios respecto a v2**: delta_max 0.10→0.05, ent_coef 0.01→0.005, neteo añadido.

**Resultados**:
- Mejor eval reward (env): +41.27 a 100k
- Delta L1 medio: 1.4 kW (56% del rango)
- Estancamiento claro desde 60k (+40.5 a +41.3)
- Matado a ~100k por falta de progreso

**Diagnóstico**: el rango reducido (2.5 kW) + entropía baja hacía correcciones
insignificantes. El neteo eliminó el "truco" de carga+descarga simultánea que
inflaba v2. Resultado más honesto pero peor.

---

## Residual SAC v4 (multiplicativo, delta_max=0.30)
| Parámetro | Valor |
|-----------|-------|
| Algoritmo | SAC |
| Arquitectura | Residual **multiplicativo**: flow × (1 + delta × 0.30) |
| Observación | 112 dims |
| Acción | Box(4) in [-1,1] |
| delta_max | 0.30 → ±30% de cada flujo MPC |
| DAWN warmup | 100,000 |
| LR | 1e-4 |
| Buffer | 100k |
| Batch size | 256 |
| GAMMA | 0.99 |
| TAU | 0.005 |
| ent_coef | 0.002 |
| Red actor/critic | 112→256→256→4 |
| Pasos totales | 1M |
| Entorno | EnergyEnvContinuo con neteo |
| Fixes | Memory leak LP corregido (buffers pre-asignados + gc.collect) |

**Cambios respecto a v3**: residual aditivo→multiplicativo, delta_max 0.05→0.30,
ent_coef 0.005→0.002, warmup 50k→100k, pasos 500k→1M.

**Motivación**: el aditivo aplica la misma corrección absoluta independientemente
del flujo MPC. El multiplicativo escala proporcionalmente: si MPC dice 0, la
corrección es 0 (no se activa lo que MPC descartó). Si MPC dice 20 kW, ±30%
da ±6 kW de corrección.

**Resultados** *(en curso — 240k/1M pasos a 21/05/2026)*:
- Mejor eval reward (env): +44.17 a 60k
- Delta L1 medio: 0.4 kW — correcciones mucho más pequeñas que v2/v3
- Estancamiento en ~+43.3 desde 80k
- Arranque mejor que v2/v3 (+41.84 a 20k vs +34.83/+38.88)

**Comparativa eval reward (env) por pasos**:

| Step | v2 (adit 0.10) | v3 (adit 0.05) | v4 (mult 0.30) |
|-----:|:-:|:-:|:-:|
| 20k | +34.83 | +38.88 | +41.84 |
| 40k | +39.42 | +40.09 | +43.77 |
| 60k | +42.91 | +40.77 | +44.17 |
| 80k | +43.57 | +40.54 | +43.07 |
| 100k | +43.72 | +41.27 | +43.66 |
| 140k | +45.00 | — | +42.40 |
| 200k | +45.63 | — | +43.16 |
| 280k | +45.41 | — | — |

---

## Residual SAC v5 (multiplicativo, delta_max=0.15)
| Parámetro | Valor |
|-----------|-------|
| Algoritmo | SAC |
| Arquitectura | Residual **multiplicativo**: flow × (1 + delta × 0.15) |
| Observación | 112 dims (108 base + 4 flujos MPC normalizados) |
| Acción | Box(4) in [-1,1] |
| delta_max | 0.15 → ±15% de cada flujo MPC |
| DAWN warmup | 100,000 |
| LR | 1e-4 |
| Buffer | 100k |
| Batch size | 256 |
| GAMMA | 0.99 |
| TAU | 0.005 |
| ent_coef | 0.01 |
| Red actor/critic | 112→256→256→4 |
| Pasos totales | 1M (en curso) |
| Entorno | EnergyEnvContinuo con neteo + ruido precios 3 capas |
| Seed | 42 |

**Cambios respecto a v4**: delta_max 0.30→0.15 (más conservador, correcciones menores),
ent_coef 0.002→0.01 (más exploración, MPC ahora tiene ruido en precios).

**Motivación**: v4 con delta_max=0.30 hacía correcciones demasiado grandes y se estancaba.
Reducir a ±15% obliga a correcciones finas sobre un MPC ya optimizado. La entropía más alta
compensa el rango reducido permitiendo explorar mejor dentro del ±15%.

**Resultados** *(en curso — 240k/1M pasos a 21/05/2026)*:
- Mejor eval reward (env): +45.71 a 240k (New best, tendencia alcista)
- Delta L1 medio: 0.14-0.17 kW — correcciones muy finas
- SoC medio: 0.43-0.49 [0.10-0.90]
- Sin estancamiento visible (a diferencia de v4 que se estancó en 80k)

**Comparativa eval reward (env) por pasos**:

| Step | v4 (mult 0.30) | v5 (mult 0.15) |
|-----:|:-:|:-:|
| 20k | +41.84 | +44.39 |
| 40k | +43.77 | +44.78 |
| 60k | +44.17 | +44.92 |
| 80k | +43.07 | +45.01 |
| 100k | +43.66 | +45.44 |
| 120k | — | +45.21 |
| 140k | +42.40 | +45.48 |
| 200k | +43.16 | +45.03 |
| 240k | — | +45.71 |

**eval_unificada (3 seeds × 50 semanas = 150 semanas, best_model a 240k)**:

| Seed | MPC realista | ResidualSAC v5 | Δ vs MPC |
|-----:|:-:|:-:|:-:|
| 42 | +48.71 | +49.33 | **+0.62** |
| 1337 | +48.70 | +48.90 | **+0.20** |
| 2024 | +48.78 | +49.45 | **+0.68** |
| **Agregado** | **+48.73** | **+49.23** | **+0.50** |

Backup modelo: `models/best_model_v5_240k.zip`

---

## Residual SAC v5b (multiplicativo, delta_max=0.20)
| Parámetro | Valor |
|-----------|-------|
| Algoritmo | SAC |
| Arquitectura | Residual **multiplicativo**: flow × (1 + delta × 0.20) |
| Observación | 112 dims (108 base + 4 flujos MPC normalizados) |
| Acción | Box(4) in [-1,1] |
| delta_max | 0.20 → ±20% de cada flujo MPC |
| DAWN warmup | 100,000 |
| LR | 1e-4 |
| Buffer | 100k |
| Batch size | 256 |
| GAMMA | 0.99 |
| TAU | 0.005 |
| ent_coef | 0.01 |
| Red actor/critic | 112→256→256→4 |
| Pasos totales | 1M |
| Entorno | EnergyEnvContinuo con neteo + ruido precios 3 capas |
| Seed | 42 |
| Tag | v5b (modelos en `models/v5b/`, logs en `logs/v5b/`) |

**Cambios respecto a v5**: delta_max 0.15→0.20 (más rango de corrección).
Resto de hiperparámetros idénticos.

**Motivación**: v5 con delta_max=0.15 supera al MPC (+0.50 EUR/sem a 240k),
pero el rango ±15% puede ser demasiado conservador. v5b prueba ±20% para ver
si un rango intermedio entre v4 (30%) y v5 (15%) mejora la convergencia.

**Resultados**: *(en curso — lanzado 22/05/2026)*

---

## Referencia eval_unificada (50 semanas eval, protocolo idéntico)
| Controlador | EUR/semana | Notas |
|-------------|-----------|-------|
| MPC oráculo | **+57.71** | Techo teórico (forecast perfecto) |
| **Residual SAC v5** | **+49.23** | **Agregado 3 seeds (150 sem), best a 240k, en curso** |
| MPC realista (AR(1)) | **+48.73** | Cota práctica (agregado 3 seeds) |
| Residual SAC v2 | +48.80 | Env sin neteo, OOM a 280k (no comparable) |
| DQN_13 | +45.23 | 101 dims, [256,128], gamma=0.995 |
| DQN_20 | +45.07 | 101 dims, [64,64], gamma=0.99 |
| DQN_14 | +41.71 | n-step=24, no convergido |
| IDLE | 0.00 | Baseline |
