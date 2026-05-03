# CHANGELOG.md — Registro de Progreso del TFG

> Al iniciar sesión: "Lee el CHANGELOG.md para ver dónde nos quedamos."
> Al terminar una tarea: añadir entrada con fecha, descripción técnica y siguiente paso lógico.

---

## [2026-05-03] — v8: reward shaping (baseline IDLE) para eliminar ruido meteorológico

### Problema detectado en DQN_7

El DQN_7 (1.5M pasos, config validada) NO convergió:
- Media final: +10 EUR/semana (inestable, oscila entre -200 y +190)
- Ratio señal/ruido: **0.15** — la varianza del clima (±68 EUR/sem) ahogaba
  la contribución marginal del agente (~10 EUR/sem)

**Causa raíz:** La recompensa `R_t = vendido*precio_venta - comprado*precio_compra - degradación`
incluye el coste/ingreso de fondo de la comunidad (que depende del clima, NO del agente).
El DQN no puede separar su contribución del ruido incontrolable con una red 64x64.

### Solución: Reward shaping con baseline IDLE

Nueva recompensa: `R_t = beneficio_accion - beneficio_IDLE`
- `beneficio_IDLE = excedente * precio_venta - deficit * precio_compra`
- Es stateless (no tiene batería → no tiene SoC → cálculo exacto)
- NO cambia la política óptima (restar constante independiente de acción)
- Elimina 100% de la varianza meteorológica

### Verificación (200 episodios)

| Métrica | Antes (reward absoluta) | Después (reward marginal) |
|---|---|---|
| Std episodio IDLE | 68 EUR/sem | **0.00 EUR/sem** |
| Std episodio random | 74 EUR/sem | **27 EUR/sem** |
| Ratio señal/ruido estimado | 0.15 | **3–5** |

### Cambios en archivos

- **`src/core/simulador.py`**: `ejecutar_accion_fisica()` ahora calcula y devuelve
  `beneficio_marginal = beneficio - beneficio_idle` además de `beneficio` (absoluto).
- **`src/envs/energy_env.py`**: `reward = resultado["beneficio_marginal"]` (antes: `resultado["beneficio"]`).

### Modelos invalidados

DQN_7 es INVÁLIDO (entrenado con reward ruidosa). Todos los modelos DQN_1 a DQN_7 descartados.

### Siguiente paso lógico
1. Ejecutar entrenamiento v8 definitivo (1.5M pasos): `python src/main.py`
2. HU-28: notebook análisis de resultados
3. HU-14: exportar modelo a ONNX

---

## [2026-05-03] — v7: fisica realista de bateria (eficiencia, limites SoC, autodescarga)

### Mejoras en el motor fisico (`simulador.py`)

**1. Eficiencia de carga/descarga**
- Carga: 95% (de 10 kWh de solar/red, la bateria almacena 9.5 kWh)
- Descarga: 95% (la bateria pierde 10 kWh, entrega 9.5 kWh utiles)
- Round-trip: 0.95 x 0.95 = 90.25% (estandar Li-ion)
- Antes: 100% (irreal, sin perdidas)
- Impacto: ciclar la bateria innecesariamente tiene un coste real adicional a la degradacion

**2. Limites operativos de SoC**
- SOC_MIN = 10%, SOC_MAX = 90%
- La bateria solo opera entre 10 y 90 kWh de los 100 kWh fisicos (80 kWh utiles)
- Antes: 0-100% (irreal, acorta vida util de la bateria)
- Impacto: el agente no puede vaciar ni llenar completamente la bateria

**3. Autodescarga**
- 0.004% por hora (~3% mensual, tipico Li-ion)
- Se aplica al inicio de cada paso, antes de la accion
- Antes: 0% (irreal pero efecto pequeno: ~0.67% por semana)

**4. Adaptacion de la logica de acciones**
- `espacio_libre` ahora respeta SOC_MAX (antes usaba capacidad total)
- `bateria_disponible` ahora respeta SOC_MIN (antes podia llegar a 0)
- Carga limitada por `espacio_libre / eficiencia` (necesitas mas energia de entrada para llenar el mismo espacio)
- Descarga entrega `descargado * eficiencia` kWh utiles (se pierde un 5%)

### Verificacion (20 semanas aleatorias)

| Estrategia | Antes (sin perdidas) | Ahora (con perdidas) |
|---|---|---|
| IDLE | -5.84 EUR/sem | -54.73 EUR/sem |
| Autoconsumo simple | +40.12 EUR/sem | -0.84 EUR/sem |
| Mejora autoconsumo vs IDLE | +45.96 EUR/sem | **+53.89 EUR/sem** |

La bateria sigue aportando valor (+54 EUR/sem vs IDLE). Los valores absolutos bajan
porque la eficiencia < 100% encarece los ciclos.

### Siguiente paso logico
1. Ejecutar entrenamiento v7 definitivo (1.5M pasos): `python src/main.py`
2. HU-28: notebook analisis de resultados

---

## [2026-05-03] — v6: modelo de precios asimetrico REAL (regulacion espanola)

### Bug critico corregido — modelo simetrico era INCORRECTO

**Problema:** La v5 usaba `ingresos = vendido * precio_kwh` (modelo simetrico: compra = venta).
Esto era incorrecto porque `precio_kwh` es el PVPC completo (ind. ESIOS 1001), que incluye
energia + peajes + cargos. Un vendedor de excedentes NO recibe todo eso, solo el componente
mayorista. El modelo simetrico permitia al agente explotar la bateria (dump DESCARGAR_RED 100%
desde SoC 0.5 y ganar dinero gratis).

**Solucion:** Modelo asimetrico con datos REALES de ESIOS:
- **Compra de red**: `precio_kwh` — PVPC completo (ind. 1001). Media: 0.217 EUR/kWh
- **Venta excedentes**: `precio_excedente` — compensacion simplificada (ind. 1739,
  RD 244/2019 Art.14: Pmh - CDSVh). Media: 0.132 EUR/kWh (~60% del PVPC)
- Spread medio compra-venta: 0.085 EUR/kWh (peajes+cargos que paga el consumidor)
- En NINGUNA hora del dataset el precio de venta supera al de compra

### Cambios en archivos

**1. Nuevo dato raw: `data/raw/compensacion_autoconsumo_esios_2021_2023.csv`**
- Indicador ESIOS 1739 (precio energia excedentaria autoconsumo, compensacion simplificada)
- 22.657 filas, mismo periodo jun-2021 a dic-2023, Espana

**2. ETL (`generar_dataset_final.py`)**
- Añadido procesamiento de `compensacion_autoconsumo_esios_2021_2023.csv`
- Dataset ahora tiene 4 columnas: `[consumo_total, generacion_total, precio_kwh, precio_excedente]`
- Shape: 22.646 x 4

**3. Simulador (`simulador.py`)**
- `ejecutar_accion_fisica()`: lee `precio_compra` (PVPC) y `precio_venta` (excedentaria)
- `ingresos = vendido * precio_venta` (antes: `vendido * precio`)
- `gastos = comprado * precio_compra` (antes: `comprado * precio`)
- `get_data_window()`: devuelve 4 columnas (antes 3)

**4. Entorno (`energy_env.py`)**
- Observacion ampliada: 76 -> **101 dimensiones**
  - 5 actuales: SoC, precio_compra, precio_venta, excedente, deficit
  - 96 pronostico: 24h x 4 vars (consumo, generacion, precio_compra, precio_venta)

**5. Entrenamiento (`main.py`) — v6**
- Docstring actualizado
- Red: 101 -> 64 -> 64 -> 13

### Verificacion de cordura economica (20 semanas aleatorias)

| Estrategia | Recompensa media/semana |
|---|---|
| IDLE (sin bateria) | -5.84 EUR |
| DESCARGAR_RED 100% (dump) | +0.23 EUR (ya NO es exploit) |
| Autoconsumo simple (solar->casa) | **+40.12 EUR** |

- Autoconsumo supera IDLE en +46 EUR/semana: correcto, la bateria aporta valor real
- Dump apenas supera IDLE (+6 EUR): solo por el SoC inicial de 0.5, se agota en 2 horas
- El DQN debe superar el autoconsumo simple aprendiendo arbitraje temporal

### Todos los modelos anteriores (DQN_1 a DQN_5) son INVALIDOS
Entrenados con precios incorrectos. Necesario reentrenar desde cero.

### Siguiente paso logico
1. Ejecutar entrenamiento v6 definitivo (1.5M pasos): `python src/main.py`
2. HU-28: notebook analisis de resultados
3. HU-14: exportar modelo a ONNX

---

## [2025-12-08] — FASE 2 COMPLETADA: Motor de Simulación y Entorno RL

### Implementado
- **`src/utils/generar_dataset_final.py`**: Pipeline ETL completo.
  - Procesa consumo ESIOS (coeficientes 2.0TD) → 15 vecinos con perturbación estocástica (α~U(0.8,1.2), β~N(1,0.1)).
  - Escala producción PVGIS a 50 kWp.
  - Convierte precios ESIOS de €/MWh a €/kWh.
  - Genera `data/processed/dataset_final.csv` (8760 filas × 3 columnas).

- **`src/core/simulador.py`**: Motor físico `ComunidadSimulador`.
  - Batería: 100 kWh, inversor 50 kW, SoC continuo [0,1].
  - 13 acciones: IDLE + 4 estrategias × 3 niveles de potencia (33%/66%/100%).
  - Función de degradación no lineal (factores I² y SoC⁴).
  - Método `ejecutar_accion_fisica(action_idx, step)` → `{beneficio, soc, comprado}`.

- **`src/envs/energy_env.py`**: Entorno `EnergyEnv(gym.Env)`.
  - Observation space: Box(76,) — 4 vars actuales + 72 pronóstico (24h × 3 vars).
  - Action space: Discrete(13).
  - Episodios de 168 pasos (1 semana), inicio aleatorio.
  - Validado con `stable_baselines3.common.env_checker` ✅.

### Siguiente paso lógico
Implementar `src/main.py`: instanciar `EnergyEnv`, configurar agente `DQN` de Stable Baselines3 y lanzar entrenamiento con callbacks de logging.

---

## [2026-04-30] — Revisión y corrección de bugs en simulador.py

### Bugs corregidos
- **Bug crítico — doble descuento en recompensa** (`simulador.py:157`):
  La fórmula `ahorro + ingresos - gastos` restaba `comprado*precio` dos veces.
  Corregido a cash flow puro: `beneficio = ingresos - gastos - coste_deg`.
  
- **Bug menor — SoC incorrecto en cálculo de degradación** (`simulador.py:155`):
  Se usaba el SoC post-operación para evaluar el stress. Corregido a SoC medio
  `(soc_antes + soc_después) / 2` para reflejar el stress real durante el ciclo.

- **Comentario incorrecto en `test_rapido`**:
  Acción 2 estaba etiquetada como "Vender Excedente" (lógica antigua).
  Corregido a "CARGAR_SOLAR 66%".

### Nota
Los PDFs de `doc/foundation/` reflejan la fórmula antigua con `ahorro`. Actualizar en la redacción de la memoria.

### Siguiente paso lógico
Implementar `src/main.py` con entrenamiento DQN (Stable Baselines3).

---

## [2026-05-01] — Diagrama Arquitectura por Capas: añadida Capa 5 SaaS

### Implementado
- **`doc/diagramas/arquitectura_capas.drawio`**: añadida **Capa 5 · SaaS** sin tocar capas existentes.
  - Bloque `Base de Datos PostgreSQL` (cilindro): almacena perfiles de vecinos e historial de resultados.
  - Bloque `Motor Post-Procesado (Determinista)`: cálculo `ahorro_total × cuota_vecino`.
  - Sub-contenedor `Frontend Multi-Rol (Streamlit)` con dos salidas: Dashboard Admin y Dashboard Vecino.
  - Flecha **Capa 4 → PostgreSQL**: "Envío de métricas (Ahorro global, SoC)".
  - Flecha **dataset_final.csv → Streamlit**: "Lectura de perfiles de consumo".
  - Flecha interna: PostgreSQL → Motor → Streamlit.

### Nota
El SaaS (Streamlit + PostgreSQL) es un componente **secundario** del TFG añadido para cerrar el ciclo de vida del dato ante el tribunal. El núcleo del proyecto sigue siendo el agente DQN y el Gemelo Digital.

### Siguiente paso lógico
Continuar con el resto de diagramas pendientes (secuencia, casos de uso, MER, etc.) o implementar `src/main.py`.

---

## [2026-05-02] — v5: datos multi-anio, fix precio venta a mercado, hiperparametros ajustados

### Cambios criticos

**1. Fix precio de venta en `simulador.py` (bug conceptual)**
- ANTES: `ingresos = vendido * 0.05` (precio fijo, incorrecto)
- DESPUES: `ingresos = vendido * precio` (precio horario de mercado)
- Motivo: segun RD 244/2019, los excedentes se compensan al precio horario OMIE.
  Para una VPP con bateria, el modelo simetrico (compra = venta = precio mercado)
  es la simplificacion estandar en la literatura academica de DQN + bateria.
- Eliminada constante `PRECIO_VENTA_EXCEDENTE`.
- Consecuencia: todos los modelos anteriores (DQN_1 a DQN_4) son INVALIDOS.
  DESCARGAR_RED ahora es rentable (vende a ~0.15 en vez de 0.05) y el agente
  debe aprender arbitraje temporal (cargar barato, vender/descargar caro).

**2. Dataset multi-anio en `generar_dataset_final.py`**
- Nuevos datos raw: jun-2021 a dic-2023 (consumo, precios y solar de ESIOS/PVGIS)
- Dataset: 22.646 horas x 3 columnas (era 8.760 horas, 2.6x mas datos)
- Semilla fija (seed=42) para reproducibilidad de perfiles de vecinos
- Eliminado hardcode de 8760 horas

**3. Hiperparametros ajustados en `main.py` (v5)**
- `TOTAL_TIMESTEPS = 1_500_000` (era 1M — mas datos + problema mas complejo)
- `BUFFER_SIZE = 200_000` (era 100k — mas diversidad con 2.6x mas datos)
- `EVAL_FREQ = 15_000` (era 10k — 100 evals en 1.5M pasos)
- Resto mantenido: lr=1e-4, 64x64, gamma=0.99, exploration_frac=0.4, norm_obs=True

### Modelo economico del simulador (documentacion)

La comunidad energetica opera con estos precios:
- **Compra de red**: precio_kwh horario del dataset (PVPC/OMIE)
- **Venta a red**: mismo precio_kwh horario (modelo simetrico)
- **P2P entre vecinos**: implicito en el modelo agregado (el DQN gestiona el balance
  neto de la comunidad). El reparto individual se calcula en post-procesado (Capa 5 SaaS)
  usando Mid-Market Rate.

Formula de recompensa: `R_t = vendido * precio - comprado * precio - coste_degradacion`

### Verificacion
- `env_checker`: OK con nuevo dataset
- Precio de venta verificado: excedente de 2.01 kWh a 0.1157 EUR/kWh = 0.2324 EUR
  (antes con 0.05 fijo habria sido 0.1004 EUR)

### Siguiente paso logico
1. Ejecutar entrenamiento v5 definitivo (1.5M pasos, ~2-2.5 horas)
2. HU-28: notebook analisis de resultados
3. HU-14: exportar modelo a ONNX

---

## [2026-05-01] — main.py v4: subir a 1M pasos tras analisis cientifico (HU-12)

### Contexto — Analisis cientifico de 4 entrenamientos
Se analizaron los 4 runs (DQN_1 test, DQN_2 v1 300k, DQN_3 v2 500k, DQN_4 v3 500k):

| Run | Config | Mejor Eval | Tendencia 2a mitad |
|-----|--------|------------|---------------------|
| DQN_2 (v1) | 64x64, sin norm | +27.1 (200k) | Inestable |
| DQN_3 (v2) | 256x256, norm_obs+reward | +10.3 | Plana (overfitting) |
| DQN_4 (v3) | 64x64, norm_obs only | +21.6 (240k) | **+3.8/100k, mejorando** |

### Hallazgos clave
- DQN_4 sigue mejorando a 500k pasos (pendiente positiva +3.8 reward/100k)
- Proyeccion lineal: ~+35 de recompensa a 1M pasos
- La config v3 (64x64, norm_obs, no norm_reward) es la mejor encontrada
- Preocupacion: accion dominante DESCARGAR_CASA 100% — posible estrategia suboptima

### Cambio
- `TOTAL_TIMESTEPS` subido de 500k a **1M** en `src/main.py`

### Leccion aprendida
No asumir meseta sin evidencia cuantitativa. El analisis de pendiente en la 2a mitad
del entrenamiento es critico para decidir si añadir mas pasos.

### Siguiente paso logico
1. Ejecutar entrenamiento v4 (1M pasos, ~90 min)
2. Mañana: datos multi-año 2020-2023 (4x datos) → adaptar ETL y subir a 1.5M-2M pasos
3. HU-28: notebook analisis de resultados

---

## [2026-05-01] — main.py v3: revertir a 64x64, quitar norm_reward (HU-12, HU-13)

### Contexto
La v2 (256x256, VecNormalize con norm_reward, 500k) fue PEOR que la v1:
- v2 mejor eval: +10.3 vs v1 mejor eval: +27.1
- Red grande overfitteaba con 8760 horas de datos
- norm_reward distorsionaba la señal de aprendizaje

### Cambios v3
- **Revertir red a 64x64**: 3.8k params vs 88k — generaliza mejor con dataset pequeño
- **Quitar norm_reward**: mantener solo VecNormalize para observaciones (norm_obs=True)
- **eval_episodes subido a 20**: medias de evaluación más fiables (reduce ruido de semanas buenas/malas)
- **Hiperparámetros revertidos a v1**: lr=1e-4, buffer=100k, batch=64, exploration=0.4
- **500k pasos**: suficiente para convergencia con 1 año

### Lección aprendida
Con 8760 horas (1 año) de datos, la mejora real vendrá de añadir datos multi-año
(2020-2023, 4x más datos), no de redes más grandes ni más pasos. Planificado para
la próxima sesión: adaptar el ETL y descargar datos de ESIOS para 2020-2022.

---

## [2026-05-01] — main.py v2: red 256x256, VecNormalize, 1M pasos (HU-12, HU-13)

### Contexto
La v1 (300k pasos, red 64x64, sin normalización) fue el primer intento de entrenamiento.
Resultados de v1: recompensa subió de -139 a -7, mejor evaluación +27.1 en paso 200k,
pero la política no estabilizó (oscilaba entre +27 y -40 en la segunda mitad).

### Mejoras implementadas en v2
- **Red neuronal más grande**: 76 -> 256 -> 256 -> 13 (era 64x64).
  Permite capturar patrones estacionales complejos y arbitraje por hora del día.
- **VecNormalize**: normalización en línea de observaciones (running mean/std).
  Las 76 variables tenían escalas muy diferentes (SoC 0-1, consumo 0-200 kWh).
  Ahora todas tienen media ~0 y desviación ~1. Se guarda `vec_normalize.pkl`.
- **1M pasos** (era 300k): la v1 convergía en paso 200k pero necesitaba más tiempo.
- **Learning rate 5e-5** (era 1e-4): más lento pero más estable.
- **Exploración 50%** (era 40%): explora durante más tiempo antes de explotar.
- **Buffer 200k, batch 128**: gradientes más suaves y experiencias más diversas.

### Siguiente paso lógico
Ejecutar entrenamiento v2 completo (~45-60 min con 1M pasos).
Después: HU-28 — notebook de análisis de resultados.

---

## [2026-05-01] — main.py v1: primer entrenamiento DQN (HU-12, HU-13)

### Rama
`feature/entrenamiento-dqn` (creada desde `develop`)

### Implementado
- **`src/main.py`**: orquestador completo del entrenamiento DQN.
  - Hiperparámetros: lr=1e-4, buffer=100k, learning_starts=10k, batch=64, gamma=0.99,
    exploration 1.0→0.05 en el 40% del entrenamiento, target_update=1000, train_freq=4.
  - `EvalCallback`: evalúa cada 10k pasos en entorno separado, guarda `models/best_model.zip`.
  - `MetricasCallback` (personalizado): loguea en TensorBoard SoC medio, energía comprada
    media y acción más frecuente cada 1000 pasos.
  - Entornos envueltos con `Monitor` → genera `logs/train.monitor.csv` y `logs/eval.monitor.csv`.
  - `sys.path` configurado para ejecutarse desde cualquier ubicación.
- **`requirements.txt`**: añadido `tensorboard` (necesario para el logging de SB3).

### Verificado
Test de 20.000 pasos: recompensa media mejora de -95.9 (primera mitad) a -44.8 (segunda mitad).
`best_model.zip` y `dqn_sgec.zip` generados correctamente. TensorBoard operativo.

### Siguiente paso lógico
Ejecutar entrenamiento real completo (300k pasos): `python src/main.py` desde la carpeta TFG/.
Después: HU-28 — notebook de análisis de resultados (curva de aprendizaje, comparación con baselines).

---
<!-- Añadir nuevas entradas ARRIBA de esta línea, en orden cronológico inverso -->
