# CHANGELOG.md — Registro de Progreso del TFG

> Al iniciar sesión: "Lee el CHANGELOG.md para ver dónde nos quedamos."
> Al terminar una tarea: añadir entrada con fecha, descripción técnica y siguiente paso lógico.

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
