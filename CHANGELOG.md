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

## [Próxima sesión] — FASE 3: Entrenamiento del Agente DQN

### Por hacer
- [ ] Escribir `src/main.py` con DQN de SB3 (configurar `learning_rate`, `buffer_size`, `batch_size`, `exploration_fraction`).
- [ ] Añadir `EvalCallback` para guardar el mejor modelo.
- [ ] Entrenar el agente (~500k-1M timesteps según capacidad del hardware).
- [ ] Guardar modelo entrenado en `models/dqn_comunidad_energia.zip`.

---
<!-- Añadir nuevas entradas ARRIBA de esta línea, en orden cronológico inverso -->
