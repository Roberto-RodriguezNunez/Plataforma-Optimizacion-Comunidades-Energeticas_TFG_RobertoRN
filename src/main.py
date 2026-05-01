"""
main.py — Orquestador de entrenamiento del agente DQN (v2)
============================================================
HU-12: Entrenar agente DQN con Stable Baselines3
HU-13: Monitorizar el progreso del entrenamiento

Cambios respecto a v1 (300k pasos, red 64x64, sin normalización):
  - Red neuronal más grande: 256x256 (captura patrones estacionales complejos)
  - VecNormalize: normaliza observaciones para igualar la escala de las 76 variables
  - 1M de pasos (más tiempo para estabilizar la política)
  - Exploración más larga (50% del entrenamiento) y learning rate más bajo (5e-5)

Ejecución desde la raíz del proyecto (carpeta TFG/):
    python src/main.py

Para ver las curvas en TensorBoard (mientras entrena o después):
    tensorboard --logdir logs/
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.energy_env import EnergyEnv


# ──────────────────────────────────────────────────────────────────
#  HIPERPARÁMETROS (v2)
#  Para un test rápido: TOTAL_TIMESTEPS = 10_000, LEARNING_STARTS = 2_000
# ──────────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS   = 1_000_000   # v1 era 300k — más tiempo para estabilizar
LEARNING_RATE     = 5e-5        # v1 era 1e-4 — más lento pero más estable
BUFFER_SIZE       = 200_000     # v1 era 100k — más experiencias en el buffer
LEARNING_STARTS   = 20_000      # v1 era 10k — más exploración inicial
BATCH_SIZE        = 128         # v1 era 64 — gradientes más suaves
GAMMA             = 0.99        # Sin cambio — horizonte largo
EXPLORATION_FRAC  = 0.5         # v1 era 0.4 — explora durante más tiempo
EXPLORATION_FINAL = 0.05        # Sin cambio
TARGET_UPDATE     = 2_000       # v1 era 1k — actualiza red objetivo menos frecuente
TRAIN_FREQ        = 4           # Sin cambio

# Red neuronal: 76 → 256 → 256 → 13 (v1 era 64x64)
NET_ARCH          = [256, 256]

EVAL_FREQ         = 20_000      # v1 era 10k — evaluaciones menos frecuentes (cada eval es lenta)
EVAL_EPISODES     = 10          # v1 era 5 — más episodios por evaluación = media más fiable

MODEL_DIR  = os.path.join(ROOT, "models")
LOG_DIR    = os.path.join(ROOT, "logs")
MODEL_NAME = "dqn_sgec"


# ──────────────────────────────────────────────────────────────────
#  CALLBACK PERSONALIZADO — métricas adicionales en TensorBoard
# ──────────────────────────────────────────────────────────────────

class MetricasCallback(BaseCallback):
    """
    Registra en TensorBoard métricas que SB3 no loguea por defecto:
    - SoC medio de los últimos episodios
    - Energía comprada a red media
    - Distribución de acciones (acción más frecuente)
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._soc_buffer      = []
        self._comprado_buffer = []
        self._acciones        = np.zeros(13, dtype=int)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "soc" in info:
                self._soc_buffer.append(info["soc"])
            if "comprado" in info:
                self._comprado_buffer.append(info["comprado"])

        accion = self.locals.get("actions")
        if accion is not None:
            self._acciones[int(accion[0])] += 1

        if self.num_timesteps % 1_000 == 0 and self._soc_buffer:
            self.logger.record("custom/soc_medio",      np.mean(self._soc_buffer))
            self.logger.record("custom/comprado_medio", np.mean(self._comprado_buffer))
            self.logger.record("custom/accion_mas_freq", int(np.argmax(self._acciones)))
            self._soc_buffer.clear()
            self._comprado_buffer.clear()

        return True


# ──────────────────────────────────────────────────────────────────
#  FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────

def make_env():
    """Crea una instancia de EnergyEnv envuelta en DummyVecEnv."""
    return DummyVecEnv([lambda: Monitor(EnergyEnv())])


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    # ── 1. Verificar entorno ──────────────────────────────────────
    print("-" * 60)
    print("1. Verificando entorno con env_checker...")
    check_env(EnergyEnv(), warn=True)
    print("   Entorno OK.\n")

    # ── 2. Crear entornos con VecNormalize ────────────────────────
    #
    #  VecNormalize aplica normalización en línea (running mean/std)
    #  a las observaciones. Esto iguala la escala de las 76 variables:
    #    - SoC (0-1), precio (0.05-0.30), consumo (0-200 kWh)
    #  todas pasan a tener media ~0 y desviación ~1.
    #
    #  norm_reward=True normaliza también la recompensa, lo que
    #  estabiliza el entrenamiento cuando R_t varía mucho entre episodios.
    #  clip_obs/clip_reward evitan outliers extremos.
    #
    print("2. Instanciando entornos con VecNormalize...")
    env = VecNormalize(
        make_env(),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    eval_env = VecNormalize(
        make_env(),
        norm_obs=True,
        norm_reward=False,    # No normalizar recompensa en eval (queremos el valor real)
        clip_obs=10.0,
    )
    print("   Entornos listos (observaciones normalizadas).\n")

    # ── 3. Configurar agente DQN ──────────────────────────────────
    print("3. Configurando agente DQN...")
    print(f"   Red neuronal: 76 -> {NET_ARCH[0]} -> {NET_ARCH[1]} -> 13")
    print(f"   Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Exploration: {EXPLORATION_FRAC*100:.0f}% del entrenamiento")

    model = DQN(
        policy                 = "MlpPolicy",
        env                    = env,
        learning_rate          = LEARNING_RATE,
        buffer_size            = BUFFER_SIZE,
        learning_starts        = LEARNING_STARTS,
        batch_size             = BATCH_SIZE,
        gamma                  = GAMMA,
        exploration_fraction   = EXPLORATION_FRAC,
        exploration_final_eps  = EXPLORATION_FINAL,
        target_update_interval = TARGET_UPDATE,
        train_freq             = TRAIN_FREQ,
        policy_kwargs          = {"net_arch": NET_ARCH},
        verbose                = 1,
        tensorboard_log        = LOG_DIR,
    )
    print()

    # ── 4. Callbacks ──────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = MODEL_DIR,
        log_path             = LOG_DIR,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = EVAL_EPISODES,
        deterministic        = True,
        verbose              = 1,
    )
    metricas_callback = MetricasCallback()

    # ── 5. Entrenar ───────────────────────────────────────────────
    print("-" * 60)
    print(f"4. Iniciando entrenamiento ({TOTAL_TIMESTEPS:,} pasos)...")
    print(f"   TensorBoard: tensorboard --logdir {LOG_DIR}")
    print("-" * 60)

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [eval_callback, metricas_callback],
        progress_bar    = False,
    )

    # ── 6. Guardar modelo y estadísticas de normalización ─────────
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save(model_path)

    # Guardar las estadísticas de VecNormalize (media y std de las observaciones).
    # Son necesarias para reproducir el comportamiento del modelo en inferencia.
    env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))

    print(f"\nModelo final guardado en: {model_path}.zip")
    print(f"Mejor modelo guardado en: {os.path.join(MODEL_DIR, 'best_model.zip')}")
    print(f"Estadisticas VecNormalize: {os.path.join(MODEL_DIR, 'vec_normalize.pkl')}")


if __name__ == "__main__":
    main()
