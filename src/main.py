"""
main.py — Orquestador de entrenamiento del agente DQN (v3)
============================================================
HU-12: Entrenar agente DQN con Stable Baselines3
HU-13: Monitorizar el progreso del entrenamiento

Cambios respecto a v2:
  - Revertir a red 64x64 (mejor con dataset pequeño de 8760 horas)
  - Mantener VecNormalize solo para observaciones (quitar norm_reward)
  - 500k pasos (suficiente para convergencia con 1 año de datos)
  - Subir eval_episodes a 20 para medias de evaluación más fiables

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
#  HIPERPARÁMETROS (v3)
#  Para un test rápido: TOTAL_TIMESTEPS = 10_000, LEARNING_STARTS = 2_000
# ──────────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS   = 500_000     # 500k con dataset 2023 (subir a 1M-2M con multi-año)
LEARNING_RATE     = 1e-4        # Revertido a v1 — funcionaba bien
BUFFER_SIZE       = 100_000     # Revertido a v1
LEARNING_STARTS   = 10_000      # Revertido a v1
BATCH_SIZE        = 64          # Revertido a v1
GAMMA             = 0.99        # Sin cambio
EXPLORATION_FRAC  = 0.4         # Revertido a v1 — 50% era demasiado lento
EXPLORATION_FINAL = 0.05        # Sin cambio
TARGET_UPDATE     = 1_000       # Revertido a v1
TRAIN_FREQ        = 4           # Sin cambio

# Red neuronal: 76 → 64 → 64 → 13 (revertido — 256x256 overfitteaba con 8760h)
NET_ARCH          = [64, 64]

EVAL_FREQ         = 10_000      # Cada 10k pasos
EVAL_EPISODES     = 20          # Subido de 5/10 a 20 — reduce varianza en la media de eval

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
    #  Solo normaliza observaciones (norm_obs=True).
    #  norm_reward=False: la recompensa NO se normaliza.
    #  Razon: normalizar la recompensa distorsiona la señal de aprendizaje
    #  y dificulta comparar evaluaciones entre entrenamientos.
    print("2. Instanciando entornos con VecNormalize (solo observaciones)...")
    env = VecNormalize(
        make_env(),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    eval_env = VecNormalize(
        make_env(),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    print("   Entornos listos.\n")

    # ── 3. Configurar agente DQN ──────────────────────────────────
    print("3. Configurando agente DQN...")
    print(f"   Red neuronal: 76 -> {NET_ARCH[0]} -> {NET_ARCH[1]} -> 13")
    print(f"   Total timesteps: {TOTAL_TIMESTEPS:,}")

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
    env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))

    print(f"\nModelo final guardado en: {model_path}.zip")
    print(f"Mejor modelo guardado en: {os.path.join(MODEL_DIR, 'best_model.zip')}")
    print(f"Estadisticas VecNormalize: {os.path.join(MODEL_DIR, 'vec_normalize.pkl')}")


if __name__ == "__main__":
    main()
