"""
main.py — Orquestador de entrenamiento del agente DQN
======================================================
HU-12: Entrenar agente DQN con Stable Baselines3
HU-13: Monitorizar el progreso del entrenamiento

Ejecución desde la raíz del proyecto (carpeta TFG/):
    python src/main.py

Para ver las curvas en TensorBoard (mientras entrena o después):
    tensorboard --logdir logs/
"""

import os
import sys

# Garantizar que la raíz del proyecto está en sys.path
# independientemente de desde dónde se ejecute el script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.envs.energy_env import EnergyEnv


# ──────────────────────────────────────────────────────────────────
#  HIPERPARÁMETROS
#  Para un test rápido usa TOTAL_TIMESTEPS = 5_000 y LEARNING_STARTS = 1_000
# ──────────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS   = 300_000   # Subir a 500k-1M para entrenamiento serio
LEARNING_RATE     = 1e-4      # Tasa de aprendizaje de la red Q
BUFFER_SIZE       = 100_000   # Tamaño del replay buffer
LEARNING_STARTS   = 10_000    # Pasos de exploración pura antes de entrenar
BATCH_SIZE        = 64        # Muestras por actualización de gradiente
GAMMA             = 0.99      # Factor de descuento (episodios de 168 pasos → horizonte largo)
EXPLORATION_FRAC  = 0.4       # Fracción del entrenamiento con ε decreciente (1.0 → 0.05)
EXPLORATION_FINAL = 0.05      # ε mínimo al final del entrenamiento
TARGET_UPDATE     = 1_000     # Pasos entre copias de Q → Q_target
TRAIN_FREQ        = 4         # Pasos del entorno entre actualizaciones de la red

EVAL_FREQ         = 10_000    # Cada cuántos pasos evaluar (guardará el mejor modelo)
EVAL_EPISODES     = 5         # Episodios por ronda de evaluación

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
        # SB3 almacena la info del último paso en self.locals["infos"]
        for info in self.locals.get("infos", []):
            if "soc" in info:
                self._soc_buffer.append(info["soc"])
            if "comprado" in info:
                self._comprado_buffer.append(info["comprado"])

        accion = self.locals.get("actions")
        if accion is not None:
            self._acciones[int(accion[0])] += 1

        # Volcar métricas cada 1000 pasos
        if self.num_timesteps % 1_000 == 0 and self._soc_buffer:
            self.logger.record("custom/soc_medio",      np.mean(self._soc_buffer))
            self.logger.record("custom/comprado_medio", np.mean(self._comprado_buffer))
            self.logger.record("custom/accion_mas_freq", int(np.argmax(self._acciones)))
            self._soc_buffer.clear()
            self._comprado_buffer.clear()

        return True  # True = continuar entrenamiento


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    # ── 1. Verificar que el entorno cumple la API de Gymnasium ────
    print("-" * 60)
    print("1. Verificando entorno con env_checker...")
    check_env(EnergyEnv(), warn=True)
    print("   Entorno OK.\n")

    # ── 2. Crear entornos (Monitor envuelve y loguea recompensas) ─
    print("2. Instanciando entornos de entrenamiento y evaluacion...")
    env      = Monitor(EnergyEnv(), filename=os.path.join(LOG_DIR, "train"))
    eval_env = Monitor(EnergyEnv(), filename=os.path.join(LOG_DIR, "eval"))
    print("   Entornos listos.\n")

    # ── 3. Configurar agente DQN ──────────────────────────────────
    print("3. Configurando agente DQN...")
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
        verbose                = 1,
        tensorboard_log        = LOG_DIR,
    )
    print(f"   Política: {model.policy}\n")

    # ── 4. Callbacks ──────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = MODEL_DIR,       # Guarda best_model.zip
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
        progress_bar    = False,   # Poner True si tienes instalado 'rich' o 'tqdm'
    )

    # ── 6. Guardar modelo final ───────────────────────────────────
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save(model_path)
    print(f"\nModelo final guardado en: {model_path}.zip")
    print(f"Mejor modelo guardado en: {os.path.join(MODEL_DIR, 'best_model.zip')}")


if __name__ == "__main__":
    main()
