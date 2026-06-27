"""
main_dqn.py -- Entrenamiento del agente DQN (configuracion DQN_13)
===================================================================
Portado desde feature/dqn-optimizaciones para tener todos los
algoritmos entrenables desde la rama principal.

Hiperparametros (DQN_13 — mejor resultado historico: +45.23 EUR/sem):
  - NET_ARCH: [256, 128] (~60k params)
  - BUFFER_SIZE: 500k (diversidad estacional)
  - GAMMA: 0.995 (credito 48h)
  - EXPLORATION_FRAC: 0.5 (1.5M exploracion + 1.5M explotacion)
  - 3M steps totales

Ejecucion desde la raiz del proyecto (carpeta TFG/):
    python src/main_dqn.py
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
from src.training.multiseed import entrenar_multiseed


# ------------------------------------------------------------------
#  HIPERPARAMETROS (DQN_13)
# ------------------------------------------------------------------

TOTAL_TIMESTEPS   = 3_000_000
LEARNING_RATE     = 1e-4
BUFFER_SIZE       = 500_000
LEARNING_STARTS   = 10_000
BATCH_SIZE        = 64
GAMMA             = 0.995
EXPLORATION_FRAC  = 0.5
EXPLORATION_FINAL = 0.05
TARGET_UPDATE     = 1_500
TRAIN_FREQ        = 4
NET_ARCH          = [256, 128]

EVAL_FREQ         = 20_000
EVAL_EPISODES     = 50

MODEL_DIR  = os.path.join(ROOT, "models")
LOG_DIR    = os.path.join(ROOT, "logs")
MODEL_NAME = "dqn_sgec"


# ------------------------------------------------------------------
#  CALLBACK — metricas adicionales en TensorBoard
# ------------------------------------------------------------------

class MetricasCallback(BaseCallback):
    """Registra SoC, economia y distribucion de acciones en TensorBoard."""

    _GRUPOS = {
        "IDLE":           [0],
        "CARGAR_SOLAR":   [1],
        "CARGAR_MIXTA":   [2, 3, 4],
        "DESCARGAR_CASA": [5],
        "DESCARGAR_RED":  [6, 7, 8],
    }

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._soc_buffer        = []
        self._comprado_buffer   = []
        self._cargado_buffer    = []
        self._descargado_buffer = []
        self._acciones          = np.zeros(9, dtype=np.int64)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "soc" in info:
                self._soc_buffer.append(info["soc"])
            if "comprado" in info:
                self._comprado_buffer.append(info["comprado"])
            if "cargado" in info:
                self._cargado_buffer.append(info["cargado"])
            if "descargado" in info:
                self._descargado_buffer.append(info["descargado"])

        accion = self.locals.get("actions")
        if accion is not None:
            self._acciones[int(accion[0])] += 1

        if self.num_timesteps % 1_000 == 0 and self._soc_buffer:
            self.logger.record("custom/soc_medio",      np.mean(self._soc_buffer))
            self.logger.record("custom/soc_minimo",     np.min(self._soc_buffer))
            self.logger.record("custom/soc_maximo",     np.max(self._soc_buffer))
            self.logger.record("custom/cargado_medio",
                               np.mean(self._cargado_buffer) if self._cargado_buffer else 0.0)
            self.logger.record("custom/descargado_medio",
                               np.mean(self._descargado_buffer) if self._descargado_buffer else 0.0)
            self.logger.record("custom/comprado_medio", np.mean(self._comprado_buffer))

            total = self._acciones.sum()
            if total > 0:
                for grupo, indices in self._GRUPOS.items():
                    pct = 100.0 * self._acciones[indices].sum() / total
                    self.logger.record(f"acciones/{grupo}_pct", pct)
                for i in range(9):
                    self.logger.record(f"acciones/accion_{i:02d}_pct",
                                       100.0 * self._acciones[i] / total)
                self.logger.record("acciones/dominante_idx", int(np.argmax(self._acciones)))
                self.logger.record("acciones/dominante_pct",
                                   100.0 * self._acciones.max() / total)

            try:
                self.logger.record("custom/epsilon", self.model.exploration_rate)
            except AttributeError:
                pass

            self._soc_buffer.clear()
            self._comprado_buffer.clear()
            self._cargado_buffer.clear()
            self._descargado_buffer.clear()
            self._acciones[:] = 0

        return True


# ------------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------------

def make_envs(seed):
    """Factoría de entornos DQN (discreto) envueltos en VecNormalize."""
    def _mk():
        return Monitor(EnergyEnv())
    train_env = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False, clip_obs=10.0)
    return train_env, eval_env


def make_model(train_env, seed):
    """Factoría del modelo DQN (con semilla aplicada)."""
    return DQN(
        policy                 = "MlpPolicy",
        env                    = train_env,
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
        verbose                = 0,
        seed                   = seed,
        tensorboard_log        = LOG_DIR,
    )


def main(version="v1", seeds=(42, 1337, 2024)):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    check_env(EnergyEnv(), warn=True)
    entrenar_multiseed(
        algo="dqn", version=version, seeds=list(seeds),
        make_envs=make_envs, make_model=make_model,
        total_timesteps=TOTAL_TIMESTEPS,
        eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES,
        extra_callbacks=lambda m: [MetricasCallback()],
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    a = p.parse_args()
    main(version=a.version, seeds=a.seeds)
