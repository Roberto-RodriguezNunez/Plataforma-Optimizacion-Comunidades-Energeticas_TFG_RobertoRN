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
from src.training.callbacks import DiscreteMetricasCallback


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

# MetricasCallback (discreto) ahora vive en src/training/callbacks.py
# (DiscreteMetricasCallback, compartido con PPO).


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
        extra_callbacks=lambda m: [DiscreteMetricasCallback()],
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    a = p.parse_args()
    main(version=a.version, seeds=a.seeds)
