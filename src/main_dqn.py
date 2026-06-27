"""
main_dqn.py -- Entrenamiento del agente DQN (familia A de la suite)
===================================================================
Registry-driven: los hiperparámetros de cada versión (DQN-1..6) se leen de
config/experimentos.yaml vía src/training/registro.py. El runner multi-semilla
(entrenar_multiseed) y la factoría make_envs/make_model se mantienen.

N-step: si la versión pide n_step>1 se cablea NStepReplayBuffer
(gamma del modelo = base_gamma**n_step, el buffer acumula el retorno N-step).

Ejecución desde la raíz del proyecto (carpeta TFG/):
    python src/main_dqn.py --version DQN-2 --seeds 42 1337 2024
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.energy_env import EnergyEnv
from src.training.multiseed import entrenar_multiseed
from src.training.callbacks import DiscreteMetricasCallback
from src.training.registro import cargar_version, seeds_comunes

LOG_DIR    = os.path.join(ROOT, "logs")
MODEL_DIR  = os.path.join(ROOT, "models")

# Config de la versión en curso (la fija main() desde la registry).
_CFG = None


def make_envs(seed):
    """Factoría de entornos DQN (discreto) envueltos en VecNormalize."""
    def _mk():
        return Monitor(EnergyEnv())
    train_env = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False, clip_obs=10.0)
    return train_env, eval_env


def make_model(train_env, seed):
    """Factoría del modelo DQN según la versión de la registry (con N-step opcional)."""
    c = _CFG
    base_gamma = float(c["gamma"])
    n_step = int(c.get("n_step", 1))

    kwargs = dict(
        policy                 = "MlpPolicy",
        env                    = train_env,
        learning_rate          = float(c["learning_rate"]),
        buffer_size            = int(c["buffer_size"]),
        learning_starts        = int(c["learning_starts"]),
        batch_size             = int(c["batch_size"]),
        gamma                  = base_gamma,
        exploration_fraction   = float(c["exploration_fraction"]),
        exploration_final_eps  = float(c["exploration_final_eps"]),
        target_update_interval = int(c["target_update_interval"]),
        train_freq             = int(c["train_freq"]),
        policy_kwargs          = {"net_arch": c["net_arch"]},
        verbose                = 0,
        seed                   = seed,
        tensorboard_log        = LOG_DIR,
    )

    if n_step > 1:
        from src.buffers.nstep_replay_buffer import NStepReplayBuffer
        kwargs["gamma"] = base_gamma ** n_step          # γ^n para el bootstrap
        kwargs["replay_buffer_class"] = NStepReplayBuffer
        kwargs["replay_buffer_kwargs"] = {"n_steps": n_step, "base_gamma": base_gamma}

    return DQN(**kwargs)


def main(version="DQN-2", seeds=None, total_timesteps=None):
    global _CFG
    _CFG = cargar_version("dqn", version)
    train_seeds, _eval_seeds, eval_episodes = seeds_comunes()
    if seeds is None:
        seeds = train_seeds
    if total_timesteps is None:
        total_timesteps = int(_CFG["total_timesteps"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    check_env(EnergyEnv(), warn=True)

    entrenar_multiseed(
        algo="dqn", version=version, seeds=list(seeds),
        make_envs=make_envs, make_model=make_model,
        total_timesteps=total_timesteps,
        eval_freq=int(_CFG["eval_freq"]), n_eval_episodes=eval_episodes,
        extra_callbacks=lambda m: [DiscreteMetricasCallback()],
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Entrena DQN (familia A) multi-semilla.")
    p.add_argument("--version", default="DQN-2")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--timesteps", type=int, default=None, help="Override total timesteps (smoke).")
    a = p.parse_args()
    main(version=a.version, seeds=a.seeds, total_timesteps=a.timesteps)
