"""
N-step: si la versión pide n_step>1 se cablea NStepReplayBuffer
(gamma del modelo = base_gamma**n_step, el buffer acumula el retorno N-step).

Ejecución desde la raíz del proyecto (carpeta TFG/):
    python src/training/mains/main_dqn.py --version DQN-2 --seeds 42 1337 2024
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.energy_env_discreto import EnergyEnvDiscreto
from src.training.callbacks import DiscreteMetricasCallback
from src.training.scaffold_mains import entrenar, cli, LOG_DIR

FAMILIA = "dqn"


def _build(cfg, total_timesteps):
    def make_envs():
        def _mk():
            return Monitor(EnergyEnvDiscreto())
        train_env = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False, clip_obs=10.0)
        return train_env, eval_env

    def make_model(train_env, seed):
        base_gamma = float(cfg["gamma"])
        n_step = int(cfg.get("n_step", 1))
        kwargs = dict(
            policy                 = "MlpPolicy",
            env                    = train_env,
            learning_rate          = float(cfg["learning_rate"]),
            buffer_size            = int(cfg["buffer_size"]),
            learning_starts        = int(cfg["learning_starts"]),
            batch_size             = int(cfg["batch_size"]),
            gamma                  = base_gamma,
            exploration_fraction   = float(cfg["exploration_fraction"]),
            exploration_final_eps  = float(cfg["exploration_final_eps"]),
            target_update_interval = int(cfg["target_update_interval"]),
            train_freq             = int(cfg["train_freq"]),
            policy_kwargs          = {"net_arch": cfg["net_arch"]},
            verbose                = 0,
            seed                   = seed,
            tensorboard_log        = LOG_DIR,
        )
        if n_step > 1:
            from src.training.nstep_buffer import NStepReplayBuffer
            kwargs["gamma"] = base_gamma ** n_step          # γ^n para el bootstrap
            kwargs["replay_buffer_class"] = NStepReplayBuffer
            kwargs["replay_buffer_kwargs"] = {"n_steps": n_step, "base_gamma": base_gamma}
        return DQN(**kwargs)

    return {
        "make_envs": make_envs,
        "make_model": make_model,
        "extra_callbacks": lambda m: [DiscreteMetricasCallback()],
        "pre_entreno": lambda: check_env(EnergyEnvDiscreto(), warn=True),
    }


def main(version="DQN-2", seeds=None, total_timesteps=None):
    entrenar(FAMILIA, _build, version=version, seeds=seeds, total_timesteps=total_timesteps)


if __name__ == "__main__":
    cli(FAMILIA, _build, version_default="DQN-2",
        description="Entrena DQN (familia A) multi-semilla.")
