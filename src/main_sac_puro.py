"""
main_sac_puro.py -- SAC continuo PURO sobre EnergyEnvContinuo (familia F5)
=========================================================================
SAC directo sobre los 4 flujos (Box-4), SIN MPC base, SIN residual y SIN DAWN
warmup. Aísla el mérito de la arquitectura residual: si rinde por debajo del
Residual SAC, el valor lo aporta el residual, no SAC en sí.

Registry-driven (familia sac_puro, versión F5).

    python src/main_sac_puro.py --version F5 --seeds 42 1337 2024
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.energy_env_continuo import EnergyEnvContinuo
from src.training.scaffold_mains import entrenar, cli, LOG_DIR

FAMILIA = "sac_puro"


def _build(cfg, total_timesteps):
    def make_envs():
        def _mk(mode):
            return lambda: Monitor(EnergyEnvContinuo(forecast_noise=True, mode=mode))
        train_env = VecNormalize(DummyVecEnv([_mk('train')]),
                                 norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env = VecNormalize(DummyVecEnv([_mk('eval')]),
                                norm_obs=True, norm_reward=False, clip_obs=10.0)
        return train_env, eval_env

    def make_model(train_env, seed):
        return SAC(
            policy='MlpPolicy', env=train_env,
            learning_rate=float(cfg['learning_rate']),
            buffer_size=int(cfg['buffer_size']),
            batch_size=int(cfg['batch_size']),
            gamma=float(cfg['gamma']), tau=float(cfg['tau']),
            ent_coef=cfg['ent_coef'], target_entropy=cfg['target_entropy'],
            learning_starts=int(cfg['learning_starts']),     # exploración inicial (sin DAWN)
            policy_kwargs={'net_arch': cfg['net_arch']},
            verbose=0, seed=seed, device='cpu', tensorboard_log=LOG_DIR,
        )

    return {"make_envs": make_envs, "make_model": make_model}


def main(version="F5", seeds=None, total_timesteps=None):
    entrenar(FAMILIA, _build, version=version, seeds=seeds, total_timesteps=total_timesteps)


if __name__ == "__main__":
    cli(FAMILIA, _build, version_default="F5",
        description="Entrena SAC puro continuo (F5) multi-semilla.")
