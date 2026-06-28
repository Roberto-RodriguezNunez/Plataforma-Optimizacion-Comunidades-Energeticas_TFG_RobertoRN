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
from src.training.multiseed import entrenar_multiseed
from src.training.registro import cargar_version, seeds_comunes, seeds_para_version

LOG_DIR   = os.path.join(ROOT, "logs")
MODEL_DIR = os.path.join(ROOT, "models")

_VCFG = None


def make_envs(seed):
    def _mk(mode):
        return lambda: Monitor(EnergyEnvContinuo(forecast_noise=True, mode=mode))
    train_env = VecNormalize(DummyVecEnv([_mk('train')]),
                             norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = VecNormalize(DummyVecEnv([_mk('eval')]),
                            norm_obs=True, norm_reward=False, clip_obs=10.0)
    return train_env, eval_env


def make_model(train_env, seed):
    c = _VCFG
    return SAC(
        policy='MlpPolicy', env=train_env,
        learning_rate=float(c['learning_rate']),
        buffer_size=int(c['buffer_size']),
        batch_size=int(c['batch_size']),
        gamma=float(c['gamma']), tau=float(c['tau']),
        ent_coef=c['ent_coef'], target_entropy=c['target_entropy'],
        learning_starts=int(c['learning_starts']),     # exploración inicial (sin DAWN)
        policy_kwargs={'net_arch': c['net_arch']},
        verbose=0, seed=seed, device='cpu', tensorboard_log=LOG_DIR,
    )


def main(version="F5", seeds=None, total_timesteps=None):
    global _VCFG
    _VCFG = cargar_version("sac_puro", version)
    _train_seeds, _es, eval_episodes = seeds_comunes()
    if seeds is None:
        seeds = seeds_para_version("sac_puro", version)
    if total_timesteps is None:
        total_timesteps = int(_VCFG['total_timesteps'])

    os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    entrenar_multiseed(
        algo="sac_puro", version=version, seeds=list(seeds),
        make_envs=make_envs, make_model=make_model,
        total_timesteps=total_timesteps,
        eval_freq=int(_VCFG['eval_freq']), n_eval_episodes=eval_episodes,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Entrena SAC puro continuo (F5) multi-semilla.")
    p.add_argument("--version", default="F5")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--timesteps", type=int, default=None)
    a = p.parse_args()
    main(version=a.version, seeds=a.seeds, total_timesteps=a.timesteps)
