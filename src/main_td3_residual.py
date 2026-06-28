"""
main_td3_residual.py -- TD3 residual sobre MPC (familia D de la suite)
=====================================================================
Mismo entorno residual que el Residual SAC (ResidualEnv sobre MPC + DAWN warmup),
pero con TD3 (off-policy determinista, doble crítico). Sirve para justificar
empíricamente la elección de SAC frente a TD3.

Registry-driven (TD3-1..4 en config/experimentos.yaml). Reutiliza dawn_warmup y
el patrón factoría de main_residual_sac.

    python src/main_td3_residual.py --version TD3-1 --seeds 42 1337 2024
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.benchmarks.mpc_benchmark import crear_mpc
from src.envs.energy_env_continuo import EnergyEnvContinuo
from src.envs.residual_env import ResidualEnv
from src.training.multiseed import entrenar_multiseed
from src.training.callbacks import ResidualMetricasCallback
from src.training.registro import cargar_version, seeds_comunes, seeds_para_version
from src.training.dawn_warmup import dawn_warmup         # warmup MPC compartido
from src.training.action_noise import construir_action_noise

LOG_DIR   = os.path.join(ROOT, "logs")
MODEL_DIR = os.path.join(ROOT, "models")

_VCFG = None
_MPC_SHARED = None


def _get_mpc():
    global _MPC_SHARED
    if _MPC_SHARED is None:
        _MPC_SHARED = crear_mpc()
    return _MPC_SHARED


def _make_residual_env(mpc, mode='train'):
    inner = EnergyEnvContinuo(forecast_noise=True, mode=mode)
    renv = ResidualEnv(inner, mpc, delta_max=_VCFG['delta_max'],
                       residual_mode=_VCFG.get('residual_mode', 'mult'))
    return Monitor(renv)


def make_envs(seed):
    mpc = _get_mpc()
    train_env = VecNormalize(DummyVecEnv([lambda: _make_residual_env(mpc, 'train')]),
                             norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = VecNormalize(DummyVecEnv([lambda: _make_residual_env(mpc, 'eval')]),
                            norm_obs=True, norm_reward=False, clip_obs=10.0)
    return train_env, eval_env


def make_model(train_env, seed):
    c = _VCFG
    return TD3(
        policy='MlpPolicy', env=train_env,
        learning_rate=float(c['learning_rate']),
        buffer_size=int(c['buffer_size']),
        batch_size=int(c['batch_size']),
        gamma=float(c['gamma']), tau=float(c['tau']),
        policy_delay=int(c['policy_delay']),
        target_policy_noise=float(c['target_policy_noise']),
        target_noise_clip=float(c['target_noise_clip']),
        action_noise=construir_action_noise(c.get('action_noise'), 4),
        learning_starts=0,                       # se aprende tras el DAWN warmup
        policy_kwargs={'net_arch': c['net_arch']},
        verbose=0, seed=seed, device='cpu', tensorboard_log=LOG_DIR,
    )


def warmup(model, train_env, seed):
    dawn_warmup(model, _get_mpc(), train_env,
                warmup_steps=int(_VCFG['dawn_warmup_steps']), seed=seed,
                delta_max=_VCFG['delta_max'], residual_mode=_VCFG.get('residual_mode', 'mult'))
    import gc; gc.collect()


def main(version="TD3-1", seeds=None, total_timesteps=None):
    global _VCFG
    _VCFG = cargar_version("td3_residual", version)
    _train_seeds, _es, eval_episodes = seeds_comunes()
    if seeds is None:
        seeds = seeds_para_version("td3_residual", version)
    if total_timesteps is None:
        total_timesteps = int(_VCFG['total_timesteps'])

    usar_warmup = warmup if int(_VCFG['dawn_warmup_steps']) > 0 else None
    os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    entrenar_multiseed(
        algo="td3_residual", version=version, seeds=list(seeds),
        make_envs=make_envs, make_model=make_model,
        total_timesteps=total_timesteps,
        eval_freq=int(_VCFG['eval_freq']), n_eval_episodes=eval_episodes,
        warmup=usar_warmup,
        extra_callbacks=lambda m: [ResidualMetricasCallback()],
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Entrena TD3 residual (familia D) multi-semilla.")
    p.add_argument("--version", default="TD3-1")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--timesteps", type=int, default=None)
    a = p.parse_args()
    main(version=a.version, seeds=a.seeds, total_timesteps=a.timesteps)
