"""
main_ddpg_residual.py -- DDPG residual sobre MPC (familia E de la suite)
=======================================================================
Igual que TD3 residual pero con DDPG (sin doble crítico ni target policy
smoothing). Evidencia del límite inferior del control continuo off-policy sin
la regularización de SAC/TD3 (baja prioridad; soporta por descarte SAC).

Registry-driven (DDPG-1..2). Reutiliza dawn_warmup y el patrón factoría.

    python src/main_ddpg_residual.py --version DDPG-2 --seeds 42 1337 2024
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import DDPG
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


def make_envs():
    mpc = _get_mpc()
    train_env = VecNormalize(DummyVecEnv([lambda: _make_residual_env(mpc, 'train')]),
                             norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = VecNormalize(DummyVecEnv([lambda: _make_residual_env(mpc, 'eval')]),
                            norm_obs=True, norm_reward=False, clip_obs=10.0)
    return train_env, eval_env


def make_model(train_env, seed):
    c = _VCFG
    return DDPG(
        policy='MlpPolicy', env=train_env,
        learning_rate=float(c['learning_rate']),
        buffer_size=int(c['buffer_size']),
        batch_size=int(c['batch_size']),
        gamma=float(c['gamma']), tau=float(c['tau']),
        action_noise=construir_action_noise(c.get('action_noise'), 4),
        learning_starts=0,
        policy_kwargs={'net_arch': c['net_arch']},
        verbose=0, seed=seed, device='cpu', tensorboard_log=LOG_DIR,
    )


def warmup(model, train_env, seed):
    dawn_warmup(model, _get_mpc(), train_env,
                warmup_steps=int(_VCFG['dawn_warmup_steps']), seed=seed,
                delta_max=_VCFG['delta_max'], residual_mode=_VCFG.get('residual_mode', 'mult'))
    import gc; gc.collect()


def main(version="DDPG-2", seeds=None, total_timesteps=None):
    global _VCFG
    _VCFG = cargar_version("ddpg_residual", version)
    _train_seeds, _es, eval_episodes = seeds_comunes()
    if seeds is None:
        seeds = seeds_para_version("ddpg_residual", version)
    if total_timesteps is None:
        total_timesteps = int(_VCFG['total_timesteps'])

    usar_warmup = warmup if int(_VCFG['dawn_warmup_steps']) > 0 else None
    os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    entrenar_multiseed(
        algo="ddpg_residual", version=version, seeds=list(seeds),
        make_envs=make_envs, make_model=make_model,
        total_timesteps=total_timesteps,
        eval_freq=int(_VCFG['eval_freq']), n_eval_episodes=eval_episodes,
        warmup=usar_warmup,
        extra_callbacks=lambda m: [ResidualMetricasCallback()],
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Entrena DDPG residual (familia E) multi-semilla.")
    p.add_argument("--version", default="DDPG-2")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--timesteps", type=int, default=None)
    a = p.parse_args()
    main(version=a.version, seeds=a.seeds, total_timesteps=a.timesteps)
