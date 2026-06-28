"""
main_residual_sac.py — Entrenamiento del agente Residual SAC sobre MPC
=====================================================================
Fase 2 del TFG: el SAC aprende correcciones Delta_a sobre la accion
del MPC realista resuelto online en cada step.

Protocolo DAWN (Data-Anchored Warmup):
  1. Pre-llenar el replay buffer con 50k transiciones MPC puro (delta=0).
  2. Entrenar SAC durante 1M pasos con exploracion sobre delta.
  3. El critic aprende primero que delta=0 da reward MPC, luego explora.

Ejecucion desde la raiz del proyecto (carpeta TFG/):
    python src/main_residual_sac.py [--version 80kwh] [--seeds 42 1337 2024]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.benchmarks.mpc_benchmark import LinearMPC, ComunidadSimulador, DATASET_PATH, crear_mpc
from src.envs.energy_env_continuo import EnergyEnvContinuo
from src.envs.residual_env import ResidualEnv
from src.training.multiseed import entrenar_multiseed
from src.training.callbacks import ResidualMetricasCallback
from src.training.registro import cargar_version, seeds_comunes, seeds_para_version
from src.training.dawn_warmup import dawn_warmup

# --- Cargar configuracion ---
_CONFIG_PATH = os.path.join(ROOT, 'config', 'system.yaml')
with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
    _CFG = yaml.safe_load(f)

_SAC_CFG = _CFG['residual_sac']      # infra (norm_obs, norm_reward, clip_obs)
_MPC_CFG = _CFG['mpc']

# Config de la versión en curso (la fija main() desde la registry de experimentos).
_VCFG = None

EPISODE_LENGTH = _MPC_CFG['duracion_episodio']
MODEL_DIR = os.path.join(ROOT, 'models')
LOG_DIR = os.path.join(ROOT, 'logs')


# ──────────────────────────────────────────────────────────────────
#  SEEDED EVAL CALLBACK — mismas 50 semanas en cada evaluacion
# ──────────────────────────────────────────────────────────────────

# SeededEvalCallback ahora vive en src/training/multiseed.py (compartido por
# todos los algoritmos, usado por el runner entrenar_multiseed).


# ──────────────────────────────────────────────────────────────────
#  METRICAS CALLBACK — SoC, delta, a_mpc en TensorBoard
# ──────────────────────────────────────────────────────────────────

# ResidualMetricasCallback ahora vive en src/training/callbacks.py (compartido).


# DAWN warmup ahora vive en src/training/dawn_warmup.py (compartido por el
# Residual SAC, TD3 residual y DDPG residual).


# ──────────────────────────────────────────────────────────────────
#  FUNCIONES DE CREACION DE ENTORNOS
# ──────────────────────────────────────────────────────────────────

def _make_residual_env(mpc, mode='train'):
    """Crea un ResidualEnv (modo residual de la versión) envuelto en Monitor."""
    inner = EnergyEnvContinuo(forecast_noise=True, mode=mode)
    renv = ResidualEnv(inner, mpc, delta_max=_VCFG['delta_max'],
                       residual_mode=_VCFG.get('residual_mode', 'mult'))
    return Monitor(renv)


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

_MPC_SHARED = None


def _get_mpc():
    """MPC compartido (fuente única crear_mpc; reutilizable entre semillas)."""
    global _MPC_SHARED
    if _MPC_SHARED is None:
        _MPC_SHARED = crear_mpc()
    return _MPC_SHARED


def make_envs(seed):
    """Factoría de entornos Residual SAC (ResidualEnv sobre MPC) con VecNormalize."""
    mpc = _get_mpc()
    train_env = VecNormalize(
        DummyVecEnv([lambda: _make_residual_env(mpc, mode='train')]),
        norm_obs=_SAC_CFG['norm_obs'], norm_reward=_SAC_CFG['norm_reward'],
        clip_obs=_SAC_CFG['clip_obs'],
    )
    eval_env = VecNormalize(
        DummyVecEnv([lambda: _make_residual_env(mpc, mode='eval')]),
        norm_obs=_SAC_CFG['norm_obs'], norm_reward=_SAC_CFG['norm_reward'],
        clip_obs=_SAC_CFG['clip_obs'],
    )
    return train_env, eval_env


def make_model(train_env, seed):
    """Factoría del modelo SAC según la versión de la registry."""
    c = _VCFG
    # Sin DAWN (F4-nodawn) hace falta exploración inicial → learning_starts>0.
    learning_starts = 0 if int(c['dawn_warmup_steps']) > 0 else 10_000
    return SAC(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=float(c['learning_rate']),
        buffer_size=int(c['buffer_size']),
        batch_size=int(c['batch_size']),
        gamma=float(c['gamma']),
        tau=float(c['tau']),
        ent_coef=c['ent_coef'],
        target_entropy=c['target_entropy'],
        learning_starts=learning_starts,
        policy_kwargs={'net_arch': c['net_arch']},
        verbose=0,
        seed=seed,
        device='cpu',
        tensorboard_log=LOG_DIR,
    )


def warmup(model, train_env, seed):
    """DAWN warmup: llena el buffer con MPC puro y calibra VecNormalize."""
    dawn_warmup(
        model, _get_mpc(), train_env,
        warmup_steps=int(_VCFG['dawn_warmup_steps']), seed=seed,
        delta_max=_VCFG['delta_max'], residual_mode=_VCFG.get('residual_mode', 'mult'),
    )
    import gc; gc.collect()


def main(version="SAC-C", seeds=None, total_timesteps=None):
    global _VCFG
    _VCFG = cargar_version("residual_sac", version)
    _train_seeds, _eval_seeds, eval_episodes = seeds_comunes()
    if seeds is None:
        seeds = seeds_para_version("residual_sac", version)
    if total_timesteps is None:
        total_timesteps = int(_VCFG['total_timesteps'])

    # F6: σ del pronóstico es propiedad del entorno → se fija por-run (global).
    if 'sigma_consumo' in _VCFG:
        import src.core.forecast as _fc
        _fc.SIGMA_CONS_BASE = float(_VCFG['sigma_consumo'])
        print(f"  [F6] SIGMA_CONS_BASE = {_fc.SIGMA_CONS_BASE}")

    # DAWN off (F4-nodawn) → no se pasa warmup al runner.
    usar_warmup = warmup if int(_VCFG['dawn_warmup_steps']) > 0 else None

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    entrenar_multiseed(
        algo="residual_sac", version=version, seeds=list(seeds),
        make_envs=make_envs, make_model=make_model,
        total_timesteps=total_timesteps,
        eval_freq=int(_VCFG['eval_freq']), n_eval_episodes=eval_episodes,
        warmup=usar_warmup,
        extra_callbacks=lambda m: [ResidualMetricasCallback()],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrena Residual SAC sobre MPC (familia F) multi-semilla.')
    parser.add_argument('--version', default='SAC-C',
                        help='Versión de la registry (SAC-A/B/C, F2-*, F4-*, F6-*).')
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Override total timesteps (smoke).')
    args = parser.parse_args()
    main(version=args.version, seeds=args.seeds, total_timesteps=args.timesteps)
