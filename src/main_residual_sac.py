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

# --- Cargar configuracion ---
_CONFIG_PATH = os.path.join(ROOT, 'config', 'system.yaml')
with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
    _CFG = yaml.safe_load(f)

_SAC_CFG = _CFG['residual_sac']
_MPC_CFG = _CFG['mpc']

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


# ──────────────────────────────────────────────────────────────────
#  DAWN WARMUP — pre-llenado del buffer con transiciones MPC puro
# ──────────────────────────────────────────────────────────────────

def dawn_warmup(model, mpc, train_env, warmup_steps, seed=42):
    """
    Pre-llena el replay buffer de SAC con transiciones donde delta=0
    (accion pura del MPC) Y calibra VecNormalize con las observaciones.

    Esto ancla el critic al rendimiento MPC antes de que SAC empiece
    a explorar deltas. Ademas, las running statistics de VecNormalize
    (obs_rms) se inicializan con datos reales en lugar de empezar
    desde cero.

    Args:
        model: Instancia de SAC (su replay_buffer se modifica in-place).
        mpc: LinearMPC para el wrapper.
        train_env: VecNormalize wrapping del entorno de entrenamiento.
        warmup_steps: Numero de transiciones a generar.
        seed: Semilla para reproducibilidad.
    """
    print(f"\n  DAWN warmup: generando {warmup_steps:,} transiciones MPC puro...")
    np.random.seed(seed)

    # Crear un ResidualEnv temporal para generar transiciones
    inner = EnergyEnvContinuo(forecast_noise=True, mode='train')
    renv = ResidualEnv(inner, mpc, delta_max=_SAC_CFG['delta_max'])

    obs, _ = renv.reset(seed=seed)
    n_episodes = 0
    obs_buffer = []

    for i in range(warmup_steps):
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # delta 4D = 0
        next_obs, reward, terminated, truncated, info = renv.step(action)

        # SB3 replay buffer: add(obs, next_obs, action, reward, done, infos)
        model.replay_buffer.add(
            obs.reshape(1, -1),
            next_obs.reshape(1, -1),
            action.reshape(1, -1),
            np.array([reward]),
            np.array([terminated]),
            [info],
        )

        obs_buffer.append(obs)

        if terminated or truncated:
            obs, _ = renv.reset()
            n_episodes += 1
        else:
            obs = next_obs

    print(f"    {warmup_steps:,} transiciones, {n_episodes} episodios completos.")
    print(f"    Buffer size: {model.replay_buffer.size()}")

    # Calibrar VecNormalize con las observaciones del warmup
    print("    Calibrando VecNormalize con obs del warmup...")
    obs_all = np.array(obs_buffer, dtype=np.float32)
    train_env.obs_rms.mean = obs_all.mean(axis=0)
    train_env.obs_rms.var = obs_all.var(axis=0)
    train_env.obs_rms.count = len(obs_all)
    print(f"    obs_rms calibrado con {len(obs_all):,} observaciones.")


# ──────────────────────────────────────────────────────────────────
#  FUNCIONES DE CREACION DE ENTORNOS
# ──────────────────────────────────────────────────────────────────

def _make_residual_env(mpc, mode='train'):
    """Crea un ResidualEnv envuelto en Monitor."""
    inner = EnergyEnvContinuo(forecast_noise=True, mode=mode)
    renv = ResidualEnv(inner, mpc, delta_max=_SAC_CFG['delta_max'])
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
    """Factoría del modelo SAC (con semilla aplicada)."""
    return SAC(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=_SAC_CFG['learning_rate'],
        buffer_size=_SAC_CFG['buffer_size'],
        batch_size=_SAC_CFG['batch_size'],
        gamma=_SAC_CFG['gamma'],
        tau=_SAC_CFG['tau'],
        ent_coef=_SAC_CFG['ent_coef'],
        target_entropy=_SAC_CFG['target_entropy'],
        learning_starts=0,  # se aprende inmediatamente tras el warmup
        policy_kwargs={'net_arch': _SAC_CFG['net_arch']},
        verbose=0,
        seed=seed,
        device='cpu',
        tensorboard_log=LOG_DIR,
    )


def warmup(model, train_env, seed):
    """DAWN warmup: llena el buffer con MPC puro y calibra VecNormalize."""
    dawn_warmup(
        model, _get_mpc(), train_env,
        warmup_steps=_SAC_CFG['dawn_warmup_steps'], seed=seed,
    )
    import gc; gc.collect()


def main(version="v1", seeds=(42, 1337, 2024), total_timesteps=None):
    if total_timesteps is None:
        total_timesteps = _SAC_CFG['total_timesteps']
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    entrenar_multiseed(
        algo="residual_sac", version=version, seeds=list(seeds),
        make_envs=make_envs, make_model=make_model,
        total_timesteps=total_timesteps,
        eval_freq=_SAC_CFG['eval_freq'], n_eval_episodes=_SAC_CFG['eval_episodes'],
        warmup=warmup,
        extra_callbacks=lambda m: [ResidualMetricasCallback()],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrena Residual SAC sobre MPC (multi-semilla).')
    parser.add_argument('--version', default='v1')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 1337, 2024])
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total timesteps (default: config)')
    parser.add_argument('--delta-max', type=float, default=None)
    parser.add_argument('--ent-coef', type=float, default=None)
    parser.add_argument('--warmup', type=int, default=None,
                        help='DAWN warmup steps override (default: config)')
    args = parser.parse_args()
    if args.delta_max is not None:
        _SAC_CFG['delta_max'] = args.delta_max
    if args.ent_coef is not None:
        _SAC_CFG['ent_coef'] = args.ent_coef
    if args.warmup is not None:
        _SAC_CFG['dawn_warmup_steps'] = args.warmup
    main(version=args.version, seeds=args.seeds, total_timesteps=args.timesteps)
