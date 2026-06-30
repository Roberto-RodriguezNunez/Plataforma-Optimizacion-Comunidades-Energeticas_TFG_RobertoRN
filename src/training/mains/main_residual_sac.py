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
    python src/training/mains/main_residual_sac.py [--version SAC-C] [--seeds 42 1337 2024]
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import SAC

from src.controllers.mpc import crear_mpc
from src.core.config import cargar_system as _cargar_system
from src.training.callbacks import ResidualMetricasCallback
from src.training.scaffold_mains import entrenar, cli, LOG_DIR
from src.training.residual_factories import make_residual_envs, make_dawn_warmup

FAMILIA = "residual_sac"

# Infra de normalización (norm_obs, norm_reward, clip_obs) desde system.yaml.
_SAC_CFG = _cargar_system()['residual_sac']


def _build(cfg, total_timesteps):
    # F6: σ del pronóstico es propiedad del entorno → se fija por-run (global).
    if 'sigma_consumo' in cfg:
        import src.core.forecast as _fc
        _fc.SIGMA_CONS_BASE = float(cfg['sigma_consumo'])
        print(f"  [F6] SIGMA_CONS_BASE = {_fc.SIGMA_CONS_BASE}")

    mpc = crear_mpc()   # MPC compartido por make_envs y warmup (misma instancia)

    def make_envs():
        return make_residual_envs(
            cfg, mpc,
            norm_obs=_SAC_CFG['norm_obs'], norm_reward=_SAC_CFG['norm_reward'],
            clip_obs=_SAC_CFG['clip_obs'],
        )

    def make_model(train_env, seed):
        # Sin DAWN (F4-nodawn) hace falta exploración inicial → learning_starts>0.
        learning_starts = 0 if int(cfg['dawn_warmup_steps']) > 0 else 10_000
        return SAC(
            policy='MlpPolicy',
            env=train_env,
            learning_rate=float(cfg['learning_rate']),
            buffer_size=int(cfg['buffer_size']),
            batch_size=int(cfg['batch_size']),
            gamma=float(cfg['gamma']),
            tau=float(cfg['tau']),
            ent_coef=cfg['ent_coef'],
            target_entropy=cfg['target_entropy'],
            learning_starts=learning_starts,
            policy_kwargs={'net_arch': cfg['net_arch']},
            verbose=0,
            seed=seed,
            device='cpu',
            tensorboard_log=LOG_DIR,
        )

    return {
        "make_envs": make_envs,
        "make_model": make_model,
        "extra_callbacks": lambda m: [ResidualMetricasCallback()],
        "warmup": make_dawn_warmup(cfg, mpc),
    }


def main(version="SAC-C", seeds=None, total_timesteps=None):
    entrenar(FAMILIA, _build, version=version, seeds=seeds, total_timesteps=total_timesteps)


if __name__ == '__main__':
    cli(FAMILIA, _build, version_default="SAC-C",
        description="Entrena Residual SAC sobre MPC (familia F) multi-semilla.")
