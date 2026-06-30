"""
main_ddpg_residual.py -- DDPG residual sobre MPC (familia E de la suite)
=======================================================================
Igual que TD3 residual pero con DDPG (sin doble crítico ni target policy
smoothing). Evidencia del límite inferior del control continuo off-policy sin
la regularización de SAC/TD3 (baja prioridad; soporta por descarte SAC).

Registry-driven (DDPG-1..2). Reutiliza las factorías residuales compartidas.

    python src/training/mains/main_ddpg_residual.py --version DDPG-2 --seeds 42 1337 2024
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import DDPG

from src.controllers.mpc import crear_mpc
from src.training.action_noise import construir_action_noise
from src.training.callbacks import ResidualMetricasCallback
from src.training.scaffold_mains import entrenar, cli, LOG_DIR
from src.training.residual_factories import make_residual_envs, make_dawn_warmup

FAMILIA = "ddpg_residual"


def _build(cfg, total_timesteps):
    mpc = crear_mpc()   # compartido por make_envs y warmup

    def make_envs():
        return make_residual_envs(cfg, mpc)

    def make_model(train_env, seed):
        return DDPG(
            policy='MlpPolicy', env=train_env,
            learning_rate=float(cfg['learning_rate']),
            buffer_size=int(cfg['buffer_size']),
            batch_size=int(cfg['batch_size']),
            gamma=float(cfg['gamma']), tau=float(cfg['tau']),
            action_noise=construir_action_noise(cfg.get('action_noise'), 4),
            learning_starts=0,
            policy_kwargs={'net_arch': cfg['net_arch']},
            verbose=0, seed=seed, device='cpu', tensorboard_log=LOG_DIR,
        )

    return {
        "make_envs": make_envs,
        "make_model": make_model,
        "extra_callbacks": lambda m: [ResidualMetricasCallback()],
        "warmup": make_dawn_warmup(cfg, mpc),
    }


def main(version="DDPG-2", seeds=None, total_timesteps=None):
    entrenar(FAMILIA, _build, version=version, seeds=seeds, total_timesteps=total_timesteps)


if __name__ == "__main__":
    cli(FAMILIA, _build, version_default="DDPG-2",
        description="Entrena DDPG residual (familia E) multi-semilla.")
