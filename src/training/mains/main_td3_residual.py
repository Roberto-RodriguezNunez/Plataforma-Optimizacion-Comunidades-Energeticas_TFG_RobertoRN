"""
main_td3_residual.py -- TD3 residual sobre MPC (familia D de la suite)
=====================================================================
Mismo entorno residual que el Residual SAC (ResidualEnv sobre MPC + DAWN warmup),
pero con TD3 (off-policy determinista, doble crítico). Sirve para justificar
empíricamente la elección de SAC frente a TD3.

Registry-driven (TD3-1..4 en config/experimentos.yaml). Reutiliza las factorías
residuales compartidas (make_residual_envs + make_dawn_warmup).

    python src/main_td3_residual.py --version TD3-1 --seeds 42 1337 2024
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import TD3

from src.controllers.mpc import crear_mpc
from src.training.action_noise import construir_action_noise
from src.training.callbacks import ResidualMetricasCallback
from src.training.scaffold_mains import entrenar, cli, LOG_DIR
from src.training.residual_factories import make_residual_envs, make_dawn_warmup

FAMILIA = "td3_residual"


def _build(cfg, total_timesteps):
    mpc = crear_mpc()   # compartido por make_envs y warmup

    def make_envs():
        return make_residual_envs(cfg, mpc)

    def make_model(train_env, seed):
        return TD3(
            policy='MlpPolicy', env=train_env,
            learning_rate=float(cfg['learning_rate']),
            buffer_size=int(cfg['buffer_size']),
            batch_size=int(cfg['batch_size']),
            gamma=float(cfg['gamma']), tau=float(cfg['tau']),
            policy_delay=int(cfg['policy_delay']),
            target_policy_noise=float(cfg['target_policy_noise']),
            target_noise_clip=float(cfg['target_noise_clip']),
            action_noise=construir_action_noise(cfg.get('action_noise'), 4),
            learning_starts=0,                       # se aprende tras el DAWN warmup
            policy_kwargs={'net_arch': cfg['net_arch']},
            verbose=0, seed=seed, device='cpu', tensorboard_log=LOG_DIR,
        )

    return {
        "make_envs": make_envs,
        "make_model": make_model,
        "extra_callbacks": lambda m: [ResidualMetricasCallback()],
        "warmup": make_dawn_warmup(cfg, mpc),
    }


def main(version="TD3-1", seeds=None, total_timesteps=None):
    entrenar(FAMILIA, _build, version=version, seeds=seeds, total_timesteps=total_timesteps)


if __name__ == "__main__":
    cli(FAMILIA, _build, version_default="TD3-1",
        description="Entrena TD3 residual (familia D) multi-semilla.")
