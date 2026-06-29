"""
main_ppo.py -- Entrenamiento del agente PPO discreto (familia B de la suite)
============================================================================
Registry-driven: los hiperparámetros de cada versión (PPO-D1..5) se leen de
config/experimentos.yaml. learning_rate y ent_coef admiten escalar (constante)
o {init, final} (decay lineal); el EntCoefScheduler solo se añade si la versión
pide decay de entropía.

Ejecución desde la raíz del proyecto (carpeta TFG/):
    python src/main_ppo.py --version PPO-D4 --seeds 42 1337 2024
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.energy_env import EnergyEnv
from src.training.callbacks import DiscreteMetricasCallback, EntCoefScheduler
from src.training.registry import es_decay, schedule_lineal
from src.training.scaffold_mains import entrenar, cli, LOG_DIR

FAMILIA = "ppo"


def _build(cfg, total_timesteps):
    def make_envs():
        def _mk():
            return Monitor(EnergyEnv())
        train_env = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False, clip_obs=10.0)
        return train_env, eval_env

    def make_model(train_env, seed):
        ent = cfg["ent_coef"]
        ent_init = float(ent["init"]) if es_decay(ent) else float(ent)
        return PPO(
            policy        = "MlpPolicy",
            env           = train_env,
            learning_rate = schedule_lineal(cfg["learning_rate"]),
            n_steps       = int(cfg["n_steps"]),
            batch_size    = int(cfg["batch_size"]),
            n_epochs      = int(cfg["n_epochs"]),
            gamma         = float(cfg["gamma"]),
            gae_lambda    = float(cfg["gae_lambda"]),
            clip_range    = float(cfg["clip_range"]),
            ent_coef      = ent_init,
            vf_coef       = float(cfg["vf_coef"]),
            policy_kwargs = {"net_arch": {"pi": cfg["net_arch_pi"], "vf": cfg["net_arch_vf"]}},
            verbose       = 0,
            seed          = seed,
            device        = "cpu",
            tensorboard_log = LOG_DIR,
        )

    def extra_callbacks(model):
        cbs = [DiscreteMetricasCallback()]
        ent = cfg["ent_coef"]
        if es_decay(ent):
            cbs.append(EntCoefScheduler(ent["init"], ent["final"], total_timesteps))
        return cbs

    return {
        "make_envs": make_envs,
        "make_model": make_model,
        "extra_callbacks": extra_callbacks,
        "pre_entreno": lambda: check_env(EnergyEnv(), warn=True),
    }


def main(version="PPO-D4", seeds=None, total_timesteps=None):
    entrenar(FAMILIA, _build, version=version, seeds=seeds, total_timesteps=total_timesteps)


if __name__ == "__main__":
    cli(FAMILIA, _build, version_default="PPO-D4",
        description="Entrena PPO discreto (familia B) multi-semilla.")
