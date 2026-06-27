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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.energy_env import EnergyEnv
from src.training.multiseed import entrenar_multiseed
from src.training.callbacks import DiscreteMetricasCallback, EntCoefScheduler
from src.training.registro import cargar_version, seeds_comunes, es_decay, schedule_lineal

LOG_DIR    = os.path.join(ROOT, "logs")
MODEL_DIR  = os.path.join(ROOT, "models")

_CFG = None
_TOTAL = None


def make_envs(seed):
    """Factoría de entornos PPO (discreto) envueltos en VecNormalize."""
    def _mk():
        return Monitor(EnergyEnv())
    train_env = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False, clip_obs=10.0)
    return train_env, eval_env


def make_model(train_env, seed):
    """Factoría del modelo PPO según la versión (lr/ent const o decay)."""
    c = _CFG
    ent = c["ent_coef"]
    ent_init = float(ent["init"]) if es_decay(ent) else float(ent)
    return PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = schedule_lineal(c["learning_rate"]),
        n_steps       = int(c["n_steps"]),
        batch_size    = int(c["batch_size"]),
        n_epochs      = int(c["n_epochs"]),
        gamma         = float(c["gamma"]),
        gae_lambda    = float(c["gae_lambda"]),
        clip_range    = float(c["clip_range"]),
        ent_coef      = ent_init,
        vf_coef       = float(c["vf_coef"]),
        policy_kwargs = {"net_arch": {"pi": c["net_arch_pi"], "vf": c["net_arch_vf"]}},
        verbose       = 0,
        seed          = seed,
        device        = "cpu",
        tensorboard_log = LOG_DIR,
    )


def _extra_callbacks(model):
    cbs = [DiscreteMetricasCallback()]
    ent = _CFG["ent_coef"]
    if es_decay(ent):
        cbs.append(EntCoefScheduler(ent["init"], ent["final"], _TOTAL))
    return cbs


def main(version="PPO-D4", seeds=None, total_timesteps=None):
    global _CFG, _TOTAL
    _CFG = cargar_version("ppo", version)
    train_seeds, _eval_seeds, eval_episodes = seeds_comunes()
    if seeds is None:
        seeds = train_seeds
    _TOTAL = int(total_timesteps) if total_timesteps else int(_CFG["total_timesteps"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    check_env(EnergyEnv(), warn=True)

    entrenar_multiseed(
        algo="ppo", version=version, seeds=list(seeds),
        make_envs=make_envs, make_model=make_model,
        total_timesteps=_TOTAL,
        eval_freq=int(_CFG["eval_freq"]), n_eval_episodes=eval_episodes,
        extra_callbacks=_extra_callbacks,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Entrena PPO discreto (familia B) multi-semilla.")
    p.add_argument("--version", default="PPO-D4")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--timesteps", type=int, default=None, help="Override total timesteps (smoke).")
    a = p.parse_args()
    main(version=a.version, seeds=a.seeds, total_timesteps=a.timesteps)
