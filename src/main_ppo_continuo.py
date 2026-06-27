"""
main_ppo_continuo.py -- PPO continuo sobre EnergyEnvContinuo (familia C)
=======================================================================
PPO on-policy directo sobre los 4 flujos (Box-4), SIN MPC. Aísla "continuo vs
discreto": un on-policy continuo tampoco supera al MPC ni puede usar demos
(DAWN) → confirma que el mérito es del residual off-policy, no del espacio
continuo en sí.

Registry-driven (PPO-C1..3). lr/ent admiten const o decay (como en main_ppo);
action_std_init se traduce a log_std_init.

    python src/main_ppo_continuo.py --version PPO-C1 --seeds 42 1337 2024
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.energy_env_continuo import EnergyEnvContinuo
from src.training.multiseed import entrenar_multiseed
from src.training.registro import cargar_version, seeds_comunes, es_decay, schedule_lineal
from src.training.callbacks import EntCoefScheduler   # compartido (decay de entropía)

LOG_DIR   = os.path.join(ROOT, "logs")
MODEL_DIR = os.path.join(ROOT, "models")

_VCFG = None
_TOTAL = None


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
    ent = c["ent_coef"]
    ent_init = float(ent["init"]) if es_decay(ent) else float(ent)
    # action_std_init → log_std_init (desviación inicial de la gaussiana)
    if "action_std_init" in c:
        log_std_init = math.log(float(c["action_std_init"]))
    else:
        log_std_init = float(c.get("log_std_init", -0.7))
    return PPO(
        policy="MlpPolicy", env=train_env,
        learning_rate=schedule_lineal(c["learning_rate"]),
        n_steps=int(c["n_steps"]), batch_size=int(c["batch_size"]),
        n_epochs=int(c["n_epochs"]), gamma=float(c["gamma"]),
        gae_lambda=float(c["gae_lambda"]), clip_range=float(c["clip_range"]),
        ent_coef=ent_init, vf_coef=float(c["vf_coef"]),
        policy_kwargs={"net_arch": {"pi": c["net_arch_pi"], "vf": c["net_arch_vf"]},
                       "log_std_init": log_std_init},
        verbose=0, seed=seed, device="cpu", tensorboard_log=LOG_DIR,
    )


def _extra_callbacks(model):
    ent = _VCFG["ent_coef"]
    if es_decay(ent):
        return [EntCoefScheduler(ent["init"], ent["final"], _TOTAL)]
    return []


def main(version="PPO-C1", seeds=None, total_timesteps=None):
    global _VCFG, _TOTAL
    _VCFG = cargar_version("ppo_continuo", version)
    train_seeds, _es, eval_episodes = seeds_comunes()
    if seeds is None:
        seeds = train_seeds
    _TOTAL = int(total_timesteps) if total_timesteps else int(_VCFG["total_timesteps"])

    os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    entrenar_multiseed(
        algo="ppo_continuo", version=version, seeds=list(seeds),
        make_envs=make_envs, make_model=make_model,
        total_timesteps=_TOTAL,
        eval_freq=int(_VCFG["eval_freq"]), n_eval_episodes=eval_episodes,
        extra_callbacks=_extra_callbacks,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Entrena PPO continuo (familia C) multi-semilla.")
    p.add_argument("--version", default="PPO-C1")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--timesteps", type=int, default=None)
    a = p.parse_args()
    main(version=a.version, seeds=a.seeds, total_timesteps=a.timesteps)
