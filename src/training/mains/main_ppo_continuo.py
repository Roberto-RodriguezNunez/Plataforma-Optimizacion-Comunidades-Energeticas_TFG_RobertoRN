"""
main_ppo_continuo.py -- PPO continuo sobre EnergyEnvContinuo (familia C)
=======================================================================
PPO on-policy directo sobre los 4 flujos (Box-4), SIN MPC. Aísla "continuo vs
discreto": un on-policy continuo tampoco supera al MPC ni puede usar demos
(DAWN) → confirma que el mérito es del residual off-policy, no del espacio
continuo en sí.

Registry-driven (PPO-C1..3). lr/ent admiten const o decay (como en main_ppo);
action_std_init se traduce a log_std_init.

    python src/training/mains/main_ppo_continuo.py --version PPO-C1 --seeds 42 1337 2024
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.energy_env_continuo import EnergyEnvContinuo
from src.training.callbacks import EntCoefScheduler   # compartido (decay de entropía)
from src.training.registry import es_decay, schedule_lineal
from src.training.scaffold_mains import entrenar, cli, LOG_DIR

FAMILIA = "ppo_continuo"


def _build(cfg, total_timesteps):
    def make_envs():
        def _mk(mode):
            return lambda: Monitor(EnergyEnvContinuo(forecast_noise=True, mode=mode))
        train_env = VecNormalize(DummyVecEnv([_mk('train')]),
                                 norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env = VecNormalize(DummyVecEnv([_mk('eval')]),
                                norm_obs=True, norm_reward=False, clip_obs=10.0)
        return train_env, eval_env

    def make_model(train_env, seed):
        ent = cfg["ent_coef"]
        ent_init = float(ent["init"]) if es_decay(ent) else float(ent)
        # action_std_init → log_std_init (desviación inicial de la gaussiana)
        if "action_std_init" in cfg:
            log_std_init = math.log(float(cfg["action_std_init"]))
        else:
            log_std_init = float(cfg.get("log_std_init", -0.7))
        return PPO(
            policy="MlpPolicy", env=train_env,
            learning_rate=schedule_lineal(cfg["learning_rate"]),
            n_steps=int(cfg["n_steps"]), batch_size=int(cfg["batch_size"]),
            n_epochs=int(cfg["n_epochs"]), gamma=float(cfg["gamma"]),
            gae_lambda=float(cfg["gae_lambda"]), clip_range=float(cfg["clip_range"]),
            ent_coef=ent_init, vf_coef=float(cfg["vf_coef"]),
            policy_kwargs={"net_arch": {"pi": cfg["net_arch_pi"], "vf": cfg["net_arch_vf"]},
                           "log_std_init": log_std_init},
            verbose=0, seed=seed, device="cpu", tensorboard_log=LOG_DIR,
        )

    def extra_callbacks(model):
        ent = cfg["ent_coef"]
        if es_decay(ent):
            return [EntCoefScheduler(ent["init"], ent["final"], total_timesteps)]
        return []

    return {
        "make_envs": make_envs,
        "make_model": make_model,
        "extra_callbacks": extra_callbacks,
    }


def main(version="PPO-C1", seeds=None, total_timesteps=None):
    entrenar(FAMILIA, _build, version=version, seeds=seeds, total_timesteps=total_timesteps)


if __name__ == "__main__":
    cli(FAMILIA, _build, version_default="PPO-C1",
        description="Entrena PPO continuo (familia C) multi-semilla.")
