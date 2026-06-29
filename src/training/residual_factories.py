"""factorias_residual.py — Factorías compartidas de los mains residuales.

residual_sac / td3_residual / ddpg_residual repetían la MISMA construcción del
ResidualEnv sobre MPC y el MISMO warmup DAWN; solo cambia el algoritmo SB3.
Aquí viven una sola vez. (El esqueleto/CLI común está en scaffold_mains.py.)
"""
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.energy_env import EnergyEnvContinuo
from src.envs.residual_env import ResidualEnv
from src.training.dawn_warmup import dawn_warmup


def make_residual_envs(cfg, mpc, norm_obs=True, norm_reward=False, clip_obs=10.0):
    """(train_env, eval_env): ResidualEnv sobre MPC envuelto en VecNormalize.

    delta_max y residual_mode salen del cfg de la versión. Los parámetros de
    normalización se exponen porque residual_sac los lee de system.yaml mientras
    td3/ddpg usan los valores por defecto (idénticos en la práctica).
    """
    def _mk(mode):
        inner = EnergyEnvContinuo(forecast_noise=True, mode=mode)
        renv = ResidualEnv(inner, mpc, delta_max=cfg["delta_max"],
                           residual_mode=cfg.get("residual_mode", "mult"))
        return Monitor(renv)

    train_env = VecNormalize(DummyVecEnv([lambda: _mk("train")]),
                             norm_obs=norm_obs, norm_reward=norm_reward, clip_obs=clip_obs)
    eval_env = VecNormalize(DummyVecEnv([lambda: _mk("eval")]),
                            norm_obs=norm_obs, norm_reward=norm_reward, clip_obs=clip_obs)
    return train_env, eval_env


def make_dawn_warmup(cfg, mpc):
    """warmup(model, train_env, seed) del protocolo DAWN, o None si la versión
    no lo pide (dawn_warmup_steps == 0)."""
    if int(cfg["dawn_warmup_steps"]) <= 0:
        return None

    def warmup(model, train_env, seed):
        dawn_warmup(model, mpc, train_env,
                    warmup_steps=int(cfg["dawn_warmup_steps"]), seed=seed,
                    delta_max=cfg["delta_max"],
                    residual_mode=cfg.get("residual_mode", "mult"))
        import gc
        gc.collect()

    return warmup
