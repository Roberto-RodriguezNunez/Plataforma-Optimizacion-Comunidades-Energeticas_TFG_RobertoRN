"""
action_noise.py — Construcción del ruido de exploración para TD3/DDPG
=====================================================================
Traduce la spec de la registry ({tipo: normal|ou|none, sigma: ...}) al objeto
ActionNoise de SB3. Compartido por main_td3_residual y main_ddpg_residual.
"""

import numpy as np


def construir_action_noise(spec, n_actions: int = 4):
    """spec = {'tipo': 'normal'|'ou'|'none', 'sigma': float} → ActionNoise | None."""
    spec = spec or {"tipo": "none"}
    tipo = spec.get("tipo", "none")
    if tipo == "none":
        return None

    from stable_baselines3.common.noise import (
        NormalActionNoise, OrnsteinUhlenbeckActionNoise,
    )
    sigma = float(spec.get("sigma", 0.1))
    mean = np.zeros(n_actions)
    sig = sigma * np.ones(n_actions)
    if tipo == "ou":
        return OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sig)
    return NormalActionNoise(mean=mean, sigma=sig)
