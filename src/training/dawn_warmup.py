"""
dawn_warmup.py — DAWN warmup (Data-Anchored Warmup) compartido
==============================================================
Pre-llena el replay buffer de un agente off-policy (SAC, TD3, DDPG) con
transiciones donde delta=0 (acción pura del MPC) y calibra VecNormalize con esas
observaciones. Ancla el critic al rendimiento del MPC antes de explorar deltas.

Vive en src/training/ porque lo comparten main_residual_sac, main_td3_residual y
main_ddpg_residual (antes estaba dentro de main_residual_sac y se importaba desde
ahí, lo cual era un pequeño smell).
"""

import numpy as np

from src.envs.energy_env import EnergyEnvContinuo
from src.envs.residual_env import ResidualEnv


def dawn_warmup(model, mpc, train_env, warmup_steps, seed=42,
                delta_max=0.15, residual_mode='mult'):
    """
    Pre-llena el replay buffer con transiciones MPC puro (delta=0) y calibra
    VecNormalize. Sirve para cualquier agente off-policy con replay_buffer.

    Args:
        model: agente SB3 (SAC/TD3/DDPG); su replay_buffer se modifica in-place.
        mpc: LinearMPC para el wrapper residual.
        train_env: VecNormalize del entorno de entrenamiento (se calibra su obs_rms).
        warmup_steps: nº de transiciones a generar.
        seed: semilla para reproducibilidad.
        delta_max / residual_mode: del ResidualEnv temporal. Como delta=0, no
            afectan a las transiciones (mpc puro), pero se pasan por coherencia.
    """
    print(f"\n  DAWN warmup: generando {warmup_steps:,} transiciones MPC puro...")
    np.random.seed(seed)

    # ResidualEnv temporal para generar transiciones (delta=0 → acción MPC pura)
    inner = EnergyEnvContinuo(forecast_noise=True, mode='train')
    renv = ResidualEnv(inner, mpc, delta_max=delta_max, residual_mode=residual_mode)

    obs, _ = renv.reset(seed=seed)
    n_episodes = 0
    obs_buffer = []

    for _ in range(warmup_steps):
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # delta 4D = 0
        next_obs, reward, terminated, truncated, info = renv.step(action)

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
