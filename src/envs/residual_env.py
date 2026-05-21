"""
residual_env.py — Wrapper residual multiplicativo 4D para SAC sobre MPC
=======================================================================
Interpone el MPC entre la salida del SAC (delta 4D) y el entorno continuo.
En cada step:
  1. Avanza AR(1) (replicando el timing de correr_episodios)
  2. Resuelve MPC online con SoC real y forecast ruidoso
  3. Combina: flow_final = max(0, mpc_flow * (1 + delta * delta_max))
  4. Ejecuta en EnergyEnvContinuo

Residual multiplicativo: las correcciones escalan con la magnitud del flujo.
  - Si MPC dice 0, la correccion es 0 (no se activa lo que MPC descarto).
  - Si MPC dice 20 kW, con delta_max=0.30 la correccion es ±6 kW.
  - delta=0 pasa los flujos MPC intactos.

La observacion se aumenta con 4 features MPC normalizadas (dim 112).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.benchmarks.mpc_benchmark import aplicar_ruido_ar1


class ResidualEnv(gym.Wrapper):
    """
    Wrapper residual multiplicativo 4D sobre EnergyEnvContinuo.

    El agente SAC controla delta in [-1, 1]^4.
    flow_final = max(0, mpc_flow * (1 + delta * delta_max))
    La observacion incluye los 4 flujos MPC normalizados (/P_MAX).

    Args:
        env: Instancia de EnergyEnvContinuo.
        mpc: Instancia de LinearMPC (se resuelve online en cada step).
        delta_max: Fraccion multiplicativa. 0.30 = ±30% de cada flujo MPC.
    """

    def __init__(self, env, mpc, delta_max: float = 0.30):
        super().__init__(env)
        self.mpc = mpc
        self.delta_max = delta_max
        self._P_MAX = env.simulador.POTENCIA_INVERSOR

        # Obs aumentada: 108 original + 4 MPC features = 112
        low = np.full(112, -np.inf, dtype=np.float32)
        high = np.full(112, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Accion del agente: delta 4D in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Calcular accion MPC para la obs inicial (AR(1) error=0 tras reset)
        mpc_action = self._compute_mpc_action()
        P = self._P_MAX
        mpc_feat = np.array([
            mpc_action['P_carga_solar'] / P,
            mpc_action['P_carga_red'] / P,
            mpc_action['P_descarga_casa'] / P,
            mpc_action['P_descarga_red'] / P,
        ], dtype=np.float32)
        obs_aug = np.append(obs, mpc_feat)

        return obs_aug, info

    def step(self, delta_action):
        sim = self.env.simulador

        # Invalidar cache de factores de precio del step anterior
        self.env._precio_noise_factors = None

        # 1. Avanzar AR(1) manualmente (replicar timing de correr_episodios:
        #    AR(1) avanza ANTES de resolver MPC en cada paso)
        if self.env.forecast_noise:
            self.env._error_solar = (
                self.env._RHO_SOLAR * self.env._error_solar
                + np.sqrt(1 - self.env._RHO_SOLAR ** 2) * self.env._noise()
            )
            self.env._error_cons = (
                self.env._RHO_CONS * self.env._error_cons
                + np.sqrt(1 - self.env._RHO_CONS ** 2) * self.env._noise()
            )

        # 2. Resolver MPC online con SoC real y forecast ruidoso
        mpc_action = self._compute_mpc_action()
        cs_mpc = mpc_action['P_carga_solar']
        cm_mpc = mpc_action['P_carga_red']
        dc_mpc = mpc_action['P_descarga_casa']
        dr_mpc = mpc_action['P_descarga_red']

        # 3. Residual multiplicativo: flow * (1 + delta * delta_max)
        delta = np.clip(delta_action, -1.0, 1.0)
        dm = self.delta_max
        cs_f = max(0.0, cs_mpc * (1.0 + delta[0] * dm))
        cm_f = max(0.0, cm_mpc * (1.0 + delta[1] * dm))
        dc_f = max(0.0, dc_mpc * (1.0 + delta[2] * dm))
        dr_f = max(0.0, dr_mpc * (1.0 + delta[3] * dm))

        # 4. Ejecutar en entorno continuo (skip AR(1) porque ya lo avanzamos)
        self.env._skip_next_ar1 = True
        obs, reward, terminated, truncated, info = self.env.step(
            np.array([cs_f, cm_f, dc_f, dr_f], dtype=np.float32)
        )

        # 5. Aumentar obs con 4 features MPC normalizadas (/P_MAX)
        P = self._P_MAX
        mpc_feat = np.array(
            [cs_mpc / P, cm_mpc / P, dc_mpc / P, dr_mpc / P],
            dtype=np.float32,
        )
        obs_aug = np.append(obs, mpc_feat)  # 108 + 4 = 112

        # 6. Info adicional para logging y analisis
        info['a_mpc'] = [cs_mpc, cm_mpc, dc_mpc, dr_mpc]
        info['a_final'] = [cs_f, cm_f, dc_f, dr_f]
        info['delta_applied'] = [
            cs_f - cs_mpc, cm_f - cm_mpc, dc_f - dc_mpc, dr_f - dr_mpc
        ]

        return obs_aug, reward, terminated, truncated, info

    def _compute_mpc_action(self) -> dict:
        sim = self.env.simulador
        window = sim.get_data_window(sim.current_step, horizon=24).copy()
        if self.env.forecast_noise:
            window = aplicar_ruido_ar1(
                window, self.env._error_solar, self.env._error_cons
            )
            # Ruido de precio: generar factores y cachearlos en el env
            # para que _get_obs() use los mismos (coherencia MPC↔agente).
            # _compute_mpc_action se llama ANTES de _get_obs en cada step.
            hora_actual = self.env._get_hora_actual_from_step(sim.current_step)
            self.env._precio_noise_factors = self.env._generar_precio_noise(
                hora_actual
            )
            # MPC window[h] = hora h adelante → factors[h] (offset=0)
            for h in range(24):
                f = self.env._precio_noise_factors[h]
                if f != 1.0:
                    window[h, 2] = max(0.0, window[h, 2] * f)
                    window[h, 3] = max(0.0, window[h, 3] * f)
        state = {'soc': sim.soc, 'step': sim.current_step}
        return self.mpc.solve(state, window)
