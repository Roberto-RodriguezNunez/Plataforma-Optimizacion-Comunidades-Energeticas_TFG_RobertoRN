"""
energy_env_continuo.py — Entorno Gymnasium con accion continua 4D
=================================================================
Variante de EnergyEnv con espacio de accion Box(4) en [0, P_MAX].
La accion vectorial representa los 4 flujos de bateria en kW:
    action = (cs_kw, cm_kw, dc_kw, dr_kw)
    cs = carga desde solar, cm = carga desde red
    dc = descarga a casa, dr = descarga a red

La fisica es IDENTICA a simular_hora_mpc (clipping, bateria, economia).
La observacion y recompensa son identicas al EnergyEnv discreto.

Uso principal: entorno base para el wrapper ResidualEnv (Residual SAC).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.envs.energy_env import EnergyEnv
from src.core.forecast import avanzar_ar1


class EnergyEnvContinuo(EnergyEnv):
    """
    Entorno con accion continua para control de bateria comunitaria.

    Hereda de EnergyEnv: misma observacion (108-dim), mismo reset(),
    misma logica AR(1) de pronostico, mismo split train/eval.

    Sobreescribe:
      - action_space: Box(low=0, high=P_MAX, shape=(4,))
      - step(): recibe 4 flujos en kW, fisica identica a simular_hora_mpc

    Args:
        forecast_noise: Si True, aplica ruido AR(1) al pronostico.
        mode: 'train', 'eval' o 'all'.
    """

    def __init__(self, forecast_noise: bool = True, mode: str = 'all',
                 rng=None):
        super().__init__(forecast_noise=forecast_noise, mode=mode, rng=rng)

        # Sobreescribir el espacio de accion discreto por continuo 4D
        P_MAX = self.simulador.POTENCIA_INVERSOR
        self.action_space = spaces.Box(
            low=0.0, high=P_MAX, shape=(4,), dtype=np.float32
        )

        # Flag para que ResidualEnv pueda controlar el avance AR(1)
        self._skip_next_ar1 = False

    def step(self, action):
        """
        Ejecuta una hora con accion continua de 4 flujos.

        Args:
            action: ndarray shape (4,), valores en [0, P_MAX] kW.
                    (cs, cm, dc, dr) = carga_solar, carga_red,
                    descarga_casa, descarga_red.

        Returns:
            obs, reward, terminated, truncated, info
        """
        # 0. Avanzar estado AR(1) (a menos que el wrapper lo haya hecho)
        if not self._skip_next_ar1 and self.forecast_noise:
            self._error_solar, self._error_cons = avanzar_ar1(
                self._error_solar, self._error_cons, self._noise
            )
        self._skip_next_ar1 = False  # Resetear flag

        # 1-6. Física de la batería — FUENTE ÚNICA compartida con el MPC
        #      (sim.aplicar_fisica_4flujos == lo que ejecuta simular_hora_mpc)
        sim = self.simulador
        r = sim.aplicar_fisica_4flujos(action[0], action[1], action[2], action[3])
        reward = r['beneficio_marginal']

        # 7. Correccion de coste inicial (simetria con valor terminal)
        if self.steps_in_episode == 0:
            reward -= self._pendiente_coste_inicial

        # 8. Avanzar tiempo
        sim.current_step += 1
        self.steps_in_episode += 1

        # 9. Comprobar fin de episodio
        terminated = (self.steps_in_episode >= self.EPISODE_LENGTH)
        truncated = False

        # 10. Valor terminal
        if terminated:
            soc_util = max(0, sim.soc - sim.SOC_MIN)
            energia_restante = soc_util * sim.BATERIA_CAPACIDAD
            datos_hora = sim.get_data_window(sim.current_step - 1, horizon=1)[0]
            precio_compra_actual = datos_hora[2]
            reward += energia_restante * precio_compra_actual * sim.EFICIENCIA_DESCARGA

        # Info
        info = {
            "soc": r['soc'],
            "beneficio": r['beneficio'],
            "comprado": r['comprado'],
            "cargado": r['cargado'],
            "descargado": r['descargado'],
        }

        return self._get_obs(), reward, terminated, truncated, info
