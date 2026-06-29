"""
energy_env_discreto.py — Entorno Gymnasium con accion discreta (9 acciones)
===========================================================================
Hermano de EnergyEnvContinuo: ambos heredan el andamiaje de EnergyEnvBase.
La accion 0-8 se decodifica (ComunidadSimulador.accion_a_flujos) y se ejecuta
con la MISMA fisica de 4 flujos (aplicar_fisica_4flujos) que usa el continuo.

Lo usan los agentes discretos (DQN / PPO).
"""

from gymnasium import spaces

from src.envs.energy_env_base import EnergyEnvBase
from src.core.forecast import avanzar_ar1


class EnergyEnvDiscreto(EnergyEnvBase):
    """Entorno con accion discreta de 9 estrategias (ver justificacion_9_acciones.md)."""

    def __init__(self, forecast_noise: bool = True, mode: str = 'all', rng=None):
        super().__init__(forecast_noise=forecast_noise, mode=mode, rng=rng)
        # --- ACCIONES: 9 (ver justificacion_9_acciones.md) ---
        self.action_space = spaces.Discrete(9)

    def step(self, action):
        # Invalidar cache de factores de precio (se regeneran en _get_obs)
        self._precio_noise_factors = None

        # 0. Avanzar estado AR(1) del error de pronóstico antes de construir la obs
        if self.forecast_noise:
            self._error_solar, self._error_cons = avanzar_ar1(
                self._error_solar, self._error_cons, self._noise
            )

        # 1. Ejecutar en el simulador (decode 0-8 + fisica 4 flujos)
        resultado = self.simulador.ejecutar_accion_fisica(action, self.simulador.current_step)

        # 2. Recompensa (marginal vs IDLE)
        reward = resultado["beneficio_marginal"]
        if self.steps_in_episode == 0:
            reward -= self._pendiente_coste_inicial

        # 3. Avanzar tiempo
        self.simulador.current_step += 1
        self.steps_in_episode += 1

        # 4. Fin de episodio
        terminated = (self.steps_in_episode >= self.EPISODE_LENGTH)
        truncated = False

        # 5. Valor terminal: la energía restante en batería no se pierde
        if terminated:
            soc_util = max(0, self.simulador.soc - self.simulador.SOC_MIN)
            energia_restante = soc_util * self.simulador.BATERIA_CAPACIDAD
            datos_hora = self.simulador.get_data_window(self.simulador.current_step - 1, horizon=1)[0]
            precio_compra_actual = datos_hora[2]
            reward += energia_restante * precio_compra_actual * self.simulador.EFICIENCIA_DESCARGA

        info = {
            "soc":        resultado["soc"],
            "beneficio":  resultado["beneficio"],
            "comprado":   resultado["comprado"],
            "cargado":    resultado["cargado"],
            "descargado": resultado["descargado"],
        }
        return self._get_obs(), reward, terminated, truncated, info
