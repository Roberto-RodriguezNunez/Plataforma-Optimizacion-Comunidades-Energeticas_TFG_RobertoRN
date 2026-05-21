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

    def __init__(self, forecast_noise: bool = True, mode: str = 'all'):
        super().__init__(forecast_noise=forecast_noise, mode=mode)

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
            self._error_solar = (
                self._RHO_SOLAR * self._error_solar
                + np.sqrt(1 - self._RHO_SOLAR ** 2) * np.random.normal()
            )
            self._error_cons = (
                self._RHO_CONS * self._error_cons
                + np.sqrt(1 - self._RHO_CONS ** 2) * np.random.normal()
            )
        self._skip_next_ar1 = False  # Resetear flag

        # 1. Leer datos de la hora actual
        sim = self.simulador
        step = sim.current_step
        row = sim.df.iloc[step]
        gen = row['generacion_total']
        cons = row['consumo_total']
        precio_compra = row['precio_kwh']
        precio_venta = row['precio_excedente']

        balance = gen - cons
        exc_disp = max(0.0, balance)
        def_cub = max(0.0, -balance)

        # 2. Autodescarga
        sim.soc *= (1 - sim.AUTODESCARGA_POR_HORA)
        bateria_kwh = sim.soc * sim.BATERIA_CAPACIDAD
        espacio_libre = max(0.0, sim.SOC_MAX * sim.BATERIA_CAPACIDAD - bateria_kwh)
        bat_disponible = max(0.0, bateria_kwh - sim.SOC_MIN * sim.BATERIA_CAPACIDAD)

        # 3. Recibir 4 flujos directamente (kW)
        cs = max(0.0, float(action[0]))
        cm = max(0.0, float(action[1]))
        dc = max(0.0, float(action[2]))
        dr = max(0.0, float(action[3]))

        # 3b. Netear carga vs descarga — inversor bidireccional ejecuta
        #     potencia neta, carga y descarga simultanea es imposible.
        #     Preserva el ratio interno (solar/red o casa/red).
        carga_bruta = cs + cm
        descarga_bruta = dc + dr
        net = carga_bruta - descarga_bruta

        if net >= 0:
            ratio_solar = cs / carga_bruta if carga_bruta > 0 else 0.0
            cs = net * ratio_solar
            cm = net * (1 - ratio_solar)
            dc, dr = 0.0, 0.0
        else:
            ratio_casa = dc / descarga_bruta if descarga_bruta > 0 else 0.0
            dc = abs(net) * ratio_casa
            dr = abs(net) * (1 - ratio_casa)
            cs, cm = 0.0, 0.0

        # 4. Recortar por estado real (identico a simular_hora_mpc lineas 372-379)
        cs = min(cs, exc_disp, espacio_libre / sim.EFICIENCIA_CARGA)
        cm = min(cm, max(0.0, espacio_libre / sim.EFICIENCIA_CARGA - cs))
        carga_total = cs + cm

        dc = min(dc, bat_disponible)
        dr = min(dr, max(0.0, bat_disponible - dc))
        descarga_total = dc + dr

        # 5. Actualizar bateria
        soc_antes = sim.soc
        bateria_kwh += carga_total * sim.EFICIENCIA_CARGA - descarga_total
        sim.soc = float(np.clip(bateria_kwh / sim.BATERIA_CAPACIDAD, 0.0, 1.0))

        # 6. Economia (identica a simular_hora_mpc)
        comprado = max(0.0, def_cub - dc * sim.EFICIENCIA_DESCARGA) + cm
        vendido = (exc_disp - cs) + dr * sim.EFICIENCIA_DESCARGA

        ingresos = vendido * precio_venta
        gastos = comprado * precio_compra

        # Degradacion no lineal completa
        energia_movida = carga_total + descarga_total
        soc_medio = (soc_antes + sim.soc) / 2
        coste_deg = sim.calcular_degradacion_no_lineal(energia_movida, soc_medio)

        beneficio = ingresos - gastos - coste_deg

        # Baseline IDLE
        beneficio_idle = exc_disp * precio_venta - def_cub * precio_compra
        reward = beneficio - beneficio_idle

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
            "soc": sim.soc,
            "beneficio": beneficio,
            "comprado": comprado,
            "cargado": carga_total,
            "descargado": descarga_total,
        }

        return self._get_obs(), reward, terminated, truncated, info
