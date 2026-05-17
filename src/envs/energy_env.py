import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.core.simulador import ComunidadSimulador

class EnergyEnv(gym.Env):
    """
    Entorno Gymnasium compatible con Stable-Baselines3.

    El vector de observación incluye 24h de pronóstico con ruido realista:
    - generacion_total: ruido multiplicativo creciente con el horizonte
        sigma: 5% (hora+1) → 25% (hora+24), típico de forecasting solar NWP
    - consumo_total: ruido multiplicativo fijo ~10%
        (varianza se atenúa por agregación de 15 vecinos)
    - precio_kwh, precio_excedente: sin ruido
        (PVPC e ind. 1739 publicados por REE/ESIOS el día anterior)

    Parámetro forecast_noise=False desactiva el ruido (útil para benchmarks).
    """
    # Sigmas base de cada variable en la ventana de pronóstico
    # Orden columnas dataset: [consumo, generacion, precio_kwh, precio_excedente]
    _SIGMA_CONS_BASE  = 0.10   # 10% MAE — relativamente plano con el horizonte
    _SIGMA_SOL_H1     = 0.05   # 5%  error en hora+1
    _SIGMA_SOL_H24    = 0.25   # 25% error en hora+24  (crece linealmente)

    def __init__(self, forecast_noise: bool = True):
        super(EnergyEnv, self).__init__()

        self.forecast_noise = forecast_noise

        # Instanciar el motor físico
        # Asegúrate de haber ejecutado la Tarea 1 para tener este archivo
        self.simulador = ComunidadSimulador('data/processed/dataset_final.csv')
        
        # --- ACCIONES: 9 (ver justificacion_9_acciones.md) ---
        self.action_space = spaces.Discrete(9)
        
        # --- ESTADO: 101 Variables ---
        # 5 actuales (SoC, Precio_compra, Precio_venta, Excedente, Deficit)
        # + 96 futuras (24h * 4 variables: Consumo, Generacion, Precio_compra, Precio_venta)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(101,), dtype=np.float32
        )
        
        # Configuración del Episodio
        self.EPISODE_LENGTH = 24 * 7  # Episodios de 1 semana
        self.steps_in_episode = 0
        self._pendiente_coste_inicial = 0.0  # Se resta en el primer step

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Elegir inicio aleatorio con margen suficiente
        max_start = self.simulador.max_steps - self.EPISODE_LENGTH - 25
        start_step = np.random.randint(0, max_start)
        
        # Resetear simulador
        self.simulador.current_step = start_step
        self.simulador.soc = self.simulador.SOC_INICIAL
        self.steps_in_episode = 0

        # Calcular coste de la energía inicial (se resta en el primer step)
        soc_util = max(0, self.simulador.SOC_INICIAL - self.simulador.SOC_MIN)
        energia_inicial = soc_util * self.simulador.BATERIA_CAPACIDAD
        datos_hora = self.simulador.get_data_window(start_step, horizon=1)[0]
        precio_compra = datos_hora[2]
        self._pendiente_coste_inicial = energia_inicial * precio_compra * self.simulador.EFICIENCIA_DESCARGA

        return self._get_obs(), {}

    def _get_obs(self):
        """
        Construye el vector de estado completo (101 dimensiones).

        Primeras 5: estado actual exacto (SoC, precios, excedente, déficit).
        Siguientes 96: pronóstico 24h con ruido en generacion y consumo.
          - Los precios se dejan exactos (publicados por REE el día anterior).
          - El ruido del solar crece linealmente con el horizonte.
        """
        t = self.simulador.current_step

        # 1. Datos actuales — sin ruido (estado físico observado ahora mismo)
        datos_hoy = self.simulador.get_data_window(t, horizon=1)[0]
        cons, gen, precio_compra, precio_venta = datos_hoy

        balance = gen - cons
        exc = max(0, balance)
        def_ = abs(min(0, balance))

        # 2. Pronóstico 24h futuras
        # Columnas: [consumo, generacion, precio_kwh, precio_excedente]
        window_future = self.simulador.get_data_window(t + 1, horizon=24).copy()

        if self.forecast_noise:
            for h in range(24):
                # sigma solar crece linealmente: 5% (h=0) → 25% (h=23)
                sigma_sol = self._SIGMA_SOL_H1 + h * (
                    (self._SIGMA_SOL_H24 - self._SIGMA_SOL_H1) / 23
                )
                # Consumo: sigma fija (patrón agregado más estable)
                sigma_cons = self._SIGMA_CONS_BASE

                # Ruido multiplicativo; clip a 0 (no puede haber valores negativos)
                window_future[h, 1] = max(  # generacion
                    0.0, window_future[h, 1] * (1.0 + np.random.normal(0, sigma_sol))
                )
                window_future[h, 0] = max(  # consumo
                    0.0, window_future[h, 0] * (1.0 + np.random.normal(0, sigma_cons))
                )
                # columnas 2 y 3 (precios) — sin tocar

        forecast_flat = window_future.flatten()  # 24 * 4 = 96 valores

        # 3. Concatenar (5 + 96 = 101)
        obs = np.concatenate((
            [self.simulador.soc, precio_compra, precio_venta, exc, def_],
            forecast_flat,
        ))
        return obs.astype(np.float32)

    def step(self, action):
        # 1. Ejecutar en el simulador
        resultado = self.simulador.ejecutar_accion_fisica(action, self.simulador.current_step)
        
        # 2. Obtener Recompensa (marginal vs IDLE para reducir varianza)
        reward = resultado["beneficio_marginal"]

        # Restar valor de la energía inicial en el primer paso (simetría con valor terminal)
        if self.steps_in_episode == 0:
            reward -= self._pendiente_coste_inicial
        
        # 3. Avanzar tiempo
        self.simulador.current_step += 1
        self.steps_in_episode += 1
        
        # 4. Comprobar fin
        terminated = (self.steps_in_episode >= self.EPISODE_LENGTH)
        truncated = False

        # 5. Valor terminal: la energía en batería al final no se pierde
        # Valorada al precio medio de compra evitada (PVPC medio del dataset)
        if terminated:
            soc_util = max(0, self.simulador.soc - self.simulador.SOC_MIN)
            energia_restante = soc_util * self.simulador.BATERIA_CAPACIDAD
            # Valorar a precio de compra actual (lo que costaría recargarla)
            datos_hora = self.simulador.get_data_window(self.simulador.current_step - 1, horizon=1)[0]
            precio_compra_actual = datos_hora[2]
            reward += energia_restante * precio_compra_actual * self.simulador.EFICIENCIA_DESCARGA

        # Info extra (útil para gráficas luego)
        info = {
            "soc":        resultado["soc"],
            "beneficio":  resultado["beneficio"],
            "comprado":   resultado["comprado"],
            "cargado":    resultado["cargado"],
            "descargado": resultado["descargado"],
        }

        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        pass