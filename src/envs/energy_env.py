import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
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

    Features temporales (6 dims): codificación cíclica sin/cos de hora, día_semana
    y mes — permiten al agente aprender patrones intradiarios de precio/solar.

    Parámetro forecast_noise=False desactiva el ruido (útil para benchmarks).
    """

    # Ruta al dataset (misma que usa ComunidadSimulador)
    _DATASET_PATH = 'data/processed/dataset_final.csv'
    # Sigmas base de cada variable en la ventana de pronóstico
    # Orden columnas dataset: [consumo, generacion, precio_kwh, precio_excedente]
    _SIGMA_CONS_BASE  = 0.10   # 10% MAE — relativamente plano con el horizonte
    _SIGMA_SOL_H1     = 0.05   # 5%  error en hora+1
    _SIGMA_SOL_H24    = 0.25   # 25% error en hora+24  (crece linealmente)

    # Autocorrelación temporal del error de pronóstico (proceso AR(1))
    # Solar: ρ=0.7 — las nubes persisten varias horas
    # Consumo: ρ=0.3 — más variable, patron horario domina sobre inercia
    _RHO_SOLAR = 0.7
    _RHO_CONS  = 0.3

    def __init__(self, forecast_noise: bool = True, mode: str = 'all'):
        super(EnergyEnv, self).__init__()

        self.forecast_noise = forecast_noise
        # Estados AR(1) del error de pronóstico (se resetean en cada episodio)
        self._error_solar = 0.0
        self._error_cons  = 0.0

        # Instanciar el motor físico (con split temporal si procede)
        self.simulador = ComunidadSimulador(self._DATASET_PATH, mode=mode)

        # Cargar índice temporal para features sin/cos (hora, día_semana, mes).
        # El CSV debe tener una columna 'fecha' (generada por generar_dataset_final.py).
        try:
            _raw = pd.read_csv(self._DATASET_PATH, parse_dates=['fecha'])
            if mode != 'all' and 'fecha' in _raw.columns:
                if mode == 'train':
                    _raw = _raw[_raw['fecha'] <= '2023-06-30 23:00:00']
                elif mode == 'test':
                    _raw = _raw[_raw['fecha'] >= '2023-09-01 00:00:00']
                _raw = _raw.reset_index(drop=True)
            self._timestamps = _raw['fecha']
        except Exception:
            self._timestamps = None

        # --- ACCIONES: 9 (ver justificacion_9_acciones.md) ---
        self.action_space = spaces.Discrete(9)

        # --- ESTADO: 108 Variables ---
        # 5 actuales (SoC, Precio_compra, Precio_venta, Excedente, Deficit)
        # + 96 futuras (24h * 4 variables: Consumo, Generacion, Precio_compra, Precio_venta)
        # + 6 temporales (sin/cos hora, sin/cos dia_semana, sin/cos mes)
        # + 1 margen_solar (solar excedente que desbordará la batería en 24h / CAP)
        #     Alta → D_RED útil para hacer hueco; Cero → D_RED probablemente innecesario
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(108,), dtype=np.float32
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

        # SoC inicial aleatorio entre SOC_MIN+5% y SOC_MAX-5%
        # Evita extremos para no empezar con batería bloqueada
        soc_inicial = np.random.uniform(
            self.simulador.SOC_MIN + 0.05,
            self.simulador.SOC_MAX - 0.05,
        )

        self.simulador.current_step = start_step
        self.simulador.soc = soc_inicial
        self.steps_in_episode = 0

        # Calcular coste de la energía inicial usando el SoC sorteado real.
        # Se resta en el primer step para que el agente no reciba un "regalo"
        # por empezar con la batería cargada. Simétrico con el valor terminal.
        soc_util = max(0, soc_inicial - self.simulador.SOC_MIN)
        energia_inicial = soc_util * self.simulador.BATERIA_CAPACIDAD
        datos_hora = self.simulador.get_data_window(start_step, horizon=1)[0]
        precio_compra = datos_hora[2]
        self._pendiente_coste_inicial = (
            energia_inicial * precio_compra * self.simulador.EFICIENCIA_DESCARGA
        )

        # Resetear estados AR(1) del error de pronóstico
        self._error_solar = 0.0
        self._error_cons  = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        """
        Construye el vector de estado completo (108 dimensiones).

        Primeras 5: estado actual exacto (SoC, precios, excedente, déficit).
        Siguientes 96: pronóstico 24h con ruido en generacion y consumo.
          - Los precios se dejan exactos (publicados por REE el día anterior).
          - El ruido del solar crece linealmente con el horizonte.
        Siguientes 6: codificación cíclica del tiempo (sin/cos hora, día_semana, mes).
          - Permiten al agente aprender patrones intradiarios de precio y solar.
        Última 1: margen_solar — solar previsto que desbordará la batería en 24h
          (normalizado por capacidad). Alta → D_RED hace hueco útil; 0 → D_RED innecesario.
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
            # _error_solar y _error_cons son estados AR(1) N(0,1) actualizados
            # en step(). Aquí se escalan por el sigma de cada hora del horizonte.
            for h in range(24):
                sigma_sol = self._SIGMA_SOL_H1 + h * (
                    (self._SIGMA_SOL_H24 - self._SIGMA_SOL_H1) / 23
                )
                window_future[h, 1] = max(  # generacion solar
                    0.0, window_future[h, 1] * (1.0 + self._error_solar * sigma_sol)
                )
                window_future[h, 0] = max(  # consumo
                    0.0, window_future[h, 0] * (1.0 + self._error_cons * self._SIGMA_CONS_BASE)
                )
                # columnas 2 y 3 (precios PVPC e ind.1739) — sin tocar

        forecast_flat = window_future.flatten()  # 24 * 4 = 96 valores

        # 3. Features temporales — codificación cíclica sin/cos (6 dims)
        if self._timestamps is not None and t < len(self._timestamps):
            ts = self._timestamps.iloc[t]
            hora    = ts.hour
            dia_sem = ts.dayofweek   # 0=lunes … 6=domingo
            mes     = ts.month - 1   # 0–11
        else:
            # Fallback: estimación por paso (asume step=0 → hora 0, día 0)
            hora    = t % 24
            dia_sem = (t // 24) % 7
            mes     = 0

        temp_feats = np.array([
            np.sin(2 * np.pi * hora    / 24),
            np.cos(2 * np.pi * hora    / 24),
            np.sin(2 * np.pi * dia_sem /  7),
            np.cos(2 * np.pi * dia_sem /  7),
            np.sin(2 * np.pi * mes     / 12),
            np.cos(2 * np.pi * mes     / 12),
        ], dtype=np.float32)

        # 4. Margen solar — excedente que desbordará la batería en 24h (1 dim)
        # Columnas window_future: [consumo, generacion, precio_kwh, precio_excedente]
        # Usamos la ventana SIN ruido para que la señal sea limpia (precios ya lo son;
        # para solar usamos window_future antes del ruido — recalculamos con datos crudos)
        window_clean = self.simulador.get_data_window(t + 1, horizon=24)
        solar_exc_24h = float(np.sum(np.maximum(0.0, window_clean[:, 1] - window_clean[:, 0])))
        espacio_bat   = max(0.0, (self.simulador.SOC_MAX - self.simulador.soc)
                           * self.simulador.BATERIA_CAPACIDAD)
        margen_solar  = np.float32(
            max(0.0, solar_exc_24h - espacio_bat) / self.simulador.BATERIA_CAPACIDAD
        )

        # 5. Concatenar (5 + 96 + 6 + 1 = 108)
        obs = np.concatenate((
            [self.simulador.soc, precio_compra, precio_venta, exc, def_],
            forecast_flat,
            temp_feats,
            [margen_solar],
        ))
        return obs.astype(np.float32)

    def step(self, action):
        # 0. Avanzar estado AR(1) del error de pronóstico antes de construir la obs
        #    ε_t = ρ·ε_{t-1} + √(1-ρ²)·N(0,1)  →  varianza estacionaria = 1
        if self.forecast_noise:
            self._error_solar = (
                self._RHO_SOLAR * self._error_solar
                + np.sqrt(1 - self._RHO_SOLAR ** 2) * np.random.normal()
            )
            self._error_cons = (
                self._RHO_CONS * self._error_cons
                + np.sqrt(1 - self._RHO_CONS ** 2) * np.random.normal()
            )

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