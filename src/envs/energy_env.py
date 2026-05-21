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
    _SIGMA_CONS_BASE  = 0.15   # 15% MAE — relativamente plano con el horizonte
    # Justificación σ=0.15: el perfil 2.0TD (ESIOS) es la media nacional de
    # millones de hogares. Con solo 15 viviendas la diversificación es menor y
    # la variabilidad local no está capturada. La literatura reporta errores de
    # 10-20% para grupos pequeños (Haben et al. 2016). [VERIFICAR CITA]
    _SIGMA_SOL_H1     = 0.05   # 5%  error en hora+1
    _SIGMA_SOL_H24    = 0.25   # 25% error en hora+24  (crece linealmente)

    # Autocorrelación temporal del error de pronóstico (proceso AR(1))
    # Solar: ρ=0.7 — las nubes persisten varias horas
    # Consumo: ρ=0.3 — más variable, patron horario domina sobre inercia
    _RHO_SOLAR = 0.7
    _RHO_CONS  = 0.3

    # Ruido de precio PVPC — modelo 3 capas
    # PVPC se publica a las 20:30h del día anterior por REE.
    # Capa 1: ya publicado → exacto. Capa 2: OMIE intradiario. Capa 3: estimación.
    _HORA_PUBLICACION    = 20.5
    _SIGMA_PRECIO_INTRA  = 0.05   # Capa 2: ~5% error (mercados intradiarios OMIE)
    _SIGMA_PRECIO_STAT   = 0.15   # Capa 3: ~15% error (estimación estadística)
    _RHO_PRECIO_INTRA    = 0.5    # Capa 2: ρ=0.5 (correcciones OMIE frecuentes)
    _RHO_PRECIO_STAT     = 0.7    # Capa 3: ρ=0.7 (condiciones mercado persisten)
    _MARGEN_INTRA_H      = 6      # Horas cubiertas por OMIE intraday

    def __init__(self, forecast_noise: bool = True, mode: str = 'all',
                 rng=None):
        super(EnergyEnv, self).__init__()

        self.forecast_noise = forecast_noise
        self._mode = mode
        # RNG: si se pasa un np.random.Generator seeded, se usa para todo
        # el ruido AR(1). Si None, usa np.random global (backward compatible).
        self._rng = rng
        # Estados AR(1) del error de pronóstico (se resetean en cada episodio)
        self._error_solar = 0.0
        self._error_cons  = 0.0
        # Factores de ruido de precio 3 capas (generados una vez por step,
        # reutilizados por _get_obs y por ResidualEnv._compute_mpc_action)
        self._precio_noise_factors = None  # None = no hay ruido cacheado

        # Instanciar el motor físico (con split por semanas si procede)
        self.simulador = ComunidadSimulador(self._DATASET_PATH, mode=mode)

        # Índice para iteración determinista sobre semanas eval
        self._eval_idx = 0

        # Cargar índice temporal para features sin/cos (hora, día_semana, mes).
        # El CSV debe tener una columna 'fecha' (generada por generar_dataset_final.py).
        # Se usa el dataset completo (no filtrado) — el split está en los pools
        # de semanas del simulador, no en el dataframe.
        try:
            _raw = pd.read_csv(self._DATASET_PATH, parse_dates=['fecha'])
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

        # Elegir semana de inicio según el modo
        if self._mode == 'eval':
            # Determinista: recorre las 50 semanas eval en orden cíclico
            start_step = self.simulador.semanas_eval[
                self._eval_idx % len(self.simulador.semanas_eval)]
            self._eval_idx += 1
        elif self._mode == 'train':
            # Aleatorio: muestrear una semana del pool train
            start_step = self.simulador.semanas_train[
                np.random.randint(len(self.simulador.semanas_train))]
        else:
            # mode='all': comportamiento original
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
        self._precio_noise_factors = None

        return self._get_obs(), {}

    def _noise(self):
        """Genera N(0,1) usando el rng seeded si existe, o np.random global."""
        if self._rng is not None:
            return self._rng.standard_normal()
        return np.random.normal()

    def _generar_precio_noise(self, hora_actual):
        """
        Genera factores multiplicativos de ruido de precio para 24 posiciones
        de forecast (offsets 0..23 horas adelante desde hora_actual).

        El AR(1) avanza POR HORA dentro del forecast (draw nuevo cada hora),
        con ρ distinto por capa. Esto cambia el ranking de precios entre
        horas, no solo el nivel general.

        Exactamente 24 draws — coincide con aplicar_ruido_precio_3capas()
        del MPC benchmark (offset=0), garantizando coherencia RNG.

        - MPC (offset=0): usa factors[0..23] directamente
        - _get_obs (offset=1): usa factors[h+1] con clamp a 23
        """
        if hora_actual >= self._HORA_PUBLICACION:
            horas_publicadas = 24 + (24 - hora_actual)
        else:
            horas_publicadas = 24 - hora_actual

        factors = np.ones(24)
        eps = 0.0
        for h_ahead in range(24):
            if h_ahead < horas_publicadas:
                continue  # Capa 1: precio publicado exacto
            elif h_ahead < horas_publicadas + self._MARGEN_INTRA_H:
                rho = self._RHO_PRECIO_INTRA    # Capa 2: ρ=0.5
                sigma = self._SIGMA_PRECIO_INTRA
            else:
                rho = self._RHO_PRECIO_STAT     # Capa 3: ρ=0.7
                sigma = self._SIGMA_PRECIO_STAT
            eps = rho * eps + np.sqrt(1 - rho**2) * self._noise()
            factors[h_ahead] = 1.0 + sigma * eps
        return factors

    def _get_hora_actual_from_step(self, step):
        """Obtiene hora del día (0-23) para un step."""
        if self._timestamps is not None and step < len(self._timestamps):
            return self._timestamps.iloc[step].hour
        return step % 24

    def _get_obs(self):
        """
        Construye el vector de estado completo (108 dimensiones).

        Primeras 5: estado actual exacto (SoC, precios, excedente, déficit).
        Siguientes 96: pronóstico 24h con ruido en generacion, consumo y precios.
          - El ruido del solar crece linealmente con el horizonte.
          - Los precios usan modelo 3 capas según publicación PVPC.
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

        # Obtener hora actual del día (necesario para ruido precios y features)
        if self._timestamps is not None and t < len(self._timestamps):
            ts = self._timestamps.iloc[t]
            hora    = ts.hour
            dia_sem = ts.dayofweek   # 0=lunes … 6=domingo
            mes     = ts.month - 1   # 0–11
        else:
            hora    = t % 24
            dia_sem = (t // 24) % 7
            mes     = 0

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

            # Ruido de precio — modelo 3 capas según publicación PVPC
            # Los factores se generan una vez por step y se cachean en
            # _precio_noise_factors para que ResidualEnv._compute_mpc_action()
            # y _get_obs() usen exactamente los mismos valores.
            if self._precio_noise_factors is None:
                self._precio_noise_factors = self._generar_precio_noise(hora)
            # window_future[h] = hora h+1 adelante → factors[min(h+1, 23)]
            for h in range(24):
                f = self._precio_noise_factors[min(h + 1, 23)]
                if f != 1.0:
                    window_future[h, 2] = max(0.0, window_future[h, 2] * f)
                    window_future[h, 3] = max(0.0, window_future[h, 3] * f)

        forecast_flat = window_future.flatten()  # 24 * 4 = 96 valores

        # 3. Features temporales — codificación cíclica sin/cos (6 dims)

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
        # Invalidar cache de factores de precio (se regeneran en _get_obs)
        self._precio_noise_factors = None

        # 0. Avanzar estado AR(1) del error de pronóstico antes de construir la obs
        #    ε_t = ρ·ε_{t-1} + √(1-ρ²)·N(0,1)  →  varianza estacionaria = 1
        if self.forecast_noise:
            self._error_solar = (
                self._RHO_SOLAR * self._error_solar
                + np.sqrt(1 - self._RHO_SOLAR ** 2) * self._noise()
            )
            self._error_cons = (
                self._RHO_CONS * self._error_cons
                + np.sqrt(1 - self._RHO_CONS ** 2) * self._noise()
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