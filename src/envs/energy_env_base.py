"""
energy_env_base.py — Andamiaje común de los entornos Gymnasium
==============================================================
Toda la lógica compartida entre el entorno discreto y el continuo vive aquí:
reset (semana + SoC inicial + coste inicial), ruido AR(1) de pronóstico,
construcción de la observación (108 dims) y gestión de episodio.

NO define action_space ni step(): cada subclase concreta los aporta
(EnergyEnvDiscreto · EnergyEnvContinuo), de modo que ninguna depende de la otra.

El vector de observación incluye 24h de pronóstico con ruido realista (σ solar
5%→25%, consumo ~15%, precio 3 capas) + 6 features temporales sin/cos + margen
solar. forecast_noise=False desactiva el ruido (oráculo / benchmark).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.core.simulator import ComunidadSimulador
from src.core.forecast import ventana_observada, generar_factores_precio
from src.core.observation import build_obs, OBS_DIM_BASE


class EnergyEnvBase(gym.Env):
    """
    Base abstracta: estado, reset, ruido AR(1) y observación compartidos.
    No instanciable por sí sola (no fija action_space ni step).
    """

    # Ruta al dataset (misma que usa ComunidadSimulador)
    _DATASET_PATH = 'data/processed/dataset_final.csv'
    # Las constantes de ruido AR(1)/precio viven en la FUENTE ÚNICA src.core.forecast
    # (cargadas de config/system.yaml); el env consume sus funciones, no las redefine.

    def __init__(self, forecast_noise: bool = True, mode: str = 'all', rng=None):
        super().__init__()

        self.forecast_noise = forecast_noise
        self._mode = mode
        # RNG: si se pasa un np.random.Generator seeded, se usa para todo
        # el ruido AR(1). Si None, usa np.random global (backward compatible).
        self._rng = rng
        # Estados AR(1) del error de pronóstico (se resetean en cada episodio)
        self._error_solar = 0.0
        self._error_cons = 0.0
        # Factores de ruido de precio 3 capas (generados una vez por step,
        # reutilizados por _get_obs y por ResidualEnv._compute_mpc_action)
        self._precio_noise_factors = None  # None = no hay ruido cacheado

        # Instanciar el motor físico (con split por semanas si procede)
        self.simulador = ComunidadSimulador(self._DATASET_PATH, mode=mode)

        # Índice para iteración determinista sobre semanas eval
        self._eval_idx = 0

        # Índice temporal para features sin/cos (hora, día_semana, mes).
        try:
            _raw = pd.read_csv(self._DATASET_PATH, parse_dates=['fecha'])
            self._timestamps = _raw['fecha']
        except Exception:
            self._timestamps = None

        # --- ESTADO: 108 dims (5 actuales + 96 forecast + 6 temporales + 1 margen) ---
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM_BASE,), dtype=np.float32
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
        soc_inicial = np.random.uniform(
            self.simulador.SOC_MIN + 0.05,
            self.simulador.SOC_MAX - 0.05,
        )

        self.simulador.current_step = start_step
        self.simulador.soc = soc_inicial
        self.steps_in_episode = 0

        # Coste de la energía inicial (se resta en el primer step; simétrico con
        # el valor terminal) usando el SoC sorteado real.
        soc_util = max(0, soc_inicial - self.simulador.SOC_MIN)
        energia_inicial = soc_util * self.simulador.BATERIA_CAPACIDAD
        datos_hora = self.simulador.get_data_window(start_step, horizon=1)[0]
        precio_compra = datos_hora[2]
        self._pendiente_coste_inicial = (
            energia_inicial * precio_compra * self.simulador.EFICIENCIA_DESCARGA
        )

        # Resetear estados AR(1) del error de pronóstico
        self._error_solar = 0.0
        self._error_cons = 0.0
        self._precio_noise_factors = None

        return self._get_obs(), {}

    def _noise(self):
        """Genera N(0,1) usando el rng seeded si existe, o np.random global."""
        if self._rng is not None:
            return self._rng.standard_normal()
        return np.random.normal()

    def _get_hora_actual_from_step(self, step):
        """Obtiene hora del día (0-23) para un step."""
        if self._timestamps is not None and step < len(self._timestamps):
            return self._timestamps.iloc[step].hour
        return step % 24

    def _get_obs(self):
        """
        Vector de estado (108 dims) vía la ÚNICA fuente `observation.build_obs`
        sobre la ventana de la ÚNICA fuente `forecast.ventana_observada`. Así
        entreno == evaluación == producción, y MPC y agente reciben la misma
        ventana con el mismo ruido (sin lookahead).
        """
        t = self.simulador.current_step
        hora = self.simulador.get_tiempo(t)[0]

        # Factores de ruido de precio — una vez por step, cacheados para que
        # ResidualEnv._compute_mpc_action() y _get_obs() usen los mismos.
        if self.forecast_noise and self._precio_noise_factors is None:
            self._precio_noise_factors = generar_factores_precio(hora, self._noise)
        factores = self._precio_noise_factors if self.forecast_noise else None

        window = ventana_observada(
            self.simulador, t, self._error_solar, self._error_cons,
            factores, self.forecast_noise,
        )
        state = {'soc': self.simulador.soc, 'step': t}
        return build_obs(state, window, self.simulador)

    def step(self, action):
        raise NotImplementedError("Lo implementan EnergyEnvDiscreto / EnergyEnvContinuo")

    def render(self):
        pass
