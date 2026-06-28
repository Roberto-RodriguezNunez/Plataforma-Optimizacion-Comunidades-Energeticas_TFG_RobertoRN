"""
discrete_rl_controller.py — Controlador para agentes RL discretos (DQN/PPO)
============================================================================
Convierte la acción discreta (0-8) de un agente DQN o PPO al formato
de 4 flujos que espera eval_unificada.py.

La decodificación replica la lógica de simulador.ejecutar_accion_fisica()
pero solo devuelve los flujos intendidos, sin ejecutar física
(simular_hora_mpc se encarga en eval_unificada).

Uso:
    controller = DiscreteRLController('models/dqn_sgec.zip', sim,
                                       vec_normalize_path='models/vec_normalize.pkl',
                                       algo='DQN')
    action = controller.solve(state, forecast)
"""

import os
import sys
from typing import Dict, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.controllers.base import BaseController
from src.core.obs_builder import build_obs

# Mapa de 9 acciones discretas (idéntico a simulador.py)
ACTION_MAP = {
    0: ("IDLE",           0.0),
    1: ("CARGAR_SOLAR",   1.0),
    2: ("CARGAR_MIXTA",   0.33),
    3: ("CARGAR_MIXTA",   0.66),
    4: ("CARGAR_MIXTA",   1.0),
    5: ("DESCARGAR_CASA", 1.0),
    6: ("DESCARGAR_RED",  0.33),
    7: ("DESCARGAR_RED",  0.66),
    8: ("DESCARGAR_RED",  1.0),
}


class DiscreteRLController(BaseController):
    """
    Controlador para agentes DQN/PPO con acción discreta (9 acciones).

    En solve():
      1. Construye obs 108-dim desde state + forecast
      2. Normaliza con VecNormalize stats
      3. Predice acción discreta (0-8)
      4. Decodifica a dict de 4 flujos kW

    Args:
        model_path: Ruta al modelo SB3 (.zip).
        sim: ComunidadSimulador (para parámetros físicos y datos).
        vec_normalize_path: Ruta a VecNormalize stats (.pkl).
        algo: 'DQN' o 'PPO'.
    """

    def __init__(
        self,
        model_path: str,
        sim,
        vec_normalize_path: Optional[str] = None,
        algo: str = 'DQN',
    ):
        self._sim = sim
        self._algo = algo.upper()
        self._P_MAX = sim.POTENCIA_INVERSOR
        self._EFF_C = sim.EFICIENCIA_CARGA
        self._EFF_D = sim.EFICIENCIA_DESCARGA

        # Cargar modelo SB3
        if self._algo == 'DQN':
            from stable_baselines3 import DQN
            self._model = DQN.load(model_path, device='cpu')
        else:
            from stable_baselines3 import PPO
            self._model = PPO.load(model_path, device='cpu')

        # Cargar stats de normalización
        self._obs_rms = None
        self._clip_obs = 10.0
        self._obs_dim = 108  # default
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            import pickle
            with open(vec_normalize_path, 'rb') as f:
                vec_norm = pickle.load(f)
            self._obs_rms = vec_norm.obs_rms
            self._clip_obs = vec_norm.clip_obs
            self._obs_dim = len(vec_norm.obs_rms.mean)

    def solve(self, state: Dict, forecast: np.ndarray) -> Dict[str, float]:
        # 1. Construir obs 108-dim
        obs = self._build_obs(state, forecast)

        # 2. Normalizar
        if self._obs_rms is not None:
            obs = np.clip(
                (obs - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + 1e-8),
                -self._clip_obs, self._clip_obs
            ).astype(np.float32)

        # 3. Predecir acción discreta
        action_idx, _ = self._model.predict(obs, deterministic=True)
        action_idx = int(action_idx)

        # 4. Decodificar a 4 flujos
        return self._decode_action(action_idx, state)

    def nombre(self) -> str:
        return f"{self._algo}"

    def _decode_action(self, action_idx: int, state: Dict) -> Dict[str, float]:
        """
        Convierte acción discreta (0-8) a dict de 4 flujos kW.
        Replica la lógica de simulador.ejecutar_accion_fisica().
        """
        sim = self._sim
        step = state['step']
        soc = state['soc']

        # Datos de la hora actual
        row = sim.df.iloc[step]
        gen = row['generacion_total']
        cons = row['consumo_total']
        balance = gen - cons
        exc_disp = max(0.0, balance)
        def_cub = max(0.0, -balance)

        # Estado batería (con autodescarga, igual que simular_hora_mpc)
        soc_ad = soc * (1 - sim.AUTODESCARGA_POR_HORA)
        bateria_kwh = soc_ad * sim.BATERIA_CAPACIDAD
        espacio_libre = max(0.0, sim.SOC_MAX * sim.BATERIA_CAPACIDAD - bateria_kwh)
        bat_disponible = max(0.0, bateria_kwh - sim.SOC_MIN * sim.BATERIA_CAPACIDAD)

        estrategia, nivel = ACTION_MAP[action_idx]
        P_obj = nivel * self._P_MAX

        cs, cm, dc, dr = 0.0, 0.0, 0.0, 0.0

        if estrategia == "IDLE":
            pass  # No hay flujos de batería

        elif estrategia == "CARGAR_SOLAR":
            max_entrada = espacio_libre / self._EFF_C
            cs = min(exc_disp, max_entrada)

        elif estrategia == "CARGAR_MIXTA":
            max_entrada = espacio_libre / self._EFF_C
            carga_total = min(max_entrada, P_obj)
            cs = min(exc_disp, carga_total)
            cm = carga_total - cs

        elif estrategia == "DESCARGAR_CASA":
            dc = min(def_cub / self._EFF_D, bat_disponible)

        elif estrategia == "DESCARGAR_RED":
            descarga_total = min(bat_disponible, P_obj)
            energia_util = descarga_total * self._EFF_D
            para_casa = min(energia_util, def_cub)
            para_red = energia_util - para_casa
            dc = para_casa / self._EFF_D if self._EFF_D > 0 else 0.0
            dr = para_red / self._EFF_D if self._EFF_D > 0 else 0.0

        return {
            'P_carga_solar': cs,
            'P_carga_red': cm,
            'P_descarga_casa': dc,
            'P_descarga_red': dr,
        }

    def _build_obs(self, state: Dict, forecast: np.ndarray) -> np.ndarray:
        """
        Construye la observación delegando en la ÚNICA fuente `build_obs`
        (la misma que usan el entreno y el controller SAC), garantizando que
        DQN/PPO ven exactamente el mismo forecast/obs que el resto del sistema.

        La obs base es de 108 dims con el orden [5 estado | 96 forecast |
        6 temporal | 1 margen]. Las dimensiones legacy (101 = 5+96, 107 = 5+96+6)
        son un prefijo exacto, por lo que basta con recortar para modelos antiguos.
        """
        obs = build_obs(state, forecast, self._sim)  # 108-dim, fuente única
        if self._obs_dim < 108:
            obs = obs[:self._obs_dim]                # legacy 101/107 = prefijo
        return obs.astype(np.float32)
