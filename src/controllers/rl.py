"""
rl.py — Controladores RL de inferencia directa sobre el entorno (sin MPC)
=========================================================================
Dos controladores que mapean la observación a flujos directamente con un modelo
SB3, sin MPC base ni corrección residual:

  - DiscreteRLController: agentes discretos (DQN/PPO) — acción 0-8 → 4 flujos
    vía `ComunidadSimulador.accion_a_flujos` (fuente única del decode).
  - ContinuousController: agentes continuos (SAC puro / PPO continuo) — la
    política emite los 4 flujos en [0, P_MAX] directamente.

Ambos comparten la observación con todo el sistema vía `observation.build_obs`
→ entreno == evaluación. (Antes en discrete_rl_controller.py y continuous_controller.py.)
"""

import os
import sys
from typing import Dict, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.controllers.base import BaseController
from src.core.observation import build_obs

_ALGOS_CONTINUOS = ('SAC', 'PPO')


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

        # 4. Decodificar a 4 flujos con la FUENTE ÚNICA (misma que el entreno).
        return self._decode_action(action_idx, forecast)

    def nombre(self) -> str:
        return f"{self._algo}"

    def _decode_action(self, action_idx: int, forecast: np.ndarray) -> Dict[str, float]:
        """Convierte la acción discreta (0-8) a 4 flujos kW vía `accion_a_flujos`.

        gen/cons de la hora actual se toman de `forecast[0]` (la MISMA ventana
        observada con ruido que reciben el MPC y los controladores residual/continuo),
        NO del dato real del dataset → comparación justa. Los flujos van sin recortar
        por batería: el recorte lo hace `aplicar_fisica_4flujos` (vía simular_hora_mpc),
        igual que en el entreno (cols de la ventana: 0=consumo, 1=generacion).
        """
        cons = float(forecast[0, 0])
        gen = float(forecast[0, 1])
        exc_disp = max(0.0, gen - cons)
        def_cub = max(0.0, cons - gen)

        cs, cm, dc, dr = self._sim.accion_a_flujos(
            action_idx, exc_disp, def_cub, self._P_MAX, self._EFF_C, self._EFF_D)

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
        # F5: el recorte solo es válido para dims legacy que son PREFIJO exacto de
        # la obs de 108 (101 = 5+96, 107 = 5+96+6). Si el layout cambiara, fallar
        # en vez de meter features equivocadas en silencio.
        assert self._obs_dim in (101, 107, 108), (
            f"obs_dim {self._obs_dim} no es un prefijo legacy conocido de la obs de 108"
        )
        assert len(obs) >= self._obs_dim, "build_obs devolvió menos dims de las esperadas"
        if self._obs_dim < 108:
            obs = obs[:self._obs_dim]                # legacy 101/107 = prefijo
        return obs.astype(np.float32)


class ContinuousController(BaseController):
    """
    Controlador continuo directo (sin MPC). `model.predict(obs_108) -> 4 flujos`.

    Args:
        model_path: Ruta al modelo (.zip).
        sim: ComunidadSimulador (parámetros físicos; P_MAX para el clip).
        vec_normalize_path: Stats de VecNormalize (.pkl). None → sin normalizar.
        algo: 'SAC' | 'PPO'.
    """

    def __init__(
        self,
        model_path: str,
        sim,
        vec_normalize_path: Optional[str] = None,
        algo: str = 'SAC',
    ):
        import stable_baselines3 as sb3

        algo = algo.upper()
        if algo not in _ALGOS_CONTINUOS:
            raise ValueError(f"algo '{algo}' no soportado; usa uno de {_ALGOS_CONTINUOS}")

        self._sim = sim
        self._algo = algo
        self._P_MAX = sim.POTENCIA_INVERSOR
        self._model = getattr(sb3, algo).load(model_path, device='cpu')

        self._mean = None
        self._var = None
        self._clip_obs = 10.0
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            import pickle
            with open(vec_normalize_path, 'rb') as f:
                vec_norm = pickle.load(f)
            self._mean = vec_norm.obs_rms.mean
            self._var = vec_norm.obs_rms.var
            self._clip_obs = vec_norm.clip_obs

    def solve(self, state: Dict, forecast: np.ndarray) -> Dict[str, float]:
        obs = build_obs(state, forecast, self._sim).astype(np.float32)  # 108
        if self._mean is not None:
            obs = np.clip(
                (obs - self._mean) / np.sqrt(self._var + 1e-8),
                -self._clip_obs, self._clip_obs,
            ).astype(np.float32)

        action, _ = self._model.predict(obs, deterministic=True)
        # Acción ya en [0, P_MAX] (SAC squash); clip por seguridad (PPO sin bound).
        a = np.clip(action, 0.0, self._P_MAX)
        return {
            'P_carga_solar':   float(a[0]),
            'P_carga_red':     float(a[1]),
            'P_descarga_casa': float(a[2]),
            'P_descarga_red':  float(a[3]),
        }

    def nombre(self) -> str:
        return f"Continuo{self._algo}"
