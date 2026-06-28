"""
continuous_controller.py — Controlador continuo PURO (sin MPC ni residual)
==========================================================================
Para los agentes continuos directos entrenados sobre EnergyEnvContinuo
(SAC puro y PPO continuo): la política mapea la observación de 108 dims a los
4 flujos físicos directamente, en [0, P_MAX]. No hay MPC base ni corrección δ.

Comparte la observación con todo lo demás vía obs_builder.build_obs → mismo
estado que en el entreno (entreno == evaluación).
"""

import os
from typing import Dict, Optional

import numpy as np

from src.controllers.base import BaseController
from src.core.obs_builder import build_obs

_ALGOS = ('SAC', 'PPO')


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
        if algo not in _ALGOS:
            raise ValueError(f"algo '{algo}' no soportado; usa uno de {_ALGOS}")

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
