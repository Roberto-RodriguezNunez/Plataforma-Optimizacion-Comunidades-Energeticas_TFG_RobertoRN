"""
residual_base.py — Base común del controlador Residual SAC (MPC + δ)
====================================================================
Toda la lógica residual vive aquí. Las subclases solo implementan el
backend de inferencia `_infer(obs_norm) -> delta`:

  - ResidualSACController (evaluación): carga el .zip y usa SAC.predict (torch).
  - OnnxResidualController (producción):  usa onnxruntime session.run.

Pipeline común de `solve()` (idéntico para ambos → garantiza por
construcción que evaluación == producción):
    1. MPC base                → cs, cm, dc, dr
    2. obs_builder.build_obs   → obs_108
    3. añadir 4 features MPC    → obs_112
    4. VecNormalize             → clip((x - mean) / sqrt(var + 1e-8), ±clip)
    5. _infer(obs_norm)         → delta 4D  (torch u ONNX)
    6. residual multiplicativo  → max(0, mpc_flow * (1 + delta * delta_max))

La base NO importa torch ni onnxruntime → segura para el contenedor edge.
"""

import os
import sys
from typing import Dict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.controllers.base import BaseController
from src.core.obs_builder import build_obs


class ResidualControllerBase(BaseController):
    """
    Lógica común MPC + residual multiplicativo 4D.

    Las subclases:
      - en `__init__`, tras llamar a `super().__init__`, cargan su modelo y
        fijan las stats de VecNormalize en `self._mean`, `self._var`,
        `self._clip_obs` (dejar `self._mean = None` desactiva la normalización).
      - implementan `_infer(obs_norm) -> np.ndarray (4,)`.

    Args:
        mpc:       Instancia de LinearMPC.
        sim:       ComunidadSimulador (parámetros físicos).
        delta_max: Fracción (multiplicativa en 'mult', de P_MAX en 'add'). Es un
                   hiperparámetro del modelo entrenado (debe coincidir) → sin default.
        residual_mode: 'mult' -> max(0, mpc*(1+δ*δmax)); 'add' -> max(0, mpc+δ*δmax*P_MAX).
                       Debe coincidir con el del entreno (ResidualEnv.residual_mode).
    """

    def __init__(self, mpc, sim, delta_max: float, residual_mode: str = 'mult'):
        self._mpc = mpc
        self._sim = sim
        self._delta_max = delta_max
        self._residual_mode = residual_mode
        self._P_MAX = sim.POTENCIA_INVERSOR
        # Stats VecNormalize — las fija la subclase. None → sin normalizar.
        self._mean = None
        self._var = None
        self._clip_obs = 10.0

    def solve(self, state: Dict, forecast: np.ndarray) -> Dict[str, float]:
        # 1. MPC base
        mpc_action = self._mpc.solve(state, forecast)
        cs_mpc = mpc_action['P_carga_solar']
        cm_mpc = mpc_action['P_carga_red']
        dc_mpc = mpc_action['P_descarga_casa']
        dr_mpc = mpc_action['P_descarga_red']

        # 2-3. obs_108 (fuente única) + 4 features MPC → obs_112
        obs_108 = build_obs(state, forecast, self._sim)
        P = self._P_MAX
        mpc_feat = np.array(
            [cs_mpc / P, cm_mpc / P, dc_mpc / P, dr_mpc / P],
            dtype=np.float32,
        )
        obs_112 = np.append(obs_108, mpc_feat).astype(np.float32)

        # 4. Normalización (no-op si la subclase no fijó stats)
        obs_norm = self._normalizar(obs_112)

        # 5. Inferencia (backend de la subclase)
        delta = self._infer(obs_norm)

        # 6. Residual (mult o aditivo)
        return {
            'P_carga_solar':   self._combinar(cs_mpc, delta[0]),
            'P_carga_red':     self._combinar(cm_mpc, delta[1]),
            'P_descarga_casa': self._combinar(dc_mpc, delta[2]),
            'P_descarga_red':  self._combinar(dr_mpc, delta[3]),
        }

    def _combinar(self, mpc_flow: float, delta_i) -> float:
        """Combina un flujo MPC con su corrección δ según residual_mode."""
        dm = self._delta_max
        if self._residual_mode == 'add':
            return max(0.0, mpc_flow + float(delta_i) * dm * self._P_MAX)
        return max(0.0, mpc_flow * (1.0 + float(delta_i) * dm))

    def _normalizar(self, obs_112: np.ndarray) -> np.ndarray:
        """clip((x - mean) / sqrt(var + 1e-8), ±clip). No-op si no hay stats."""
        if self._mean is None:
            return obs_112
        return np.clip(
            (obs_112 - self._mean) / np.sqrt(self._var + 1e-8),
            -self._clip_obs, self._clip_obs,
        ).astype(np.float32)

    def _infer(self, obs_norm: np.ndarray) -> np.ndarray:
        """Backend de inferencia: obs_norm (112,) → delta (4,). Lo implementa la subclase."""
        raise NotImplementedError

    def nombre(self) -> str:
        raise NotImplementedError
