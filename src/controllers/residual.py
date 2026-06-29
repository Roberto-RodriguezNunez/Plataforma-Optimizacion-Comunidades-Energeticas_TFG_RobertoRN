"""
residual.py — Controlador Residual (MPC + δ): base común + backend torch
=========================================================================
Toda la lógica residual vive en `ResidualControllerBase`. Las subclases solo
implementan el backend de inferencia `_infer(obs_norm) -> delta`:

  - ResidualSACController (evaluación): carga el .zip y usa SAC.predict (torch).
  - OnnxResidualController (producción, en src/production/onnx_inference.py):
    usa onnxruntime session.run.

Pipeline común de `solve()` (idéntico para ambos → garantiza por construcción
que evaluación == producción):
    1. MPC base                → cs, cm, dc, dr
    2. observation.build_obs   → obs_108
    3. añadir 4 features MPC    → obs_112
    4. VecNormalize             → clip((x - mean) / sqrt(var + 1e-8), ±clip)
    5. _infer(obs_norm)         → delta 4D  (torch u ONNX)
    6. residual multiplicativo  → max(0, mpc_flow * (1 + delta * delta_max))

La BASE no importa torch ni onnxruntime → segura para el contenedor edge.
(Antes en residual_base.py + residual_sac_controller.py.)
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

_ALGOS_TORCH = ('SAC', 'TD3', 'DDPG')


class ResidualControllerBase(BaseController):
    """
    Lógica común MPC + residual multiplicativo 4D.

    Las subclases:
      - en `__init__`, tras llamar a `super().__init__`, cargan su modelo y
        fijan las stats de VecNormalize en `self._mean`, `self._var`,
        `self._clip_obs` (dejar `self._mean = None` desactiva la normalización).
      - implementan `_infer(obs_norm) -> np.ndarray (4,)`.

    Args:
        mpc:       Controlador MPC con `.solve(state, forecast)`.
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


class ResidualSACController(ResidualControllerBase):
    """
    Controlador residual con inferencia torch (SB3). Sirve para cualquier
    algoritmo off-policy continuo entrenado sobre ResidualEnv (SAC, TD3, DDPG):
    todos exponen `model.predict(obs, deterministic=True) -> (delta, _)`.

    Args:
        model_path: Ruta al modelo (.zip).
        mpc: Controlador MPC con `.solve(state, forecast)`.
        sim: ComunidadSimulador (parámetros físicos).
        delta_max: Fracción (hiperparámetro del modelo entrenado).
        vec_normalize_path: Ruta a las stats de VecNormalize (.pkl). None → sin normalizar.
        algo: 'SAC' | 'TD3' | 'DDPG' (define la clase SB3 que carga el .zip).
        residual_mode: 'mult' | 'add' (debe coincidir con el del entreno).
    """

    def __init__(
        self,
        model_path: str,
        mpc,                      # cualquier controlador MPC con .solve(state, forecast)
        sim,
        delta_max: float,
        vec_normalize_path: Optional[str] = None,
        algo: str = 'SAC',
        residual_mode: str = 'mult',
    ):
        import stable_baselines3 as sb3

        algo = algo.upper()
        if algo not in _ALGOS_TORCH:
            raise ValueError(f"algo '{algo}' no soportado; usa uno de {_ALGOS_TORCH}")

        super().__init__(mpc, sim, delta_max, residual_mode=residual_mode)
        self._algo = algo
        self._model = getattr(sb3, algo).load(model_path, device='cpu')

        if vec_normalize_path and os.path.exists(vec_normalize_path):
            import pickle
            with open(vec_normalize_path, 'rb') as f:
                vec_norm = pickle.load(f)
            self._mean = vec_norm.obs_rms.mean
            self._var = vec_norm.obs_rms.var
            self._clip_obs = vec_norm.clip_obs

    def _infer(self, obs_norm: np.ndarray) -> np.ndarray:
        delta, _ = self._model.predict(obs_norm, deterministic=True)
        return delta

    def nombre(self) -> str:
        return f"Residual{self._algo}({self._residual_mode},dmax={self._delta_max})"
