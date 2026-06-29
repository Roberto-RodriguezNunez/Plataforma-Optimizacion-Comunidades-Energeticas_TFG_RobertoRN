"""
residual_sac_controller.py — Backend torch del Residual SAC (evaluación)
========================================================================
Subclase de ResidualControllerBase que carga el modelo SAC entrenado (.zip)
y las stats de VecNormalize (.pkl). Toda la lógica residual (MPC + obs +
normalización + combinación) vive en la base; aquí solo se implementa la
inferencia con torch (`SAC.predict`).

Se usa en evaluación (eval_unificada): justo tras entrenar se tiene el .zip
de torch, todavía no el .onnx. La versión desplegable es OnnxResidualController.

Residual multiplicativo:
    flow_final = max(0, mpc_flow * (1 + delta * delta_max))
Con delta=(0,0,0,0), los flujos MPC pasan intactos.
"""

import os
from typing import Optional

import numpy as np

from src.controllers.residual_base import ResidualControllerBase


_ALGOS_TORCH = ('SAC', 'TD3', 'DDPG')


class ResidualSACController(ResidualControllerBase):
    """
    Controlador residual con inferencia torch (SB3). Sirve para cualquier
    algoritmo off-policy continuo entrenado sobre ResidualEnv (SAC, TD3, DDPG):
    todos exponen `model.predict(obs, deterministic=True) -> (delta, _)`.

    Args:
        model_path: Ruta al modelo (.zip).
        mpc: Instancia de LinearMPC.
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
