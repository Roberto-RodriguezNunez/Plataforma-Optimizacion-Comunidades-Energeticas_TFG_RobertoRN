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
from src.benchmarks.mpc_benchmark import LinearMPC


class ResidualSACController(ResidualControllerBase):
    """
    Residual SAC con inferencia torch (SB3).

    Args:
        model_path: Ruta al modelo SAC (.zip).
        mpc: Instancia de LinearMPC.
        sim: ComunidadSimulador (parámetros físicos).
        delta_max: Fracción multiplicativa (hiperparámetro del modelo entrenado).
        vec_normalize_path: Ruta a las stats de VecNormalize (.pkl). None → sin normalizar.
    """

    def __init__(
        self,
        model_path: str,
        mpc: LinearMPC,
        sim,
        delta_max: float,
        vec_normalize_path: Optional[str] = None,
    ):
        from stable_baselines3 import SAC

        super().__init__(mpc, sim, delta_max)
        self._model = SAC.load(model_path, device='cpu')

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
        return f"ResidualSAC(dmax={self._delta_max})"
