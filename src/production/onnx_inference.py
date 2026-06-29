"""
onnx_inference.py — Backend ONNX del Residual SAC (producción / edge)
=====================================================================
Subclase de ResidualControllerBase con inferencia sobre ONNX Runtime, sin
SB3, Gymnasium ni torch. Comparte toda la lógica residual (MPC + obs +
normalización + combinación) con ResidualSACController vía la base común →
equivalencia por construcción (la verifica además test_onnx_equivalence).

Dependencias del contenedor edge:
  numpy, scipy, onnxruntime, pyyaml
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.controllers.residual_base import ResidualControllerBase
from src.core.obs_builder import OBS_DIM_RESIDUAL

# Rutas por defecto (relativas a ROOT)
_DEFAULT_ONNX = os.path.join(ROOT, 'models', 'residual_sac_actor.onnx')
_DEFAULT_NPZ  = os.path.join(ROOT, 'models', 'vec_normalize_v5_1M.npz')


class OnnxResidualController(ResidualControllerBase):
    """
    Residual SAC en producción, con inferencia ONNX (sin SB3 ni torch).

    Args:
        mpc:         Instancia de LinearMPC.
        sim:         ComunidadSimulador.
        onnx_path:   Ruta al .onnx (exportado por export_onnx.py).
        npz_path:    Ruta al .npz con mean, var, clip_obs (exportado por export_onnx.py).
        delta_max:   Fracción multiplicativa (hiperparámetro del modelo entrenado).
    """

    def __init__(
        self,
        mpc,
        sim,
        delta_max: float,
        onnx_path: str = _DEFAULT_ONNX,
        npz_path: str = _DEFAULT_NPZ,
    ):
        super().__init__(mpc, sim, delta_max)

        # ONNX session (CPU provider, 1 hilo → determinista)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=['CPUExecutionProvider'],
        )

        # Stats VecNormalize (exportadas en el .npz)
        npz = np.load(npz_path)
        self._mean     = npz['mean'].astype(np.float32)    # (112,)
        self._var      = npz['var'].astype(np.float32)     # (112,)
        self._clip_obs = float(npz['clip_obs'])

    def _infer(self, obs_norm: np.ndarray) -> np.ndarray:
        delta = self._session.run(
            ['delta'],
            {'obs': obs_norm.reshape(1, OBS_DIM_RESIDUAL)},
        )[0][0]  # (1, 4) → (4,)
        return delta

    def nombre(self) -> str:
        return f"OnnxResidualSAC(dmax={self._delta_max})"
