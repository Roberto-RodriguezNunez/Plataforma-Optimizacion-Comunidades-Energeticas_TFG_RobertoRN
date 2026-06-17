"""
onnx_inference.py — Controlador Residual SAC sobre ONNX Runtime
================================================================
OnnxResidualController replica exactamente la ruta de inferencia de
ResidualSACController pero sin SB3, Gymnasium ni torch. Solo requiere:
  numpy, scipy, onnxruntime, pyyaml

Dependencias del contenedor edge:
  numpy, scipy, onnxruntime, pyyaml
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.controllers.base import BaseController
from src.production.obs_builder import build_obs

# Rutas por defecto (relativas a ROOT)
_DEFAULT_ONNX = os.path.join(ROOT, 'models', 'residual_sac_actor.onnx')
_DEFAULT_NPZ  = os.path.join(ROOT, 'models', 'vec_normalize_v5_1M.npz')


class OnnxResidualController(BaseController):
    """
    Controlador Residual SAC en producción, sin SB3 ni torch.

    Ruta de inferencia idéntica a ResidualSACController.solve():
      1. MPC base → cs, cm, dc, dr
      2. obs_builder.build_obs → obs_108
      3. Añadir MPC features → obs_112
      4. VecNormalize: clip((x - mean) / sqrt(var + 1e-8), -clip, clip)
      5. ONNX session.run → delta (1, 4)
      6. Residual multiplicativo: max(0, mpc_flow * (1 + delta * delta_max))

    Args:
        onnx_path:   Ruta al archivo .onnx (exportado por export_onnx.py).
        npz_path:    Ruta al .npz con mean, var, clip_obs (exportado por export_onnx.py).
        mpc:         Instancia de LinearMPC.
        sim:         ComunidadSimulador.
        delta_max:   Fracción multiplicativa. Default 0.15 (valor v5).
    """

    def __init__(
        self,
        mpc,
        sim,
        onnx_path: str = _DEFAULT_ONNX,
        npz_path: str = _DEFAULT_NPZ,
        delta_max: float = 0.15,
    ):
        self._mpc = mpc
        self._sim = sim
        self._delta_max = delta_max
        self._P_MAX = sim.POTENCIA_INVERSOR

        # ONNX session (CPU provider)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=['CPUExecutionProvider'],
        )

        # Estadísticas VecNormalize
        npz = np.load(npz_path)
        self._mean     = npz['mean'].astype(np.float32)    # (112,)
        self._var      = npz['var'].astype(np.float32)     # (112,)
        self._clip_obs = float(npz['clip_obs'])

    def solve(self, state: Dict, forecast: np.ndarray) -> Dict[str, float]:
        # 1. MPC base
        mpc_action = self._mpc.solve(state, forecast)
        cs_mpc = mpc_action['P_carga_solar']
        cm_mpc = mpc_action['P_carga_red']
        dc_mpc = mpc_action['P_descarga_casa']
        dr_mpc = mpc_action['P_descarga_red']

        # 2. obs_108 (función pura compartida con ResidualSACController)
        obs_108 = build_obs(state, forecast, self._sim)

        # 3. Añadir MPC features → obs_112
        P = self._P_MAX
        mpc_feat = np.array(
            [cs_mpc / P, cm_mpc / P, dc_mpc / P, dr_mpc / P],
            dtype=np.float32,
        )
        obs_112 = np.append(obs_108, mpc_feat).astype(np.float32)

        # 4. VecNormalize — misma fórmula que ResidualSACController
        obs_norm = np.clip(
            (obs_112 - self._mean) / np.sqrt(self._var + 1e-8),
            -self._clip_obs, self._clip_obs,
        ).astype(np.float32)

        # 5. Inferencia ONNX
        delta = self._session.run(
            ['delta'],
            {'obs': obs_norm.reshape(1, 112)},
        )[0][0]  # (1, 4) → (4,)

        # 6. Residual multiplicativo
        dm = self._delta_max
        return {
            'P_carga_solar':   max(0.0, cs_mpc * (1.0 + float(delta[0]) * dm)),
            'P_carga_red':     max(0.0, cm_mpc * (1.0 + float(delta[1]) * dm)),
            'P_descarga_casa': max(0.0, dc_mpc * (1.0 + float(delta[2]) * dm)),
            'P_descarga_red':  max(0.0, dr_mpc * (1.0 + float(delta[3]) * dm)),
        }

    def nombre(self) -> str:
        return f"OnnxResidualSAC(dmax={self._delta_max})"
