"""
residual_sac_controller.py — Controlador Residual SAC multiplicativo 4D
=======================================================================
Combina un modelo SAC entrenado con el LinearMPC para producir acciones
a traves de la interfaz estandar solve(state, forecast) -> dict.

Residual multiplicativo:
    flow_final = max(0, mpc_flow * (1 + delta * delta_max))
Con delta=(0,0,0,0), los flujos MPC pasan intactos.
"""

import os
import sys
from typing import Dict, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.controllers.base import BaseController
from src.benchmarks.mpc_benchmark import LinearMPC


class ResidualSACController(BaseController):
    """
    Controlador que combina MPC + SAC residual multiplicativo 4D.

    En solve():
      1. Resuelve el MPC con (state, forecast)
      2. Construye obs 112-dim (108 del entorno + 4 MPC features)
      3. Predice delta 4D con el SAC (determinista)
      4. Combina: flow_final = max(0, mpc_flow * (1 + delta * delta_max))

    Args:
        model_path: Ruta al modelo SAC (.zip).
        mpc: Instancia de LinearMPC.
        sim: ComunidadSimulador (para parametros fisicos).
        delta_max: Fraccion multiplicativa. 0.30 = ±30% de cada flujo MPC.
        vec_normalize_path: Ruta a las stats de VecNormalize (.pkl).
    """

    def __init__(
        self,
        model_path: str,
        mpc: LinearMPC,
        sim,
        delta_max: float = 0.30,
        vec_normalize_path: Optional[str] = None,
    ):
        from stable_baselines3 import SAC

        self._mpc = mpc
        self._sim = sim
        self._delta_max = delta_max
        self._P_MAX = sim.POTENCIA_INVERSOR

        self._model = SAC.load(model_path, device='cpu')

        self._obs_rms = None
        self._clip_obs = 10.0
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            import pickle
            with open(vec_normalize_path, 'rb') as f:
                vec_norm = pickle.load(f)
            self._obs_rms = vec_norm.obs_rms
            self._clip_obs = vec_norm.clip_obs

    def solve(self, state: Dict, forecast: np.ndarray) -> Dict[str, float]:
        # 1. Resolver MPC
        mpc_action = self._mpc.solve(state, forecast)
        cs_mpc = mpc_action['P_carga_solar']
        cm_mpc = mpc_action['P_carga_red']
        dc_mpc = mpc_action['P_descarga_casa']
        dr_mpc = mpc_action['P_descarga_red']

        # 2. Construir obs 112-dim (108 base + 4 MPC features)
        obs_108 = self._build_obs(state, forecast)
        P = self._P_MAX
        mpc_feat = np.array(
            [cs_mpc / P, cm_mpc / P, dc_mpc / P, dr_mpc / P],
            dtype=np.float32,
        )
        obs_112 = np.append(obs_108, mpc_feat)

        # 3. Normalizar si hay stats
        if self._obs_rms is not None:
            obs_112 = np.clip(
                (obs_112 - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + 1e-8),
                -self._clip_obs, self._clip_obs
            ).astype(np.float32)

        # 4. Predecir delta 4D (determinista)
        delta, _ = self._model.predict(obs_112, deterministic=True)
        dm = self._delta_max

        # 5. Residual multiplicativo
        return {
            'P_carga_solar':   max(0.0, cs_mpc * (1.0 + delta[0] * dm)),
            'P_carga_red':     max(0.0, cm_mpc * (1.0 + delta[1] * dm)),
            'P_descarga_casa': max(0.0, dc_mpc * (1.0 + delta[2] * dm)),
            'P_descarga_red':  max(0.0, dr_mpc * (1.0 + delta[3] * dm)),
        }

    def nombre(self) -> str:
        return f"ResidualSAC(dmax={self._delta_max})"

    def _build_obs(self, state: Dict, forecast: np.ndarray) -> np.ndarray:
        soc = state['soc']
        step = state['step']

        datos_hoy = self._sim.get_data_window(step, horizon=1)[0]
        cons, gen, precio_compra, precio_venta = datos_hoy
        balance = gen - cons
        exc = max(0.0, balance)
        def_ = abs(min(0.0, balance))

        H = min(24, len(forecast))
        forecast_padded = np.zeros((24, 4), dtype=np.float32)
        forecast_padded[:H] = forecast[:H]
        forecast_flat = forecast_padded.flatten()

        hora = step % 24
        dia_sem = (step // 24) % 7
        mes = 0
        temp_feats = np.array([
            np.sin(2 * np.pi * hora / 24),
            np.cos(2 * np.pi * hora / 24),
            np.sin(2 * np.pi * dia_sem / 7),
            np.cos(2 * np.pi * dia_sem / 7),
            np.sin(2 * np.pi * mes / 12),
            np.cos(2 * np.pi * mes / 12),
        ], dtype=np.float32)

        window_clean = self._sim.get_data_window(step + 1, horizon=24)
        solar_exc_24h = float(np.sum(np.maximum(0.0, window_clean[:, 1] - window_clean[:, 0])))
        espacio_bat = max(0.0, (self._sim.SOC_MAX - soc) * self._sim.BATERIA_CAPACIDAD)
        margen_solar = np.float32(
            max(0.0, solar_exc_24h - espacio_bat) / self._sim.BATERIA_CAPACIDAD
        )

        obs = np.concatenate((
            [soc, precio_compra, precio_venta, exc, def_],
            forecast_flat,
            temp_feats,
            [margen_solar],
        ))
        return obs.astype(np.float32)
