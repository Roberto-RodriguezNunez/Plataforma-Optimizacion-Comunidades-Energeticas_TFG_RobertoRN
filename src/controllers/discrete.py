"""
discrete.py — Controlador para agentes RL discretos (DQN/PPO)
============================================================================
Convierte la acción discreta (0-8) de un agente DQN o PPO al formato
de 4 flujos que espera evaluation/unified.py.

La decodificación replica la lógica de simulador.ejecutar_accion_fisica()
pero solo devuelve los flujos intendidos, sin ejecutar física
(simular_hora_mpc se encarga en la evaluación).
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
# El decode discreto (acción -> 4 flujos) es ComunidadSimulador.accion_a_flujos
# (staticmethod, fuente única), llamado vía self._sim.


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

        # Cargar modelo SB3. custom_objects sustituye los schedules pickled
        # (lr/clip/exploration), que son EXCLUSIVOS del entreno: evita el fallo
        # al cargar modelos entrenados con otra version de Python (los closures
        # pickled no son portables entre versiones) sin alterar la inferencia.
        _train_only = {'lr_schedule': (lambda _: 0.0), 'learning_rate': 0.0}
        if self._algo == 'DQN':
            from stable_baselines3 import DQN
            self._model = DQN.load(model_path, device='cpu',
                                   custom_objects={**_train_only,
                                                   'exploration_schedule': (lambda _: 0.0)})
        else:
            from stable_baselines3 import PPO
            self._model = PPO.load(model_path, device='cpu',
                                   custom_objects={**_train_only,
                                                   'clip_range': (lambda _: 0.2)})

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
