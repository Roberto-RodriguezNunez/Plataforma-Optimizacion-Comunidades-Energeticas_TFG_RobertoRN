"""
nstep_replay_buffer.py — ReplayBuffer con retornos N-step para DQN
===================================================================
Extiende ReplayBuffer de SB3 añadiendo acumulación de retornos de N pasos
antes de almacenar cada transición.

Motivación — problema de crédito tardío:
  Con backup 1-step estándar, la señal de Q(s_t) tarda N actualizaciones
  en propagarse desde la recompensa obtenida N pasos después. En episodios
  de 168h con ciclos carga/descarga de 24-48h, esto ralentiza mucho la
  convergencia.

Fórmula del update N-step:
  Q_target = Σ_{i=0}^{n-1} γ^i · r_{t+i}  +  γ^n · max_a Q(s_{t+n}, a)

  donde la primera parte (retorno acumulado R_n) se almacena como "reward"
  en el buffer, y el segundo término (bootstrap) usa γ^n como gamma del modelo.

Uso correcto con DQN de SB3:
  model = DQN(
      ...
      gamma                = base_gamma ** n_steps,   # ← γ^n para bootstrap
      replay_buffer_class  = NStepReplayBuffer,
      replay_buffer_kwargs = {"n_steps": n_steps, "base_gamma": base_gamma},
  )

Gestión de fronteras de episodio:
  Cuando done=True, se vacía la cola pendiente. Las transiciones cercanas
  al final del episodio usan el retorno acumulado hasta el terminal,
  sin cruzar la frontera con el episodio siguiente.
"""

from collections import deque
from typing import Any, Dict, List, Union

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer


class NStepReplayBuffer(ReplayBuffer):
    """
    ReplayBuffer que acumula retornos de N pasos antes de almacenar.

    Args:
        buffer_size:            Capacidad total del buffer.
        observation_space:      Espacio de observación del entorno.
        action_space:           Espacio de acción del entorno.
        device:                 Dispositivo torch ("auto", "cpu", "cuda").
        n_envs:                 Número de entornos paralelos (1 para DummyVecEnv).
        optimize_memory_usage:  Optimización de memoria de SB3 (por defecto False).
        n_steps:                Número de pasos para el retorno acumulado.
        base_gamma:             Factor de descuento base del problema.
                                El modelo DQN recibe gamma=base_gamma**n_steps.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        n_steps: int = 8,
        base_gamma: float = 0.99,
        **kwargs,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            **kwargs,
        )
        self.n_steps    = n_steps
        self.base_gamma = base_gamma
        self._pending: deque = deque()

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Guardar copia de la transición en cola temporal
        self._pending.append({
            "obs":      obs.copy(),
            "next_obs": next_obs.copy(),
            "action":   action.copy(),
            "reward":   np.array(reward, dtype=np.float32),
            "done":     done.copy(),
            "infos":    infos,
        })

        # Cuando tenemos n_steps acumulados, enviar la más antigua al buffer
        if len(self._pending) >= self.n_steps:
            self._flush_oldest()

        # Al fin del episodio, vaciar todas las transiciones pendientes
        # (usan retornos acortados, sin cruzar la frontera del episodio)
        if done[0]:
            while self._pending:
                self._flush_oldest()

    def _flush_oldest(self) -> None:
        """
        Extrae la transición más antigua de la cola y la almacena en el buffer
        padre con el retorno N-step acumulado.
        """
        if not self._pending:
            return

        first = self._pending[0]

        # Acumular retorno descontado hasta el primer terminal o fin de ventana
        R             = 0.0
        last_next_obs = first["next_obs"]
        last_done     = first["done"]
        last_infos    = first["infos"]

        for i, trans in enumerate(self._pending):
            R            += (self.base_gamma ** i) * float(trans["reward"][0])
            last_next_obs = trans["next_obs"]
            last_done     = trans["done"]
            last_infos    = trans["infos"]
            if trans["done"][0]:
                # Episodio terminado: el bootstrap se anula (1 - done = 0)
                break

        super().add(
            first["obs"],
            last_next_obs,
            first["action"],
            np.array([R], dtype=np.float32),
            last_done,
            last_infos,
        )

        self._pending.popleft()
