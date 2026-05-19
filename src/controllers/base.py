"""
base.py — Interfaz abstracta para controladores de la comunidad energética
==========================================================================
Patrón Strategy: todos los controladores (heurístico, MPC, DQN, PPO,
Residual SAC) heredan de BaseController y exponen el mismo método solve().

Esto permite que el bucle de evaluación sea agnóstico al controlador:

    controller = LinearMPC(config)        # o HeuristicController(), etc.
    for step in range(168):
        action = controller.solve(state, forecast)
        state = simulator.apply(action)
"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class BaseController(ABC):
    """
    Interfaz abstracta para controladores energéticos.

    Todos los controladores reciben el mismo estado y pronóstico,
    y devuelven las mismas claves de acción para comparación justa.
    """

    @abstractmethod
    def solve(self, state: Dict, forecast: np.ndarray) -> Dict[str, float]:
        """
        Calcula la acción óptima para el paso actual.

        Args:
            state: Diccionario con el estado actual del sistema.
                Claves mínimas:
                    'soc': float — Estado de carga actual [0, 1].
                    'step': int — Índice temporal actual en el dataset.
            forecast: Array (H, 4) con el pronóstico del horizonte.
                Columnas: [consumo, generacion, precio_kwh, precio_excedente].
                Puede ser perfecto (oráculo) o ruidoso (realista).

        Returns:
            Diccionario con los flujos energéticos para el paso h=0:
                'P_carga_solar':   kWh de excedente solar → batería (lado input).
                'P_carga_red':     kWh comprados de red → batería (lado input).
                'P_descarga_casa': kWh extraídos batería → casas (lado batería).
                'P_descarga_red':  kWh extraídos batería → red (lado batería).
        """
        ...

    @abstractmethod
    def nombre(self) -> str:
        """Nombre legible del controlador para logs y tablas."""
        ...
