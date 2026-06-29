"""
heuristic_controller.py — Controlador heurístico basado en reglas
=================================================================
Estrategia: almacenar excedente solar en batería, usar batería para
cubrir déficit. Sin arbitraje temporal (no compra de red para batería,
no vende batería a red).

Es la heurística más natural para un usuario doméstico sin optimización.
"""

from src.controllers.base import BaseController
from src.config import bateria as _bateria_cfg

_BAT = _bateria_cfg()


class HeuristicController(BaseController):
    """
    Controlador heurístico: almacena solar, cubre déficit con batería.

    Reglas:
      - Si hay excedente solar: cargar batería (sin comprar de red).
      - Si hay déficit: descargar batería a casas (sin vender a red).
      - Nunca hace arbitraje temporal con la red.

    Usa state['soc'] proporcionado por el bucle de evaluación para
    calcular los límites de la batería en cada paso.
    """

    def __init__(self):
        self.cap = _BAT['capacidad_kwh']
        self.eff_c = _BAT['eficiencia_carga']
        self.eff_d = _BAT['eficiencia_descarga']
        self.soc_min = _BAT['soc_min']
        self.soc_max = _BAT['soc_max']

    def solve(self, state, forecast):
        soc = state['soc']
        cons, gen = forecast[0, 0], forecast[0, 1]

        bal = gen - cons
        exc = max(0.0, bal)
        dfc = max(0.0, -bal)

        bat_kwh = soc * self.cap
        espacio = max(0.0, self.soc_max * self.cap - bat_kwh)
        disponible = max(0.0, bat_kwh - self.soc_min * self.cap)

        if exc > 0:
            cs = min(exc, espacio / self.eff_c)
        else:
            cs = 0.0

        if dfc > 0:
            dc = min(dfc / self.eff_d, disponible)
        else:
            dc = 0.0

        return {
            'P_carga_solar': cs,
            'P_carga_red': 0.0,
            'P_descarga_casa': dc,
            'P_descarga_red': 0.0,
        }

    def nombre(self):
        return "Heuristico"
