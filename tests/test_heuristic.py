"""
test_heuristic.py — Invariantes del controlador heurístico
==========================================================
El HeuristicController es el baseline sin arbitraje: almacena excedente solar
y cubre déficit con la batería, sin comprar de red para la batería ni verter
batería a red. Estos tests fijan sus invariantes de decisión (respeto de los
límites de SoC, eficiencias y ausencia de arbitraje) usando los parámetros
reales de config/system.yaml, los mismos que carga el controlador.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.controllers.heuristic import HeuristicController
from src.core.config import bateria

_BAT = bateria()
CAP = _BAT['capacidad_kwh']
EFF_C = _BAT['eficiencia_carga']
EFF_D = _BAT['eficiencia_descarga']
SOC_MIN = _BAT['soc_min']
SOC_MAX = _BAT['soc_max']


def _forecast(cons, gen):
    """Forecast (H, 4) con [consumo, generacion, precio_kwh, precio_excedente];
    el heurístico solo mira la primera fila."""
    f = np.zeros((24, 4))
    f[0, 0] = cons
    f[0, 1] = gen
    return f


def _solve(soc, cons, gen):
    return HeuristicController().solve({'soc': soc, 'step': 0},
                                       _forecast(cons, gen))


class TestSinArbitraje:
    def test_nunca_usa_la_red_para_la_bateria(self):
        """P_carga_red y P_descarga_red son 0 en cualquier estado."""
        for soc in (SOC_MIN, 0.50, SOC_MAX):
            for cons, gen in ((10.0, 2.0), (2.0, 10.0), (5.0, 5.0)):
                a = _solve(soc, cons, gen)
                assert a['P_carga_red'] == 0.0
                assert a['P_descarga_red'] == 0.0

    def test_balance_neutro_no_actua(self):
        """Sin excedente ni déficit, todos los flujos son 0."""
        a = _solve(0.50, 5.0, 5.0)
        assert all(v == 0.0 for v in a.values())


class TestCargaConExcedente:
    def test_carga_todo_el_excedente_si_hay_espacio(self):
        a = _solve(0.50, 4.0, 10.0)  # excedente 6 kWh, espacio de sobra
        assert a['P_carga_solar'] == pytest.approx(6.0)
        assert a['P_descarga_casa'] == 0.0

    def test_espacio_limita_la_carga(self):
        """Con 1,6 kWh de hueco, carga como máximo 1,6/η_c del lado AC."""
        soc = SOC_MAX - 1.6 / CAP
        a = _solve(soc, 4.0, 10.0)  # excedente 6 > hueco
        assert a['P_carga_solar'] == pytest.approx(1.6 / EFF_C)

    def test_bateria_llena_no_carga(self):
        a = _solve(SOC_MAX, 4.0, 10.0)
        assert a['P_carga_solar'] == pytest.approx(0.0)


class TestDescargaConDeficit:
    def test_cubre_el_deficit_con_perdidas(self):
        """Para entregar 8 kWh a las casas extrae 8/η_d de la batería."""
        a = _solve(0.50, 10.0, 2.0)  # déficit 8 kWh, batería con reserva
        assert a['P_descarga_casa'] == pytest.approx(8.0 / EFF_D)
        assert a['P_carga_solar'] == 0.0

    def test_reserva_limita_la_descarga(self):
        """Con 4 kWh sobre el mínimo, no extrae más de 4 kWh."""
        soc = SOC_MIN + 4.0 / CAP
        a = _solve(soc, 50.0, 0.0)  # déficit muy superior a la reserva
        assert a['P_descarga_casa'] == pytest.approx(4.0)

    def test_bateria_en_minimo_no_descarga(self):
        a = _solve(SOC_MIN, 10.0, 2.0)
        assert a['P_descarga_casa'] == pytest.approx(0.0)
