"""
data_feed.py — Conectores de datos para el sistema de gestión energética
=========================================================================
ABC DataFeed con la misma semántica que ComunidadSimulador.get_data_window.

Implementaciones:
  - HistoricalDataFeed  : envuelve el dataset histórico (default).
  - SimulatedLiveFeed   : emula ESIOS / Open-Meteo / datadis con datos
                          simulados. Interfaz lista para pasar a HTTP real
                          sin tocar el resto del sistema (solo rellenar los
                          tres métodos _fetch_*).

Selección por variable de entorno FEED_MODE (historico | simulado).
APIs implementadas pero no activadas con datos reales.
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.benchmarks.mpc_benchmark import (
    ComunidadSimulador,
    DATASET_PATH,
    _get_hora_actual,
    _RHO_SOLAR,
    _RHO_CONS,
)
from src.core.forecast import (
    aplicar_ruido_ventana, generar_factores_precio, avanzar_ar1,
)


# ──────────────────────────────────────────────────────────────────────────────
#  ABC
# ──────────────────────────────────────────────────────────────────────────────

class DataFeed(ABC):
    """
    Interfaz de fuente de datos horaria.

    get_window devuelve una ventana de horizonte H con columnas:
      [consumo_total (kWh), generacion_total (kWh),
       precio_kwh (EUR/kWh), precio_excedente (EUR/kWh)]
    """

    @abstractmethod
    def get_window(self, step: int, horizon: int = 24) -> np.ndarray:
        """Retorna (horizon, 4) float array."""

    @abstractmethod
    def get_current(self, step: int) -> dict:
        """Retorna la fila actual como dict con las 4 columnas."""


# ──────────────────────────────────────────────────────────────────────────────
#  HistoricalDataFeed — dataset histórico sin modificaciones
# ──────────────────────────────────────────────────────────────────────────────

class HistoricalDataFeed(DataFeed):
    """Envuelve ComunidadSimulador — usa dataset_final.csv directamente."""

    def __init__(self, data_path: str = DATASET_PATH):
        self._sim = ComunidadSimulador(data_path)

    @property
    def sim(self) -> ComunidadSimulador:
        return self._sim

    def get_window(self, step: int, horizon: int = 24) -> np.ndarray:
        return self._sim.get_data_window(step, horizon)

    def get_current(self, step: int) -> dict:
        row = self._sim.get_data_window(step, horizon=1)[0]
        return {
            'consumo_total':     float(row[0]),
            'generacion_total':  float(row[1]),
            'precio_kwh':        float(row[2]),
            'precio_excedente':  float(row[3]),
        }


# ──────────────────────────────────────────────────────────────────────────────
#  SimulatedLiveFeed — emulación de APIs en tiempo real
# ──────────────────────────────────────────────────────────────────────────────

class SimulatedLiveFeed(DataFeed):
    """
    Emula ESIOS / Open-Meteo / datadis con datos del dataset histórico.

    Los métodos _fetch_* tienen la firma de una integración HTTP real.
    Para activar datos reales: rellenar esos tres métodos sin tocar el resto.

    En esta implementación, las tres fuentes devuelven datos simulados
    del dataset_final.csv con ruido de precios 3 capas para realismo.
    """

    def __init__(self, data_path: str = DATASET_PATH, seed: int = 42):
        self._sim = ComunidadSimulador(data_path)
        self._rng = np.random.default_rng(seed)
        # Estado AR(1) para ruido de consumo/solar — evoluciona paso a paso
        # igual que en el bucle de entrenamiento de mpc_benchmark.py
        self._error_solar = 0.0
        self._error_cons  = 0.0

    @property
    def sim(self) -> ComunidadSimulador:
        return self._sim

    # ── Métodos privados con firma de integración real ────────────────────────

    def _fetch_esios(self, step: int, horizon: int) -> np.ndarray:
        """
        Obtiene precios PVPC y compensación de excedentes del indicador ESIOS.
        Implementación actual: datos simulados del dataset histórico.
        Activar HTTP real: llamar a la API de ESIOS (indicadores 1001 y 1739)
        y devolver array (horizon, 2) con [precio_kwh, precio_excedente].
        """
        window = self._sim.get_data_window(step, horizon)
        return window[:, 2:4].copy()  # (H, 2): precio_kwh, precio_excedente

    def _fetch_open_meteo(self, step: int, horizon: int) -> np.ndarray:
        """
        Obtiene previsión de irradiancia solar de Open-Meteo.
        Implementación actual: generación solar simulada del dataset histórico.
        Activar HTTP real: llamar a Open-Meteo API y convertir irradiancia
        a generación solar con factor de planta de la instalación.
        Devuelve array (horizon,) con generacion_total estimada en kWh.
        """
        window = self._sim.get_data_window(step, horizon)
        return window[:, 1].copy()  # (H,): generacion_total

    def _fetch_datadis(self, step: int, horizon: int) -> np.ndarray:
        """
        Obtiene datos de consumo de la plataforma datadis.
        Implementación actual: consumo simulado del dataset histórico.
        Activar HTTP real: llamar a la API de datadis con CUPS de las viviendas
        y agregar el consumo comunitario.
        Devuelve array (horizon,) con consumo_total en kWh.
        """
        window = self._sim.get_data_window(step, horizon)
        return window[:, 0].copy()  # (H,): consumo_total

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def get_window(self, step: int, horizon: int = 24) -> np.ndarray:
        """
        Ensambla la ventana desde las tres fuentes simuladas y aplica ambos
        tipos de ruido — igual que el bucle de entrenamiento en mpc_benchmark:
          1. AR(1) sobre consumo y generación solar (estado persistente entre llamadas)
          2. Ruido 3 capas sobre precios
        """
        consumo    = self._fetch_datadis(step, horizon)      # (H,)
        generacion = self._fetch_open_meteo(step, horizon)   # (H,)
        precios    = self._fetch_esios(step, horizon)         # (H, 2)

        H = len(consumo)
        window = np.zeros((H, 4), dtype=np.float64)
        window[:, 0] = consumo
        window[:, 1] = generacion
        window[:, 2] = precios[:, 0]
        window[:, 3] = precios[:, 1]

        # Avanzar AR(1) y aplicar ruido con la FUENTE ÚNICA (hora actual = primer
        # paso de pronóstico, con ruido), idéntico a correr_episodios/eval/env.
        self._error_solar, self._error_cons = avanzar_ar1(
            self._error_solar, self._error_cons, self._rng.standard_normal)
        hora_actual = _get_hora_actual(step)
        factores = generar_factores_precio(hora_actual, self._rng.standard_normal)
        aplicar_ruido_ventana(window, self._error_solar, self._error_cons, factores)
        return window

    def get_current(self, step: int) -> dict:
        row = self.get_window(step, horizon=1)[0]
        return {
            'consumo_total':     float(row[0]),
            'generacion_total':  float(row[1]),
            'precio_kwh':        float(row[2]),
            'precio_excedente':  float(row[3]),
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_feed(mode: str | None = None, data_path: str = DATASET_PATH) -> DataFeed:
    """
    Retorna la implementación de DataFeed según FEED_MODE.

    Args:
        mode: 'historico' | 'simulado'. Si None, lee FEED_MODE del entorno.
        data_path: Ruta al dataset_final.csv.
    """
    if mode is None:
        mode = os.environ.get('FEED_MODE', 'historico')
    if mode == 'simulado':
        return SimulatedLiveFeed(data_path)
    return HistoricalDataFeed(data_path)
