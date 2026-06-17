"""
obs_builder.py — Construcción del vector de observación de 108 dimensiones
===========================================================================
Función pura extraída de ResidualSACController._build_obs. Es la única fuente
de verdad para construir las 108 dims base compartidas por ResidualSACController
y OnnxResidualController, garantizando que ambas rutas de inferencia son idénticas.

Sin dependencias de SB3, Gymnasium ni torch — apto para el contenedor edge.
"""

import numpy as np


def build_obs(state: dict, forecast: np.ndarray, sim) -> np.ndarray:
    """
    Construye el vector de observación base de 108 dimensiones (float32).

    Estructura:
      [0:5]    Estado y precios actuales   (5 dims)
      [5:101]  Forecast 24h aplanado       (96 dims = 24 × 4 cols)
      [101:107] Features temporales cíclicos (6 dims)
      [107]    Margen solar a 24h          (1 dim)

    Args:
        state:    {'soc': float, 'step': int}
        forecast: (H, 4) float array — [consumo, generacion, precio_kwh, precio_exc]
        sim:      ComunidadSimulador — acceso a get_data_window y parámetros físicos

    Returns:
        np.ndarray de shape (108,) dtype float32
    """
    soc = state['soc']
    step = state['step']

    # Bloque 1 — estado y precios actuales (5 dims)
    datos_hoy = sim.get_data_window(step, horizon=1)[0]
    cons, gen, precio_compra, precio_venta = datos_hoy
    balance = gen - cons
    exc = max(0.0, balance)
    def_ = abs(min(0.0, balance))

    # Bloque 2 — forecast 24h aplanado (96 dims)
    H = min(24, len(forecast))
    forecast_padded = np.zeros((24, 4), dtype=np.float32)
    forecast_padded[:H] = forecast[:H]
    forecast_flat = forecast_padded.flatten()

    # Bloque 3 — features temporales cíclicos (6 dims)
    hora = step % 24
    dia_sem = (step // 24) % 7
    mes = 0  # hardcoded durante el entrenamiento; mantener igual para inferencia
    temp_feats = np.array([
        np.sin(2 * np.pi * hora / 24),
        np.cos(2 * np.pi * hora / 24),
        np.sin(2 * np.pi * dia_sem / 7),
        np.cos(2 * np.pi * dia_sem / 7),
        np.sin(2 * np.pi * mes / 12),
        np.cos(2 * np.pi * mes / 12),
    ], dtype=np.float32)

    # Bloque 4 — margen solar a 24h (1 dim)
    window_clean = sim.get_data_window(step + 1, horizon=24)
    solar_exc_24h = float(np.sum(np.maximum(0.0, window_clean[:, 1] - window_clean[:, 0])))
    espacio_bat = max(0.0, (sim.SOC_MAX - soc) * sim.BATERIA_CAPACIDAD)
    margen_solar = np.float32(
        max(0.0, solar_exc_24h - espacio_bat) / sim.BATERIA_CAPACIDAD
    )

    obs = np.concatenate((
        np.array([soc, precio_compra, precio_venta, exc, def_], dtype=np.float32),
        forecast_flat,
        temp_feats,
        np.array([margen_solar], dtype=np.float32),
    ))
    return obs.astype(np.float32)
