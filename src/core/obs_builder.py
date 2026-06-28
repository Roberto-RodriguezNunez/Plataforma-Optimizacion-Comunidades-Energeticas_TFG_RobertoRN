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

    ÚNICA fuente de verdad de la obs: la usa por igual el entorno de entreno
    (`energy_env._get_obs`) y las rutas de inferencia (ResidualSACController /
    OnnxResidualController), de modo que entreno == evaluación == producción.

    Opera SOBRE LA VENTANA recibida (`forecast`), que ya viene de la fuente única
    `forecast.ventana_observada` con la convención: forecast[0] = presente EXACTO
    (neto medido en el CUPS único), forecast[k>=1] con ruido. Así el presente es
    exacto para TODOS los controladores y el futuro lleva el mismo ruido.

    Estructura:
      [0:5]    Estado y precios actuales   (5 dims) — derivado de forecast[0]
      [5:101]  Forecast 24h aplanado       (96 dims = 24 × 4 cols)
      [101:107] Features temporales cíclicos (6 dims) — desde sim.get_tiempo
      [107]    Margen solar a 24h          (1 dim) — sobre la MISMA ventana

    Args:
        state:    {'soc': float, 'step': int}
        forecast: (H, 4) float array — [consumo, generacion, precio_kwh, precio_exc]
        sim:      ComunidadSimulador — acceso a tiempo y parámetros físicos

    Returns:
        np.ndarray de shape (108,) dtype float32
    """
    soc = state['soc']
    step = state['step']
    window = np.asarray(forecast, dtype=np.float64)

    # Bloque 1 — estado y precios actuales (5 dims), desde forecast[0] (hora actual del pronóstico)
    cons, gen, precio_compra, precio_venta = window[0]
    balance = gen - cons
    exc = max(0.0, balance)
    def_ = abs(min(0.0, balance))

    # Bloque 2 — forecast 24h aplanado (96 dims)
    H = min(24, len(window))
    forecast_padded = np.zeros((24, 4), dtype=np.float32)
    forecast_padded[:H] = window[:H]
    forecast_flat = forecast_padded.flatten()

    # Bloque 3 — features temporales cíclicos (6 dims) — fuente única sim.get_tiempo
    hora, dia_sem, mes = sim.get_tiempo(step)
    temp_feats = np.array([
        np.sin(2 * np.pi * hora / 24),
        np.cos(2 * np.pi * hora / 24),
        np.sin(2 * np.pi * dia_sem / 7),
        np.cos(2 * np.pi * dia_sem / 7),
        np.sin(2 * np.pi * mes / 12),
        np.cos(2 * np.pi * mes / 12),
    ], dtype=np.float32)

    # Bloque 4 — margen solar a 24h (1 dim), sobre la MISMA ventana (sin fuga de futuro)
    solar_exc_24h = float(np.sum(np.maximum(0.0, forecast_padded[:, 1] - forecast_padded[:, 0])))
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
