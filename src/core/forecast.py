"""
forecast.py — Fuente ÚNICA del pronóstico observado (ventana 24h con ruido).
Compartida por env, MPC y residual_env → por construcción ven los MISMOS datos.

Invariantes que NO se deben romper:
  - window[0] (hora actual) lleva ruido: es el PRIMER paso de pronóstico, NO un
    dato exacto → sin lookahead. La física/recompensa sí usan el valor real (lo
    lee el simulador, no esta ventana).
  - Las funciones de ventana son PURAS (deterministas dados error_solar/_cons y
    precio_factores). El avance AR(1) y los factores de precio (que consumen RNG)
    van aparte, para reconstruir la misma ventana varias veces por paso (MPC + obs).
  - Ruido: solar AR(1) creciente (σ 5%→25%), consumo AR(1) fijo (15%), precio en
    3 capas según publicación del PVPC.
"""

import numpy as np

from src.core.config import pronostico as _pronostico

# --- Constantes de ruido (ÚNICA definición; desde config/system.yaml vía
#     src.core.config — antes hardcodeadas aquí y duplicadas en env y mpc) ---
_PRON = _pronostico()
_PRECIO = _PRON.get("precio", {})

SIGMA_CONS_BASE = _PRON["sigma_consumo"]    # 15% MAE consumo, ~plano con el horizonte
SIGMA_SOL_H1    = _PRON["sigma_solar_h1"]   # 5%  error solar en el primer paso de pronóstico
SIGMA_SOL_H24   = _PRON["sigma_solar_h24"]  # 25% error solar en el último paso del horizonte
RHO_SOLAR = _PRON["rho_solar"]              # nubes persisten varias horas
RHO_CONS  = _PRON["rho_consumo"]            # consumo más variable, patrón horario domina

# Precio PVPC — modelo 3 capas (publicación a las 20:30 del día anterior)
HORA_PUBLICACION    = _PRECIO.get("hora_publicacion", 20.5)
SIGMA_PRECIO_INTRA  = _PRECIO.get("sigma_intradiario", 0.05)   # Capa 2: OMIE intradiario
SIGMA_PRECIO_STAT   = _PRECIO.get("sigma_estadistico", 0.15)   # Capa 3: estimación estadística
RHO_PRECIO_INTRA    = _PRECIO.get("rho_intradiario", 0.5)
RHO_PRECIO_STAT     = _PRECIO.get("rho_estadistico", 0.7)
MARGEN_INTRA_H      = _PRECIO.get("margen_intradiario_h", 6)   # horas cubiertas por OMIE intraday

HORIZON = 24


def avanzar_ar1(error_solar, error_cons, noise_fn):
    """Avanza un paso los estados AR(1) del error de pronóstico solar/consumo.

        ε_t = ρ·ε_{t-1} + √(1-ρ²)·N(0,1)   (varianza estacionaria = 1)

    Devuelve (error_solar, error_cons) actualizados. Consume 2 draws de noise_fn.
    """
    error_solar = RHO_SOLAR * error_solar + np.sqrt(1 - RHO_SOLAR ** 2) * noise_fn()
    error_cons  = RHO_CONS  * error_cons  + np.sqrt(1 - RHO_CONS  ** 2) * noise_fn()
    return error_solar, error_cons


def sigma_solar(k, horizon=HORIZON):
    """σ del ruido solar en el paso k del horizonte (crece lineal 5%→25%)."""
    return SIGMA_SOL_H1 + k * ((SIGMA_SOL_H24 - SIGMA_SOL_H1) / (horizon - 1))


def generar_factores_precio(hora_actual, noise_fn, horizon=HORIZON):
    """Genera los factores multiplicativos de ruido de precio (3 capas).

    factores[0] = 1.0 (presente exacto). factores[k] para k>=1 según capa.
    El AR(1) interno avanza hora a hora con ρ distinto por capa, de modo que
    cambia el ranking de precios entre horas. Consume RNG (noise_fn).
    """
    if hora_actual >= HORA_PUBLICACION:
        horas_publicadas = 24 + (24 - hora_actual)
    else:
        horas_publicadas = 24 - hora_actual

    factores = np.ones(horizon)
    eps = 0.0
    for k in range(horizon):              # k=0 = hora actual (normalmente publicada → exacta)
        if k < horas_publicadas:
            continue                      # Capa 1: precio publicado exacto
        elif k < horas_publicadas + MARGEN_INTRA_H:
            rho, sigma = RHO_PRECIO_INTRA, SIGMA_PRECIO_INTRA
        else:
            rho, sigma = RHO_PRECIO_STAT, SIGMA_PRECIO_STAT
        eps = rho * eps + np.sqrt(1 - rho ** 2) * noise_fn()
        factores[k] = 1.0 + sigma * eps
    return factores


def aplicar_ruido_ventana(window, error_solar, error_cons, precio_factores):
    """Aplica el ruido (cons/gen desde k=0; precio por capas vía precio_factores)
    a una ventana ya ensamblada. Modifica y devuelve `window`."""
    horizon = len(window)
    for k in range(horizon):
        s_sol = sigma_solar(k, horizon)
        window[k, 1] = max(0.0, window[k, 1] * (1.0 + error_solar * s_sol))            # generacion
        window[k, 0] = max(0.0, window[k, 0] * (1.0 + error_cons * SIGMA_CONS_BASE))   # consumo
        f = precio_factores[k]
        if f != 1.0:
            window[k, 2] = max(0.0, window[k, 2] * f)   # precio_kwh
            window[k, 3] = max(0.0, window[k, 3] * f)   # precio_excedente
    return window


def ventana_observada(sim, step, error_solar, error_cons, precio_factores,
                      con_ruido, horizon=HORIZON):
    """Ventana (horizon, 4) = [consumo, generacion, precio_kwh, precio_excedente]
    que ven TODOS los controladores. PURA (determinista dados los errores y
    factores). con_ruido=False → ventana cruda (oráculo, previsión perfecta)."""
    window = sim.get_data_window(step, horizon=horizon).copy()
    if not con_ruido:
        return window
    return aplicar_ruido_ventana(window, error_solar, error_cons, precio_factores)
