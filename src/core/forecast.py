"""
forecast.py — Fuente ÚNICA de verdad del pronóstico observado
=============================================================
Toda la lógica de "qué ventana de pronóstico ve un controlador" vive aquí.
La usan por igual el entorno de RL (`energy_env`), el benchmark MPC
(`mpc_benchmark`) y el wrapper residual (`residual_env`), de modo que es
IMPOSIBLE por construcción que reciban datos distintos → comparación justa.

Convención (única y documentada):
  - El controlador decide la acción de cada hora a partir de su PRONÓSTICO, no
    del valor realizado. Por eso TODA la ventana, incluida la hora actual
    (window[0]), lleva ruido de pronóstico: la hora actual es el PRIMER paso de
    pronóstico (σ mínima), no un dato exacto. Esto evita cualquier lookahead y
    garantiza que el MPC y el agente reciben EXACTAMENTE la misma información.
    La física y la recompensa sí usan el valor REAL de la hora (el resultado
    realizado), que se lee directamente del simulador, no de esta ventana.
  - Ruido (k = 0..H-1):
       * solar:   ruido multiplicativo AR(1) que crece con el horizonte
                  (σ: 5% en k=0 → 25% en k=23).
       * consumo: ruido multiplicativo AR(1) fijo (15%).
       * precio:  modelo de 3 capas según publicación del PVPC (la hora actual
                  suele estar publicada → exacta).

Las funciones de construcción de ventana son PURAS (deterministas dadas
`error_solar`, `error_cons` y `precio_factores`). El avance del estado AR(1)
y la generación de factores de precio (que consumen RNG) están separados, para
que un mismo paso pueda construir la ventana varias veces (MPC base + obs del
agente) y obtener exactamente el mismo resultado.
"""

import numpy as np

# --- Constantes de ruido (ÚNICA definición; antes duplicadas en env y mpc) ---
SIGMA_CONS_BASE = 0.15   # 15% MAE consumo, ~plano con el horizonte
SIGMA_SOL_H1    = 0.05   # 5%  error solar en el primer paso de pronóstico
SIGMA_SOL_H24   = 0.25   # 25% error solar en el último paso del horizonte
RHO_SOLAR = 0.7          # nubes persisten varias horas
RHO_CONS  = 0.3          # consumo más variable, patrón horario domina

# Precio PVPC — modelo 3 capas (publicación a las 20:30 del día anterior)
HORA_PUBLICACION    = 20.5
SIGMA_PRECIO_INTRA  = 0.05   # Capa 2: OMIE intradiario
SIGMA_PRECIO_STAT   = 0.15   # Capa 3: estimación estadística
RHO_PRECIO_INTRA    = 0.5
RHO_PRECIO_STAT     = 0.7
MARGEN_INTRA_H      = 6      # horas cubiertas por OMIE intraday

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
    """σ del ruido solar para el paso k del horizonte.

    k=0 = hora actual: es el PRIMER paso de pronóstico (σ mínima), NO un dato
    exacto. El controlador decide la acción de la hora a partir de su pronóstico
    (no conoce el consumo/generación realizados de la hora hasta que termina);
    la física/recompensa sí usan el valor real. Así no hay lookahead y el MPC y
    el agente reciben exactamente la misma información.
    """
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
    """Aplica el ruido de pronóstico a una ventana YA ENSAMBLADA.

    El ruido de consumo/generación se aplica DESDE k=0 (la hora actual es el
    primer paso de pronóstico, con σ mínima): el controlador decide a partir del
    pronóstico, no del valor realizado → sin lookahead, e idéntico para MPC y
    agente. El precio de la hora actual queda exacto si ya está publicado
    (lo gestionan los `precio_factores`, 3 capas). Modifica y devuelve `window`.
    Usada por `ventana_observada` (env/MPC) y por el feed de producción
    (`api/data_feed`).
    """
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
    """ÚNICA fuente de verdad de la ventana (horizon, 4) que ven TODOS los
    controladores. Columnas: [consumo, generacion, precio_kwh, precio_excedente].

    Función PURA: dado (error_solar, error_cons, precio_factores) el resultado
    es determinista, así que env y MPC obtienen byte a byte la misma ventana.

    Todas las horas (incluida la actual, k=0) llevan ruido de pronóstico — el
    presente es el primer paso de pronóstico, no un dato exacto. Si
    con_ruido=False (oráculo) devuelve la ventana cruda (previsión perfecta).
    """
    window = sim.get_data_window(step, horizon=horizon).copy()
    if not con_ruido:
        return window
    return aplicar_ruido_ventana(window, error_solar, error_cons, precio_factores)
