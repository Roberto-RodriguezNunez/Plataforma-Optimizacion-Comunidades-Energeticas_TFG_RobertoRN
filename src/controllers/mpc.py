"""
mpc.py — Librería del MPC: LinearMPC (horizonte deslizante 24h) + crear_mpc.
============================================================================
MPC lineal (LP) usado como controlador BaseController (cota de referencia del RL)
y como base del residual. El benchmark ejecutable (runner + main) está en
src/evaluation/mpc_benchmark.py. Variantes: oráculo (previsión perfecta) y
realista (ruido AR(1) idéntico al RL).

Formulación LP (linprog minimiza):
  - Variables (4 × H = 96): cs[h], cm[h], dc[h], dr[h]  ∀h ∈ {0..H-1}
  - Objetivo: min -beneficio_marginal_vs_IDLE + valor_terminal
  - Restricciones:
      cs[h] + cm[h] ≤ P_MAX                     (potencia carga)
      dc[h] + dr[h] ≤ P_MAX                     (potencia descarga)
      SOC_MIN × CAP ≤ E[h+1] ≤ SOC_MAX × CAP   (límites SoC con autodescarga)
  - Bounds: cs ≤ exc, dc ≤ dfc/η_d, cm ≤ P_MAX, dr ≤ P_MAX

  Coeficientes de la función objetivo (derivación):
    Sea IDLE: vender excedente a pv, comprar déficit a pc.
    Beneficio marginal de cada variable:
      cs[h]: pierde venta solar (pv) + degradación (DEG)          → coef = +(pv + DEG)
      cm[h]: paga compra red (pc) + degradación (DEG)             → coef = +(pc + DEG)
      dc[h]: ahorra compra (pc × η_d) − degradación (DEG)        → coef = −(pc×η_d − DEG)
      dr[h]: ingresa venta (pv × η_d) − degradación (DEG)        → coef = −(pv×η_d − DEG)

  Valor terminal (si activado):
    V_T = λ × p_T × η_d × E[H]
    Añade a c_obj:
      cs[h], cm[h]: −λ × p_T × η_d × η_c × α^(H-1-h)   (cargar aumenta E[H])
      dc[h], dr[h]: +λ × p_T × η_d × α^(H-1-h)          (descargar reduce E[H])
"""

import os
import sys
import time
from typing import Dict, Optional

import numpy as np
from scipy.optimize import linprog

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.controllers.base import BaseController
from src.core.simulator import ComunidadSimulador
from src.core.forecast import (
    ventana_observada, generar_factores_precio, avanzar_ar1,
)
from src.core.config import cargar_system as _cargar_system

# --- Cargar configuración (fuente única src.core.config) ---
_CFG = _cargar_system()

# Parámetros de configuración
_BAT    = _CFG['bateria']
_MPC    = _CFG['mpc']
_PRON   = _CFG['pronostico']

DATASET_PATH   = _CFG['rutas']['dataset_final']
EPISODE_LENGTH = _MPC['duracion_episodio']
N_EPISODES     = _MPC['n_episodios']
HORIZON        = _MPC['horizonte']
SEED           = _CFG['etl']['semilla']
SOC_INICIAL    = _BAT['soc_inicial']

# Parámetros AR(1) — idénticos a energy_env.py
_SIGMA_SOL_H1    = _PRON['sigma_solar_h1']
_SIGMA_SOL_H24   = _PRON['sigma_solar_h24']
_SIGMA_CONS_BASE = _PRON['sigma_consumo']
_RHO_SOLAR       = _PRON['rho_solar']
_RHO_CONS        = _PRON['rho_consumo']

# Parámetros de ruido de precio — modelo 3 capas
# Justificación: PVPC se publica a las 20:30h del día anterior por REE.
# Antes de esa hora, las horas del día siguiente no están disponibles.
# Capa 1: ya publicado → exacto. Capa 2: OMIE intradiario → σ bajo.
# Capa 3: estimación estadística → σ alto.
_PRECIO_CFG         = _PRON.get('precio', {})
_HORA_PUBLICACION   = _PRECIO_CFG.get('hora_publicacion', 20.5)
_SIGMA_PRECIO_INTRA = _PRECIO_CFG.get('sigma_intradiario', 0.05)
_SIGMA_PRECIO_STAT  = _PRECIO_CFG.get('sigma_estadistico', 0.15)
_RHO_PRECIO_INTRA   = _PRECIO_CFG.get('rho_intradiario', 0.5)
_RHO_PRECIO_STAT    = _PRECIO_CFG.get('rho_estadistico', 0.7)
_MARGEN_INTRA_H     = _PRECIO_CFG.get('margen_intradiario_h', 6)

# Timestamps para obtener hora_actual del día (necesario para modelo 3 capas)
import pandas as _pd_bench
_TIMESTAMPS_BENCH = None
try:
    _df_ts = _pd_bench.read_csv(
        os.path.join(ROOT, DATASET_PATH), usecols=['fecha'], parse_dates=['fecha']
    )
    _TIMESTAMPS_BENCH = _df_ts['fecha']
except Exception:
    pass


def _get_hora_actual(step):
    """Obtiene la hora del día (0-23) para un step del dataset."""
    if _TIMESTAMPS_BENCH is not None and step < len(_TIMESTAMPS_BENCH):
        return _TIMESTAMPS_BENCH.iloc[step].hour
    return step % 24


# =====================================================================
# LinearMPC — Controlador MPC lineal con horizonte deslizante
# =====================================================================

class LinearMPC(BaseController):
    """
    Controlador MPC basado en programación lineal (LP) con horizonte 24h.

    El LP se resuelve cada hora con el SoC real medido del simulador.
    Solo se aplica la acción del primer paso (h=0) — horizonte rodante.

    Args:
        sim: Instancia de ComunidadSimulador (para parámetros físicos).
        use_terminal_value: Si True, añade valor terminal al LP.
        terminal_lambda: Peso del valor terminal (0.0 = desactivado).
        terminal_price_mode: 'ultimo', 'media_movil' o 'mediana_historica'.
        k_deg_lin: Coste lineal de degradación (EUR/kWh). Si None, usa
                   sim.COSTE_DEGRADACION_BASE.
    """

    def __init__(
        self,
        sim: ComunidadSimulador,
        use_terminal_value: bool = True,
        terminal_lambda: float = 1.0,
        terminal_price_mode: str = 'ultimo',
        k_deg_lin: Optional[float] = None,
    ):
        self._sim = sim
        self._use_tv = use_terminal_value
        self._lambda = terminal_lambda if use_terminal_value else 0.0
        self._tv_mode = terminal_price_mode
        self._k_deg = k_deg_lin if k_deg_lin is not None else sim.COSTE_DEGRADACION_BASE

        # Cachear parámetros del simulador
        self._EFF_C = sim.EFICIENCIA_CARGA
        self._EFF_D = sim.EFICIENCIA_DESCARGA
        self._CAP   = sim.BATERIA_CAPACIDAD
        self._P_MAX = sim.POTENCIA_INVERSOR
        self._ALPHA = 1.0 - sim.AUTODESCARGA_POR_HORA

        # Métricas de rendimiento del solver (buffer circular, no crece)
        self._solve_times_max = 1000
        self._solve_times = []

        # Pre-asignar matrices LP (H=24 fijo: 4*24=96 variables, 4*24=96 filas)
        H = HORIZON
        n = 4 * H
        n_rows = 4 * H  # 2 potencia + 2 SoC por hora
        self._A_ub_buf = np.zeros((n_rows, n), dtype=np.float64)
        self._b_ub_buf = np.zeros(n_rows, dtype=np.float64)
        self._c_obj_buf = np.zeros(n, dtype=np.float64)
        self._bounds_lo = np.zeros(n, dtype=np.float64)
        self._bounds_hi = np.zeros(n, dtype=np.float64)

        # Pre-calcular la estructura fija de A_ub (potencia carga/descarga)
        for h in range(H):
            # Potencia carga: cs[h] + cm[h] ≤ P_MAX
            self._A_ub_buf[4*h, 4*h] = 1.0
            self._A_ub_buf[4*h, 4*h+1] = 1.0
            # Potencia descarga: dc[h] + dr[h] ≤ P_MAX
            self._A_ub_buf[4*h+1, 4*h+2] = 1.0
            self._A_ub_buf[4*h+1, 4*h+3] = 1.0

    def solve(self, state: Dict, forecast: np.ndarray) -> Dict[str, float]:
        """
        Resuelve el LP y devuelve las acciones para h=0.

        Args:
            state: {'soc': float, 'step': int}
            forecast: (H, 4) array [consumo, generacion, precio_kwh, precio_excedente]

        Returns:
            {'P_carga_solar', 'P_carga_red', 'P_descarga_casa', 'P_descarga_red'}
        """
        cs, cm, dc, dr = self._solve_lp(state['soc'], forecast)
        return {
            'P_carga_solar':   cs,
            'P_carga_red':     cm,
            'P_descarga_casa': dc,
            'P_descarga_red':  dr,
        }

    def nombre(self) -> str:
        tv_str = f"λ={self._lambda}" if self._use_tv else "sin_TV"
        return f"MPC_LP({tv_str}, k_deg={self._k_deg:.4f})"

    def stats_solver(self) -> Dict:
        """Estadísticas del tiempo de resolución del solver."""
        if not self._solve_times:
            return {}
        t = np.array(self._solve_times)
        return {
            'media_ms':  float(t.mean() * 1000),
            'max_ms':    float(t.max() * 1000),
            'mediana_ms': float(np.median(t) * 1000),
            'n_solves':  len(t),
        }

    # -----------------------------------------------------------------
    # LP interno
    # -----------------------------------------------------------------
    def _solve_lp(self, soc_actual: float, window: np.ndarray):
        """
        LP de horizonte H sobre la ventana recibida.

        Variables por hora h (4 × H = 96 total):
          cs[h]: kWh de excedente solar → cargador (lado input inversor)
          cm[h]: kWh comprados de red → cargador (lado input inversor)
          dc[h]: kWh extraídos batería → casas (lado batería)
          dr[h]: kWh extraídos batería → red (lado batería)

        Retorna (cs0, cm0, dc0, dr0) para la hora actual.
        """
        t0 = time.perf_counter()

        H      = min(HORIZON, len(window))
        EFF_C  = self._EFF_C
        EFF_D  = self._EFF_D
        CAP    = self._CAP
        P_MAX  = self._P_MAX
        DEG    = self._k_deg
        ALPHA  = self._ALPHA
        LAM    = self._lambda

        consumo    = window[:H, 0]
        generacion = window[:H, 1]
        precio_c   = window[:H, 2]   # PVPC (compra)
        precio_v   = window[:H, 3]   # Excedentaria (venta)

        balance = generacion - consumo
        exc = np.maximum(0,  balance)
        dfc = np.maximum(0, -balance)

        def i_cs(h): return 4 * h
        def i_cm(h): return 4 * h + 1
        def i_dc(h): return 4 * h + 2
        def i_dr(h): return 4 * h + 3

        n = 4 * H

        # =============================================================
        # FUNCIÓN OBJETIVO (linprog minimiza) — reutiliza buffer
        # =============================================================
        c_obj = self._c_obj_buf
        c_obj[:] = 0.0
        for h in range(H):
            pv = precio_v[h]
            pc = precio_c[h]
            c_obj[i_cs(h)] =  pv + DEG             # pierde venta solar + degradación
            c_obj[i_cm(h)] =  pc + DEG             # paga compra red + degradación
            c_obj[i_dc(h)] = -(pc * EFF_D - DEG)   # ahorra compra × η_d − degradación
            c_obj[i_dr(h)] = -(pv * EFF_D - DEG)   # ingresa venta × η_d − degradación

        # Valor terminal: V_T = λ × p_T × η_d × E[H]
        if LAM > 0:
            p_terminal = self._precio_terminal(precio_c, H)
            tv_base = LAM * p_terminal * EFF_D
            for h in range(H):
                decay = ALPHA ** (H - 1 - h)
                tv_charge    = tv_base * EFF_C * decay
                tv_discharge = tv_base * decay
                c_obj[i_cs(h)] -= tv_charge
                c_obj[i_cm(h)] -= tv_charge
                c_obj[i_dc(h)] += tv_discharge
                c_obj[i_dr(h)] += tv_discharge

        # =============================================================
        # BOUNDS — reutiliza buffers
        # =============================================================
        lo = self._bounds_lo
        hi = self._bounds_hi
        lo[:] = 0.0
        for h in range(H):
            hi[i_cs(h)] = exc[h]
            hi[i_cm(h)] = P_MAX
            hi[i_dc(h)] = dfc[h] / EFF_D if dfc[h] > 0 else 0.0
            hi[i_dr(h)] = P_MAX
        bounds = list(zip(lo, hi))

        # =============================================================
        # RESTRICCIONES (A_ub × x ≤ b_ub) — reutiliza buffers
        # =============================================================
        A_ub = self._A_ub_buf
        b_ub = self._b_ub_buf
        E0 = soc_actual * CAP

        for h in range(H):
            # Potencia carga/descarga (estructura fija, solo actualizar b)
            b_ub[4*h]   = P_MAX
            b_ub[4*h+1] = P_MAX

            # SoC acumulado con autodescarga
            E0_decayed = (ALPHA ** (h + 1)) * E0
            row_idx_max = 4*h + 2
            row_idx_min = 4*h + 3

            # Limpiar filas SoC (la estructura cambia con h)
            A_ub[row_idx_max, :] = 0.0
            A_ub[row_idx_min, :] = 0.0

            for k in range(h + 1):
                decay = ALPHA ** (h - k)
                A_ub[row_idx_max, i_cs(k)] =  EFF_C * decay
                A_ub[row_idx_max, i_cm(k)] =  EFF_C * decay
                A_ub[row_idx_max, i_dc(k)] = -1.0   * decay
                A_ub[row_idx_max, i_dr(k)] = -1.0   * decay

                A_ub[row_idx_min, i_cs(k)] = -EFF_C * decay
                A_ub[row_idx_min, i_cm(k)] = -EFF_C * decay
                A_ub[row_idx_min, i_dc(k)] =  1.0   * decay
                A_ub[row_idx_min, i_dr(k)] =  1.0   * decay

            b_ub[row_idx_max] = self._sim.SOC_MAX * CAP - E0_decayed
            b_ub[row_idx_min] = E0_decayed - self._sim.SOC_MIN * CAP

        # =============================================================
        # RESOLVER
        # =============================================================
        res = linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'disp': False},
        )

        elapsed = time.perf_counter() - t0
        if len(self._solve_times) >= self._solve_times_max:
            self._solve_times.pop(0)
        self._solve_times.append(elapsed)

        if not res.success:
            # Diagnóstico para debugging (no debería ocurrir en operación normal)
            raise RuntimeError(
                f"LP infactible: status={res.status}, message='{res.message}', "
                f"soc={soc_actual:.4f}, E0={E0:.2f} kWh, "
                f"exc_h0={exc[0]:.2f}, dfc_h0={dfc[0]:.2f}, "
                f"pc_h0={precio_c[0]:.4f}, pv_h0={precio_v[0]:.4f}"
            )

        x = res.x
        return x[i_cs(0)], x[i_cm(0)], x[i_dc(0)], x[i_dr(0)]

    def _precio_terminal(self, precio_c: np.ndarray, H: int) -> float:
        """Calcula el precio de referencia para el valor terminal."""
        if self._tv_mode == 'ultimo':
            # Modo "a": precio de compra en la última hora del horizonte.
            # Coherente con la corrección terminal del simulador (energy_env.py:237).
            return float(precio_c[H - 1])
        elif self._tv_mode == 'media_movil':
            # Modo "b": media de precio de compra en el horizonte.
            return float(precio_c[:H].mean())
        elif self._tv_mode == 'mediana_historica':
            # Modo "c": aproxima la mediana histórica con la mediana del
            # horizonte actual (modo alternativo; la config usa "ultimo").
            return float(np.median(precio_c[:H]))
        else:
            raise ValueError(f"modo_precio desconocido: {self._tv_mode}")


# =====================================================================
# Funciones auxiliares de ejecución
# =====================================================================

# El ruido de pronóstico (AR(1) solar/consumo + precio 3 capas) vive ahora en la
# FUENTE ÚNICA src/core/forecast.py (avanzar_ar1, generar_factores_precio,
# aplicar_ruido_ventana, ventana_observada). Las antiguas aplicar_ruido_ar1 y
# aplicar_ruido_precio_3capas se eliminaron al unificar (eran equivalentes).


def simular_hora_mpc(sim, cs, cm, dc, dr):
    """
    Aplica los flujos del LP al simulador con física completa.
    Retorna (beneficio_absoluto, beneficio_marginal_vs_idle).
    """
    # Física desde la FUENTE ÚNICA (idéntica a la del entorno continuo del SAC)
    r = sim.aplicar_fisica_4flujos(cs, cm, dc, dr)
    sim.current_step += 1
    return r['beneficio'], r['beneficio_marginal']


def simular_semana_idle(sim_idle, start):
    """IDLE: vende excedente y compra déficit sin usar batería."""
    sim_idle.current_step = start
    total = 0.0
    for _ in range(EPISODE_LENGTH):
        row = sim_idle.df.iloc[sim_idle.current_step]
        gen  = row['generacion_total']
        cons = row['consumo_total']
        bal  = gen - cons
        total += max(0, bal) * row['precio_excedente'] - max(0, -bal) * row['precio_kwh']
        sim_idle.current_step += 1
    return total


def crear_mpc(sim=None):
    """ÚNICA factoría del MPC desde config — garantiza que entreno, evaluación y
    producción usan EXACTAMENTE el mismo controlador MPC (mismas condiciones).

    Si no se pasa `sim`, crea uno fresco desde DATASET_PATH (entreno/eval).
    En producción se pasa el sim del feed (`crear_mpc(feed.sim)`).
    """
    if sim is None:
        sim = ComunidadSimulador(DATASET_PATH)
    tv = _MPC['valor_terminal']
    return LinearMPC(
        sim,
        use_terminal_value=tv['activado'],
        terminal_lambda=tv['lambda'],
        terminal_price_mode=tv['modo_precio'],
        k_deg_lin=_MPC['k_deg_lin'],
    )
