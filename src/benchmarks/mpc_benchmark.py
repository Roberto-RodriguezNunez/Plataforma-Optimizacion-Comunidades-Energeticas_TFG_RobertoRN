"""
mpc_benchmark.py — Benchmark MPC de horizonte deslizante (24h)
===============================================================
Implementa un MPC lineal (LP) como controlador BaseController para
contextualizar los resultados del RL.

Dos variantes ejecutables desde una sola clase LinearMPC:
  1. Oráculo (forecast_mode="oraculo"):  previsión perfecta → cota superior teórica.
  2. Realista (forecast_mode="realista"): ruido AR(1) idéntico al RL → comparación justa.

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

Ejecución desde la raíz del proyecto (carpeta TFG/):
    python src/benchmarks/mpc_benchmark.py
"""

import os
import sys
import time
from typing import Dict, Optional

import numpy as np
import yaml
from scipy.optimize import linprog

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.controllers.base import BaseController
from src.core.simulador import ComunidadSimulador
from src.core.forecast import (
    ventana_observada, generar_factores_precio, avanzar_ar1,
)

# --- Cargar configuración ---
_CONFIG_PATH = os.path.join(ROOT, 'config', 'system.yaml')
with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
    _CFG = yaml.safe_load(f)

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
            # Modo "c": mediana global (se debería calcular offline sobre train).
            # Placeholder: usa la mediana del horizonte actual.
            return float(np.median(precio_c[:H]))
        else:
            raise ValueError(f"modo_precio desconocido: {self._tv_mode}")


# =====================================================================
# Funciones auxiliares de ejecución
# =====================================================================

def aplicar_ruido_ar1(window, error_solar, error_cons):
    """
    Aplica el mismo ruido AR(1) que energy_env._get_obs() a la ventana de 24h.
    Precios (columnas 2 y 3) no se tocan — publicados por REE el día anterior.
    """
    w = window.copy()
    for h in range(len(w)):
        sigma_sol = _SIGMA_SOL_H1 + h * ((_SIGMA_SOL_H24 - _SIGMA_SOL_H1) / 23)
        w[h, 1] = max(0.0, w[h, 1] * (1.0 + error_solar * sigma_sol))
        w[h, 0] = max(0.0, w[h, 0] * (1.0 + error_cons * _SIGMA_CONS_BASE))
    return w


def aplicar_ruido_precio_3capas(window, hora_actual, noise_fn,
                                start_offset=0):
    """
    Aplica ruido AR(1) a precios con modelo de 3 capas según publicación PVPC.
    El estado AR(1) eps avanza HORA A HORA dentro del forecast, con un draw
    nuevo por cada hora y ρ distinto por capa. Esto hace que el ranking de
    precios entre horas cambie, no solo el nivel general.

    El PVPC del día siguiente se publica a las 20:30h del día anterior por REE.
    Antes de esa publicación, el agente/MPC no conoce los precios exactos.

    Capas:
      1. Horas ya publicadas (precio exacto conocido): sin ruido.
      2. Hasta 6h post-publicación (OMIE intradiario): σ=0.05, ρ=0.5.
      3. Resto del horizonte (estimación estadística): σ=0.15, ρ=0.7.

    Args:
        window: (H, 4+) array. Columnas 2 y 3 se modifican in-place.
        hora_actual: Hora del día (0-23) del paso de decisión actual.
        noise_fn: Callable que devuelve N(0,1). Puede ser rng.standard_normal
                  (seeded, para MPC/eval) o np.random.normal (global, para env).
        start_offset: 0 si window[0]=hora actual (MPC/eval),
                      1 si window[0]=hora+1 (energy_env._get_obs).
    """
    if hora_actual >= _HORA_PUBLICACION:
        horas_publicadas = 24 + (24 - hora_actual)
    else:
        horas_publicadas = 24 - hora_actual

    eps = 0.0

    for h in range(len(window)):
        hora_forecast = h + start_offset  # horas adelante desde hora_actual

        if hora_forecast < horas_publicadas:
            # Capa 1: precio publicado exacto — sin ruido
            continue
        elif hora_forecast < horas_publicadas + _MARGEN_INTRA_H:
            # Capa 2: OMIE intradiario — ruido bajo, ρ=0.5
            rho = _RHO_PRECIO_INTRA
            sigma = _SIGMA_PRECIO_INTRA
        else:
            # Capa 3: estimación estadística — ruido alto, ρ=0.7
            rho = _RHO_PRECIO_STAT
            sigma = _SIGMA_PRECIO_STAT

        # Avanzar AR(1) por hora: eps_h = ρ·eps_{h-1} + √(1-ρ²)·N(0,1)
        eps = rho * eps + np.sqrt(1 - rho**2) * noise_fn()
        factor = 1.0 + sigma * eps
        window[h, 2] = max(0.0, window[h, 2] * factor)  # precio_kwh
        window[h, 3] = max(0.0, window[h, 3] * factor)  # precio_excedente


def simular_hora_mpc(sim, cs, cm, dc, dr):
    """
    Aplica los flujos del LP al simulador con física completa.
    Retorna (beneficio_absoluto, beneficio_marginal_vs_idle).
    """
    step = sim.current_step
    row  = sim.df.iloc[step]
    gen  = row['generacion_total']
    cons = row['consumo_total']
    precio_compra = row['precio_kwh']
    precio_venta  = row['precio_excedente']

    balance  = gen - cons
    exc_disp = max(0,  balance)
    def_cub  = max(0, -balance)

    EFF_C = sim.EFICIENCIA_CARGA
    EFF_D = sim.EFICIENCIA_DESCARGA
    CAP   = sim.BATERIA_CAPACIDAD
    AUTO  = sim.AUTODESCARGA_POR_HORA

    # Autodescarga
    sim.soc *= (1 - AUTO)
    bateria_kwh    = sim.soc * CAP
    espacio_libre  = max(0, sim.SOC_MAX * CAP - bateria_kwh)
    bat_disponible = max(0, bateria_kwh - sim.SOC_MIN * CAP)

    # Netear carga vs descarga — inversor bidireccional ejecuta potencia
    # neta, carga y descarga simultánea es imposible (coherente con
    # energy_env_continuo.step líneas 100-116).
    carga_bruta = cs + cm
    descarga_bruta = dc + dr
    net = carga_bruta - descarga_bruta
    if net >= 0:
        ratio_solar = cs / carga_bruta if carga_bruta > 0 else 0.0
        cs = net * ratio_solar
        cm = net * (1 - ratio_solar)
        dc, dr = 0.0, 0.0
    else:
        ratio_casa = dc / descarga_bruta if descarga_bruta > 0 else 0.0
        dc = abs(net) * ratio_casa
        dr = abs(net) * (1 - ratio_casa)
        cs, cm = 0.0, 0.0

    # Recortar por estado real de la batería
    cs = min(cs, exc_disp, espacio_libre / EFF_C)
    cm = min(cm, max(0, espacio_libre / EFF_C - cs))
    carga_total = cs + cm

    dc = min(dc, bat_disponible)
    dr = min(dr, max(0, bat_disponible - dc))
    descarga_total = dc + dr

    # Actualizar batería
    soc_antes = sim.soc
    bateria_kwh += carga_total * EFF_C - descarga_total
    sim.soc = float(np.clip(bateria_kwh / CAP, 0.0, 1.0))

    # Flujos económicos
    comprado = max(0, def_cub - dc * EFF_D) + cm
    vendido  = (exc_disp - cs) + dr * EFF_D

    ingresos = vendido * precio_venta
    gastos   = comprado * precio_compra

    # Degradación no lineal completa (igual que simulador)
    soc_medio      = (soc_antes + sim.soc) / 2
    energia_movida = carga_total + descarga_total
    coste_deg      = sim.calcular_degradacion_no_lineal(energia_movida, soc_medio)

    beneficio      = ingresos - gastos - coste_deg
    beneficio_idle = exc_disp * precio_venta - def_cub * precio_compra

    sim.current_step += 1
    return beneficio, beneficio - beneficio_idle


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


def correr_episodios(
    sim, sim_idle, rng, mpc: LinearMPC, forecast_mode: str = 'oraculo',
    pool: str = 'eval'
):
    """
    Ejecuta episodios sobre un pool de semanas y devuelve arrays de beneficios.

    Args:
        sim: Simulador principal (se modifica soc y current_step).
        sim_idle: Simulador para calcular baseline IDLE (solo lectura + step).
        rng: Generador de números aleatorios (numpy).
        mpc: Instancia de LinearMPC configurada.
        forecast_mode: 'oraculo' (datos perfectos) o 'realista' (ruido AR(1)).
        pool: 'eval' (semanas eval, determinista) o 'train' (semanas train,
              muestreo aleatorio N_EPISODES veces).
    """
    con_ruido = (forecast_mode == 'realista')

    if pool == 'eval':
        starts = sim.semanas_eval
    else:
        starts = [sim.semanas_train[int(rng.integers(len(sim.semanas_train)))]
                  for _ in range(N_EPISODES)]

    bens_mpc, bens_idle, bens_marg = [], [], []

    for ep, start in enumerate(starts):

        # IDLE
        ben_idle = simular_semana_idle(sim_idle, start)

        # MPC
        sim.current_step = start
        sim.soc = SOC_INICIAL
        error_solar = 0.0
        error_cons  = 0.0
        ben_mpc  = 0.0
        ben_marg = 0.0

        for _ in range(EPISODE_LENGTH):
            # Pronóstico desde la FUENTE ÚNICA: hora actual = primer paso de
            # pronóstico (con ruido). Idéntico a lo que ve el agente (energy_env).
            if con_ruido:
                error_solar, error_cons = avanzar_ar1(
                    error_solar, error_cons, rng.standard_normal)
                hora_actual = _get_hora_actual(sim.current_step)
                factores = generar_factores_precio(hora_actual, rng.standard_normal)
            else:
                factores = None
            window = ventana_observada(
                sim, sim.current_step, error_solar, error_cons,
                factores, con_ruido, horizon=HORIZON)

            # Resolver LP con SoC real medido y ejecutar
            state = {'soc': sim.soc, 'step': sim.current_step}
            action = mpc.solve(state, window)
            cs = action['P_carga_solar']
            cm = action['P_carga_red']
            dc = action['P_descarga_casa']
            dr = action['P_descarga_red']

            b, bm = simular_hora_mpc(sim, cs, cm, dc, dr)
            ben_mpc  += b
            ben_marg += bm

        bens_mpc.append(ben_mpc)
        bens_idle.append(ben_idle)
        bens_marg.append(ben_marg)

        if (ep + 1) % 100 == 0:
            print(f"     episodio {ep + 1}/{len(starts)} completado")

    return np.array(bens_mpc), np.array(bens_idle), np.array(bens_marg)


def imprimir_resultado(label, bens_mpc, bens_idle, bens_marg):
    print(f"\n  [{label}]")
    print(f"    IDLE puro  : {bens_idle.mean():+.2f} +/- {bens_idle.std():.2f} EUR/semana")
    print(f"    MPC total  : {bens_mpc.mean():+.2f} +/- {bens_mpc.std():.2f} EUR/semana")
    print(f"    MPC - IDLE : {bens_marg.mean():+.2f} +/- {bens_marg.std():.2f} EUR/semana"
          "  <- misma metrica que DQN reward")


def calibrate_deg(
    sim, rng,
    k_grid=None,
    n_semanas: int = 20,
    terminal_lambda: float = 0.0,
):
    """
    Calibra K_DEG_LIN ejecutando MPC oráculo con distintos valores sobre
    n_semanas del pool train. Retorna el K que maximiza beneficio marginal.

    NOTA: usa semanas del pool train del simulador.

    Args:
        sim: ComunidadSimulador (con pools calculados).
        rng: Generador aleatorio.
        k_grid: Lista de valores a probar.
        n_semanas: Número de semanas de calibración (del pool train).
        terminal_lambda: Lambda para valor terminal durante calibración.

    Returns:
        (mejor_k, resultados_dict)
    """
    if k_grid is None:
        k_grid = [0.003, 0.004, 0.005, 0.006, 0.008, 0.010]

    # Muestrear semanas del pool train
    train_pool = sim.semanas_train
    n_semanas = min(n_semanas, len(train_pool))
    idxs = rng.choice(len(train_pool), n_semanas, replace=False)
    starts = [train_pool[i] for i in idxs]

    resultados = {}
    for k in k_grid:
        mpc = LinearMPC(sim, use_terminal_value=(terminal_lambda > 0),
                        terminal_lambda=terminal_lambda, k_deg_lin=k)
        total_marg = 0.0
        for start in starts:
            sim.current_step = start
            sim.soc = SOC_INICIAL
            sem_marg = 0.0
            for _ in range(EPISODE_LENGTH):
                # Calibración determinista (sin ruido) vía la fuente única
                window = ventana_observada(
                    sim, sim.current_step, 0.0, 0.0, None,
                    con_ruido=False, horizon=HORIZON)
                state = {'soc': sim.soc, 'step': sim.current_step}
                action = mpc.solve(state, window)
                _, bm = simular_hora_mpc(
                    sim, action['P_carga_solar'], action['P_carga_red'],
                    action['P_descarga_casa'], action['P_descarga_red'])
                sem_marg += bm
            total_marg += sem_marg

        media = total_marg / n_semanas
        resultados[k] = media
        print(f"     K_DEG={k:.4f} -> beneficio marginal medio = {media:.2f} EUR/sem")

    mejor_k = max(resultados, key=resultados.get)
    print(f"\n  Mejor K_DEG_LIN = {mejor_k:.4f} ({resultados[mejor_k]:.2f} EUR/sem)")
    return mejor_k, resultados


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 62)
    print("BENCHMARK MPC -- horizonte 24h")
    print("=" * 62)

    sim      = ComunidadSimulador(DATASET_PATH)
    sim_idle = ComunidadSimulador(DATASET_PATH)

    n_eval  = len(sim.semanas_eval)
    n_train = len(sim.semanas_train)
    print(f"\n  Split por semanas: {n_train} train + {n_eval} eval")

    # Configuracion del MPC desde YAML
    tv_cfg = _MPC['valor_terminal']
    use_tv = tv_cfg['activado']
    lam    = tv_cfg['lambda']
    k_deg  = _MPC['k_deg_lin']

    print(f"  Horizonte:      {HORIZON}h")
    print(f"  Valor terminal: {'activado' if use_tv else 'desactivado'}"
          f" (lam={lam}, modo='{tv_cfg['modo_precio']}')")
    print(f"  K_DEG_LIN:      {k_deg}")
    print(f"  SoC inicial:    {SOC_INICIAL}")

    mpc = LinearMPC(
        sim,
        use_terminal_value=use_tv,
        terminal_lambda=lam,
        terminal_price_mode=tv_cfg['modo_precio'],
        k_deg_lin=k_deg,
    )

    rng_perf  = np.random.default_rng(SEED)
    rng_ruido = np.random.default_rng(SEED)

    print(f"\n  Ejecutando MPC oraculo sobre {n_eval} semanas eval...")
    mpc_p, idle_p, marg_p = correr_episodios(
        sim, sim_idle, rng_perf, mpc, forecast_mode='oraculo', pool='eval')

    print(f"\n  Ejecutando MPC realista sobre {n_eval} semanas eval...")
    mpc_r, idle_r, marg_r = correr_episodios(
        sim, sim_idle, rng_ruido, mpc, forecast_mode='realista', pool='eval')

    print("\n" + "=" * 62)
    print(f"RESULTADOS ({n_eval} semanas eval, muestreo aleatorio)")
    print("=" * 62)
    imprimir_resultado("MPC oraculo -- cota superior teorica", mpc_p, idle_p, marg_p)
    imprimir_resultado("MPC realista AR(1) -- comparacion justa con RL", mpc_r, idle_r, marg_r)

    print()
    gap = marg_p.mean() - marg_r.mean()
    print(f"  Coste del ruido para MPC: {gap:.2f} EUR/sem")

    # Estadísticas del solver
    stats = mpc.stats_solver()
    if stats:
        print(f"\n  Solver LP (HiGHS):")
        print(f"    Tiempo medio:   {stats['media_ms']:.2f} ms")
        print(f"    Tiempo máximo:  {stats['max_ms']:.2f} ms")
        print(f"    Tiempo mediana: {stats['mediana_ms']:.2f} ms")

    print()
    print(f"  El DQN reward es beneficio_marginal (ya descuenta IDLE).")
    print(f"  Objetivo DQN: acercarse a {marg_r.mean():.1f} EUR/sem (MPC con mismo ruido).")
    print("=" * 62)


if __name__ == "__main__":
    main()
