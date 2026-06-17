"""
eval_escenarios.py — Comparativa economica de escenarios energeticos
====================================================================
Calcula el beneficio economico semanal (media +/- std sobre 50 semanas
eval, seed=42) para 7 escenarios que varian en paneles, comunidad,
bateria y controlador.

Metricas reportadas:
  - Absoluto: balance economico semanal (ingresos - gastos).
  - vs B:     beneficio marginal respecto al escenario B (paneles sin
              bateria), que es la referencia natural para cuantificar
              el valor anadido de cada decision.

Escenarios:
  A  Sin paneles, sin bateria
  B  Paneles, sin bateria, sin comunidad
  G  Paneles, sin bateria, con comunidad (beta_i=1/15)
  H  Paneles, bateria comunitaria IDLE
  I  Paneles, bateria + heuristica
  J  Paneles, bateria + MPC
  K  Paneles, bateria + Residual SAC

Uso:
    python src/eval_escenarios.py                          # A-J (sin SAC)
    python src/eval_escenarios.py --skip-battery           # solo A, B, G
    python src/eval_escenarios.py --sac-model models/sac.zip --sac-norm models/norm.pkl
"""

import argparse
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.benchmarks.mpc_benchmark import (
    ComunidadSimulador, DATASET_PATH, EPISODE_LENGTH,
    simular_semana_idle,
)
from src.eval_unificada import (
    evaluar_controlador, IdleController, crear_mpc, crear_residual_sac,
)
from src.controllers.heuristic_controller import HeuristicController

_CONFIG_PATH = os.path.join(ROOT, 'config', 'system.yaml')
with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
    _CFG = yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────
#  ESCENARIOS SIN BATERIA (aritmetica sobre dataset_final.csv)
# ──────────────────────────────────────────────────────────────────

def escenario_a(df, starts):
    """A: Sin paneles, sin bateria — toda la demanda se compra de red."""
    bens = []
    for start in starts:
        chunk = df.iloc[start : start + EPISODE_LENGTH]
        bens.append(-(chunk['consumo_total'] * chunk['precio_kwh']).sum())
    return np.array(bens)


def escenario_b(df, starts):
    """B: Con paneles, sin bateria — autoconsumo directo + venta excedentes.

    Equivalente a simular_semana_idle() del benchmark MPC.
    """
    bens = []
    for start in starts:
        chunk = df.iloc[start : start + EPISODE_LENGTH]
        bal = chunk['generacion_total'] - chunk['consumo_total']
        total = (bal.clip(lower=0) * chunk['precio_excedente']
                 - (-bal).clip(lower=0) * chunk['precio_kwh']).sum()
        bens.append(total)
    return np.array(bens)


def escenario_g(df, starts):
    """G: Con paneles, sin bateria, con comunidad (beta_i=1/15 iguales).

    Limitacion: con datos ya agregados y coeficientes de reparto iguales,
    el resultado es identico a B. La diferencia real apareceria con
    coeficientes dinamicos o datos individuales por vivienda.
    """
    return escenario_b(df, starts)


# ──────────────────────────────────────────────────────────────────
#  ESCENARIOS CON BATERIA (reutilizan eval_unificada)
# ──────────────────────────────────────────────────────────────────

def escenario_h():
    """H: Bateria comunitaria IDLE — existe pero no se gestiona."""
    res = evaluar_controlador(IdleController(), forecast_mode='realista')
    return res['bens_abs']


def escenario_i():
    """I: Bateria comunitaria + controlador heuristico."""
    ctrl = HeuristicController()
    res = evaluar_controlador(ctrl, forecast_mode='realista')
    return res['bens_abs']


def escenario_j():
    """J: Bateria comunitaria + MPC (pronostico realista)."""
    mpc = crear_mpc()
    res = evaluar_controlador(mpc, forecast_mode='realista')
    return res['bens_abs']


def escenario_k(model_path, norm_path, delta_max):
    """K: Bateria comunitaria + Residual SAC."""
    ctrl = crear_residual_sac(model_path, norm_path, delta_max)
    res = evaluar_controlador(ctrl, forecast_mode='realista')
    return res['bens_abs']


# ──────────────────────────────────────────────────────────────────
#  FUNCION PRINCIPAL
# ──────────────────────────────────────────────────────────────────

def evaluar_todos_escenarios(sac_model=None, sac_norm=None, delta_max=None,
                              skip_battery=False):
    """
    Evalua todos los escenarios y devuelve dict {clave: np.array(50)}.
    Cada array contiene el beneficio absoluto semanal de las 50 semanas eval.
    """
    sim = ComunidadSimulador(DATASET_PATH)
    df = sim.df
    starts = sim.semanas_eval

    resultados = {}

    # --- Escenarios sin bateria (rapidos) ---
    print("  Calculando escenario A (sin paneles)...")
    resultados['A'] = escenario_a(df, starts)

    print("  Calculando escenario B (paneles, sin bateria)...")
    resultados['B'] = escenario_b(df, starts)

    # Verificar equivalencia B == simular_semana_idle
    sim_check = ComunidadSimulador(DATASET_PATH)
    for i, start in enumerate(starts):
        idle_val = simular_semana_idle(sim_check, start)
        diff = abs(resultados['B'][i] - idle_val)
        assert diff < 1e-6, \
            f"Semana {i}: escenario_b={resultados['B'][i]:.6f} != idle={idle_val:.6f}"
    print("    Verificado: escenario_b == simular_semana_idle (50/50 semanas)")

    print("  Calculando escenario G (comunidad, beta=1/15)...")
    resultados['G'] = escenario_g(df, starts)

    if not skip_battery:
        print("  Calculando escenario H (bateria IDLE)...")
        resultados['H'] = escenario_h()

        print("  Calculando escenario I (heuristico)...")
        resultados['I'] = escenario_i()

        print("  Calculando escenario J (MPC realista)...")
        resultados['J'] = escenario_j()

        if sac_model:
            print("  Calculando escenario K (Residual SAC)...")
            if delta_max is None:
                delta_max = _CFG['residual_sac']['delta_max']
            resultados['K'] = escenario_k(sac_model, sac_norm, delta_max)

    return resultados


# ──────────────────────────────────────────────────────────────────
#  TABLA DE RESULTADOS
# ──────────────────────────────────────────────────────────────────

_NOMBRES = {
    'A': 'Sin paneles, sin bateria',
    'B': 'Paneles, sin bateria, sin comunidad',
    'G': 'Paneles, sin bateria, con comunidad',
    'H': 'Paneles, bateria comunitaria IDLE',
    'I': 'Paneles, bateria + heuristica',
    'J': 'Paneles, bateria + MPC',
    'K': 'Paneles, bateria + SAC',
}


def imprimir_tabla(resultados):
    """Imprime tabla con beneficio absoluto y marginal vs B."""
    ref_b = resultados['B'].mean()

    print()
    print("=" * 90)
    print("COMPARATIVA ECONOMICA — 50 semanas eval, seed=42")
    print("=" * 90)
    print(f"  {'Esc':<4} {'Descripcion':<40} {'EUR/sem':>10} {'Std':>8} {'vs B EUR/sem':>14}")
    print("  " + "-" * 80)

    for key in ['A', 'B', 'G', 'H', 'I', 'J', 'K']:
        if key not in resultados:
            continue
        arr = resultados[key]
        m, s = arr.mean(), arr.std()
        vs_b = m - ref_b
        print(f"  {key:<4} {_NOMBRES[key]:<40} {m:>+10.2f} {s:>8.2f} {vs_b:>+14.2f}")

    print("  " + "-" * 80)
    print()
    print("  Notas:")
    print("  [1] G = B: datos agregados + beta_i=1/15 iguales -> resultado identico.")
    print("      Diferencia real con coeficientes dinamicos o datos por vivienda.")
    print("  [2] H = B: bateria IDLE no genera valor (sin coste fijo O&M modelado).")
    print("  [3] 'vs B' = beneficio marginal respecto a tener solo paneles")
    print("      sin bateria. Cuantifica el valor anadido de cada decision.")
    print("=" * 90)


# ──────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Comparativa economica de escenarios energeticos')
    parser.add_argument('--sac-model', type=str, default=None,
                        help='Ruta al modelo SAC (.zip)')
    parser.add_argument('--sac-norm', type=str, default=None,
                        help='Ruta a VecNormalize stats (.pkl)')
    parser.add_argument('--delta-max', type=float, default=None,
                        help='Delta max del SAC (default: config)')
    parser.add_argument('--skip-battery', action='store_true',
                        help='Solo calcular A, B, G (rapido, sin simulacion)')
    args = parser.parse_args()

    print("=" * 90)
    print("EVALUACION DE ESCENARIOS ECONOMICOS")
    print("=" * 90)
    print(f"  Semanas eval:  50")
    print(f"  Seed:          42")
    print(f"  Dataset:       {DATASET_PATH}")
    print()

    resultados = evaluar_todos_escenarios(
        sac_model=args.sac_model,
        sac_norm=args.sac_norm,
        delta_max=args.delta_max,
        skip_battery=args.skip_battery,
    )
    imprimir_tabla(resultados)


if __name__ == '__main__':
    main()
