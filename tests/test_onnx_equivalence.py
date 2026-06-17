"""
test_onnx_equivalence.py — Equivalencia numérica ResidualSAC vs ONNX
=====================================================================
Verifica que OnnxResidualController produce flujos idénticos a
ResidualSACController para ~200 observaciones representativas.

Skip automático si los artefactos ONNX no existen aún.
Generar con: python -m src.production.export_onnx
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.benchmarks.mpc_benchmark import (
    ComunidadSimulador,
    DATASET_PATH,
    LinearMPC,
    simular_hora_mpc,
    aplicar_ruido_ar1,
    aplicar_ruido_precio_3capas,
    _get_hora_actual,
    SOC_INICIAL,
    SEED,
    HORIZON,
)

ONNX_PATH = os.path.join(ROOT, 'models', 'residual_sac_actor.onnx')
NPZ_PATH  = os.path.join(ROOT, 'models', 'vec_normalize_v5_1M.npz')
MODEL_ZIP = os.path.join(ROOT, 'models', 'best_model_v5_1M.zip')
VEC_NORM  = os.path.join(ROOT, 'models', 'vec_normalize_v5_1M.pkl')
DELTA_MAX = 0.15   # valor v5

_onnx_available = os.path.exists(ONNX_PATH) and os.path.exists(NPZ_PATH)
_sb3_available  = os.path.exists(MODEL_ZIP) and os.path.exists(VEC_NORM)

pytestmark = pytest.mark.skipif(
    not (_onnx_available and _sb3_available),
    reason='Artefactos ONNX/SB3 no encontrados. Ejecutar: python -m src.production.export_onnx',
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def controllers():
    from src.controllers.residual_sac_controller import ResidualSACController
    from src.production.onnx_inference import OnnxResidualController

    sim_sb3  = ComunidadSimulador(DATASET_PATH)
    sim_onnx = ComunidadSimulador(DATASET_PATH)

    mpc_sb3 = LinearMPC(sim_sb3)
    mpc_onnx = LinearMPC(sim_onnx)

    sb3_ctrl = ResidualSACController(
        model_path=MODEL_ZIP,
        mpc=mpc_sb3,
        sim=sim_sb3,
        delta_max=DELTA_MAX,
        vec_normalize_path=VEC_NORM,
    )
    onnx_ctrl = OnnxResidualController(
        mpc=mpc_onnx,
        sim=sim_onnx,
        onnx_path=ONNX_PATH,
        npz_path=NPZ_PATH,
        delta_max=DELTA_MAX,
    )
    return sb3_ctrl, onnx_ctrl, sim_sb3, sim_onnx


# ── Test principal ────────────────────────────────────────────────────────────

def test_onnx_equivalencia_flujos(controllers):
    """
    Compara flujos de ambos controladores en ~200 pasos representativos.
    Tolerancia: max |Δflujo| < 1e-3 kWh/h.
    """
    sb3_ctrl, onnx_ctrl, sim_sb3, sim_onnx = controllers

    rng = np.random.default_rng(SEED)
    error_solar = error_cons = 0.0

    starts = ComunidadSimulador(DATASET_PATH).semanas_eval[:4]  # 4 semanas, 672 pasos
    max_delta = 0.0
    n_comparaciones = 0

    for start in starts:
        sim_sb3.current_step  = start
        sim_onnx.current_step = start
        sim_sb3.soc  = SOC_INICIAL
        sim_onnx.soc = SOC_INICIAL

        for _ in range(168):
            step = sim_sb3.current_step

            error_solar = (0.7 * error_solar + np.sqrt(1 - 0.49) * rng.standard_normal())
            error_cons  = (0.3 * error_cons  + np.sqrt(1 - 0.09) * rng.standard_normal())

            # Un solo forecast ruidoso — mismo input para ambos controladores
            window = sim_sb3.get_data_window(step, HORIZON).copy()
            aplicar_ruido_ar1(window, error_solar, error_cons)
            aplicar_ruido_precio_3capas(window, _get_hora_actual(step), rng.standard_normal)

            state_sb3  = {'soc': sim_sb3.soc,  'step': step}
            state_onnx = {'soc': sim_onnx.soc, 'step': step}

            a_sb3  = sb3_ctrl.solve(state_sb3,  window)
            a_onnx = onnx_ctrl.solve(state_onnx, window)

            for key in ['P_carga_solar', 'P_carga_red', 'P_descarga_casa', 'P_descarga_red']:
                diff = abs(a_sb3[key] - a_onnx[key])
                if diff > max_delta:
                    max_delta = diff

            simular_hora_mpc(sim_sb3,  a_sb3['P_carga_solar'],  a_sb3['P_carga_red'],
                             a_sb3['P_descarga_casa'],  a_sb3['P_descarga_red'])
            simular_hora_mpc(sim_onnx, a_onnx['P_carga_solar'], a_onnx['P_carga_red'],
                             a_onnx['P_descarga_casa'], a_onnx['P_descarga_red'])

            n_comparaciones += 1

    print(f'\n  Pasos comparados: {n_comparaciones}  max|Δflujo|={max_delta:.2e} kWh/h')
    assert max_delta < 1e-3, (
        f'Divergencia ONNX vs SB3: max|Δflujo|={max_delta:.2e} kWh/h ≥ 1e-3'
    )


def test_onnx_beneficio_semanal(controllers):
    """
    Beneficio semanal agregado no difiere más de 0.1 EUR entre SB3 y ONNX.
    Ejecuta una semana completa.
    """
    sb3_ctrl, onnx_ctrl, sim_sb3, sim_onnx = controllers

    rng = np.random.default_rng(SEED + 1)
    start = ComunidadSimulador(DATASET_PATH).semanas_eval[0]
    error_solar = error_cons = 0.0

    sim_sb3.current_step  = start
    sim_onnx.current_step = start
    sim_sb3.soc  = SOC_INICIAL
    sim_onnx.soc = SOC_INICIAL

    ben_sb3 = ben_onnx = 0.0

    for _ in range(168):
        step = sim_sb3.current_step
        error_solar = 0.7 * error_solar + np.sqrt(1 - 0.49) * rng.standard_normal()
        error_cons  = 0.3 * error_cons  + np.sqrt(1 - 0.09) * rng.standard_normal()

        window = sim_sb3.get_data_window(step, HORIZON).copy()
        aplicar_ruido_ar1(window, error_solar, error_cons)
        aplicar_ruido_precio_3capas(window, _get_hora_actual(step), rng.standard_normal)

        a_sb3  = sb3_ctrl.solve({'soc': sim_sb3.soc,  'step': step}, window)
        a_onnx = onnx_ctrl.solve({'soc': sim_onnx.soc, 'step': step}, window)

        _, bm_sb3  = simular_hora_mpc(sim_sb3,
                         a_sb3['P_carga_solar'],  a_sb3['P_carga_red'],
                         a_sb3['P_descarga_casa'], a_sb3['P_descarga_red'])
        _, bm_onnx = simular_hora_mpc(sim_onnx,
                         a_onnx['P_carga_solar'],  a_onnx['P_carga_red'],
                         a_onnx['P_descarga_casa'], a_onnx['P_descarga_red'])
        ben_sb3  += bm_sb3
        ben_onnx += bm_onnx

    gap = abs(ben_sb3 - ben_onnx)
    print(f'\n  Beneficio SB3={ben_sb3:.4f}  ONNX={ben_onnx:.4f}  Δ={gap:.4f} EUR/sem')
    assert gap < 0.1, f'Gap beneficio SB3 vs ONNX: {gap:.4f} EUR ≥ 0.10 EUR'
