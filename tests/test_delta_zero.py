"""
test_delta_zero.py — Verifica que Δa=(0,0,0,0) via ResidualEnv
reproduce el MPC standalone.

Test 1 (oráculo): sin ruido, ambos paths deben dar resultado IDÉNTICO.
Test 2 (realista): con ruido y RNG unificado, deben dar resultado IDÉNTICO.
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
from src.benchmarks.mpc_benchmark import (
    LinearMPC, ComunidadSimulador, DATASET_PATH,
    simular_semana_idle,
    SOC_INICIAL, SEED, EPISODE_LENGTH,
)
from src.envs.energy_env_continuo import EnergyEnvContinuo
from src.envs.residual_env import ResidualEnv

_CONFIG_PATH = os.path.join(ROOT, 'config', 'system.yaml')
with open(_CONFIG_PATH) as f:
    _CFG = yaml.safe_load(f)
_MPC_CFG = _CFG['mpc']


def crear_mpc():
    sim = ComunidadSimulador(DATASET_PATH)
    tv_cfg = _MPC_CFG['valor_terminal']
    return LinearMPC(
        sim,
        use_terminal_value=tv_cfg['activado'],
        terminal_lambda=tv_cfg['lambda'],
        terminal_price_mode=tv_cfg['modo_precio'],
        k_deg_lin=_MPC_CFG['k_deg_lin'],
    )


def test_mpc_standalone(mode='realista'):
    """MPC via eval_unificada path."""
    from src.eval_unificada import evaluar_controlador
    mpc = crear_mpc()
    res = evaluar_controlador(mpc, forecast_mode=mode)
    return res['bens_marg']


def test_residual_delta_zero(forecast_noise=True):
    """
    ResidualEnv con delta=(0,0,0,0) sobre 50 semanas eval.
    Fuerza SoC=0.50 al inicio de cada episodio (igual que eval_unificada).
    Usa rng seeded idéntico al de eval_unificada para ruido exacto.
    """
    mpc = crear_mpc()
    # Crear env SIN rng para que el reset no consuma draws del rng seeded.
    # El rng se inyecta DESPUÉS del reset, justo antes del loop.
    inner = EnergyEnvContinuo(forecast_noise=forecast_noise, mode='eval')
    renv = ResidualEnv(inner, mpc, delta_max=0.15)

    sim = inner.simulador
    starts = sim.semanas_eval
    delta_zero = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Crear rng seeded idéntico al de eval_unificada
    rng = np.random.default_rng(SEED)

    bens_marg = []

    for start in starts:
        # Reset sin consumir draws del rng seeded
        obs, _ = renv.reset()

        # Forzar estado idéntico a eval_unificada al inicio de episodio
        inner.simulador.current_step = start
        inner.simulador.soc = SOC_INICIAL
        inner.steps_in_episode = 0
        inner._precio_noise_factors = None
        inner._error_solar = 0.0
        inner._error_cons = 0.0

        # Inyectar rng seeded AHORA (después del reset, antes del loop)
        # para que las draws del loop coincidan con eval_unificada
        inner._rng = rng

        total_ben = 0.0
        for _ in range(EPISODE_LENGTH):
            obs, reward, terminated, truncated, info = renv.step(delta_zero)
            total_ben += info['beneficio']

        sim_idle = ComunidadSimulador(DATASET_PATH)
        ben_idle = simular_semana_idle(sim_idle, start)
        bens_marg.append(total_ben - ben_idle)

    return np.array(bens_marg)


if __name__ == '__main__':
    print("=" * 60)
    print("VERIFICACION: Delta=0 via ResidualEnv vs MPC standalone")
    print("=" * 60)

    # --- Test 1: ORÁCULO (sin ruido) ---
    print("\n  TEST 1: ORÁCULO (sin ruido — verifica fisica pura)")
    print("  " + "-" * 56)

    print("    MPC oráculo (eval_unificada)...")
    mpc_orac = test_mpc_standalone(mode='oraculo')
    print(f"    MPC:         {mpc_orac.mean():+.2f} +/- {mpc_orac.std():.2f} EUR/sem")

    print("    ResidualEnv delta=0 (sin ruido)...")
    res_orac = test_residual_delta_zero(forecast_noise=False)
    print(f"    ResidualEnv: {res_orac.mean():+.2f} +/- {res_orac.std():.2f} EUR/sem")

    diff_orac = abs(mpc_orac.mean() - res_orac.mean())
    print(f"    Diferencia: {diff_orac:.4f} EUR/sem")
    if diff_orac <= 0.1:
        print("    PASA — fisica identica (< 0.1 EUR/sem)")
    else:
        print("    FALLA — fisica difiere!")

    # --- Test 2: REALISTA (con ruido — RNG unificado) ---
    print("\n  TEST 2: REALISTA (con ruido — RNG unificado)")
    print("  " + "-" * 56)

    print("    MPC realista (eval_unificada)...")
    mpc_real = test_mpc_standalone(mode='realista')
    print(f"    MPC:         {mpc_real.mean():+.2f} +/- {mpc_real.std():.2f} EUR/sem")

    print("    ResidualEnv delta=0 (con ruido)...")
    res_real = test_residual_delta_zero(forecast_noise=True)
    print(f"    ResidualEnv: {res_real.mean():+.2f} +/- {res_real.std():.2f} EUR/sem")

    diff_real = abs(mpc_real.mean() - res_real.mean())
    print(f"    Diferencia: {diff_real:.4f} EUR/sem")
    if diff_real <= 0.05:
        print("    PASA — RNG unificado, diferencia < 0.05 EUR/sem")
    else:
        print(f"    FALLA — diferencia {diff_real:.4f} > 0.05 EUR/sem")

    print("\n" + "=" * 60)
    if diff_orac <= 0.1 and diff_real <= 0.05:
        print("  RESULTADO: VERIFICACION SUPERADA")
    else:
        print("  RESULTADO: VERIFICACION FALLIDA")
    print("=" * 60)
