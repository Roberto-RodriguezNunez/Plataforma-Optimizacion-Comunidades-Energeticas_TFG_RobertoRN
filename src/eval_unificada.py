"""
eval_unificada.py — Evaluación unificada de todos los controladores
===================================================================
Protocolo idéntico al de correr_episodios del MPC benchmark:
  - 50 semanas eval (sim.semanas_eval), deterministas
  - SoC inicial = config.bateria.soc_inicial (0.50)
  - RNG: np.random.default_rng(SEED) para AR(1)
  - Reward = sum(beneficio_marginal) de 168 horas (sin pendiente ni terminal)

Esto garantiza que delta=0 via ResidualSAC = MPC realista exactamente.

Uso:
    python src/eval_unificada.py
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.benchmarks.mpc_benchmark import (
    LinearMPC, ComunidadSimulador, DATASET_PATH,
    simular_hora_mpc, simular_semana_idle, aplicar_ruido_ar1,
    aplicar_ruido_precio_3capas, _get_hora_actual,
    SOC_INICIAL, SEED, EPISODE_LENGTH, HORIZON,
    _RHO_SOLAR, _RHO_CONS,
)
from src.controllers.base import BaseController

# Cargar config
_CONFIG_PATH = os.path.join(ROOT, 'config', 'system.yaml')
with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
    _CFG = yaml.safe_load(f)
_MPC_CFG = _CFG['mpc']


# ──────────────────────────────────────────────────────────────────
#  PROTOCOLO DE EVALUACIÓN
# ──────────────────────────────────────────────────────────────────

def evaluar_controlador(
    controller: BaseController,
    forecast_mode: str = 'realista',
    seed: int = SEED,
) -> Dict:
    """
    Evalúa un controlador sobre las 50 semanas eval con protocolo
    idéntico a correr_episodios del MPC benchmark.

    Args:
        controller: Cualquier BaseController (MPC, SAC, heurístico...).
        forecast_mode: 'oraculo' o 'realista'.
        seed: Semilla para el RNG del ruido AR(1).

    Returns:
        Dict con 'bens_marg', 'bens_abs', 'bens_idle' (arrays de 50).
    """
    sim = ComunidadSimulador(DATASET_PATH)
    sim_idle = ComunidadSimulador(DATASET_PATH)
    rng = np.random.default_rng(seed)
    con_ruido = (forecast_mode == 'realista')

    starts = sim.semanas_eval
    bens_abs, bens_idle, bens_marg = [], [], []

    for start in starts:
        # IDLE baseline
        ben_idle = simular_semana_idle(sim_idle, start)

        # Controlador
        sim.current_step = start
        sim.soc = SOC_INICIAL
        error_solar = 0.0
        error_cons = 0.0
        ben_total = 0.0
        ben_marg = 0.0

        for _ in range(EPISODE_LENGTH):
            # Avanzar AR(1) solar/consumo — mismo timing que correr_episodios
            if con_ruido:
                error_solar = (
                    _RHO_SOLAR * error_solar
                    + np.sqrt(1 - _RHO_SOLAR ** 2) * rng.standard_normal()
                )
                error_cons = (
                    _RHO_CONS * error_cons
                    + np.sqrt(1 - _RHO_CONS ** 2) * rng.standard_normal()
                )

            # Forecast (perfecto o ruidoso)
            window = sim.get_data_window(sim.current_step, horizon=HORIZON)
            if con_ruido:
                window = aplicar_ruido_ar1(window, error_solar, error_cons)
                hora_actual = _get_hora_actual(sim.current_step)
                aplicar_ruido_precio_3capas(window, hora_actual,
                                            rng.standard_normal,
                                            start_offset=0)

            # Resolver controlador
            state = {'soc': sim.soc, 'step': sim.current_step}
            action = controller.solve(state, window)

            # Ejecutar con física de simular_hora_mpc
            b, bm = simular_hora_mpc(
                sim,
                action['P_carga_solar'],
                action['P_carga_red'],
                action['P_descarga_casa'],
                action['P_descarga_red'],
            )
            ben_total += b
            ben_marg += bm

        bens_abs.append(ben_total)
        bens_idle.append(ben_idle)
        bens_marg.append(ben_marg)

    return {
        'bens_marg': np.array(bens_marg),
        'bens_abs': np.array(bens_abs),
        'bens_idle': np.array(bens_idle),
    }


def imprimir_resultado(label: str, res: Dict):
    m = res['bens_marg']
    print(f"  {label:<35s}  {m.mean():+7.2f} +/- {m.std():5.2f} EUR/sem"
          f"  (min={m.min():+.1f}, max={m.max():+.1f})")


# ──────────────────────────────────────────────────────────────────
#  CONTROLADORES DISPONIBLES
# ──────────────────────────────────────────────────────────────────

class IdleController(BaseController):
    """No usa batería. Baseline inferior."""
    def solve(self, state, forecast):
        return {'P_carga_solar': 0, 'P_carga_red': 0,
                'P_descarga_casa': 0, 'P_descarga_red': 0}
    def nombre(self):
        return "IDLE"


def crear_mpc() -> LinearMPC:
    """Crea el MPC con la config de system.yaml."""
    sim = ComunidadSimulador(DATASET_PATH)
    tv_cfg = _MPC_CFG['valor_terminal']
    return LinearMPC(
        sim,
        use_terminal_value=tv_cfg['activado'],
        terminal_lambda=tv_cfg['lambda'],
        terminal_price_mode=tv_cfg['modo_precio'],
        k_deg_lin=_MPC_CFG['k_deg_lin'],
    )


def crear_residual_sac(model_path: str, vec_norm_path: str,
                       delta_max: float) -> BaseController:
    """Crea ResidualSACController con el MPC configurado."""
    from src.controllers.residual_sac_controller import ResidualSACController
    sim = ComunidadSimulador(DATASET_PATH)
    mpc = crear_mpc()
    return ResidualSACController(
        model_path=model_path,
        mpc=mpc,
        sim=sim,
        delta_max=delta_max,
        vec_normalize_path=vec_norm_path,
    )


def crear_discrete_rl(model_path: str, vec_norm_path: str,
                      algo: str = 'DQN') -> BaseController:
    """Crea DiscreteRLController para DQN o PPO."""
    from src.controllers.discrete_rl_controller import DiscreteRLController
    sim = ComunidadSimulador(DATASET_PATH)
    return DiscreteRLController(
        model_path=model_path,
        sim=sim,
        vec_normalize_path=vec_norm_path,
        algo=algo,
    )


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluación unificada de controladores')
    parser.add_argument('--sac-model', type=str, default=None,
                        help='Ruta al modelo SAC (.zip)')
    parser.add_argument('--sac-norm', type=str, default=None,
                        help='Ruta a VecNormalize stats (.pkl)')
    parser.add_argument('--delta-max', type=float, default=None,
                        help='Delta max del SAC (default: config)')
    parser.add_argument('--dqn-model', type=str, default=None, action='append',
                        help='Ruta a modelo DQN (.zip). Repetible.')
    parser.add_argument('--ppo-model', type=str, default=None, action='append',
                        help='Ruta a modelo PPO (.zip). Repetible.')
    parser.add_argument('--rl-norm', type=str, default=None,
                        help='Ruta a VecNormalize stats para DQN/PPO (.pkl)')
    parser.add_argument('--skip-baselines', action='store_true',
                        help='No evaluar MPC oraculo/realista/IDLE')
    args = parser.parse_args()

    if args.delta_max is None:
        args.delta_max = _CFG['residual_sac']['delta_max']

    print("=" * 70)
    print("EVALUACION UNIFICADA — protocolo identico a MPC benchmark")
    print("=" * 70)
    print(f"  Semanas eval:  {len(ComunidadSimulador(DATASET_PATH).semanas_eval)}")
    print(f"  SoC inicial:   {SOC_INICIAL}")
    print(f"  Seed RNG:      {SEED}")
    print(f"  Forecast:      realista (AR(1))")
    print()

    resultados = {}

    if not args.skip_baselines:
        # 1. MPC oráculo
        print("  Evaluando MPC oraculo...")
        mpc = crear_mpc()
        res_orac = evaluar_controlador(mpc, forecast_mode='oraculo')
        resultados['MPC oraculo'] = res_orac

        # 2. MPC realista
        print("  Evaluando MPC realista...")
        res_real = evaluar_controlador(mpc, forecast_mode='realista')
        resultados['MPC realista'] = res_real

        # 3. IDLE
        print("  Evaluando IDLE...")
        res_idle = evaluar_controlador(IdleController(), forecast_mode='realista')
        resultados['IDLE'] = res_idle

    # 4. Residual SAC (si se proporcionó modelo)
    if args.sac_model:
        print(f"  Evaluando Residual SAC ({os.path.basename(args.sac_model)})...")
        sac_ctrl = crear_residual_sac(args.sac_model, args.sac_norm, args.delta_max)
        res_sac = evaluar_controlador(sac_ctrl, forecast_mode='realista')
        resultados[sac_ctrl.nombre()] = res_sac

    # 5. DQN (puede haber varios modelos)
    for dqn_path in (args.dqn_model or []):
        label = os.path.splitext(os.path.basename(dqn_path))[0]
        print(f"  Evaluando DQN ({label})...")
        dqn_ctrl = crear_discrete_rl(dqn_path, args.rl_norm, algo='DQN')
        res_dqn = evaluar_controlador(dqn_ctrl, forecast_mode='realista')
        resultados[f'DQN({label})'] = res_dqn

    # 6. PPO (puede haber varios modelos)
    for ppo_path in (args.ppo_model or []):
        label = os.path.splitext(os.path.basename(ppo_path))[0]
        print(f"  Evaluando PPO ({label})...")
        ppo_ctrl = crear_discrete_rl(ppo_path, args.rl_norm, algo='PPO')
        res_ppo = evaluar_controlador(ppo_ctrl, forecast_mode='realista')
        resultados[f'PPO({label})'] = res_ppo

    # Tabla
    print()
    print("=" * 70)
    print("RESULTADOS (50 semanas eval, beneficio marginal vs IDLE)")
    print("=" * 70)
    for label, res in resultados.items():
        imprimir_resultado(label, res)

    if not args.skip_baselines:
        gap = res_orac['bens_marg'].mean() - res_real['bens_marg'].mean()
        print(f"\n  Gap oraculo - realista: {gap:.2f} EUR/sem")
    print("=" * 70)


if __name__ == '__main__':
    main()
