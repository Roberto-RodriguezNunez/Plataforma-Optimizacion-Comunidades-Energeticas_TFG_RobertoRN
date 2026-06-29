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
    python src/reporting/eval_unificada.py
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.benchmarks.mpc_benchmark import (
    LinearMPC, ComunidadSimulador, DATASET_PATH,
    simular_hora_mpc, simular_semana_idle, _get_hora_actual, crear_mpc,
    SOC_INICIAL, SEED, EPISODE_LENGTH, HORIZON,
    _RHO_SOLAR, _RHO_CONS,
)
from src.core.forecast import (
    ventana_observada, generar_factores_precio, avanzar_ar1,
)
from src.controllers.base import BaseController
from src.config import cargar_system as _cargar_system

# Cargar config (fuente única src.config)
_CFG = _cargar_system()
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
            # Pronóstico desde la FUENTE ÚNICA (hora actual = primer paso de
            # pronóstico, con ruido), idéntico para MPC y agente — mismo timing.
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

            # Resolver controlador
            state = {'soc': sim.soc, 'step': sim.current_step}
            action = controller.solve(state, window)

            # Ejecutar con la física (aplicar_fisica_4flujos vía simular_hora_mpc).
            # F3: `bm` es el beneficio_marginal LIMPIO (sin shaping). El reward de
            # ENTRENO añade además coste_oportunidad + pendiente/terminal, por eso
            # el reward de entreno (~37) sale por debajo de esta media de eval (~39):
            # son métricas distintas a propósito, no un bug.
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


def evaluar_multiseed(controller: BaseController, forecast_mode: str,
                      seeds) -> Dict:
    """Evalúa el controlador bajo varias semillas de RUIDO y agrega.

    Devuelve el beneficio marginal POOLED (todas las semillas × 50 semanas)
    más la media por semilla, para reportar media ± std SOBRE SEMILLAS — la
    barra de error mide la robustez frente a la realización del ruido AR(1)
    (no la varianza de entrenamiento).
    """
    medias, pooled = [], []
    for sd in seeds:
        res = evaluar_controlador(controller, forecast_mode=forecast_mode, seed=sd)
        m = res['bens_marg']
        medias.append(float(m.mean()))
        pooled.append(m)
    return {
        'bens_marg': np.concatenate(pooled),
        'medias_por_seed': np.array(medias),
        'n_seeds': len(seeds),
    }


def imprimir_resultado(label: str, res: Dict):
    m = res['bens_marg']
    md = res.get('medias_por_seed')
    if md is not None and len(md) > 1:
        print(f"  {label:<35s}  {md.mean():+7.2f} +/- {md.std():5.2f} EUR/sem"
              f"  (media±std sobre {len(md)} semillas de ruido; pooled "
              f"min/max {m.min():+.1f}/{m.max():+.1f})")
    else:
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


def crear_onnx_residual_sac(onnx_path: str, npz_path: str,
                            delta_max: float) -> BaseController:
    """Crea OnnxResidualController (sin SB3/torch) con el MPC configurado."""
    from src.production.onnx_inference import OnnxResidualController
    sim = ComunidadSimulador(DATASET_PATH)
    mpc = crear_mpc()
    return OnnxResidualController(
        mpc=mpc,
        sim=sim,
        onnx_path=onnx_path,
        npz_path=npz_path,
        delta_max=delta_max,
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
    parser.add_argument('--onnx-model', type=str, default=None,
                        help='Ruta al actor ONNX (.onnx)')
    parser.add_argument('--onnx-npz', type=str, default=None,
                        help='Ruta a VecNormalize stats (.npz)')
    parser.add_argument('--skip-baselines', action='store_true',
                        help='No evaluar MPC oraculo/realista/IDLE')
    parser.add_argument('--eval-seeds', type=int, default=10,
                        help='Nº de semillas de ruido para la barra de error (default: 10)')
    args = parser.parse_args()

    if args.delta_max is None:
        args.delta_max = _CFG['residual_sac']['delta_max']

    eval_seeds = [SEED + i for i in range(args.eval_seeds)]

    print("=" * 70)
    print("EVALUACION UNIFICADA — protocolo identico a MPC benchmark")
    print("=" * 70)
    print(f"  Semanas eval:  {len(ComunidadSimulador(DATASET_PATH).semanas_eval)}")
    print(f"  SoC inicial:   {SOC_INICIAL}")
    print(f"  Semillas eval: {len(eval_seeds)} (ruido AR(1), base {SEED})")
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
        res_real = evaluar_multiseed(mpc, 'realista', eval_seeds)
        resultados['MPC realista'] = res_real

        # 3. IDLE
        print("  Evaluando IDLE...")
        res_idle = evaluar_controlador(IdleController(), forecast_mode='realista')
        resultados['IDLE'] = res_idle

    # 4. Residual SAC (si se proporcionó modelo)
    if args.sac_model:
        print(f"  Evaluando Residual SAC ({os.path.basename(args.sac_model)})...")
        sac_ctrl = crear_residual_sac(args.sac_model, args.sac_norm, args.delta_max)
        res_sac = evaluar_multiseed(sac_ctrl, 'realista', eval_seeds)
        resultados[sac_ctrl.nombre()] = res_sac

    # 5. ONNX Residual SAC
    if args.onnx_model:
        npz = args.onnx_npz or os.path.join(ROOT, 'models', 'vec_normalize_v5_1M.npz')
        print(f"  Evaluando ONNX ResidualSAC ({os.path.basename(args.onnx_model)})...")
        onnx_ctrl = crear_onnx_residual_sac(args.onnx_model, npz, args.delta_max)
        res_onnx = evaluar_multiseed(onnx_ctrl, 'realista', eval_seeds)
        resultados[onnx_ctrl.nombre()] = res_onnx

    # 6. DQN (puede haber varios modelos)
    for dqn_path in (args.dqn_model or []):
        label = os.path.splitext(os.path.basename(dqn_path))[0]
        print(f"  Evaluando DQN ({label})...")
        dqn_ctrl = crear_discrete_rl(dqn_path, args.rl_norm, algo='DQN')
        res_dqn = evaluar_multiseed(dqn_ctrl, 'realista', eval_seeds)
        resultados[f'DQN({label})'] = res_dqn

    # 6. PPO (puede haber varios modelos)
    for ppo_path in (args.ppo_model or []):
        label = os.path.splitext(os.path.basename(ppo_path))[0]
        print(f"  Evaluando PPO ({label})...")
        ppo_ctrl = crear_discrete_rl(ppo_path, args.rl_norm, algo='PPO')
        res_ppo = evaluar_multiseed(ppo_ctrl, 'realista', eval_seeds)
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
