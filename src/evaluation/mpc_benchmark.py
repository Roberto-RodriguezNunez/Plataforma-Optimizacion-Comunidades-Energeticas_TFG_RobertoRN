"""
mpc_benchmark.py — Script de benchmark del MPC (runner + calibración + main)
============================================================================
Ejecuta el MPC (oráculo y realista AR(1)) sobre las semanas eval y reporta
beneficios. La LIBRERÍA del MPC (LinearMPC, crear_mpc, helpers físicos y
constantes) vive en src/controllers/mpc.py; aquí solo el runner de episodios,
la calibración de degradación y el main del script.

    python src/evaluation/mpc_benchmark.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.controllers.mpc import (
    LinearMPC, ComunidadSimulador, crear_mpc,
    simular_hora_mpc, simular_semana_idle, _get_hora_actual,
    DATASET_PATH, SEED, EPISODE_LENGTH, N_EPISODES, HORIZON, SOC_INICIAL, _MPC,
)
from src.core.forecast import avanzar_ar1, generar_factores_precio, ventana_observada


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
