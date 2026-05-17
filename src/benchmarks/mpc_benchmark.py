"""
mpc_benchmark.py — Benchmark MPC de horizonte deslizante (24h)
===============================================================
Ejecuta dos variantes para contextualizar los resultados del DQN:

  1. MPC con ruido AR(1) idéntico al DQN  → comparación JUSTA
       Ambos reciben la misma calidad de previsión solar/consumo.
       Si DQN ≈ MPC_ruido: el agente exprime bien la información disponible.

  2. MPC con previsión perfecta           → cota superior TEÓRICA
       Máximo alcanzable si se conociera el futuro exacto.
       Sirve para acotar el margen de mejora posible.

Física idéntica al simulador en ambos casos:
  - Precios asimétricos: PVPC (ind.1001) compra, excedentaria (ind.1739) venta
  - Eficiencia carga/descarga: 95% cada dirección (round-trip 90.25%)
  - Autodescarga: ~3% mensual (0.004%/hora)
  - Degradación no lineal: factor potencia (I²) + factor SoC (extremos)

El LP interno usa degradación linealizada (necesario para LP; en la ejecución
real se aplica la degradación no lineal completa del simulador).

Ejecución desde la raíz del proyecto (carpeta TFG/):
    python src/benchmarks/mpc_benchmark.py
"""

import os
import sys
import numpy as np
from scipy.optimize import linprog

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.simulador import ComunidadSimulador

# --- CONFIGURACIÓN ---
DATASET_PATH   = 'data/processed/dataset_final.csv'
EPISODE_LENGTH = 24 * 7   # 1 semana = 168 horas
N_EPISODES     = 100
HORIZON        = 24
SEED           = 42
SOC_INICIAL    = 0.5

# Parámetros AR(1) — idénticos a energy_env.py
_SIGMA_SOL_H1    = 0.05
_SIGMA_SOL_H24   = 0.25
_SIGMA_CONS_BASE = 0.10
_RHO_SOLAR       = 0.7
_RHO_CONS        = 0.3


def aplicar_ruido_ar1(window, error_solar, error_cons):
    """
    Aplica el mismo ruido AR(1) que energy_env._get_obs() a la ventana de 24h.
    Precios (columnas 2 y 3) no se tocan — publicados por REE el día anterior.
    Devuelve la ventana ruidosa (copia).
    """
    w = window.copy()
    for h in range(len(w)):
        sigma_sol = _SIGMA_SOL_H1 + h * ((_SIGMA_SOL_H24 - _SIGMA_SOL_H1) / 23)
        w[h, 1] = max(0.0, w[h, 1] * (1.0 + error_solar * sigma_sol))   # generacion
        w[h, 0] = max(0.0, w[h, 0] * (1.0 + error_cons * _SIGMA_CONS_BASE))  # consumo
    return w


def solve_lp(soc_actual, window, sim):
    """
    LP de horizonte 24h sobre la ventana recibida (puede ser perfecta o ruidosa).

    Variables por hora h (4 × 24 = 96 total):
      cs[h]: kWh de excedente solar → cargador (lado input inversor)
      cm[h]: kWh comprados de red  → cargador (lado input inversor)
      dc[h]: kWh extraídos batería → casas (lado batería)
      dr[h]: kWh extraídos batería → red   (lado batería)

    Eficiencia (igual que simulador):
      energía almacenada  = (cs + cm) * EFF_C
      energía entregada   = (dc + dr) * EFF_D

    Retorna (cs0, cm0, dc0, dr0) para la hora actual.
    """
    H     = HORIZON
    EFF_C = sim.EFICIENCIA_CARGA
    EFF_D = sim.EFICIENCIA_DESCARGA
    CAP   = sim.BATERIA_CAPACIDAD
    P_MAX = sim.POTENCIA_INVERSOR
    DEG   = sim.COSTE_DEGRADACION_BASE   # linealizado (sin factores stress)

    consumo    = window[:H, 0]
    generacion = window[:H, 1]
    precio_c   = window[:H, 2]
    precio_v   = window[:H, 3]

    balance = generacion - consumo
    exc = np.maximum(0,  balance)
    dfc = np.maximum(0, -balance)

    def i_cs(h): return 4*h
    def i_cm(h): return 4*h + 1
    def i_dc(h): return 4*h + 2
    def i_dr(h): return 4*h + 3

    n = 4 * H

    # Objetivo: maximizar beneficio incremental sobre IDLE (linprog minimiza → negamos)
    c_obj = np.zeros(n)
    for h in range(H):
        pv = precio_v[h]; pc = precio_c[h]
        c_obj[i_cs(h)] =  pv + DEG             # pierde venta solar directa + deg
        c_obj[i_cm(h)] =  pc + DEG             # paga compra red + deg
        c_obj[i_dc(h)] = -(pc * EFF_D - DEG)   # ahorra compra - deg
        c_obj[i_dr(h)] = -(pv * EFF_D - DEG)   # ingresa venta red - deg

    # Límites de variables
    bounds = []
    for h in range(H):
        bounds.append((0, exc[h]))           # cs <= excedente
        bounds.append((0, P_MAX))           # cm <= potencia inversor
        bounds.append((0, dfc[h] / EFF_D)) # dc <= déficit (lado batería)
        bounds.append((0, P_MAX))           # dr <= potencia inversor

    # Restricciones de desigualdad
    A_ub, b_ub = [], []
    E0 = soc_actual * CAP

    for h in range(H):
        # Potencia carga total (input): cs + cm <= P_MAX
        row = np.zeros(n)
        row[i_cs(h)] = 1; row[i_cm(h)] = 1
        A_ub.append(row.copy()); b_ub.append(P_MAX)

        # Potencia descarga total (batería): dc + dr <= P_MAX
        row = np.zeros(n)
        row[i_dc(h)] = 1; row[i_dr(h)] = 1
        A_ub.append(row.copy()); b_ub.append(P_MAX)

        # SoC acumulado hasta hora h+1:
        # E[h+1] = E0 + sum_{k=0..h} [(cs+cm)*EFF_C - dc - dr]
        # (autodescarga ignorada en LP: 0.004%/h → <0.07% semanal)
        row = np.zeros(n)
        for k in range(h + 1):
            row[i_cs(k)] =  EFF_C
            row[i_cm(k)] =  EFF_C
            row[i_dc(k)] = -1.0
            row[i_dr(k)] = -1.0

        # E[h+1] <= SOC_MAX * CAP
        A_ub.append(row.copy());  b_ub.append(sim.SOC_MAX * CAP - E0)
        # E[h+1] >= SOC_MIN * CAP
        A_ub.append(-row);        b_ub.append(E0 - sim.SOC_MIN * CAP)

    res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=bounds, method='highs', options={'disp': False})

    if not res.success:
        return 0.0, 0.0, 0.0, 0.0   # fallback IDLE

    x = res.x
    return x[i_cs(0)], x[i_cm(0)], x[i_dc(0)], x[i_dr(0)]


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

    # Recortar por estado real de la batería (el LP usó el estado del paso anterior)
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


def correr_episodios(sim, sim_idle, rng, con_ruido: bool):
    """Ejecuta N_EPISODES semanas aleatorias y devuelve listas de beneficios."""
    max_start = sim.max_steps - EPISODE_LENGTH - HORIZON - 1

    bens_mpc, bens_idle, bens_marg = [], [], []

    for ep in range(N_EPISODES):
        start = int(rng.integers(0, max_start))

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
            # Avanzar estado AR(1) (igual que energy_env.step())
            if con_ruido:
                error_solar = (_RHO_SOLAR * error_solar
                               + np.sqrt(1 - _RHO_SOLAR**2) * rng.standard_normal())
                error_cons  = (_RHO_CONS  * error_cons
                               + np.sqrt(1 - _RHO_CONS**2)  * rng.standard_normal())

            # Ventana de previsión (perfecta o ruidosa)
            window = sim.get_data_window(sim.current_step, horizon=HORIZON)
            if con_ruido:
                window = aplicar_ruido_ar1(window, error_solar, error_cons)

            # Resolver LP y ejecutar
            cs, cm, dc, dr = solve_lp(sim.soc, window, sim)
            b, bm = simular_hora_mpc(sim, cs, cm, dc, dr)
            ben_mpc  += b
            ben_marg += bm

        bens_mpc.append(ben_mpc)
        bens_idle.append(ben_idle)
        bens_marg.append(ben_marg)

    return np.array(bens_mpc), np.array(bens_idle), np.array(bens_marg)


def imprimir_resultado(label, bens_mpc, bens_idle, bens_marg):
    print(f"\n  [{label}]")
    print(f"    IDLE puro  : {bens_idle.mean():+.2f} ± {bens_idle.std():.2f} €/semana")
    print(f"    MPC total  : {bens_mpc.mean():+.2f} ± {bens_mpc.std():.2f} €/semana")
    print(f"    MPC − IDLE : {bens_marg.mean():+.2f} ± {bens_marg.std():.2f} €/semana"
          "  ← misma métrica que DQN reward")


def main():
    print("=" * 62)
    print("BENCHMARK MPC — horizonte 24h")
    print("=" * 62)

    sim      = ComunidadSimulador(DATASET_PATH)
    sim_idle = ComunidadSimulador(DATASET_PATH)

    # Dos RNG con semillas distintas para que el ruido no sea idéntico
    # entre la variante ruidosa y la perfecta (más realista)
    rng_perf  = np.random.default_rng(SEED)
    rng_ruido = np.random.default_rng(SEED + 1)

    print("\nEjecutando MPC con previsión perfecta ...")
    mpc_p, idle_p, marg_p = correr_episodios(sim, sim_idle, rng_perf,  con_ruido=False)

    print("Ejecutando MPC con ruido AR(1) idéntico al DQN ...")
    mpc_r, idle_r, marg_r = correr_episodios(sim, sim_idle, rng_ruido, con_ruido=True)

    print("\n" + "=" * 62)
    print("RESULTADOS (100 semanas aleatorias cada variante)")
    print("=" * 62)
    imprimir_resultado("MPC previsión PERFECTA — cota superior teórica", mpc_p, idle_p, marg_p)
    imprimir_resultado("MPC con RUIDO AR(1)    — comparación justa con DQN", mpc_r, idle_r, marg_r)

    print()
    gap = marg_p.mean() - marg_r.mean()
    print(f"  Coste del ruido para MPC: {gap:.2f} €/sem")
    print(f"  (diferencia perfecta vs ruidosa — cuánto vale tener previsión perfecta)")
    print()
    print("  El DQN reward es beneficio_marginal (ya descuenta IDLE).")
    print(f"  Objetivo DQN: acercarse a {marg_r.mean():.1f} €/sem (MPC con mismo ruido).")
    print("=" * 62)


if __name__ == "__main__":
    main()
