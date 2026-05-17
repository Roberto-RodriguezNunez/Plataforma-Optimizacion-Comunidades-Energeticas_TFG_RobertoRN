"""
diagnostico_recompensa.py — Análisis completo del sistema de recompensa
=======================================================================
Objetivo: determinar con certeza POR QUÉ el agente siempre descarga.

Comprobaciones:
  1. Lógica física paso a paso de cada acción (¿hay bugs?)
  2. Recompensa marginal de cada acción en condiciones típicas
  3. Ciclo completo CARGAR_SOLAR → DESCARGAR_CASA: ¿es rentable neto?
  4. Retorno total no descontado por estrategia en una semana real
  5. Retorno DESCONTADO (γ=0.99) que optimiza el DQN — EL DIAGNÓSTICO CLAVE
  6. Valor del terminal bonus con descuento γ^168 — posible causa raíz

Ejecución: python src/benchmarks/diagnostico_recompensa.py
"""

import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.simulador import ComunidadSimulador

DATASET = 'data/processed/dataset_final.csv'
GAMMA   = 0.99
EPISODE_LENGTH = 168   # 24*7
SOC_INI = 0.5          # mismo que en reset() por defecto
SEED    = 42

SEP  = "=" * 65
SEP2 = "-" * 65


# ─────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────

def retorno_descontado(rewards, gamma=GAMMA):
    """Retorno descontado desde t=0: sum_t γ^t * r_t"""
    ret = 0.0
    for t, r in enumerate(rewards):
        ret += (gamma ** t) * r
    return ret


def simular_semana(sim, start, soc_ini, politica_fn, label, verbose=False):
    """
    Ejecuta una semana completa con la política dada.
    politica_fn(step_en_episodio, datos_hora, soc, exc_disp, def_cub) → action_idx
    Devuelve dict con métricas completas.
    """
    sim.current_step = start
    sim.soc = soc_ini

    # Calcular pendiente_coste_inicial (igual que energy_env.reset)
    soc_util = max(0, soc_ini - sim.SOC_MIN)
    energia_ini = soc_util * sim.BATERIA_CAPACIDAD
    datos_h0 = sim.get_data_window(start, horizon=1)[0]
    precio_h0 = datos_h0[2]
    pendiente = energia_ini * precio_h0 * sim.EFICIENCIA_DESCARGA

    rewards = []
    acciones = []
    socs = []

    for ep_step in range(EPISODE_LENGTH):
        step = sim.current_step
        row = sim.df.iloc[step]
        gen, cons  = row['generacion_total'], row['consumo_total']
        precio_c   = row['precio_kwh']
        precio_v   = row['precio_excedente']
        balance    = gen - cons
        exc_disp   = max(0, balance)
        def_cub    = abs(min(0, balance))

        accion = politica_fn(ep_step, row, sim.soc, exc_disp, def_cub)
        resultado = sim.ejecutar_accion_fisica(accion, step)

        r = resultado['beneficio_marginal']

        # Corrección coste inicial (solo en paso 0, igual que energy_env)
        if ep_step == 0:
            r -= pendiente

        rewards.append(r)
        acciones.append(accion)
        socs.append(resultado['soc'])
        sim.current_step += 1

    # Valor terminal (igual que energy_env.step cuando terminated=True)
    soc_util_final = max(0, sim.soc - sim.SOC_MIN)
    energia_final  = soc_util_final * sim.BATERIA_CAPACIDAD
    datos_ult      = sim.get_data_window(sim.current_step - 1, horizon=1)[0]
    precio_final   = datos_ult[2]
    terminal       = energia_final * precio_final * sim.EFICIENCIA_DESCARGA

    rewards[-1] += terminal   # se añade al último paso

    ret_nodisc  = sum(rewards)
    ret_disc    = retorno_descontado(rewards)
    terminal_disc = (GAMMA ** (EPISODE_LENGTH - 1)) * terminal

    if verbose:
        print(f"\n  Detalle {label}:")
        print(f"    pendiente_coste_inicial = {pendiente:.3f} EUR")
        print(f"    terminal bonus (sin desc.) = {terminal:.3f} EUR")
        print(f"    terminal bonus (con desc. γ^{EPISODE_LENGTH-1}) = {terminal_disc:.3f} EUR")
        print(f"    SoC inicial={soc_ini:.2f} → SoC final={sim.soc:.2f}")
        print(f"    Distribución acciones: {dict(zip(*np.unique(acciones, return_counts=True)))}")

    return {
        'label':       label,
        'ret_nodisc':  ret_nodisc,
        'ret_disc':    ret_disc,
        'terminal':    terminal,
        'terminal_disc': terminal_disc,
        'pendiente':   pendiente,
        'soc_final':   sim.soc,
        'rewards':     rewards,
        'acciones':    acciones,
        'socs':        socs,
    }


# ─────────────────────────────────────────────────────────────────
# POLÍTICAS
# ─────────────────────────────────────────────────────────────────

def pol_idle(ep_step, row, soc, exc, dfc):
    return 0

def pol_autoconsumo(ep_step, row, soc, exc, dfc):
    """Regla simple: carga solar si hay excedente y batería no llena;
       descarga para casa si hay déficit y batería disponible; IDLE resto."""
    if exc > 0.1 and soc < 0.88:
        return 1   # CARGAR_SOLAR
    if dfc > 0.1 and soc > 0.12:
        return 5   # DESCARGAR_CASA
    return 0       # IDLE

def pol_siempre_descargar_red(ep_step, row, soc, exc, dfc):
    return 8   # DESCARGAR_RED 100%

def pol_siempre_cargar(ep_step, row, soc, exc, dfc):
    """Siempre intenta cargar (CARGAR_MIXTA 100%). Extremo opuesto."""
    return 4   # CARGAR_MIXTA 100%

def pol_precio_alto_descarga(ep_step, row, soc, exc, dfc):
    """Precio-aware: descarga si precio está por encima de la media histórica."""
    UMBRAL_ALTO = 0.2172   # media histórica del dataset
    if exc > 0.1 and soc < 0.88:
        return 1   # CARGAR_SOLAR primero
    if soc > 0.12 and row['precio_kwh'] > UMBRAL_ALTO:
        return 5   # DESCARGAR_CASA si precio alto
    return 0


# ─────────────────────────────────────────────────────────────────
# DIAGNÓSTICO MATEMÁTICO
# ─────────────────────────────────────────────────────────────────

def analizar_recompensa_por_accion(sim, step):
    """
    Para una hora concreta, muestra el beneficio_marginal de cada acción
    y lo que pasa físicamente. Crucial para detectar sesgos.
    """
    row = sim.df.iloc[step]
    gen, cons = row['generacion_total'], row['consumo_total']
    pc, pv = row['precio_kwh'], row['precio_excedente']
    balance = gen - cons
    exc = max(0, balance)
    dfc = abs(min(0, balance))

    print(f"\n  Hora analizada: gen={gen:.2f} cons={cons:.2f} "
          f"exc={exc:.2f} dfc={dfc:.2f} kWh")
    print(f"  Precio compra={pc:.4f} precio_venta={pv:.4f} EUR/kWh")
    print(f"  SoC batería: {sim.soc:.3f}")
    print()

    nombres = {
        0: 'IDLE', 1: 'CARGAR_SOLAR', 2: 'CARGAR_MIXTA 33%',
        3: 'CARGAR_MIXTA 66%', 4: 'CARGAR_MIXTA 100%',
        5: 'DESCARGAR_CASA', 6: 'DESCARGAR_RED 33%',
        7: 'DESCARGAR_RED 66%', 8: 'DESCARGAR_RED 100%'
    }

    soc_orig = sim.soc
    for a in range(9):
        sim.current_step = step
        sim.soc = soc_orig
        r = sim.ejecutar_accion_fisica(a, step)
        delta_soc = r['soc'] - soc_orig
        print(f"  [{a}] {nombres[a]:<20} marginal={r['beneficio_marginal']:+.4f} EUR  "
              f"ΔSOC={delta_soc:+.4f}  "
              f"cargado={r['cargado']:.2f}  descargado={r['descargado']:.2f}")
        sim.soc = soc_orig  # restaurar para siguiente acción


def main():
    sim   = ComunidadSimulador(DATASET)
    rng   = np.random.default_rng(SEED)

    # Elegir una semana representativa (misma que usará el DQN)
    max_start = sim.max_steps - EPISODE_LENGTH - 26
    start = int(rng.integers(0, max_start))

    print(SEP)
    print("DIAGNÓSTICO COMPLETO DEL SISTEMA DE RECOMPENSA")
    print(SEP)
    print(f"\nSemana analizada: paso {start} a {start+EPISODE_LENGTH}")
    print(f"γ = {GAMMA}  |  γ^167 = {GAMMA**167:.4f}  ← factor descuento terminal")
    print(f"Significado: el bonus terminal vale solo {GAMMA**167*100:.1f}% de su valor nominal")

    # ── SECCIÓN 1: Recompensa por acción en horas típicas ─────────
    print(f"\n{SEP}")
    print("1. RECOMPENSA MARGINAL POR ACCIÓN EN HORAS TÍPICAS")
    print(SEP)

    # Encontrar hora con excedente solar significativo
    for s in range(start, start + EPISODE_LENGTH):
        row = sim.df.iloc[s]
        bal = row['generacion_total'] - row['consumo_total']
        if bal > 10:
            print("\n  --- HORA CON EXCEDENTE SOLAR (batería al 50%) ---")
            sim.soc = 0.5
            analizar_recompensa_por_accion(sim, s)
            break

    # Hora con déficit y batería cargada
    for s in range(start, start + EPISODE_LENGTH):
        row = sim.df.iloc[s]
        bal = row['generacion_total'] - row['consumo_total']
        if bal < -5:
            print("\n  --- HORA CON DÉFICIT (batería al 70%) ---")
            sim.soc = 0.7
            analizar_recompensa_por_accion(sim, s)
            break

    # ── SECCIÓN 2: Ciclo completo carga → descarga ────────────────
    print(f"\n{SEP}")
    print("2. CICLO COMPLETO CARGAR_SOLAR → DESCARGAR_CASA")
    print("   ¿Es rentable neto? (la pregunta clave)")
    print(SEP)

    # Encontrar 2 horas consecutivas: excedente y luego déficit
    for s in range(start, start + EPISODE_LENGTH - 5):
        r1 = sim.df.iloc[s]
        bal1 = r1['generacion_total'] - r1['consumo_total']
        if bal1 < 10:
            continue
        for s2 in range(s + 1, min(s + 15, start + EPISODE_LENGTH)):
            r2 = sim.df.iloc[s2]
            bal2 = r2['generacion_total'] - r2['consumo_total']
            if bal2 < -5:
                exc = bal1
                dfc = abs(bal2)
                pc1 = r1['precio_kwh']; pv1 = r1['precio_excedente']
                pc2 = r2['precio_kwh']
                EFF_C = sim.EFICIENCIA_CARGA; EFF_D = sim.EFICIENCIA_DESCARGA
                DEG = sim.COSTE_DEGRADACION_BASE

                # Recompensas del ciclo
                r_carga   = -exc * pv1  # renuncia a venta solar
                r_descar  = min(exc * EFF_C, dfc / EFF_D) * EFF_D * pc2  # ahorro compra
                # (deg simplificada)
                gap_pasos = s2 - s
                factor_gamma = GAMMA ** gap_pasos

                print(f"\n  Carga en paso {s} (bal={bal1:.1f} kWh, pv={pv1:.4f})")
                print(f"  Descarga en paso {s2} (bal={bal2:.1f} kWh, pc={pc2:.4f})")
                print(f"  Distancia temporal: {gap_pasos} horas (γ^{gap_pasos}={factor_gamma:.4f})")
                print(f"\n  Recompensa CARGAR_SOLAR:   {r_carga:+.4f} EUR (inmediata)")
                print(f"  Recompensa DESCARGAR_CASA: +{r_descar:.4f} EUR (en {gap_pasos}h)")
                print(f"  Valor descontado descarga:  {r_descar*factor_gamma:+.4f} EUR")
                print(f"  NETO descontado del ciclo:  {r_carga + r_descar*factor_gamma:+.4f} EUR")

                if r_carga + r_descar * factor_gamma > 0:
                    print("  ✓ RENTABLE para el DQN (Q-learning debería aprenderlo)")
                else:
                    print("  ✗ NO RENTABLE en términos descontados → DQN nunca aprenderá este ciclo")
                break
        else:
            continue
        break

    # ── SECCIÓN 3: Retorno total por estrategia ───────────────────
    print(f"\n{SEP}")
    print("3. RETORNO TOTAL POR ESTRATEGIA (misma semana, mismo SoC inicial)")
    print(SEP)

    politicas = [
        ("IDLE puro",                   pol_idle),
        ("Autoconsumo (carga→descarga)", pol_autoconsumo),
        ("Siempre DESCARGAR_RED 100%",   pol_siempre_descargar_red),
        ("Siempre CARGAR_MIXTA 100%",    pol_siempre_cargar),
        ("Precio-aware (umbral 0.2172)", pol_precio_alto_descarga),
    ]

    resultados = []
    for label, pol_fn in politicas:
        res = simular_semana(sim, start, SOC_INI, pol_fn, label, verbose=True)
        resultados.append(res)

    print(f"\n{'Estrategia':<35} {'No-disc':>10} {'Disc(γ=.99)':>12} {'Terminal(disc)':>15} {'SoC fin':>8}")
    print(SEP2)
    for r in resultados:
        print(f"  {r['label']:<33} {r['ret_nodisc']:>+10.2f} {r['ret_disc']:>+12.2f} "
              f"{r['terminal_disc']:>+15.2f} {r['soc_final']:>8.3f}")

    # ── SECCIÓN 4: Diagnóstico del terminal bonus ─────────────────
    print(f"\n{SEP}")
    print("4. DIAGNÓSTICO: TERMINAL BONUS vs INITIAL CORRECTION")
    print(SEP)

    idle_res = resultados[0]
    print(f"\n  Análisis para estrategia IDLE (debe dar 0 por diseño):")
    print(f"    Corrección inicial (restada en paso 0): {-idle_res['pendiente']:+.3f} EUR")
    print(f"    Terminal bonus (nominal):               {idle_res['terminal']:+.3f} EUR")
    print(f"    Terminal bonus (descontado γ^167):      {idle_res['terminal_disc']:+.3f} EUR")
    print(f"    Suma NO descontada:   {-idle_res['pendiente'] + idle_res['terminal']:+.3f} EUR  ← debería ser ≈0")
    print(f"    Suma DESCONTADA:      {-idle_res['pendiente'] + idle_res['terminal_disc']:+.3f} EUR  ← esto ve el DQN")

    diff = abs(-idle_res['pendiente'] + idle_res['terminal_disc'])
    print(f"\n  *** DESEQUILIBRIO que ve el DQN: {diff:.3f} EUR ***")
    if diff > 0.5:
        print(f"  *** PROBLEMA CONFIRMADO: el terminal bonus ({idle_res['terminal']:.2f} EUR)")
        print(f"  *** se descuenta a solo {idle_res['terminal_disc']:.2f} EUR para el DQN.")
        print(f"  *** El agente 'paga' {idle_res['pendiente']:.2f} EUR al inicio pero solo")
        print(f"  *** 'recupera' {idle_res['terminal_disc']:.2f} EUR en valor descontado.")
        print(f"  *** Resultado: tener batería cargada parece un COSTE para el DQN.")
    else:
        print(f"  ✓ El mecanismo inicial/terminal está equilibrado.")

    # ── SECCIÓN 5: Ranking que ve el DQN ─────────────────────────
    print(f"\n{SEP}")
    print("5. RANKING DESDE LA PERSPECTIVA DEL DQN (retorno descontado)")
    print("   ← esto es lo que el agente realmente maximiza")
    print(SEP)

    ordenados = sorted(resultados, key=lambda x: x['ret_disc'], reverse=True)
    print()
    for i, r in enumerate(ordenados):
        marca = " ← DQN elige esta" if i == 0 else ""
        marca2 = " ← ÓPTIMA real" if r['label'].startswith("Autoconsumo") else ""
        print(f"  {i+1}. {r['label']:<35} {r['ret_disc']:>+10.2f} EUR{marca}{marca2}")

    print()
    autocons_rank = next(i for i, r in enumerate(ordenados)
                         if r['label'].startswith("Autoconsumo"))
    descargar_rank = next(i for i, r in enumerate(ordenados)
                          if "DESCARGAR" in r['label'])

    if descargar_rank < autocons_rank:
        print("  *** PROBLEMA: DESCARGAR_RED supera al autoconsumo en términos DESCONTADOS.")
        print("  *** El DQN tiene incentivo matemático equivocado. No es fallo de aprendizaje,")
        print("  *** es un fallo en el diseño del sistema de recompensa con γ y episodios largos.")
        print("  *** SOLUCIÓN: potential-based reward shaping.")
    else:
        print("  ✓ El autoconsumo supera a DESCARGAR_RED en términos descontados.")
        print("  El problema es de aprendizaje (crédito, exploración), no de diseño de recompensa.")

    print(f"\n{SEP}")
    print("FIN DEL DIAGNÓSTICO")
    print(SEP)


if __name__ == "__main__":
    main()
