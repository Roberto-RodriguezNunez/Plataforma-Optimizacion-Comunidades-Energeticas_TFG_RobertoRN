"""
test_fisica_4flujos.py — Identidad de la física unificada de la batería
=======================================================================
Tras unificar la física de 4 flujos en `ComunidadSimulador.aplicar_fisica_4flujos`
(usada por energy_env_continuo del SAC y por simular_hora_mpc del MPC), este test
verifica que el resultado es IDÉNTICO a la implementación anterior (referencia
congelada) para miles de estados aleatorios — garantía de que la refactorización
no cambió el comportamiento.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.simulador import ComunidadSimulador

DATA = os.path.join(ROOT, 'data', 'processed', 'dataset_final.csv')


def _fisica_referencia(sim, cs, cm, dc, dr):
    """Copia congelada de la física antigua (energy_env_continuo / simular_hora_mpc)."""
    row = sim.df.iloc[sim.current_step]
    gen, cons = row['generacion_total'], row['consumo_total']
    pc, pv = row['precio_kwh'], row['precio_excedente']
    exc = max(0.0, gen - cons)
    deff = max(0.0, cons - gen)
    EFF_C, EFF_D = sim.EFICIENCIA_CARGA, sim.EFICIENCIA_DESCARGA
    CAP, AUTO = sim.BATERIA_CAPACIDAD, sim.AUTODESCARGA_POR_HORA
    sim.soc *= (1 - AUTO)
    bkwh = sim.soc * CAP
    libre = max(0.0, sim.SOC_MAX * CAP - bkwh)
    disp = max(0.0, bkwh - sim.SOC_MIN * CAP)
    cb, db = cs + cm, dc + dr
    net = cb - db
    if net >= 0:
        rs = cs / cb if cb > 0 else 0.0
        cs, cm, dc, dr = net * rs, net * (1 - rs), 0.0, 0.0
    else:
        rc = dc / db if db > 0 else 0.0
        dc, dr, cs, cm = abs(net) * rc, abs(net) * (1 - rc), 0.0, 0.0
    cs = min(cs, exc, libre / EFF_C)
    cm = min(cm, max(0.0, libre / EFF_C - cs))
    ct = cs + cm
    dc = min(dc, disp)
    dr = min(dr, max(0.0, disp - dc))
    dt = dc + dr
    sa = sim.soc
    bkwh += ct * EFF_C - dt
    sim.soc = float(np.clip(bkwh / CAP, 0.0, 1.0))
    comp = max(0.0, deff - dc * EFF_D) + cm
    vend = (exc - cs) + dr * EFF_D
    deg = sim.calcular_degradacion_no_lineal(ct + dt, (sa + sim.soc) / 2)
    ben = vend * pv - comp * pc - deg
    bidle = exc * pv - deff * pc
    return {'beneficio': ben, 'beneficio_marginal': ben - bidle, 'soc': sim.soc,
            'comprado': comp, 'vendido': vend, 'cargado': ct, 'descargado': dt,
            'beneficio_idle': bidle}


def test_fisica_4flujos_identica_a_referencia():
    rng = np.random.default_rng(0)
    sn = ComunidadSimulador(DATA)
    so = ComunidadSimulador(DATA)
    P = sn.POTENCIA_INVERSOR
    maxdiff = 0.0
    for _ in range(3000):
        step = int(rng.integers(0, sn.max_steps - 1))
        soc = float(rng.uniform(0.1, 0.9))
        cs, cm, dc, dr = (float(rng.uniform(0, P)) for _ in range(4))
        sn.current_step = step; sn.soc = soc
        so.current_step = step; so.soc = soc
        r = sn.aplicar_fisica_4flujos(cs, cm, dc, dr)
        ref = _fisica_referencia(so, cs, cm, dc, dr)
        for k in r:
            maxdiff = max(maxdiff, abs(r[k] - ref[k]))
    assert maxdiff < 1e-9, f"física unificada difiere de la referencia: maxdiff={maxdiff:.2e}"
