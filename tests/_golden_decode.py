"""Utilidades compartidas del golden-master de la física discreta (F6).

Genera un conjunto DETERMINISTA de casos (step × SoC × acción) y captura/compara
la salida de `ComunidadSimulador.ejecutar_accion_fisica`. Se usa para garantizar
que el refactor F6 (decode + aplicar_fisica_4flujos) preserva el comportamiento
exacto del entreno discreto.
"""
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.simulator import ComunidadSimulador  # noqa: E402

_CFG = yaml.safe_load(open(os.path.join(ROOT, "config", "system.yaml")))
DATASET_PATH = os.path.join(ROOT, _CFG["rutas"]["dataset_final"])
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_ejecutar_accion.npz")

# Rejilla determinista de casos
_SOC_GRID = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90]
_N_STEPS = 240          # steps muestreados a lo largo del dataset
_ACCIONES = list(range(9))
_CLAVES = ["beneficio", "beneficio_marginal", "soc", "comprado", "cargado", "descargado"]


def _steps_muestra(sim):
    """Steps deterministas repartidos por el dataset (evitando el final)."""
    top = sim.max_steps - 1
    return np.linspace(0, top, _N_STEPS, dtype=int).tolist()


def generar_casos(sim):
    """Lista determinista de (step, soc, accion)."""
    casos = []
    for step in _steps_muestra(sim):
        for soc in _SOC_GRID:
            for a in _ACCIONES:
                casos.append((int(step), float(soc), int(a)))
    return casos


def capturar(sim, casos):
    """Ejecuta ejecutar_accion_fisica en cada caso y devuelve matriz (N, 6)."""
    out = np.empty((len(casos), len(_CLAVES)), dtype=np.float64)
    for i, (step, soc, a) in enumerate(casos):
        sim.soc = soc
        r = sim.ejecutar_accion_fisica(a, step)
        out[i] = [r[k] for k in _CLAVES]
    return out


if __name__ == "__main__":
    sim = ComunidadSimulador(DATASET_PATH)
    casos = generar_casos(sim)
    vals = capturar(sim, casos)
    np.savez_compressed(
        GOLDEN_PATH,
        casos=np.array(casos, dtype=np.float64),
        vals=vals,
        claves=np.array(_CLAVES),
    )
    print(f"Golden capturado: {len(casos)} casos × {len(_CLAVES)} salidas -> {GOLDEN_PATH}")
