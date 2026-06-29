"""Golden-master de la física discreta (F6).

Garantiza que el refactor de `ejecutar_accion_fisica` (decode único +
`aplicar_fisica_4flujos` + shaping) preserva EXACTAMENTE el comportamiento
capturado antes del cambio (tests/golden_ejecutar_accion.npz). Si pasa, el
entreno discreto (DQN/PPO) no cambia de comportamiento.

Re-generar el golden (solo si se cambia el modelo a propósito):
    python tests/_golden_decode.py
"""
import os

import numpy as np
import pytest

from tests._golden_decode import (
    DATASET_PATH, GOLDEN_PATH, generar_casos, capturar,
)
from src.core.simulador import ComunidadSimulador


@pytest.mark.skipif(not os.path.exists(GOLDEN_PATH), reason="falta el golden; corre _golden_decode.py")
def test_ejecutar_accion_fisica_identica_al_golden():
    g = np.load(GOLDEN_PATH, allow_pickle=True)
    golden_casos, golden_vals = g["casos"], g["vals"]

    sim = ComunidadSimulador(DATASET_PATH)
    casos = generar_casos(sim)
    assert np.array_equal(np.array(casos, dtype=np.float64), golden_casos), \
        "los casos generados no coinciden con los del golden (¿cambió el dataset/rejilla?)"

    nuevos = capturar(sim, casos)

    max_abs = float(np.max(np.abs(nuevos - golden_vals)))
    assert np.allclose(nuevos, golden_vals, atol=1e-9, rtol=0.0), (
        f"la nueva ejecutar_accion_fisica difiere del golden (max |Δ| = {max_abs:.3e}). "
        "El refactor F6 NO preserva el comportamiento."
    )
