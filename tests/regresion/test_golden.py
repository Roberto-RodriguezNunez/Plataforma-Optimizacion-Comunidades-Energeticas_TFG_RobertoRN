"""test_golden.py — Verifica que el refactor NO cambia ningún número.

Recalcula el snapshot de los caminos críticos y lo compara, clave a clave,
contra el golden congelado en golden/golden.npz. Si algo se mueve, falla y dice
exactamente qué entrada y con qué desviación.

Regenerar el golden (solo cuando un cambio de comportamiento es INTENCIONADO):
    .venv/bin/python tests/regresion/snapshot.py
"""
import os

import numpy as np
import pytest

from snapshot import GOLDEN_PATH, compute_snapshot

# Caminos de numpy puro: exigimos igualdad EXACTA (bit a bit).
# Caminos con scipy/torch (MPC, DQN): toleramos epsilon numérico mínimo.
_ATOL = {
    "mpc_real_marg": 1e-9,
    "mpc_real_abs": 1e-9,
    "dqn_real_marg": 1e-9,
    "residualenv_obs0": 1e-9,
    "residualenv_obs_fin": 1e-9,
    "residualenv_reward": 1e-9,
}
_ATOL_DEFAULT = 0.0


@pytest.fixture(scope="module")
def golden():
    if not os.path.exists(GOLDEN_PATH):
        pytest.skip(f"No existe el golden ({GOLDEN_PATH}). Genéralo con "
                    f"`python tests/regresion/snapshot.py`.")
    data = np.load(GOLDEN_PATH)
    return {k: data[k] for k in data.files}


@pytest.fixture(scope="module")
def actual():
    return compute_snapshot()


def test_mismas_claves(golden, actual):
    assert set(actual.keys()) == set(golden.keys()), (
        f"Claves distintas. Falta(n): {set(golden) - set(actual)}; "
        f"sobra(n): {set(actual) - set(golden)}"
    )


@pytest.mark.parametrize("clave", [
    "obs_build", "sim_soc", "sim_benef_marg", "sim_benef",
    "energyenv_obs0", "energyenv_obs_fin", "energyenv_reward",
    "residualenv_obs0", "residualenv_obs_fin", "residualenv_reward",
    "mpc_real_marg", "mpc_real_abs", "dqn_real_marg",
])
def test_golden_por_clave(golden, actual, clave):
    if clave not in golden:
        pytest.skip(f"'{clave}' no está en el golden (p.ej. modelo ausente).")
    assert clave in actual, f"El snapshot actual no produjo '{clave}'."
    atol = _ATOL.get(clave, _ATOL_DEFAULT)
    np.testing.assert_allclose(
        actual[clave], golden[clave], rtol=0.0, atol=atol,
        err_msg=f"Regresión numérica en '{clave}' (atol={atol})",
    )
