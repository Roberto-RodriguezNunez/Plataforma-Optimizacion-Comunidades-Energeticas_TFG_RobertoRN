"""
estadistica.py — Estadística para los resultados de la suite
============================================================
IC95% por bootstrap de la media + test de Wilcoxon pareado frente a un baseline.
Usado por src/evaluation/suite.py para cuantificar significancia (F1) y robustez.
"""

import numpy as np
from scipy import stats


def ic95_bootstrap(x, n_boot: int = 10000, seed: int = 0):
    """IC95% (percentiles 2.5/97.5) de la MEDIA de x por bootstrap."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_vs(x, base):
    """p-valor del Wilcoxon pareado x vs base (mismo orden (seed,semana)).

    Devuelve NaN si las formas no casan; 1.0 si las diferencias son todas ~0.
    """
    x = np.asarray(x, dtype=float)
    base = np.asarray(base, dtype=float)
    if x.shape != base.shape or x.size == 0:
        return float("nan")
    if np.allclose(x - base, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(x, base).pvalue)
    except Exception:
        return float("nan")
