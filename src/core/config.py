"""config.py — Fuente ÚNICA de carga de config/system.yaml.

Antes, `system.yaml` se abría y parseaba por separado en `core/simulator.py`,
`controllers/heuristic.py`, `controllers/mpc.py`,
`evaluation/unified.py`... — varias "fuentes de verdad" con riesgo de
desincronizarse. Este módulo lo carga una sola vez (cacheado) y expone
accesores. NO valida ni transforma: devuelve el dict tal cual del YAML, de modo
que el comportamiento es idéntico al de las cargas anteriores.
"""
import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYSTEM_PATH = os.path.join(ROOT, "config", "system.yaml")

_cache = None


def cargar_system():
    """Devuelve el dict completo de config/system.yaml (cacheado)."""
    global _cache
    if _cache is None:
        with open(SYSTEM_PATH, "r", encoding="utf-8") as f:
            _cache = yaml.safe_load(f)
    return _cache


def bateria():
    """Sección 'bateria' (capacidad, eficiencias, SoC, degradación...)."""
    return cargar_system()["bateria"]


def pronostico():
    """Sección 'pronostico' (rho/sigma AR(1), precio 3 capas...)."""
    return cargar_system()["pronostico"]


def mpc():
    """Sección 'mpc' (horizonte, duración episodio, valor terminal...)."""
    return cargar_system()["mpc"]
