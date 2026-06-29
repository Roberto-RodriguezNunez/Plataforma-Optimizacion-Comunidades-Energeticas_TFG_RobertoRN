"""scaffold_mains.py — Esqueleto común de los src/main_*.py (Fase 2 del refactor).

Antes, los 7 `src/main_*.py` repetían ~60-70 líneas idénticas: setup de
ROOT/sys.path, LOG_DIR/MODEL_DIR, carga de la versión, resolución de semillas,
creación de carpetas, parseo de CLI y la llamada a `entrenar_multiseed`. Solo
cambiaba de verdad lo específico del algoritmo (make_envs / make_model /
callbacks / warmup).

Aquí vive ese esqueleto UNA sola vez. Cada main aporta una factoría
`build(cfg, total_timesteps) -> spec` con lo suyo:

    spec = {
        "make_envs":       callable()            -> (train_env, eval_env)   [requerido]
        "make_model":      callable(env, seed)   -> modelo SB3              [requerido]
        "extra_callbacks": callable(model)       -> [BaseCallback] | None   [opcional]
        "warmup":          callable(model, env, seed) | None                [opcional]
        "pre_entreno":     callable()            (p.ej. check_env)          [opcional]
        "algo":            str (default: familia)                          [opcional]
    }

`build` recibe el cfg de la versión (registry) ya cargado y el total_timesteps
resuelto, de modo que las factorías se cierran sobre ellos (sin globals).

NOTA: el bucle de entreno real (por semilla) NO está aquí, sino en
`multiseed.entrenar_multiseed`. Esto es solo la capa de arranque por encima.
Las factorías del entorno residual (compartidas por SAC/TD3/DDPG) están en
`factorias_residual.py`.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.multiseed import entrenar_multiseed
from src.training.registry import cargar_version, seeds_comunes, seeds_para_version

LOG_DIR = os.path.join(ROOT, "logs")
MODEL_DIR = os.path.join(ROOT, "models")


def entrenar(familia, build, version, seeds=None, total_timesteps=None):
    """Carga la versión, resuelve semillas/timesteps y entrena multi-semilla.

    Reproduce EXACTAMENTE la secuencia que tenían los mains: cargar_version →
    seeds_comunes → seeds_para_version (si no se pasan) → total_timesteps →
    makedirs → (pre_entreno) → entrenar_multiseed.
    """
    cfg = cargar_version(familia, version)
    _train_seeds, _eval_seeds, eval_episodes = seeds_comunes()
    if seeds is None:
        seeds = seeds_para_version(familia, version)
    if total_timesteps is None:
        total_timesteps = int(cfg["total_timesteps"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    spec = build(cfg, total_timesteps)

    pre = spec.get("pre_entreno")
    if pre is not None:
        pre()

    entrenar_multiseed(
        algo=spec.get("algo", familia), version=version, seeds=list(seeds),
        make_envs=spec["make_envs"], make_model=spec["make_model"],
        total_timesteps=total_timesteps,
        eval_freq=int(cfg["eval_freq"]), n_eval_episodes=eval_episodes,
        warmup=spec.get("warmup"),
        extra_callbacks=spec.get("extra_callbacks"),
    )


def cli(familia, build, version_default, description=None):
    """Parser estándar (--version/--seeds/--timesteps) + entrenar()."""
    p = argparse.ArgumentParser(
        description=description or f"Entrena {familia} multi-semilla.")
    p.add_argument("--version", default=version_default)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--timesteps", type=int, default=None,
                   help="Override total timesteps (smoke).")
    a = p.parse_args()
    entrenar(familia, build, version=a.version, seeds=a.seeds,
             total_timesteps=a.timesteps)
