"""eval_rapida.py — Evalúa UN modelo y solo imprime el resultado (NO toca resultados.csv).

Sirve para mirar la eval real de una versión o de una SEMILLA SUELTA de
models/_runs/, sin escribir ninguna fila. Útil para comprobar un entreno en
curso (la eval limpia suele dar algo más que la recompensa de entreno).

Uso:
  # versión ya consolidada (models/best/):
  .venv/bin/python experimentos/eval_rapida.py --familia td3_residual --version TD3-1

  # una semilla concreta de _runs/ (lo más cómodo, solo el número):
  .venv/bin/python experimentos/eval_rapida.py --familia td3_residual --version TD3-1 --seed 42

  # nº de semillas de ruido (default 10; usa 1 para ir rápido):
  ... --seeds 3
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from src.training.registry import cargar_version, seeds_comunes
from src.evaluation.unified import evaluar_multiseed
from src.evaluation.stats import ic95_bootstrap
from src.evaluation.suite import _construir_controller, _paths, _baseline_mpc_pooled
from src.controllers.mpc import crear_mpc
from src.core.simulator import ComunidadSimulador
import src.evaluation.suite as ES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--familia", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--seed", type=int, default=None,
                   help="evalúa la SEMILLA suelta de models/_runs/ (p.ej. --seed 42). "
                        "Sin esto usa el modelo consolidado de models/best/.")
    p.add_argument("--modelo", default=None, help="ruta .zip explícita (tiene prioridad sobre --seed)")
    p.add_argument("--vecnorm", default=None, help="ruta _vecnorm.pkl explícita")
    p.add_argument("--seeds", type=int, default=None, help="nº semillas de ruido (default: el de la suite)")
    p.add_argument("--vs-mpc", action="store_true", help="además compara con el MPC realista (más lento)")
    a = p.parse_args()

    cfg = cargar_version(a.familia, a.version)
    _, eval_seeds_n, _ = seeds_comunes()
    n = a.seeds if a.seeds else eval_seeds_n
    eval_seeds = [ES.SEED + i for i in range(n)]

    # Rutas: explícitas > semilla de _runs/ > consolidado de best/
    if a.modelo:
        model_path = a.modelo
        vecnorm = a.vecnorm
    elif a.seed is not None:
        run_dir = os.path.join(ROOT, "models", "_runs",
                               f"{cfg['algo']}_{a.version}_seed{a.seed}")
        model_path = os.path.join(run_dir, "best_model.zip")
        vecnorm = os.path.join(run_dir, "best_vecnormalize.pkl")
    else:
        model_path, vecnorm, _ = _paths(cfg["algo"], a.version)
    if not os.path.exists(model_path):
        sys.exit(f"No existe el modelo: {model_path}")
    if vecnorm and not os.path.exists(vecnorm):
        vecnorm = None

    sim = ComunidadSimulador(ES.DATASET_PATH)
    mpc = crear_mpc()
    ctrl = _construir_controller(a.familia, cfg, model_path, vecnorm, sim, mpc)

    print(f"Evaluando {a.familia}/{a.version}  ({os.path.basename(model_path)})  "
          f"con {n} semillas de ruido...")
    res = evaluar_multiseed(ctrl, "realista", eval_seeds)
    pooled = res["bens_marg"]
    lo, hi = ic95_bootstrap(pooled)
    media = float(pooled.mean())

    print("-" * 56)
    print(f"  media        = {media:.3f} €/sem")
    print(f"  IC95         = [{lo:.3f}, {hi:.3f}]")
    print(f"  std (semillas)= {res['medias_por_seed'].std():.3f}")
    if a.vs_mpc:
        base = _baseline_mpc_pooled(eval_seeds, sigma_consumo=cfg.get("sigma_consumo"))
        print(f"  vs MPC real. = {media - float(base.mean()):+.3f}")
    print("-" * 56)
    print("  (NO se ha escrito nada en resultados.csv)")


if __name__ == "__main__":
    main()
