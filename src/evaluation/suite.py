"""
eval_suite.py — Evaluación estructurada de la suite experimental
================================================================
Dado un modelo ya entrenado (models/best/{algo}_{version}.zip + _vecnorm.pkl),
construye el controller correcto según la familia, lo evalúa con N semillas de
ruido (default 10) reutilizando el protocolo de eval_unificada, calcula
media €/sem, std sobre semillas, IC95% (bootstrap) y Wilcoxon vs MPC realista, y
añade una fila a resultados/resultados.csv (+ JSON por versión).

Uso:
    python src/evaluation/suite.py --baselines                 # G1-G4 (una vez)
    python src/evaluation/suite.py --familia dqn --version DQN-2
    python src/evaluation/suite.py --todas                     # todas las versiones con artefacto

El CSV es la fuente que se vuelca a las tablas LaTeX (anexos F.1-F.6 + G).
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluation.unified import (
    evaluar_controlador, evaluar_multiseed, IdleController,
    ComunidadSimulador, DATASET_PATH, crear_mpc, SEED,
)
from src.controllers.heuristic import HeuristicController
from src.controllers.discrete import DiscreteRLController
from src.controllers.continuous import ContinuousController
from src.controllers.residual_sac_controller import ResidualSACController
from src.training.registry import cargar_version, familias, listar_versiones
from src.evaluation.stats import ic95_bootstrap, wilcoxon_vs

BEST_DIR = os.path.join(ROOT, "models", "best")
RES_DIR = os.path.join(ROOT, "resultados")
CSV_PATH = os.path.join(RES_DIR, "resultados.csv")
MPC_CACHE = os.path.join(RES_DIR, "_mpc_realista_pooled.npy")

CAMPOS = [
    "familia", "version", "algo", "controller", "media_eur_sem", "std_seeds",
    "ic95_low", "ic95_high", "n_eval_seeds", "vs_mpc_realista", "wilcoxon_p",
    "mejor_seed", "train_scores", "hiperparametros", "timestamp",
]

# Familia → ('SB3_ALGO', tipo_controller)
_ALGO_SB3 = {
    "residual_sac": "SAC", "td3_residual": "TD3", "ddpg_residual": "DDPG",
    "sac_puro": "SAC", "ppo_continuo": "PPO", "dqn": "DQN", "ppo": "PPO",
}


def _paths(algo, version):
    base = os.path.join(BEST_DIR, f"{algo}_{version}")
    return base + ".zip", base + "_vecnorm.pkl", base + "_seeds.json"


def _hiperparametros(cfg):
    """cfg de la registry sin las claves meta, como dict serializable."""
    return {k: v for k, v in cfg.items()
            if k not in ("algo", "env", "familia", "version", "nota")}


def _construir_controller(familia, cfg, model_path, vecnorm, sim, mpc):
    env = cfg["env"]
    if env == "discreto":
        return DiscreteRLController(model_path, sim, vecnorm, algo=_ALGO_SB3[familia])
    if env == "continuo":
        return ContinuousController(model_path, sim, vecnorm, algo=_ALGO_SB3[familia])
    if env == "residual":
        return ResidualSACController(
            model_path, mpc, sim, float(cfg["delta_max"]), vecnorm,
            algo=_ALGO_SB3[familia], residual_mode=cfg.get("residual_mode", "mult"),
        )
    raise ValueError(f"env '{env}' no soportado")


def _baseline_mpc_pooled(eval_seeds, sigma_consumo=None):
    """Pooled del MPC realista (para vs_mpc + Wilcoxon). Cachea el caso por defecto."""
    if sigma_consumo is None and os.path.exists(MPC_CACHE):
        return np.load(MPC_CACHE)
    mpc = crear_mpc()
    res = evaluar_multiseed(mpc, "realista", eval_seeds)
    pooled = res["bens_marg"]
    if sigma_consumo is None:
        os.makedirs(RES_DIR, exist_ok=True)
        np.save(MPC_CACHE, pooled)
    return pooled


def _escribir_fila(fila):
    os.makedirs(RES_DIR, exist_ok=True)
    nuevo = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader()
        w.writerow(fila)
    with open(os.path.join(RES_DIR, f"{fila['algo']}_{fila['version']}.json"), "w") as f:
        json.dump(fila, f, indent=2, ensure_ascii=False)
    print(f"  -> fila escrita: {fila['familia']}/{fila['version']}  "
          f"{fila['media_eur_sem']:+.2f} €/sem (vs MPC {fila['vs_mpc_realista']:+.2f}, p={fila['wilcoxon_p']})")


def evaluar_version(familia, version, eval_seeds):
    cfg = cargar_version(familia, version)
    algo = cfg["algo"]
    model_path, vecnorm, seeds_json = _paths(algo, version)
    if not os.path.exists(model_path):
        print(f"  [SALTADO] {familia}/{version}: no existe {model_path} (entrénalo primero)")
        return
    if not os.path.exists(vecnorm):
        vecnorm = None

    # F6: σ del pronóstico es propiedad del entorno → fijarla también en eval.
    sigma = cfg.get("sigma_consumo")
    if sigma is not None:
        import src.core.forecast as _fc
        _fc.SIGMA_CONS_BASE = float(sigma)
        print(f"  [F6] eval con SIGMA_CONS_BASE = {sigma}")

    sim = ComunidadSimulador(DATASET_PATH)
    mpc = crear_mpc()
    ctrl = _construir_controller(familia, cfg, model_path, vecnorm, sim, mpc)

    res = evaluar_multiseed(ctrl, "realista", eval_seeds)
    pooled = res["bens_marg"]
    medias = res["medias_por_seed"]
    base_pooled = _baseline_mpc_pooled(eval_seeds, sigma_consumo=sigma)

    lo, hi = ic95_bootstrap(pooled)
    media = float(pooled.mean())
    train_scores = {}
    mejor_seed = ""
    if os.path.exists(seeds_json):
        sj = json.load(open(seeds_json))
        train_scores = sj.get("scores", {})
        mejor_seed = sj.get("best_seed", "")

    _escribir_fila({
        "familia": familia, "version": version, "algo": algo,
        "controller": ctrl.nombre(),
        "media_eur_sem": round(media, 3), "std_seeds": round(float(medias.std()), 3),
        "ic95_low": round(lo, 3), "ic95_high": round(hi, 3),
        "n_eval_seeds": len(eval_seeds),
        "vs_mpc_realista": round(media - float(base_pooled.mean()), 3),
        "wilcoxon_p": f"{wilcoxon_vs(pooled, base_pooled):.2e}",
        "mejor_seed": mejor_seed,
        "train_scores": json.dumps(train_scores),
        "hiperparametros": json.dumps(_hiperparametros(cfg), ensure_ascii=False),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


def evaluar_baselines(eval_seeds):
    print("  Baselines G1-G4 ...")
    sim = ComunidadSimulador(DATASET_PATH)
    mpc = crear_mpc()
    filas = []

    # G1 MPC oráculo (1 semilla, previsión perfecta)
    r = evaluar_controlador(mpc, forecast_mode="oraculo")
    filas.append(("G", "G1-MPC-oraculo", "mpc", "MPC oraculo",
                  r["bens_marg"], r["bens_marg"]))
    # G2 MPC realista (multiseed) — además cachea el pooled
    base_pooled = _baseline_mpc_pooled(eval_seeds)
    rr = evaluar_multiseed(mpc, "realista", eval_seeds)
    filas.append(("G", "G2-MPC-realista", "mpc", "MPC realista",
                  rr["bens_marg"], rr["medias_por_seed"]))
    # G3 Heurístico
    rh = evaluar_multiseed(HeuristicController(), "realista", eval_seeds)
    filas.append(("G", "G3-heuristico", "heuristico", "Heuristico",
                  rh["bens_marg"], rh["medias_por_seed"]))
    # G4 IDLE (batería desconectada)
    ri = evaluar_controlador(IdleController(), forecast_mode="realista")
    filas.append(("G", "G4-idle", "idle", "IDLE", ri["bens_marg"], ri["bens_marg"]))

    for fam, ver, algo, ctrl_name, pooled, medias in filas:
        lo, hi = ic95_bootstrap(pooled)
        media = float(pooled.mean())
        _escribir_fila({
            "familia": fam, "version": ver, "algo": algo, "controller": ctrl_name,
            "media_eur_sem": round(media, 3), "std_seeds": round(float(np.std(medias)), 3),
            "ic95_low": round(lo, 3), "ic95_high": round(hi, 3),
            "n_eval_seeds": len(eval_seeds),
            "vs_mpc_realista": round(media - float(base_pooled.mean()), 3),
            "wilcoxon_p": f"{wilcoxon_vs(pooled, base_pooled):.2e}" if pooled.shape == base_pooled.shape else "n/a",
            "mejor_seed": "", "train_scores": "{}", "hiperparametros": "{}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })


def main():
    p = argparse.ArgumentParser(description="Evaluación estructurada de la suite.")
    p.add_argument("--familia", default=None)
    p.add_argument("--version", default=None)
    p.add_argument("--todas", action="store_true", help="Evalúa toda versión con artefacto.")
    p.add_argument("--baselines", action="store_true", help="Calcula G1-G4.")
    p.add_argument("--eval-seeds", type=int, default=10)
    a = p.parse_args()

    eval_seeds = [SEED + i for i in range(a.eval_seeds)]
    print("=" * 70)
    print(f"EVAL SUITE — {len(eval_seeds)} semillas de ruido (base {SEED})")
    print("=" * 70)

    if a.baselines:
        evaluar_baselines(eval_seeds)

    if a.todas:
        for fam in familias():
            for ver in listar_versiones(fam):
                evaluar_version(fam, ver, eval_seeds)
    elif a.familia and a.version:
        evaluar_version(a.familia, a.version, eval_seeds)
    elif a.familia:
        for ver in listar_versiones(a.familia):
            evaluar_version(a.familia, ver, eval_seeds)
    elif not a.baselines:
        p.error("indica --baselines, --todas, --familia X [--version Y]")

    print(f"\nResultados en: {CSV_PATH}")


if __name__ == "__main__":
    main()
