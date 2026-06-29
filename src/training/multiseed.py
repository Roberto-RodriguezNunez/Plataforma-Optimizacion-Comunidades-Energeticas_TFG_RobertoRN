"""
multiseed.py — Runner de entrenamiento compartido (multi-semilla)
=================================================================
Orquesta el entrenamiento de CUALQUIER algoritmo (DQN, PPO, Residual SAC) con
varias semillas, selecciona el mejor modelo por recompensa de evaluación y lo
guarda en una carpeta común (`models/best/`) con nomenclatura
`{algo}_{version}.zip` + `{algo}_{version}_vecnorm.pkl` + `{algo}_{version}_seeds.json`.

Cada algoritmo aporta sus FACTORÍAS (make_envs, make_model, warmup opcional);
el runner se encarga del bucle de semillas, el callback de evaluación
(reproducible y seedeado), la selección del mejor y el guardado de artefactos.
Así todos los entrenamientos comparten exactamente la misma lógica.
"""

import os
import json
import shutil

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BEST_DIR = os.path.join(ROOT, "models", "best")
RUNS_DIR = os.path.join(ROOT, "models", "_runs")


class SeededEvalCallback(EvalCallback):
    """EvalCallback que re-seedea ``np.random`` antes de cada ronda de evaluación
    (garantiza que las semanas eval y el ruido AR(1) son idénticos entre evals)
    y guarda el VecNormalize junto al best_model para poder reproducir la inferencia.
    Compartido por todos los algoritmos.
    """

    def __init__(self, *args, eval_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_seed = eval_seed

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            np.random.seed(self._eval_seed)
        prev_best = self.best_mean_reward
        result = super()._on_step()
        if self.best_mean_reward > prev_best and self.best_model_save_path is not None:
            vec = self.model.get_vec_normalize_env()
            if vec is not None:
                vec.save(os.path.join(self.best_model_save_path, "best_vecnormalize.pkl"))
        return result


def _semilla_completa(run_dir, total_timesteps, eval_freq):
    """True si la semilla TERMINÓ su entrenamiento completo.

    Se detecta por su ``evaluations.npz`` (lo escribe SB3 en cada evaluación):
    una semilla terminada tiene su última evaluación a ~``total_timesteps``; una
    interrumpida (p.ej. un apagón) la tiene por debajo. La semilla a medias NO
    supera esta comprobación y se reentrena desde cero, porque su
    ``best_model.zip`` guardado es parcial y no representa un entreno completo.
    """
    ev = os.path.join(run_dir, "evaluations.npz")
    if not os.path.exists(ev):
        return False
    try:
        ts = np.load(ev)["timesteps"]
    except Exception:
        return False
    return len(ts) > 0 and int(ts[-1]) >= int(total_timesteps) - int(eval_freq)


def _score_semilla(run_dir):
    """Recompensa de eval (equivalente a ``best_mean_reward``) de una semilla ya
    completa, recuperada de su ``evaluations.npz`` sin reentrenar."""
    ev = np.load(os.path.join(run_dir, "evaluations.npz"))
    return float(ev["results"].mean(axis=1).max())


def entrenar_multiseed(
    algo: str,
    version: str,
    seeds,
    make_envs,
    make_model,
    total_timesteps: int,
    eval_freq: int,
    n_eval_episodes: int,
    warmup=None,
    extra_callbacks=None,
    eval_seed: int = 42,
):
    """Entrena una corrida por cada semilla y guarda el MEJOR modelo en común.

    Args:
        algo:            nombre del algoritmo, p.ej. "dqn", "ppo", "residual_sac".
        version:         etiqueta de la versión entrenada (define el nombre de salida).
        seeds:           lista de semillas, p.ej. [42, 1337, 2024].
        make_envs(): -> (train_env, eval_env)  ya envueltos en VecNormalize.
                          (no recibe seed: la siembra la aplica make_model vía SB3).
        make_model(train_env, seed): -> modelo SB3 (con seed aplicado).
        warmup(model, train_env, seed): opcional (p.ej. DAWN warmup del Residual SAC).
        extra_callbacks(model): opcional -> lista de BaseCallback extra (métricas, etc.).

    Returns:
        (best_seed, resultados)  resultados = {seed: best_eval_reward}
    """
    os.makedirs(BEST_DIR, exist_ok=True)
    resultados = {}

    for seed in seeds:
        run_name = f"{algo}_{version}_seed{seed}"
        run_dir = os.path.join(RUNS_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Reanudación: si la semilla ya terminó en una ejecución previa, se
        # reutiliza su resultado sin reentrenar. La que quedó a medias (apagón)
        # no supera la comprobación y se reentrena entera (sobrescribe su parcial).
        if _semilla_completa(run_dir, total_timesteps, eval_freq):
            resultados[seed] = _score_semilla(run_dir)
            print("=" * 62)
            print(f"  {algo.upper()}  {version}  —  SEMILLA {seed}  "
                  f"[YA COMPLETA -> reutilizada, reward {resultados[seed]:.2f}]")
            print("=" * 62)
            continue

        print("=" * 62)
        print(f"  {algo.upper()}  {version}  —  SEMILLA {seed}")
        print("=" * 62)

        train_env, eval_env = make_envs()
        model = make_model(train_env, seed)
        if warmup is not None:
            warmup(model, train_env, seed)

        callbacks = [SeededEvalCallback(
            eval_env,
            best_model_save_path=run_dir,
            log_path=run_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=1,
            eval_seed=eval_seed,
        )]
        if extra_callbacks is not None:
            callbacks += extra_callbacks(model)

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=run_name,
            progress_bar=False,
        )

        resultados[seed] = float(callbacks[0].best_mean_reward)
        print(f"\n  >> semilla {seed}: best eval reward = {resultados[seed]:.2f}\n")

        try:
            train_env.close()
            eval_env.close()
        except Exception:
            pass

    # --- Selección del mejor y guardado en carpeta común ---
    best_seed = max(resultados, key=resultados.get)
    src_dir = os.path.join(RUNS_DIR, f"{algo}_{version}_seed{best_seed}")
    dst_model = os.path.join(BEST_DIR, f"{algo}_{version}.zip")
    dst_vec = os.path.join(BEST_DIR, f"{algo}_{version}_vecnorm.pkl")

    shutil.copy(os.path.join(src_dir, "best_model.zip"), dst_model)
    src_vec = os.path.join(src_dir, "best_vecnormalize.pkl")
    if os.path.exists(src_vec):
        shutil.copy(src_vec, dst_vec)

    with open(os.path.join(BEST_DIR, f"{algo}_{version}_seeds.json"), "w") as f:
        json.dump({
            "algo": algo,
            "version": version,
            "best_seed": best_seed,
            "scores": {str(s): r for s, r in resultados.items()},
        }, f, indent=2)

    print("=" * 62)
    print(f"  MEJOR semilla: {best_seed}  (eval reward {resultados[best_seed]:.2f})")
    print(f"  Guardado:  {dst_model}")
    print(f"             {dst_vec}")
    print("=" * 62)
    return best_seed, resultados
