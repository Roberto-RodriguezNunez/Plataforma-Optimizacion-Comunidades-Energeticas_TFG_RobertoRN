"""
main_ppo.py -- Entrenamiento del agente PPO (configuracion PPO_4)
=================================================================
Portado desde feature/ppo-discreto para tener todos los
algoritmos entrenables desde la rama principal.

Hiperparametros (PPO_4 — mejor resultado historico: +45.69 EUR/sem pico):
  - LR decay: 3e-4 -> 3e-5 (lineal)
  - ENT_COEF decay: 0.005 -> 0.0005 (lineal)
  - N_STEPS: 8192 (~48 episodios/rollout)
  - 4M steps totales

Ejecucion desde la raiz del proyecto (carpeta TFG/):
    python src/main_ppo.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.energy_env import EnergyEnv
from src.training.multiseed import entrenar_multiseed


# ------------------------------------------------------------------
#  HIPERPARAMETROS — PPO_4
# ------------------------------------------------------------------

TOTAL_TIMESTEPS   = 4_000_000
LR_INIT           = 3e-4
LR_FINAL          = 3e-5
N_STEPS           = 8192
BATCH_SIZE        = 128
N_EPOCHS          = 5
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
CLIP_RANGE        = 0.2
ENT_COEF_INIT     = 0.005
ENT_COEF_FINAL    = 0.0005
VF_COEF           = 0.5

NET_ARCH_PI       = [64, 64]
NET_ARCH_VF       = [64, 64]

EVAL_FREQ         = 20_000
EVAL_EPISODES     = 50
EVAL_SEED         = 42

MODEL_DIR  = os.path.join(ROOT, "models")
LOG_DIR    = os.path.join(ROOT, "logs")
MODEL_NAME = "ppo_sgec"


# ------------------------------------------------------------------
#  ENT COEF SCHEDULER
# ------------------------------------------------------------------

class EntCoefScheduler(BaseCallback):
    """Decae model.ent_coef linealmente de ENT_COEF_INIT a ENT_COEF_FINAL."""
    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / TOTAL_TIMESTEPS)
        self.model.ent_coef = float(
            ENT_COEF_INIT + frac * (ENT_COEF_FINAL - ENT_COEF_INIT)
        )
        return True


# ------------------------------------------------------------------
#  SEEDED EVAL CALLBACK
# ------------------------------------------------------------------

class SeededEvalCallback(EvalCallback):
    """Re-seedea np.random antes de cada eval para reproducibilidad."""
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            np.random.seed(EVAL_SEED)
        return super()._on_step()


# ------------------------------------------------------------------
#  CALLBACK — metricas adicionales en TensorBoard
# ------------------------------------------------------------------

class MetricasCallback(BaseCallback):
    """Registra SoC, economia y distribucion de acciones en TensorBoard."""

    _GRUPOS = {
        "IDLE":           [0],
        "CARGAR_SOLAR":   [1],
        "CARGAR_MIXTA":   [2, 3, 4],
        "DESCARGAR_CASA": [5],
        "DESCARGAR_RED":  [6, 7, 8],
    }

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._soc_buffer        = []
        self._comprado_buffer   = []
        self._cargado_buffer    = []
        self._descargado_buffer = []
        self._acciones          = np.zeros(9, dtype=np.int64)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "soc" in info:
                self._soc_buffer.append(info["soc"])
            if "comprado" in info:
                self._comprado_buffer.append(info["comprado"])
            if "cargado" in info:
                self._cargado_buffer.append(info["cargado"])
            if "descargado" in info:
                self._descargado_buffer.append(info["descargado"])

        accion = self.locals.get("actions")
        if accion is not None:
            self._acciones[int(accion[0])] += 1

        if self.num_timesteps % 1_000 == 0 and self._soc_buffer:
            self.logger.record("custom/soc_medio",      np.mean(self._soc_buffer))
            self.logger.record("custom/soc_minimo",     np.min(self._soc_buffer))
            self.logger.record("custom/soc_maximo",     np.max(self._soc_buffer))
            self.logger.record("custom/cargado_medio",
                               np.mean(self._cargado_buffer) if self._cargado_buffer else 0.0)
            self.logger.record("custom/descargado_medio",
                               np.mean(self._descargado_buffer) if self._descargado_buffer else 0.0)
            self.logger.record("custom/comprado_medio", np.mean(self._comprado_buffer))

            total = self._acciones.sum()
            if total > 0:
                for grupo, indices in self._GRUPOS.items():
                    pct = 100.0 * self._acciones[indices].sum() / total
                    self.logger.record(f"acciones/{grupo}_pct", pct)
                for i in range(9):
                    self.logger.record(f"acciones/accion_{i:02d}_pct",
                                       100.0 * self._acciones[i] / total)
                self.logger.record("acciones/dominante_idx", int(np.argmax(self._acciones)))
                self.logger.record("acciones/dominante_pct",
                                   100.0 * self._acciones.max() / total)

            if self.num_timesteps % 20_000 == 0 and total > 0:
                grupos_str = "  ".join(
                    f"{g}={100.0 * self._acciones[idxs].sum() / total:.0f}%"
                    for g, idxs in self._GRUPOS.items()
                )
                print(
                    f"[{self.num_timesteps:>9,}]  "
                    f"SoC={np.mean(self._soc_buffer):.2f}"
                    f"[{np.min(self._soc_buffer):.2f}-{np.max(self._soc_buffer):.2f}]"
                    f"  {grupos_str}",
                    flush=True,
                )

            self._soc_buffer.clear()
            self._comprado_buffer.clear()
            self._cargado_buffer.clear()
            self._descargado_buffer.clear()
            self._acciones[:] = 0

        return True


# ------------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------------

def make_envs(seed):
    """Factoría de entornos PPO (discreto) envueltos en VecNormalize."""
    def _mk():
        return Monitor(EnergyEnv())
    train_env = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=False, clip_obs=10.0)
    return train_env, eval_env


def make_model(train_env, seed):
    """Factoría del modelo PPO (con semilla y decay de LR aplicados)."""
    return PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = lambda p: LR_FINAL + p * (LR_INIT - LR_FINAL),
        n_steps       = N_STEPS,
        batch_size    = BATCH_SIZE,
        n_epochs      = N_EPOCHS,
        gamma         = GAMMA,
        gae_lambda    = GAE_LAMBDA,
        clip_range    = CLIP_RANGE,
        ent_coef      = ENT_COEF_INIT,
        vf_coef       = VF_COEF,
        policy_kwargs = {"net_arch": {"pi": NET_ARCH_PI, "vf": NET_ARCH_VF}},
        verbose       = 0,
        seed          = seed,
        device        = "cpu",
        tensorboard_log = LOG_DIR,
    )


def main(version="v1", seeds=(42, 1337, 2024)):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    check_env(EnergyEnv(), warn=True)
    entrenar_multiseed(
        algo="ppo", version=version, seeds=list(seeds),
        make_envs=make_envs, make_model=make_model,
        total_timesteps=TOTAL_TIMESTEPS,
        eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES,
        extra_callbacks=lambda m: [MetricasCallback(), EntCoefScheduler()],
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    a = p.parse_args()
    main(version=a.version, seeds=a.seeds)
