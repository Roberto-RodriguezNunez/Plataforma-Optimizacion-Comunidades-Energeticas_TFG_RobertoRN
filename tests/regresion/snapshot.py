"""snapshot.py — Generador del "golden output" de regresión.

Red de seguridad del refactor (rama refactor1): congela los outputs numéricos de
los caminos críticos del código de investigación RL para que CUALQUIER refactor
posterior pueda demostrarse NEUTRO (mismos números) en vez de confiar en que lo es.

Cada entrada del snapshot ejercita una porción real del pipeline:
  - obs_build      : obs_builder.build_obs (vector 108) — fuente única de obs.
  - sim_*          : ComunidadSimulador (física batería + economía + degradación).
  - energyenv_*    : EnergyEnv discreto (env + forecast AR(1) + obs).
  - residualenv_*  : ResidualEnv + MPC online (obs 112, coordinación AR(1)).
  - mpc_real_*     : evaluar_controlador(MPC, 'realista') sobre 2 semanas eval.
  - dqn_*          : evaluar_controlador(DQN best, 'realista') sobre 2 semanas eval.

Uso:
    # (re)generar el golden tras confirmar que HEAD está limpio:
    .venv/bin/python tests/regresion/snapshot.py
    # comparar contra el golden (lo hace el test):
    .venv/bin/pytest tests/regresion -q

DETERMINISMO: las evals usan semilla fija (SEED) para el ruido AR(1); los
rollouts de env fijan además np.random global (el SoC inicial de reset usa el
RNG global) y pasan un Generator semillado para el ruido.
"""
import os
import sys
from contextlib import contextmanager

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "golden.npz")

# Modelo DQN consolidado usado para el golden del controlador discreto.
_DQN_MODEL = os.path.join(ROOT, "models", "best", "dqn_DQN-1.zip")
_DQN_NORM = os.path.join(ROOT, "models", "best", "dqn_DQN-1_vecnorm.pkl")

# Nº de semanas eval a las que recortamos las evals "pesadas" (50 → 2) para que
# el test sea rápido sin dejar de recorrer EXACTAMENTE el mismo código.
_N_SEMANAS_GOLDEN = 2


@contextmanager
def _recortar_semanas_eval(n):
    """Recorta sim.semanas_eval a las primeras `n` durante el bloque.

    Parchea ComunidadSimulador.__init__ (la misma clase que instancia
    internamente evaluar_controlador) para no replicar su bucle aquí.
    """
    from src.core.simulator import ComunidadSimulador
    orig = ComunidadSimulador.__init__

    def parcheado(self, *a, **k):
        orig(self, *a, **k)
        self.semanas_eval = self.semanas_eval[:n]

    ComunidadSimulador.__init__ = parcheado
    try:
        yield
    finally:
        ComunidadSimulador.__init__ = orig


def _snap_obs_builder():
    from src.core.simulator import ComunidadSimulador
    from src.core.observation import build_obs
    from src.controllers.mpc import DATASET_PATH

    sim = ComunidadSimulador(DATASET_PATH)
    step = 100
    window = sim.get_data_window(step, horizon=24)
    obs = build_obs({"soc": 0.5, "step": step}, window, sim)
    return {"obs_build": np.asarray(obs, dtype=np.float64)}


def _snap_simulador():
    """Rollout determinista (sin RNG) de la física: acciones cíclicas 0..8."""
    from src.core.simulator import ComunidadSimulador
    from src.controllers.mpc import DATASET_PATH

    sim = ComunidadSimulador(DATASET_PATH)
    start = 672
    sim.soc = 0.5
    socs, marg, benef = [], [], []
    for i in range(60):
        r = sim.ejecutar_accion_fisica(i % 9, start + i)
        socs.append(r["soc"])
        marg.append(r["beneficio_marginal"])
        benef.append(r["beneficio"])
    return {
        "sim_soc": np.array(socs, dtype=np.float64),
        "sim_benef_marg": np.array(marg, dtype=np.float64),
        "sim_benef": np.array(benef, dtype=np.float64),
    }


def _snap_energy_env():
    from src.envs.energy_env import EnergyEnv

    np.random.seed(123)  # reset() sortea el SoC inicial con el RNG global
    env = EnergyEnv(forecast_noise=True, mode="eval", rng=np.random.default_rng(123))
    obs0, _ = env.reset(seed=123)
    rewards = []
    obs = obs0
    for i in range(30):
        obs, reward, term, trunc, _ = env.step(i % 9)
        rewards.append(reward)
        if term or trunc:
            break
    return {
        "energyenv_obs0": np.asarray(obs0, dtype=np.float64),
        "energyenv_obs_fin": np.asarray(obs, dtype=np.float64),
        "energyenv_reward": np.array(rewards, dtype=np.float64),
    }


def _snap_residual_env():
    from src.envs.energy_env import EnergyEnvContinuo
    from src.envs.residual_env import ResidualEnv
    from src.controllers.mpc import crear_mpc

    np.random.seed(7)
    inner = EnergyEnvContinuo(forecast_noise=True, mode="eval",
                              rng=np.random.default_rng(7))
    renv = ResidualEnv(inner, crear_mpc(), delta_max=0.30, residual_mode="mult")
    obs0, _ = renv.reset(seed=7)
    rewards = []
    obs = obs0
    deltas = [
        np.zeros(4, dtype=np.float32),
        np.array([0.2, -0.2, 0.1, -0.1], dtype=np.float32),
        np.array([-0.3, 0.0, 0.3, 0.0], dtype=np.float32),
    ]
    for i in range(20):
        obs, reward, term, trunc, _ = renv.step(deltas[i % len(deltas)])
        rewards.append(reward)
        if term or trunc:
            break
    return {
        "residualenv_obs0": np.asarray(obs0, dtype=np.float64),
        "residualenv_obs_fin": np.asarray(obs, dtype=np.float64),
        "residualenv_reward": np.array(rewards, dtype=np.float64),
    }


def _snap_mpc_eval():
    from src.evaluation.unified import evaluar_controlador, crear_mpc
    from src.controllers.mpc import SEED

    with _recortar_semanas_eval(_N_SEMANAS_GOLDEN):
        res = evaluar_controlador(crear_mpc(), forecast_mode="realista", seed=SEED)
    return {
        "mpc_real_marg": np.asarray(res["bens_marg"], dtype=np.float64),
        "mpc_real_abs": np.asarray(res["bens_abs"], dtype=np.float64),
    }


def _snap_dqn_eval():
    from src.evaluation.unified import evaluar_controlador, crear_discrete_rl
    from src.controllers.mpc import SEED

    if not os.path.exists(_DQN_MODEL):
        return {}  # sin modelo no se puede anclar este golden
    with _recortar_semanas_eval(_N_SEMANAS_GOLDEN):
        ctrl = crear_discrete_rl(_DQN_MODEL, _DQN_NORM, algo="DQN")
        res = evaluar_controlador(ctrl, forecast_mode="realista", seed=SEED)
    return {"dqn_real_marg": np.asarray(res["bens_marg"], dtype=np.float64)}


def compute_snapshot():
    """Devuelve un dict {nombre: np.ndarray float64} con todo el golden."""
    snap = {}
    snap.update(_snap_obs_builder())
    snap.update(_snap_simulador())
    snap.update(_snap_energy_env())
    snap.update(_snap_residual_env())
    snap.update(_snap_mpc_eval())
    snap.update(_snap_dqn_eval())
    return snap


def main():
    snap = compute_snapshot()
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    np.savez(GOLDEN_PATH, **snap)
    print(f"Golden guardado en {GOLDEN_PATH}")
    for k, v in snap.items():
        v = np.asarray(v)
        print(f"  {k:<22s} shape={v.shape}  [{v.flat[0]:.6g} ... {v.flat[-1]:.6g}]")


if __name__ == "__main__":
    main()
