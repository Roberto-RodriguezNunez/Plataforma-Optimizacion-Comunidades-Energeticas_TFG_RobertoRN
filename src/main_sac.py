"""
main_sac.py — Entrenamiento del agente Residual SAC sobre MPC
=============================================================
Fase 2 del TFG: el SAC aprende correcciones Delta_a sobre la accion
del MPC realista resuelto online en cada step.

Protocolo DAWN (Data-Anchored Warmup):
  1. Pre-llenar el replay buffer con 50k transiciones MPC puro (delta=0).
  2. Entrenar SAC durante 1M pasos con exploracion sobre delta.
  3. El critic aprende primero que delta=0 da reward MPC, luego explora.

Ejecucion desde la raiz del proyecto (carpeta TFG/):
    python src/main_sac.py [--seed 42] [--timesteps 1000000]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.benchmarks.mpc_benchmark import LinearMPC, ComunidadSimulador, DATASET_PATH
from src.envs.energy_env_continuo import EnergyEnvContinuo
from src.envs.residual_env import ResidualEnv

# --- Cargar configuracion ---
_CONFIG_PATH = os.path.join(ROOT, 'config', 'system.yaml')
with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
    _CFG = yaml.safe_load(f)

_SAC_CFG = _CFG['residual_sac']
_MPC_CFG = _CFG['mpc']

EPISODE_LENGTH = _MPC_CFG['duracion_episodio']
MODEL_DIR = os.path.join(ROOT, 'models')
LOG_DIR = os.path.join(ROOT, 'logs')


# ──────────────────────────────────────────────────────────────────
#  SEEDED EVAL CALLBACK — mismas 50 semanas en cada evaluacion
# ──────────────────────────────────────────────────────────────────

class SeededEvalCallback(EvalCallback):
    """
    Re-seedea np.random antes de cada ronda de evaluacion para garantizar
    que las 50 semanas eval y el ruido AR(1) son identicos entre evals.
    """
    def __init__(self, *args, eval_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_seed = eval_seed

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            np.random.seed(self._eval_seed)
        return super()._on_step()


# ──────────────────────────────────────────────────────────────────
#  METRICAS CALLBACK — SoC, delta, a_mpc en TensorBoard
# ──────────────────────────────────────────────────────────────────

class ResidualMetricasCallback(BaseCallback):
    """Registra metricas especificas del Residual SAC en TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._soc_buf = []
        self._delta_buf = []
        self._a_mpc_buf = []
        self._gc_freq = 10_000  # gc.collect() cada 10k steps

    def _on_step(self) -> bool:
        # Forzar garbage collection periódico (evita memory leak de scipy/HiGHS)
        if self.num_timesteps % self._gc_freq == 0:
            import gc
            gc.collect()

        for info in self.locals.get('infos', []):
            if 'soc' in info:
                self._soc_buf.append(info['soc'])
            if 'delta_applied' in info:
                self._delta_buf.append(info['delta_applied'])  # list of 4
            if 'a_mpc' in info:
                self._a_mpc_buf.append(info['a_mpc'])  # list of 4

        if self.num_timesteps % 1000 == 0 and self._soc_buf:
            self.logger.record('custom/soc_medio', np.mean(self._soc_buf))
            self.logger.record('custom/soc_min', np.min(self._soc_buf))
            self.logger.record('custom/soc_max', np.max(self._soc_buf))

            if self._delta_buf:
                deltas = np.array(self._delta_buf)  # (N, 4) kW
                self.logger.record('custom/delta_l1_medio',
                                   np.mean(np.abs(deltas)))
                self.logger.record('custom/delta_std', np.std(deltas))

            if self.num_timesteps % 20_000 == 0:
                delta_l1 = np.mean(np.abs(np.array(self._delta_buf))) if self._delta_buf else 0
                print(
                    f"[{self.num_timesteps:>9,}]  "
                    f"SoC={np.mean(self._soc_buf):.2f}"
                    f"[{np.min(self._soc_buf):.2f}-{np.max(self._soc_buf):.2f}]"
                    f"  |delta|_L1={delta_l1:.4f} kW",
                    flush=True,
                )

            self._soc_buf.clear()
            self._delta_buf.clear()
            self._a_mpc_buf.clear()

        return True


# ──────────────────────────────────────────────────────────────────
#  DAWN WARMUP — pre-llenado del buffer con transiciones MPC puro
# ──────────────────────────────────────────────────────────────────

def dawn_warmup(model, mpc, train_env, warmup_steps, seed=42):
    """
    Pre-llena el replay buffer de SAC con transiciones donde delta=0
    (accion pura del MPC) Y calibra VecNormalize con las observaciones.

    Esto ancla el critic al rendimiento MPC antes de que SAC empiece
    a explorar deltas. Ademas, las running statistics de VecNormalize
    (obs_rms) se inicializan con datos reales en lugar de empezar
    desde cero.

    Args:
        model: Instancia de SAC (su replay_buffer se modifica in-place).
        mpc: LinearMPC para el wrapper.
        train_env: VecNormalize wrapping del entorno de entrenamiento.
        warmup_steps: Numero de transiciones a generar.
        seed: Semilla para reproducibilidad.
    """
    print(f"\n  DAWN warmup: generando {warmup_steps:,} transiciones MPC puro...")
    np.random.seed(seed)

    # Crear un ResidualEnv temporal para generar transiciones
    inner = EnergyEnvContinuo(forecast_noise=True, mode='train')
    renv = ResidualEnv(inner, mpc, delta_max=_SAC_CFG['delta_max'])

    obs, _ = renv.reset(seed=seed)
    n_episodes = 0
    obs_buffer = []

    for i in range(warmup_steps):
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # delta 4D = 0
        next_obs, reward, terminated, truncated, info = renv.step(action)

        # SB3 replay buffer: add(obs, next_obs, action, reward, done, infos)
        model.replay_buffer.add(
            obs.reshape(1, -1),
            next_obs.reshape(1, -1),
            action.reshape(1, -1),
            np.array([reward]),
            np.array([terminated]),
            [info],
        )

        obs_buffer.append(obs)

        if terminated or truncated:
            obs, _ = renv.reset()
            n_episodes += 1
        else:
            obs = next_obs

    print(f"    {warmup_steps:,} transiciones, {n_episodes} episodios completos.")
    print(f"    Buffer size: {model.replay_buffer.size()}")

    # Calibrar VecNormalize con las observaciones del warmup
    print("    Calibrando VecNormalize con obs del warmup...")
    obs_all = np.array(obs_buffer, dtype=np.float32)
    train_env.obs_rms.mean = obs_all.mean(axis=0)
    train_env.obs_rms.var = obs_all.var(axis=0)
    train_env.obs_rms.count = len(obs_all)
    print(f"    obs_rms calibrado con {len(obs_all):,} observaciones.")


# ──────────────────────────────────────────────────────────────────
#  FUNCIONES DE CREACION DE ENTORNOS
# ──────────────────────────────────────────────────────────────────

def _make_residual_env(mpc, mode='train'):
    """Crea un ResidualEnv envuelto en Monitor."""
    inner = EnergyEnvContinuo(forecast_noise=True, mode=mode)
    renv = ResidualEnv(inner, mpc, delta_max=_SAC_CFG['delta_max'])
    return Monitor(renv)


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

def main(seed=42, total_timesteps=None, tag=None):
    if total_timesteps is None:
        total_timesteps = _SAC_CFG['total_timesteps']

    # Directorios con tag opcional para no sobreescribir otros runs
    model_dir = MODEL_DIR
    log_dir = LOG_DIR
    if tag:
        model_dir = os.path.join(MODEL_DIR, tag)
        log_dir = os.path.join(LOG_DIR, tag)

    print("=" * 62)
    print("RESIDUAL SAC — Entrenamiento")
    print("=" * 62)
    print(f"  Seed:            {seed}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Delta max:       {_SAC_CFG['delta_max']}")
    print(f"  DAWN warmup:     {_SAC_CFG['dawn_warmup_steps']:,}")
    print(f"  Buffer size:     {_SAC_CFG['buffer_size']:,}")
    print(f"  Net arch:        {_SAC_CFG['net_arch']}")
    if tag:
        print(f"  Tag:             {tag}")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Crear MPC (compartido por todos los entornos)
    sim_mpc = ComunidadSimulador(DATASET_PATH)
    tv_cfg = _MPC_CFG['valor_terminal']
    mpc = LinearMPC(
        sim_mpc,
        use_terminal_value=tv_cfg['activado'],
        terminal_lambda=tv_cfg['lambda'],
        terminal_price_mode=tv_cfg['modo_precio'],
        k_deg_lin=_MPC_CFG['k_deg_lin'],
    )
    print(f"  MPC:             {mpc.nombre().encode('ascii', 'replace').decode()}")

    # 2. Crear entornos
    print("\n  Creando entornos...")
    train_env = VecNormalize(
        DummyVecEnv([lambda: _make_residual_env(mpc, mode='train')]),
        norm_obs=_SAC_CFG['norm_obs'],
        norm_reward=_SAC_CFG['norm_reward'],
        clip_obs=_SAC_CFG['clip_obs'],
    )

    eval_env = VecNormalize(
        DummyVecEnv([lambda: _make_residual_env(mpc, mode='eval')]),
        norm_obs=_SAC_CFG['norm_obs'],
        norm_reward=_SAC_CFG['norm_reward'],
        clip_obs=_SAC_CFG['clip_obs'],
    )

    # 3. Crear agente SAC
    print("  Creando agente SAC...")
    model = SAC(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=_SAC_CFG['learning_rate'],
        buffer_size=_SAC_CFG['buffer_size'],
        batch_size=_SAC_CFG['batch_size'],
        gamma=_SAC_CFG['gamma'],
        tau=_SAC_CFG['tau'],
        ent_coef=_SAC_CFG['ent_coef'],
        target_entropy=_SAC_CFG['target_entropy'],
        learning_starts=0,  # Empezamos a aprender inmediatamente tras warmup
        policy_kwargs={'net_arch': _SAC_CFG['net_arch']},
        verbose=0,
        seed=seed,
        device='cpu',
        tensorboard_log=log_dir,
    )

    # 4. DAWN warmup (llena buffer + calibra VecNormalize)
    dawn_warmup(
        model, mpc, train_env,
        warmup_steps=_SAC_CFG['dawn_warmup_steps'],
        seed=seed,
    )

    import gc; gc.collect()  # libera objetos Python del warmup antes de entrenar

    # 5. Callbacks
    eval_callback = SeededEvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=_SAC_CFG['eval_freq'],
        n_eval_episodes=_SAC_CFG['eval_episodes'],
        deterministic=True,
        verbose=1,
        eval_seed=42,
    )
    metricas_callback = ResidualMetricasCallback()

    # 6. Entrenar
    print(f"\n  Iniciando entrenamiento ({total_timesteps:,} pasos)...")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")
    print("-" * 62)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, metricas_callback],
        progress_bar=False,
        tb_log_name=f"ResidualSAC_seed{seed}",
    )

    # 7. Guardar
    model_path = os.path.join(model_dir, f"residual_sac_seed{seed}")
    model.save(model_path)
    train_env.save(os.path.join(model_dir, f"residual_sac_vec_normalize_seed{seed}.pkl"))

    print(f"\n  Modelo guardado:     {model_path}.zip")
    print(f"  Mejor modelo:        {os.path.join(model_dir, 'best_model.zip')}")
    print(f"  VecNormalize stats:  residual_sac_vec_normalize_seed{seed}.pkl")
    print("=" * 62)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrena Residual SAC sobre MPC.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla (default: 42)')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total timesteps (default: config)')
    parser.add_argument('--delta-max', type=float, default=None,
                        help='Delta max override (default: config)')
    parser.add_argument('--ent-coef', type=float, default=None,
                        help='Entropy coef override (default: config)')
    parser.add_argument('--warmup', type=int, default=None,
                        help='DAWN warmup steps override (default: config)')
    parser.add_argument('--tag', type=str, default=None,
                        help='Tag para separar modelos/logs (ej: v5b)')
    args = parser.parse_args()
    if args.delta_max is not None:
        _SAC_CFG['delta_max'] = args.delta_max
    if args.ent_coef is not None:
        _SAC_CFG['ent_coef'] = args.ent_coef
    if args.warmup is not None:
        _SAC_CFG['dawn_warmup_steps'] = args.warmup
    main(seed=args.seed, total_timesteps=args.timesteps, tag=args.tag)
