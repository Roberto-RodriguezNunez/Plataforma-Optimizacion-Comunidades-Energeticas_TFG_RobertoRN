"""
main.py -- Orquestador de entrenamiento del agente PPO (PPO_4)
============================================================
HU-12: Entrenar agente RL con Stable Baselines3
HU-13: Monitorizar el progreso del entrenamiento

Historial:
  PPO_1: pico 39.91 EUR/sem (320k), luego colapso. ENT_COEF=0.01 demasiado alto.
  PPO_2: pico 41.82 EUR/sem (960k), luego colapso. ENT_COEF=0.003 mejoro pero
         el patron de colapso persiste — problema estructural de PPO on-policy.
  PPO_3: pico 45.69 EUR/sem (1.48M), estabilizacion ~44 EUR/sem. Sin colapso
         visible gracias al decay de ENT_COEF y rollouts largos (N_STEPS=8192).
         Plateau leve a partir de 1.5M — la politica convergió antes de los 3M.

PPO_4 = PPO_3 + LR decay + 4M pasos

  1. Learning rate decay: 3e-4 -> 3e-5 (lineal, via callable SB3)
       progress_remaining va de 1.0 (inicio) a 0.0 (fin).
       LR = LR_FINAL + progress_remaining * (LR_INIT - LR_FINAL)
       Razon: LR alto al inicio para explorar el espacio de politicas;
       LR bajo al final para consolidar sin destruir la politica aprendida.
       Complementa el decay de ENT_COEF ya presente en PPO_3.

  2. Total timesteps: 3M -> 4M
       PPO_3 mostro plateau a partir de 1.5M pero sin colapso posterior.
       1M extra con LR bajo puede permitir consolidacion adicional.
       EVAL_FREQ = 20k → 200 evaluaciones en 4M (misma granularidad).

Resto sin cambios (108 dims, ENT_COEF decay 0.005→0.0005, N_STEPS=8192,
coste oportunidad dinamico en simulador, margen_solar feature).

Ejecucion desde la raiz del proyecto (carpeta TFG/):
    python src/main.py

Para ver las curvas en TensorBoard (mientras entrena o despues):
    tensorboard --logdir logs/
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


# ------------------------------------------------------------------
#  HIPERPARAMETROS — PPO_4
#  Para un test rapido: TOTAL_TIMESTEPS = 10_000
# ------------------------------------------------------------------

TOTAL_TIMESTEPS   = 4_000_000
LR_INIT           = 3e-4        # LR al inicio del entrenamiento
LR_FINAL          = 3e-5        # LR al final del entrenamiento (decay lineal)
N_STEPS           = 8192        # ~48 episodios/rollout (4096 -> 8192)
BATCH_SIZE        = 128         # 64 -> 128 (proporcional al rollout mayor)
N_EPOCHS          = 5           # 10 -> 5 (menos sobreajuste por rollout)
GAMMA             = 0.99        # Sin cambio
GAE_LAMBDA        = 0.95        # Sin cambio
CLIP_RANGE        = 0.2         # Sin cambio
ENT_COEF_INIT     = 0.005       # Entropia inicial (mas alta que PPO_2=0.003)
ENT_COEF_FINAL    = 0.0005      # Entropia final (muy baja — explotacion estable)
VF_COEF           = 0.5         # Sin cambio

# Redes actor/critic separadas: 108 -> 64 -> 64 -> 9 (actor) / 1 (critic)
NET_ARCH_PI       = [64, 64]
NET_ARCH_VF       = [64, 64]

EVAL_FREQ         = 20_000      # Cada 20k pasos (200 evals en 4M)
EVAL_EPISODES     = 50          # 50 episodios por eval
EVAL_SEED         = 42          # Semilla fija — siempre las mismas 50 semanas

MODEL_DIR  = os.path.join(ROOT, "models")
LOG_DIR    = os.path.join(ROOT, "logs")
MODEL_NAME = "ppo_sgec"


# ──────────────────────────────────────────────────────────────────
#  ENT COEF SCHEDULER — decae linealmente ENT_COEF_INIT → ENT_COEF_FINAL
# ──────────────────────────────────────────────────────────────────

class EntCoefScheduler(BaseCallback):
    """
    Decae model.ent_coef linealmente de ENT_COEF_INIT a ENT_COEF_FINAL
    a lo largo de TOTAL_TIMESTEPS pasos.

    Razon: ENT_COEF constante alto (PPO_1=0.01) destruye la politica.
    ENT_COEF constante bajo (PPO_2=0.003) permite consolidarla pero
    sin exploracion suficiente al inicio. El decay combina ambas ventajas:
    exploracion al principio + explotacion estable al final.
    """
    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / TOTAL_TIMESTEPS)
        self.model.ent_coef = float(
            ENT_COEF_INIT + frac * (ENT_COEF_FINAL - ENT_COEF_INIT)
        )
        return True


# ──────────────────────────────────────────────────────────────────
#  SEEDED EVAL CALLBACK — mismas 50 semanas en cada evaluación
# ──────────────────────────────────────────────────────────────────

class SeededEvalCallback(EvalCallback):
    """
    EvalCallback que re-seedea np.random antes de cada ronda de evaluación.

    Problema que resuelve:
      EnergyEnv usa np.random.randint/uniform/normal (rng global de numpy).
      Sin este callback las semanas evaluadas cambian entre evals, añadiendo
      ±5 EUR/sem de varianza que enmascara el progreso real del agente.

    Con este callback, cada ronda de evaluación empieza siempre con el mismo
    estado del rng global → las 50 semanas son siempre las mismas → la curva
    eval/mean_reward refleja convergencia real, no varianza de datos.
    """
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            np.random.seed(EVAL_SEED)
        return super()._on_step()


# ──────────────────────────────────────────────────────────────────
#  CALLBACK PERSONALIZADO — métricas adicionales en TensorBoard
# ──────────────────────────────────────────────────────────────────

class MetricasCallback(BaseCallback):
    """
    Registra en TensorBoard métricas que SB3 no loguea por defecto:

    Física / batería:
    - SoC medio, mínimo y máximo del intervalo
    - Energía media cargada y descargada por paso

    Economía:
    - Energía media comprada a red

    Distribución de acciones (cada 1k pasos):
    - % IDLE, % CARGAR_SOLAR, % CARGAR_MIXTA, % DESCARGAR_CASA, % DESCARGAR_RED
    - % de cada acción individual (0-8) para máximo detalle
    """

    # Agrupación de las 9 acciones por estrategia
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
            # ── Física / batería ──────────────────────────────────
            self.logger.record("custom/soc_medio",      np.mean(self._soc_buffer))
            self.logger.record("custom/soc_minimo",     np.min(self._soc_buffer))
            self.logger.record("custom/soc_maximo",     np.max(self._soc_buffer))
            self.logger.record("custom/cargado_medio",  np.mean(self._cargado_buffer)
                               if self._cargado_buffer else 0.0)
            self.logger.record("custom/descargado_medio", np.mean(self._descargado_buffer)
                               if self._descargado_buffer else 0.0)

            # ── Economía ──────────────────────────────────────────
            self.logger.record("custom/comprado_medio", np.mean(self._comprado_buffer))

            # ── Distribución de acciones por grupo ────────────────
            total = self._acciones.sum()
            if total > 0:
                for grupo, indices in self._GRUPOS.items():
                    pct = 100.0 * self._acciones[indices].sum() / total
                    self.logger.record(f"acciones/{grupo}_pct", pct)

                # Detalle por acción individual (0-8)
                for i in range(9):
                    pct_i = 100.0 * self._acciones[i] / total
                    self.logger.record(f"acciones/accion_{i:02d}_pct", pct_i)

                # Acción dominante (diagnóstico rápido)
                self.logger.record("acciones/dominante_idx",
                                   int(np.argmax(self._acciones)))
                self.logger.record("acciones/dominante_pct",
                                   100.0 * self._acciones.max() / total)

            # ── Print a stdout cada 20k pasos (visible en Kaggle) ───
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

            # Limpiar buffers (incluido conteo de acciones → % del intervalo, no acumulado)
            self._soc_buffer.clear()
            self._comprado_buffer.clear()
            self._cargado_buffer.clear()
            self._descargado_buffer.clear()
            self._acciones[:] = 0

        return True


# ──────────────────────────────────────────────────────────────────
#  FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────

def make_env():
    """Crea una instancia de EnergyEnv envuelta en DummyVecEnv."""
    return DummyVecEnv([lambda: Monitor(EnergyEnv())])


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    # ── 1. Verificar entorno ──────────────────────────────────────
    print("-" * 60)
    print("1. Verificando entorno con env_checker...")
    check_env(EnergyEnv(), warn=True)
    print("   Entorno OK.\n")

    # ── 2. Crear entornos con VecNormalize ────────────────────────
    #  Solo normaliza observaciones (norm_obs=True).
    #  norm_reward=False: la recompensa NO se normaliza.
    print("2. Instanciando entornos con VecNormalize (solo observaciones)...")
    env = VecNormalize(
        make_env(),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    eval_env = VecNormalize(
        make_env(),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    print("   Entornos listos.\n")

    # ── 3. Configurar agente PPO ──────────────────────────────────
    print("3. Configurando agente PPO (PPO_4 — 108 dims, LR decay, ENT_COEF decay, N_STEPS=8192)...")
    print(f"   Actor:  108 -> {NET_ARCH_PI[0]} -> {NET_ARCH_PI[1]} -> 9")
    print(f"   Critic: 108 -> {NET_ARCH_VF[0]} -> {NET_ARCH_VF[1]} -> 1")
    print(f"   LR: {LR_INIT} -> {LR_FINAL} (decay lineal)")
    print(f"   ENT_COEF: {ENT_COEF_INIT} -> {ENT_COEF_FINAL} (decay lineal)")
    print(f"   N_STEPS={N_STEPS}  BATCH={BATCH_SIZE}  EPOCHS={N_EPOCHS}")
    print(f"   Total timesteps: {TOTAL_TIMESTEPS:,}")

    model = PPO(
        policy        = "MlpPolicy",
        env           = env,
        learning_rate = lambda p: LR_FINAL + p * (LR_INIT - LR_FINAL),
        n_steps       = N_STEPS,
        batch_size    = BATCH_SIZE,
        n_epochs      = N_EPOCHS,
        gamma         = GAMMA,
        gae_lambda    = GAE_LAMBDA,
        clip_range    = CLIP_RANGE,
        ent_coef      = ENT_COEF_INIT,   # EntCoefScheduler lo decae durante training
        vf_coef       = VF_COEF,
        policy_kwargs = {"net_arch": {"pi": NET_ARCH_PI, "vf": NET_ARCH_VF}},
        verbose       = 0,
        device        = "cpu",
        tensorboard_log = LOG_DIR,
    )
    print()

    # ── 4. Callbacks ──────────────────────────────────────────────
    eval_callback = SeededEvalCallback(
        eval_env,
        best_model_save_path = MODEL_DIR,
        log_path             = LOG_DIR,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = EVAL_EPISODES,
        deterministic        = True,
        verbose              = 1,
    )
    metricas_callback = MetricasCallback()
    ent_scheduler     = EntCoefScheduler()

    # ── 5. Entrenar ───────────────────────────────────────────────
    print("-" * 60)
    print(f"4. Iniciando entrenamiento PPO_4 ({TOTAL_TIMESTEPS:,} pasos)...")
    print(f"   TensorBoard: tensorboard --logdir {LOG_DIR}")
    print("-" * 60)

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [eval_callback, metricas_callback, ent_scheduler],
        progress_bar    = False,
        tb_log_name     = "PPO_4",
    )

    # ── 6. Guardar modelo y estadísticas de normalización ─────────
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save(model_path)
    env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))

    print(f"\nModelo final guardado en: {model_path}.zip")
    print(f"Mejor modelo guardado en: {os.path.join(MODEL_DIR, 'best_model.zip')}")
    print(f"Estadisticas VecNormalize: {os.path.join(MODEL_DIR, 'vec_normalize.pkl')}")


if __name__ == "__main__":
    main()
