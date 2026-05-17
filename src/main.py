"""
main.py -- Orquestador de entrenamiento del agente PPO (PPO_1)
============================================================
HU-12: Entrenar agente RL con Stable Baselines3
HU-13: Monitorizar el progreso del entrenamiento

Cambio de algoritmo: DQN -> PPO con Discrete(9)
  Motivacion: DQN lleva 12 iteraciones (DQN_1 a DQN_12) estancado en 34-39 EUR/sem.
  El MPC con ruido da 48.46. El cuello de botella es el credito tardio: DQN propaga
  valor paso a paso (1-step Bellman) y no consigue asignar credito a ciclos de
  carga/descarga de 8-48h en episodios de 168h.

  PPO con Discrete(9) resuelve esto porque GAE calcula retornos multi-step
  directamente. Mantiene las mismas 9 acciones bang-bang que DQN — sin clipping
  en limites fisicos, sin convergencia conservadora a potencias intermedias.

Sin cambios en entorno ni simulador:
  - energy_env.py: action_space = Discrete(9), misma reward marginal
  - simulador.py: misma fisica (bateria 100kWh, inversor 50kW, precios asimetricos)
  - Correccion inicial/terminal: se mantiene (propiedad de la reward, no del algoritmo)

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
#  HIPERPARAMETROS — PPO_1
#  Para un test rapido: TOTAL_TIMESTEPS = 10_000
# ------------------------------------------------------------------

TOTAL_TIMESTEPS   = 3_000_000
LEARNING_RATE     = 3e-4        # Default PPO (Adam)
N_STEPS           = 2048        # ~12 episodios por rollout (2048/168)
BATCH_SIZE        = 64          # Minibatches dentro de cada rollout
N_EPOCHS          = 10          # Pasadas por rollout (estandar PPO)
GAMMA             = 0.99        # GAE maneja horizonte largo; no necesita 0.995
GAE_LAMBDA        = 0.95        # Balance sesgo/varianza en estimador de ventaja
CLIP_RANGE        = 0.2         # Clipping PPO estandar
ENT_COEF          = 0.01        # Entropia para mantener exploracion
VF_COEF           = 0.5         # Peso del value loss (estandar)

# Redes actor/critic separadas: 101 -> 64 -> 64 -> 9 (actor) / 1 (critic)
NET_ARCH_PI       = [64, 64]
NET_ARCH_VF       = [64, 64]

EVAL_FREQ         = 20_000      # Cada 20k pasos (150 evals en 3.0M)
EVAL_EPISODES     = 50          # 50 episodios por eval

MODEL_DIR  = os.path.join(ROOT, "models")
LOG_DIR    = os.path.join(ROOT, "logs")
MODEL_NAME = "ppo_sgec"


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
    print("3. Configurando agente PPO...")
    print(f"   Actor:  101 -> {NET_ARCH_PI[0]} -> {NET_ARCH_PI[1]} -> 9")
    print(f"   Critic: 101 -> {NET_ARCH_VF[0]} -> {NET_ARCH_VF[1]} -> 1")
    print(f"   Total timesteps: {TOTAL_TIMESTEPS:,}")

    model = PPO(
        policy        = "MlpPolicy",
        env           = env,
        learning_rate = LEARNING_RATE,
        n_steps       = N_STEPS,
        batch_size    = BATCH_SIZE,
        n_epochs      = N_EPOCHS,
        gamma         = GAMMA,
        gae_lambda    = GAE_LAMBDA,
        clip_range    = CLIP_RANGE,
        ent_coef      = ENT_COEF,
        vf_coef       = VF_COEF,
        policy_kwargs = {"net_arch": {"pi": NET_ARCH_PI, "vf": NET_ARCH_VF}},
        verbose       = 1,
        tensorboard_log = LOG_DIR,
    )
    print()

    # ── 4. Callbacks ──────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = MODEL_DIR,
        log_path             = LOG_DIR,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = EVAL_EPISODES,
        deterministic        = True,
        verbose              = 1,
    )
    metricas_callback = MetricasCallback()

    # ── 5. Entrenar ───────────────────────────────────────────────
    print("-" * 60)
    print(f"4. Iniciando entrenamiento PPO ({TOTAL_TIMESTEPS:,} pasos)...")
    print(f"   TensorBoard: tensorboard --logdir {LOG_DIR}")
    print("-" * 60)

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [eval_callback, metricas_callback],
        progress_bar    = False,
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
