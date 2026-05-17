"""
main.py -- Orquestador de entrenamiento del agente DQN (v10 — DQN_12)
============================================================
HU-12: Entrenar agente DQN con Stable Baselines3
HU-13: Monitorizar el progreso del entrenamiento

Cambios respecto a v9 (DQN_12):
  - GAMMA: 0.99 -> 0.995 (causa raiz: credito tardio en ciclos carga/descarga 24-48h)
      Con gamma=0.99 un ciclo 48h vale 61.7%; con 0.995 vale 78.6% (+27% de incentivo)
      El terminal bonus sube de 18.7% a 43.2% -> sesgo contra mantener bateria se reduce
  - TOTAL_TIMESTEPS: 2.0M -> 3.0M (gamma mas alto necesita mas explotacion para converger)
  - EXPLORATION_FRAC: 0.6 -> 0.5 (exploracion acaba en 1.5M, explotacion=1.5M vs 800k antes)
  - EVAL_FREQ: sin cambio (20k -> 150 evals en 3M, mas visibilidad de la curva)
  - EVAL_EPISODES: 20 -> 50 (std del estimador se reduce ~40%; mejor trazabilidad del progreso)

Sin cambios en arquitectura (heredados de v9):
  - 9 acciones (de 13): eliminados niveles redundantes en CARGAR_SOLAR y DESCARGAR_CASA
  - Ruido AR(1) en pronostico solar y consumo (precios sin ruido, publicados por REE)
  - SoC inicial aleatorio (SOC_MIN+5% a SOC_MAX-5%), simetria inicial/terminal de valor
  - MetricasCallback: distribucion completa de acciones por grupo e individual
  - Red 64x64 (101 -> 64 -> 64 -> 9)

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
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.energy_env import EnergyEnv


# ------------------------------------------------------------------
#  HIPERPARAMETROS (v5)
#  Para un test rapido: TOTAL_TIMESTEPS = 10_000, LEARNING_STARTS = 2_000
# ------------------------------------------------------------------

TOTAL_TIMESTEPS   = 3_000_000   # 3.0M — gamma=0.995 necesita mas explotacion para converger
LEARNING_RATE     = 1e-4        # Validado: mejor que 5e-5 (v2 fue peor)
BUFFER_SIZE       = 200_000     # Subido: 2.6x mas datos -> mas diversidad en buffer
LEARNING_STARTS   = 10_000      # Sin cambio — suficiente exploracion inicial
BATCH_SIZE        = 64          # Validado: mejor que 128 (mas actualizaciones)
GAMMA             = 0.995       # Subido: credito 48h sube 61.7%->78.6%; terminal 18.7%->43.2%
EXPLORATION_FRAC  = 0.5         # Exploracion acaba en 1.5M pasos; explotacion = 1.5M
EXPLORATION_FINAL = 0.05        # Estandar DQN
TARGET_UPDATE     = 1_000       # Sin cambio
TRAIN_FREQ        = 4           # Sin cambio

# Red neuronal: 101 -> 64 -> 64 -> 9
# (101+1)*64 + (64+1)*64 + (64+1)*9 = 6528+4160+585 = 11273 params
NET_ARCH          = [64, 64]

EVAL_FREQ         = 20_000      # Cada 20k pasos (150 evals en 3.0M — mas visibilidad)
EVAL_EPISODES     = 50          # 50 episodios por eval — std estimador -40% vs 20 eps

MODEL_DIR  = os.path.join(ROOT, "models")
LOG_DIR    = os.path.join(ROOT, "logs")
MODEL_NAME = "dqn_sgec"


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
    - % de cada acción individual (0-12) para máximo detalle

    Exploración:
    - epsilon actual del agente
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

            # ── Exploración ───────────────────────────────────────
            try:
                eps = self.model.exploration_rate
                self.logger.record("custom/epsilon", eps)
            except AttributeError:
                pass

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
    #  Razon: normalizar la recompensa distorsiona la señal de aprendizaje
    #  y dificulta comparar evaluaciones entre entrenamientos.
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

    # ── 3. Configurar agente DQN ──────────────────────────────────
    print("3. Configurando agente DQN...")
    print(f"   Red neuronal: 101 -> {NET_ARCH[0]} -> {NET_ARCH[1]} -> 9  (~11273 params)")
    print(f"   Total timesteps: {TOTAL_TIMESTEPS:,}")

    model = DQN(
        policy                 = "MlpPolicy",
        env                    = env,
        learning_rate          = LEARNING_RATE,
        buffer_size            = BUFFER_SIZE,
        learning_starts        = LEARNING_STARTS,
        batch_size             = BATCH_SIZE,
        gamma                  = GAMMA,
        exploration_fraction   = EXPLORATION_FRAC,
        exploration_final_eps  = EXPLORATION_FINAL,
        target_update_interval = TARGET_UPDATE,
        train_freq             = TRAIN_FREQ,
        policy_kwargs          = {"net_arch": NET_ARCH},
        verbose                = 1,
        tensorboard_log        = LOG_DIR,
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
    print(f"4. Iniciando entrenamiento ({TOTAL_TIMESTEPS:,} pasos)...")
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
