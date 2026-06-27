"""
callbacks.py — Callbacks de métricas compartidos (TensorBoard)
==============================================================
Centraliza los callbacks de logging para no duplicarlos entre los mains:
  - DiscreteMetricasCallback: SoC, economía y distribución de acciones del
    entorno discreto (DQN y PPO). Antes duplicado casi idéntico en ambos.
  - ResidualMetricasCallback:  métricas específicas del Residual SAC
    (SoC, delta, a_mpc) + gc.collect periódico.

Son solo observabilidad (no afectan al entrenamiento ni a las "mismas
condiciones"); se unifican por mantenibilidad.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DiscreteMetricasCallback(BaseCallback):
    """Registra SoC, economía y distribución de acciones (entorno discreto de
    9 acciones) en TensorBoard. Compartido por DQN y PPO."""

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

            # epsilon (solo DQN; en PPO falla silenciosamente)
            try:
                self.logger.record("custom/epsilon", self.model.exploration_rate)
            except AttributeError:
                pass

            # Resumen periódico por consola
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


class ResidualMetricasCallback(BaseCallback):
    """Métricas específicas del Residual SAC (SoC, delta, a_mpc) en TensorBoard,
    más un gc.collect() periódico (evita memory leak de scipy/HiGHS)."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._soc_buf = []
        self._delta_buf = []
        self._a_mpc_buf = []
        self._gc_freq = 10_000

    def _on_step(self) -> bool:
        if self.num_timesteps % self._gc_freq == 0:
            import gc
            gc.collect()

        for info in self.locals.get('infos', []):
            if 'soc' in info:
                self._soc_buf.append(info['soc'])
            if 'delta_applied' in info:
                self._delta_buf.append(info['delta_applied'])
            if 'a_mpc' in info:
                self._a_mpc_buf.append(info['a_mpc'])

        if self.num_timesteps % 1000 == 0 and self._soc_buf:
            self.logger.record('custom/soc_medio', np.mean(self._soc_buf))
            self.logger.record('custom/soc_min', np.min(self._soc_buf))
            self.logger.record('custom/soc_max', np.max(self._soc_buf))

            if self._delta_buf:
                deltas = np.array(self._delta_buf)
                self.logger.record('custom/delta_l1_medio', np.mean(np.abs(deltas)))
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
