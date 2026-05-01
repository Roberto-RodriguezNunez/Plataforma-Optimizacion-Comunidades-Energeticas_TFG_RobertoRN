# Diseño y Desarrollo de una Plataforma Software para la Optimización de Comunidades Energéticas

**Trabajo de Fin de Grado (TFG)** para el Grado en Enxeñaría Informática de la Universidade de Vigo (ESEI).

- **Autor:** Roberto Rodríguez Núñez
- **Tutora:** Eva Mª Lorenzo Iglesias
- **Co-tutor:** Pedro Celard Pérez

---

## Instalación

Desde la raíz del proyecto (`TFG/`), crea un entorno virtual e instala las dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Nota (WSL en Ubuntu 22.04+):** Ubuntu bloquea instalar paquetes en el Python del sistema.
> Si `python3 -m venv` falla, instala primero el paquete necesario:
> ```bash
> sudo apt install python3.12-venv -y
> ```
> Luego vuelve a ejecutar los tres comandos de arriba.

El entorno virtual solo hay que crearlo una vez. En sesiones posteriores, actívalo con:

```bash
source .venv/bin/activate
```

Sabrás que está activo porque el prompt cambia a `(.venv) roberto@...`.

---

## Entrenamiento del agente DQN

### Requisitos previos

- Entorno virtual activo (`source .venv/bin/activate`)
- Dataset en `data/processed/dataset_final.csv` (incluido en el repo)
- Ejecutar siempre desde la raíz del proyecto (`TFG/`)

---

### Prueba rápida (~1 minuto)

Antes de lanzar el entrenamiento completo, verifica que todo funciona editando
las dos primeras constantes de `src/main.py`:

```python
TOTAL_TIMESTEPS = 5_000    # cambia de 300_000 a 5_000
LEARNING_STARTS = 1_000    # cambia de 10_000 a 1_000
```

Ejecuta:

```bash
python src/main.py
```

Si ves `Entorno OK` y los logs del entrenamiento sin errores, todo está correcto.
Restaura los valores originales antes de entrenar de verdad.

---

### Entrenamiento completo (~15-25 minutos)

Con los valores por defecto de `src/main.py` (`TOTAL_TIMESTEPS = 300_000`):

```bash
python src/main.py
```

Durante el entrenamiento verás logs periódicos con estas métricas clave:

| Métrica | Qué indica |
|---|---|
| `ep_rew_mean` | Recompensa media por episodio — debe subir progresivamente |
| `exploration_rate` | Baja de 1.0 a 0.05 según el agente gana confianza |
| `loss` | Pérdida de la red Q — oscila pero tiende a bajar |

Cada 10.000 pasos el `EvalCallback` evalúa el modelo y guarda automáticamente
el mejor en `models/best_model.zip`:

```
Eval num_timesteps=10000, episode_reward=-38.5 +/- 23.0
New best mean reward!
```

---

### Archivos generados

```
TFG/
├── models/
│   ├── best_model.zip     <- mejor política encontrada durante el entrenamiento
│   └── dqn_sgec.zip       <- modelo del último paso
└── logs/
    ├── train.monitor.csv  <- recompensa de cada episodio de entrenamiento
    ├── eval.monitor.csv   <- recompensas de los episodios de evaluación
    └── evaluations.npz    <- histórico de evaluaciones en formato numpy
```

Usa siempre `best_model.zip` para análisis y despliegue.

---

### Visualización con TensorBoard

Abre una **segunda terminal** en la carpeta `TFG/` mientras entrena (o después) y ejecuta:

```bash
tensorboard --logdir logs
```

Abre el navegador en **`http://localhost:6006`**.

Métricas disponibles en TensorBoard:

| Panel | Qué mirar |
|---|---|
| `rollout/ep_rew_mean` | Curva principal de aprendizaje — debe subir |
| `rollout/exploration_rate` | Decaimiento de epsilon |
| `train/loss` | Pérdida de la red Q |
| `eval/mean_reward` | Recompensa en evaluación (más fiable que rollout) |
| `custom/soc_medio` | SoC medio de la batería — debería estabilizarse en 0.4-0.6 |
| `custom/comprado_medio` | Energía comprada a red — si baja, el agente usa mejor la batería |
| `custom/accion_mas_freq` | Acción dominante del agente |

---

### Hiperparámetros configurables

Todos los hiperparámetros están al principio de `src/main.py` como constantes:

| Constante | Valor por defecto | Descripción |
|---|---|---|
| `TOTAL_TIMESTEPS` | 300.000 | Pasos totales de entrenamiento (subir a 500k-1M para resultados más sólidos) |
| `LEARNING_RATE` | 1e-4 | Tasa de aprendizaje de la red Q |
| `BUFFER_SIZE` | 100.000 | Tamaño del replay buffer |
| `LEARNING_STARTS` | 10.000 | Pasos de exploración pura antes de empezar a entrenar |
| `BATCH_SIZE` | 64 | Muestras por actualización de gradiente |
| `GAMMA` | 0.99 | Factor de descuento (horizonte largo, episodios de 168 pasos) |
| `EXPLORATION_FRAC` | 0.4 | Fracción del entrenamiento con epsilon decreciente |
| `EXPLORATION_FINAL` | 0.05 | Epsilon mínimo al final del entrenamiento |
| `EVAL_FREQ` | 10.000 | Cada cuántos pasos evaluar y guardar el mejor modelo |