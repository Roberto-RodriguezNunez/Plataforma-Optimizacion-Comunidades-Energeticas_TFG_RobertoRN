# Diseño y Desarrollo de una Plataforma Software para la Optimización de Comunidades Energéticas

**Trabajo de Fin de Grado (TFG)** para el Grado en Enxeñaría Informática de la Universidade de Vigo (ESEI).

- **Autor:** Roberto Rodríguez Núñez
- **Tutora:** Eva Mª Lorenzo Iglesias
- **Co-tutor:** Pedro Celard Pérez

---

## Instalación

El proyecto se ejecuta con el **Python de Windows desde PowerShell**. Las dependencias ya están instaladas en el Python de Windows. Si necesitas instalarlas en un equipo nuevo:

```powershell
pip install -r requirements.txt
```

> **Nota:** `stable-baselines3` depende de PyTorch. Para instalar la versión CPU
> (más ligera, sin CUDA), usa:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install stable-baselines3 gymnasium tensorboard pandas numpy matplotlib
> ```

---

## Entrenamiento del agente DQN

### Requisitos previos

- **PowerShell** (no WSL)
- Dependencias instaladas en el Python de Windows
- Dataset en `data/processed/dataset_final.csv` (incluido en el repo)
- Ejecutar siempre desde la raíz del proyecto (`TFG/`)

---

### Prueba rápida (~1 minuto)

Antes de lanzar el entrenamiento completo, verifica que todo funciona editando
las dos primeras constantes de `src/main.py`:

```python
TOTAL_TIMESTEPS = 10_000   # cambia de 1_000_000 a 10_000
LEARNING_STARTS = 2_000    # cambia de 20_000 a 2_000
```

Ejecuta:

```bash
python src/main.py
```

Si ves `Entorno OK` y los logs del entrenamiento sin errores, todo está correcto.
Restaura los valores originales antes de entrenar de verdad.

---

### Entrenamiento completo (~45-60 minutos)

Con los valores por defecto de `src/main.py` (`TOTAL_TIMESTEPS = 1_000_000`):

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
│   ├── best_model.zip      <- mejor política encontrada durante el entrenamiento
│   ├── dqn_sgec.zip        <- modelo del último paso
│   └── vec_normalize.pkl   <- estadísticas de normalización (imprescindible para inferencia)
└── logs/
    ├── train.monitor.csv   <- recompensa de cada episodio de entrenamiento
    ├── eval.monitor.csv    <- recompensas de los episodios de evaluación
    └── evaluations.npz     <- histórico de evaluaciones en formato numpy
```

Usa siempre `best_model.zip` + `vec_normalize.pkl` para análisis y despliegue.

---

### Visualización con TensorBoard

Abre una **segunda ventana de PowerShell** mientras entrena (o después) y ejecuta:

```powershell
cd C:\Users\nicor\OneDrive\Documents\TFGyDocumentosTFG\TFG
tensorboard --logdir logs
```

Abre el navegador en **`http://localhost:6006`**.

> **Aviso normal:** TensorBoard mostrará `TensorFlow installation not found - running with reduced feature set.`
> Es esperado — TensorBoard funciona perfectamente sin TensorFlow para este proyecto.

---

#### Qué mirar en TensorBoard

TensorBoard muestra muchos paneles. Estos son los únicos que importan, en orden de importancia:

**1. `rollout/ep_rew_mean` — LA curva principal**
La recompensa media por episodio. Es el indicador de que el agente está aprendiendo.
- Al principio estará en torno a -100 (el agente no sabe nada)
- Debe subir progresivamente hacia 0 o positivo
- Si sube = el agente aprende. Si se queda plana = problema con hiperparámetros

**2. `eval/mean_reward` — Recompensa de evaluación**
Igual que la anterior pero medida en episodios de test separados (más fiable).
Cada vez que aparece un punto nuevo aquí y es el mejor hasta ahora, se guarda `best_model.zip`.

**3. `rollout/exploration_rate` — Decaimiento de epsilon**
Baja de 1.0 a 0.05 a lo largo del entrenamiento.
- Al principio (ε=1.0): el agente elige acciones al azar, explorando
- Al final (ε=0.05): el agente usa casi siempre su red neuronal

**4. `train/loss` — Pérdida de la red Q**
Mide cuánto se equivoca la red al predecir los Q-values.
- Oscila bastante, es normal
- No debe dispararse a valores muy altos ni quedarse en 0

**5. `custom/soc_medio` — Estado de carga de la batería**
SoC medio de la batería por paso. Debería estabilizarse en 0.3-0.7.
Si se queda en 0 o en 1 constantemente, el agente está haciendo algo mal.

**6. `custom/comprado_medio` — Energía comprada a red**
Si baja con el tiempo, el agente está aprendiendo a usar mejor la batería (comprando menos a red).

> El resto de paneles que aparecen (`time/`, `train/n_updates`, etc.) son métricas
> internas de SB3 que no necesitas monitorizar.

---

### Hiperparámetros configurables

Todos los hiperparámetros están al principio de `src/main.py` como constantes:

| Constante | Valor v2 | v1 | Descripción |
|---|---|---|---|
| `TOTAL_TIMESTEPS` | 1.000.000 | 300k | Pasos totales de entrenamiento |
| `LEARNING_RATE` | 5e-5 | 1e-4 | Tasa de aprendizaje de la red Q |
| `BUFFER_SIZE` | 200.000 | 100k | Tamaño del replay buffer |
| `LEARNING_STARTS` | 20.000 | 10k | Pasos de exploración pura antes de entrenar |
| `BATCH_SIZE` | 128 | 64 | Muestras por actualización de gradiente |
| `GAMMA` | 0.99 | 0.99 | Factor de descuento (horizonte largo) |
| `EXPLORATION_FRAC` | 0.5 | 0.4 | Fracción del entrenamiento con epsilon decreciente |
| `EXPLORATION_FINAL` | 0.05 | 0.05 | Epsilon mínimo al final |
| `NET_ARCH` | [256, 256] | [64, 64] | Capas ocultas de la red neuronal |
| `EVAL_FREQ` | 20.000 | 10k | Cada cuántos pasos evaluar y guardar el mejor modelo |

---

## Entrenamiento del agente Residual SAC

El Residual SAC aprende correcciones ±15% sobre las decisiones del MPC. Se ejecuta desde **WSL** (no PowerShell) porque usa HiGHS LP en cada paso y es más estable en Linux.

### Requisitos previos

- **WSL** con el entorno virtual activado (`.venv/`)
- Ejecutar siempre desde la raíz del proyecto (`TFG/`)
- Para que el entreno sobreviva a cerrar el terminal, deshabilitar la suspensión automática de Windows:
  Panel de control → Opciones de energía → Elegir el comportamiento al cerrar la tapa → **No hacer nada**

### Entrenamiento completo (~8-9 horas, CPU)

```bash
source .venv/bin/activate
nohup python src/training/mains/main_residual_sac.py --version 80kwh --seeds 42 1337 2024 > logs/train_sac.log 2>&1 &
echo $! > logs/train_sac.pid
```

Seguir el progreso:

```bash
tail -f logs/train_sac.log
```

Ver paso actual:

```bash
python3 -c "import numpy as np; d=np.load('logs/evaluations.npz'); print(d['timesteps'][-1], '/ 1,000,000')"
```

### Archivos generados

```
TFG/
├── models/
│   ├── best_model.zip                    ← mejor checkpoint (criterio: reward medio eval)
│   ├── best_vecnormalize.pkl             ← stats VecNormalize del mejor checkpoint
│   ├── residual_sac_seed42.zip           ← checkpoint final (paso 1M)
│   └── residual_sac_vec_normalize_seed42.pkl
└── logs/
    ├── train_sac.log                     ← stdout del entrenamiento
    ├── evaluations.npz                   ← curva de eval (timesteps, rewards)
    └── ResidualSAC_seed42_1/             ← eventos TensorBoard
```

> `best_model.zip` + `best_vecnormalize.pkl` son siempre el par a usar. Se actualizan
> automáticamente durante el entreno cada vez que el modelo mejora su reward medio.

### Visualización con TensorBoard

```bash
tensorboard --logdir logs/
```

Métricas relevantes: `eval/mean_reward`, `custom/soc_medio`, `custom/delta_l1_medio`.

---

## Evaluación del modelo

Compara el Residual SAC contra MPC oráculo, MPC realista e IDLE sobre las 50 semanas hold-out:

```bash
python src/eval_unificada.py \
  --sac-model models/best_model.zip \
  --sac-norm  models/best_vecnormalize.pkl
```

Para evaluar con varias semillas de ruido (recomendado para la memoria):

```bash
python src/evaluation/scenarios.py   # próximamente
```

---

## Exportación a ONNX (despliegue en producción)

Una vez terminado el entreno y elegido el modelo a desplegar, exportar el actor a ONNX:

```bash
# Exportar best_model.zip (por defecto)
python -m src.production.export_onnx

# O especificar otro modelo explícitamente
python -m src.production.export_onnx \
  --model    models/residual_sac_seed42.zip \
  --vec-norm models/residual_sac_vec_normalize_seed42.pkl \
  --onnx-out models/residual_sac_actor.onnx \
  --npz-out  models/vec_normalize_v5_1M.npz
```

Genera dos artefactos en `models/`:

| Archivo | Tamaño | Descripción |
|---|---|---|
| `residual_sac_actor.onnx` | ~376 KB | Actor determinista, entrada `obs[batch,112]` → `delta[batch,4]` |
| `vec_normalize_v5_1M.npz` | ~2 KB | Estadísticas de normalización (mean, var, clip_obs) |

Verificar que el ONNX da los mismos resultados que el modelo SB3:

```bash
python src/eval_unificada.py \
  --sac-model  models/best_model.zip \
  --sac-norm   models/best_vecnormalize.pkl \
  --onnx-model models/residual_sac_actor.onnx \
  --onnx-npz   models/vec_normalize_v5_1M.npz \
  --skip-baselines
```

Los dos controladores deben dar el mismo reward (diferencia < 0.01 EUR/sem).

Una vez verificado, el sistema de producción carga automáticamente el ONNX desde las rutas
configuradas en `config/system.yaml` (sección `produccion:`). No hay ningún paso adicional.