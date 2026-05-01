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