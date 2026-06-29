# Manifiesto de ejecución — Suite experimental TFG

Reparto de los entrenos en varios ordenadores. **Una orden por familia y máquina:**

```bash
PYTHON=.venv/bin/python ./experimentos/run_familia.sh <familia>
```

Cada familia entrena sus versiones (3 semillas: 42, 1337, 2024), elige la mejor por
recompensa de evaluación y la guarda en `models/best/{algo}_{version}.zip`; luego la
evalúa con 10 semillas de ruido y añade la fila a `resultados/resultados.csv`.

> En este repo `python` pelado no está en PATH → usa siempre `PYTHON=.venv/bin/python`
> (o `source .venv/bin/activate` y `PYTHON=python`).

---

## Entrenar desde WSL / Linux

En WSL el script ya usa `.venv/bin/python` por defecto → **no hace falta el prefijo `PYTHON=`**:

```bash
cd /mnt/c/Users/rober/TFG
./experimentos/run_familia.sh dqn                  # familia entera (entrena + evalúa)
.venv/bin/python src/main_dqn.py --version DQN-2   # una sola versión
```

Dejarlo corriendo en segundo plano (sobrevive a cerrar la terminal + deja log):

```bash
nohup ./experimentos/run_familia.sh dqn > resultados/log_dqn.txt 2>&1 &
echo "PID: $!"
tail -f resultados/log_dqn.txt   # ver avance (Ctrl-C sale del tail, NO para el entreno)
```

> **Relanzar una familia ya entrenada NO borra nada:** la reentrena desde cero
> (sobrescribe su modelo en `models/best/` al terminar) y **añade** filas a
> `resultados/resultados.csv` (quedan duplicadas → vale la última por `familia,version`).
> No toca otras familias. Para solo re-evaluar sin reentrenar:
> `./experimentos/run_familia.sh <fam> --solo-eval`.

---

## Orden recomendado (bloqueos)

1. **PRIMERO, en cualquier máquina:** baselines G (bloquean las métricas relativas).
   ```bash
   ./experimentos/run_familia.sh G
   ```
   Genera G1 MPC-oráculo, G2 MPC-realista, G3 heurístico, G4 IDLE + cachea el MPC.
2. **En paralelo (1 familia por máquina):** A (dqn), B (ppo), C (ppo_continuo), F (residual_sac).
3. **Según liberen máquinas:** D (td3_residual), sac_puro (F5).
4. **Al final (baja prioridad):** E (ddpg_residual).

---

## Familias, comando y coste aproximado

| Familia | Comando | Versiones × 3 semillas | Coste/seed (1M-4M pasos) | Notas |
|---|---|---|---|---|
| **G** baselines | `run_familia.sh G` | — | ~5-10 min total | hazlo primero |
| **A** DQN | `run_familia.sh dqn` | 6 × 3 = 18 | ~1.5 h | discreto, 3M pasos |
| **B** PPO disc | `run_familia.sh ppo` | 5 × 3 = 15 | ~2 h | 4M pasos |
| **C** PPO cont | `run_familia.sh ppo_continuo` | 3 × 3 = 9 | ~2.5 h | 4M pasos |
| **D** TD3 resid | `run_familia.sh td3_residual` | 4 × 3 = 12 | ~9 h | MPC online por paso |
| **E** DDPG resid | `run_familia.sh ddpg_residual` | 2 × 3 = 6 | ~9 h | baja prioridad |
| **F** Residual SAC | `run_familia.sh residual_sac` | 10 × 3 = 30 | ~9 h | la más pesada |
| **F5** SAC puro | `run_familia.sh sac_puro` | 1 × 3 = 3 | ~4 h | sin MPC |

> ⚠️ Las familias residuales (D, E, F) resuelven el MPC (LP) en **cada paso** → ~9 h/seed.
> La familia F sola son ~30 entrenos × ~9 h ≈ mucho cómputo: prioriza **SAC-A/B/C** (tabla de
> arquitectura), **F5** (mérito del residual) y **F2** (ablación δ); F4/F6 si hay máquinas libres.

---

## El Residual SAC 80 kWh ya en curso = SAC-C

El entreno lanzado (`models/best/residual_sac_80kwh.*`, 3 semillas) **es** la versión SAC-C
(δ=0.15, ent=0.01, DAWN=100k, mult). Cuando termine, renómbralo para que la suite lo reconozca:

```bash
cd models/best
for ext in zip _vecnorm.pkl _seeds.json; do
  mv "residual_sac_80kwh${ext}" "residual_sac_SAC-C${ext}"
done
# y evalúalo:
.venv/bin/python src/eval_suite.py --familia residual_sac --version SAC-C
```
Así **no hay que reentrenar SAC-C**. Al lanzar `run_familia.sh residual_sac`, sáltate SAC-C
(o usa `--solo-eval` tras tener todas) para no repetirlo.

---

## Unir resultados de varias máquinas

Cada máquina genera su propio `resultados/resultados.csv`. Para juntarlos: copia los CSV a una
carpeta y concaténalos quitando cabeceras repetidas (la fila más reciente por `familia,version`
es la buena):

```bash
# en la máquina que agrega:
head -1 maquina1.csv > resultados.csv
tail -n +2 -q maquina*.csv >> resultados.csv
```
Pásame `resultados.csv` y relleno/recalculo las tablas de la memoria.

---

## Smoke antes de una tanda larga (opcional)

Comprueba que una versión arranca sin gastar horas:
```bash
.venv/bin/python src/main_dqn.py --version DQN-2 --seeds 42 --timesteps 5000
```
