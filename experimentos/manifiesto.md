# Manifiesto de ejecución — Suite experimental TFG

Reparto de los entrenos en varios ordenadores. **Una orden por familia y máquina:**

```bash
./experimentos/entrenar.sh <familia>
```

`entrenar.sh` entrena cada versión de la familia (3 semillas: 42, 1337, 2024), elige la
mejor por recompensa de evaluación y la guarda en `models/best/{algo}_{version}.zip`;
acto seguido la evalúa con 10 semillas de ruido y añade su fila a
`resultados/resultados.csv`. **Es reanudable** (ver abajo).

> En este repo `python` pelado no está en PATH. `entrenar.sh` ya usa `.venv/bin/python`
> por defecto, así que **no hace falta prefijo**. Para otro intérprete:
> `PYTHON=python3 ./experimentos/entrenar.sh dqn`.

---

## Entrenar desde WSL / Linux

```bash
cd /ruta/al/proyecto                                             # raíz del repositorio
./experimentos/entrenar.sh dqn                                   # familia entera (entrena + evalúa)
.venv/bin/python src/training/mains/main_dqn.py --version DQN-2  # entrenar UNA versión (sin evaluar)
.venv/bin/python src/evaluation/suite.py --familia dqn --version DQN-2   # evaluar UNA versión
```

Dejarlo en segundo plano (sobrevive a cerrar la terminal + deja log):

```bash
nohup ./experimentos/entrenar.sh dqn > resultados/log_dqn.txt 2>&1 &
echo "PID: $!"
tail -f resultados/log_dqn.txt   # ver avance (Ctrl-C sale del tail, NO para el entreno)
```

> **`entrenar.sh` es REANUDABLE — relanzar la MISMA orden es seguro y NO repite trabajo:**
> - **Salta** las versiones que ya tienen fila en `resultados/resultados.csv` (`[HECHA] -> salto`).
> - **Reutiliza** las semillas ya completas de una versión a medias (mira `models/_runs/`)
>   y solo **reentrena la que quedó cortada** (p.ej. un apagón).
>
> Tras un corte, basta con volver a lanzar `./experimentos/entrenar.sh <fam>`: continúa solo.
> Para **re-evaluar** una versión ya hecha sin reentrenar, llama directamente a la suite:
> `.venv/bin/python src/evaluation/suite.py --familia <fam> --version <V>` (añade fila;
> vale la última por `familia,version`).

---

## Orden recomendado (bloqueos)

1. **PRIMERO, en cualquier máquina:** baselines G (bloquean las métricas relativas).
   ```bash
   ./experimentos/entrenar.sh G
   ```
   Genera G1 MPC-oráculo, G2 MPC-realista, G3 heurístico, G4 IDLE + cachea el MPC.
2. **En paralelo (1 familia por máquina):** A (dqn), B (ppo), C (ppo_continuo), F (residual_sac).
3. **Según liberen máquinas:** D (td3_residual), sac_puro (F5).
4. **Al final (baja prioridad):** E (ddpg_residual).

---

## Familias, comando y coste aproximado

| Familia | Comando | Versiones × 3 semillas | Coste/seed (1M-4M pasos) | Notas |
|---|---|---|---|---|
| **G** baselines | `entrenar.sh G` | — | ~5-10 min total | hazlo primero |
| **A** DQN | `entrenar.sh dqn` | 6 × 3 = 18 | ~1.5 h | discreto, 3M pasos |
| **B** PPO disc | `entrenar.sh ppo` | 5 × 3 = 15 | ~2 h | 4M pasos |
| **C** PPO cont | `entrenar.sh ppo_continuo` | 3 × 3 = 9 | ~2.5 h | 4M pasos |
| **D** TD3 resid | `entrenar.sh td3_residual` | 4 × 3 = 12 | ~9 h | MPC online por paso |
| **E** DDPG resid | `entrenar.sh ddpg_residual` | 2 × 3 = 6 | ~9 h | baja prioridad |
| **F** Residual SAC | `entrenar.sh residual_sac` | 10 × 3 = 30 | ~9 h | la más pesada |
| **F5** SAC puro | `entrenar.sh sac_puro` | 1 × 3 = 3 | ~4 h | sin MPC |

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
.venv/bin/python src/evaluation/suite.py --familia residual_sac --version SAC-C
```
Así **no hay que reentrenar SAC-C**. Al lanzar `entrenar.sh residual_sac`, como SAC-C ya
tendrá fila en `resultados.csv`, se saltará sola.

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
.venv/bin/python src/training/mains/main_dqn.py --version DQN-2 --seeds 42 --timesteps 5000
```

---

## (Una vez) Regenerar el dataset desde las fuentes ESIOS/PVGIS

```bash
.venv/bin/python experimentos/build_dataset.py [--semilla 42]
```
Genera `data/processed/dataset_final.csv` + `metadata.json`. Normalmente NO hace falta:
el dataset ya está versionado.
