# LeaLink — Optimización de Comunidades Energéticas con RL

**TFG** — Grado en Enxeñaría Informática, ESEI (Universidade de Vigo), 2025/2026.
**Autor:** Roberto Rodríguez Núñez · **Tutora:** Eva Mª Lorenzo Iglesias · **Co-tutor:** Pedro Celard Pérez
**Memoria:** [`doc/memoria_latex/memoria_tipo2.pdf`](doc/memoria_latex/memoria_tipo2.pdf)

Plataforma que optimiza la batería compartida (80 kWh) de una comunidad de autoconsumo
fotovoltaico colectivo (15 viviendas, 42 kWp), con dos partes:

- **Investigación:** gemelo digital con 22.646 h de datos reales (ESIOS + PVGIS, 2021-2023);
  comparación rigurosa de RL (DQN, PPO, SAC, TD3, DDPG) contra un MPC como línea base.
- **Producción:** el mejor modelo (**Residual SAC**: correcciones ±15 % sobre el MPC) exportado
  a ONNX en un runtime edge (compatible Raspberry Pi, sin PyTorch) + plataforma web SaaS
  (Flask + PostgreSQL), todo orquestado con Docker Compose.

**Resultado central:** el Residual SAC supera al MPC realista en **+0,66 €/semana**
(Wilcoxon p = 1,45×10⁻²³) con garantía de suelo (en el peor caso actúa como el MPC).
Ningún RL puro superó al MPC.

| Controlador | €/semana | Δ vs MPC realista |
|---|---:|---:|
| MPC oráculo (cota superior) | 51,45 | +8,06 |
| **Residual SAC (SAC-C)** ← desplegado | **44,05** | **+0,66** |
| MPC realista (línea base) | 43,39 | 0,00 |
| Mejor DQN / mejor PPO | 39,36 / 39,35 | ≈ −4 |
| Heurístico / SAC puro | 30,49 / 26,94 | −12,90 / −16,44 |

Barrido completo (31 configuraciones): [`resultados/resultados.csv`](resultados/resultados.csv).
Diagramas de arquitectura: [`doc/diagramas/`](doc/diagramas/)
([investigación](doc/diagramas/arquitectura_1_investigacion.drawio.png) ·
[producción](doc/diagramas/arquitectura_2_produccion.drawio.png)).

---

## Estructura del proyecto

```
├── config/                # system.yaml (física, MPC, ruido) + experimentos.yaml (registry de experimentos)
├── data/                  # raw/ (ESIOS+PVGIS) y processed/dataset_final.csv  [INCLUIDOS]
├── src/
│   ├── core/              # Simulador, pronósticos, config
│   ├── envs/              # Entornos Gymnasium (discreto, continuo, residual)
│   ├── controllers/       # MPC (LP HiGHS), heurístico, RL, residuales
│   ├── training/          # mains/ (7 algoritmos), multiseed, registry
│   ├── evaluation/        # suite.py, unified.py, estadística (bootstrap, Wilcoxon)
│   ├── production/        # edge_loop, export_onnx, inferencia ONNX
│   └── saas/              # App Flask completa (con sus propios tests y requirements)
├── experimentos/          # build_dataset.py, entrenar.sh, eval_rapida.py, manifiesto.md
├── models/                # Modelos entrenados + ONNX de producción  [INCLUIDOS]
├── resultados/            # resultados.csv + 35 JSON del barrido  [INCLUIDOS]
├── tests/                 # Suite pytest (física, MPC, entornos, ONNX, regresión)
├── docker/                # Dockerfile.edge
├── docker-compose.yml     # postgres + web (SaaS) + edge
└── doc/                   # Memoria (PDF + LaTeX) y 11 diagramas
```

Dataset, modelos entrenados y resultados **van incluidos**: todo es ejecutable sin re-entrenar.
Se generan al usar el proyecto (no incluidos): `logs/`, `models/_runs/`, `.env`, `.venv/`.

---

## Requisitos e instalación

- **Docker + Docker Compose** — para la puesta en producción (Paso 2). Sirve cualquier
  sistema: Windows, macOS o Linux.
- **Python 3.10–3.12** (solo CPU) en **Linux o WSL** — solo para el entrenamiento (Paso 1).

Para el entrenamiento, prepara el entorno de Python **en Linux/WSL**, desde la raíz del proyecto.
Los comandos `apt` son para **Debian/Ubuntu** (en otra distro usa su gestor de paquetes):

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip   # venv y pip (no vienen de serie)

python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # torch CPU (evita bajar CUDA)
pip install -r requirements.txt
```

> Los modelos y los resultados **ya vienen incluidos**. Si solo quieres ver la plataforma
> funcionando, salta directamente al **Paso 2** (producción); el Paso 1 solo hace falta para
> reproducir el entrenamiento desde cero.

---

## Paso 1 · Entrenar y evaluar

`entrenar.sh` entrena una familia (3 semillas), consolida el mejor modelo **y lo evalúa**
(50 semanas × 10 semillas de ruido, IC95 bootstrap, Wilcoxon vs MPC realista), escribiendo
`resultados/resultados.csv`. Entrenamiento y evaluación en un solo comando.

```bash
chmod +x experimentos/entrenar.sh

# 1º los baselines: calculan las referencias (MPC, heurístico, IDLE) que usan las demás
./experimentos/entrenar.sh G

# 2º las familias de RL, una a una (las residuales son lentas, ~9 h/semilla)
./experimentos/entrenar.sh dqn
./experimentos/entrenar.sh ppo
./experimentos/entrenar.sh ppo_continuo
./experimentos/entrenar.sh td3_residual
./experimentos/entrenar.sh ddpg_residual
./experimentos/entrenar.sh sac_puro
./experimentos/entrenar.sh residual_sac
```

- **Reanudable:** si se interrumpe, relanza el mismo comando y continúa donde iba.
- Curvas de entrenamiento: `tensorboard --logdir logs/`.
- Protocolo completo del barrido: [`experimentos/manifiesto.md`](experimentos/manifiesto.md).

> ⚠️ **Para reproducir desde cero, primero vacía los artefactos incluidos.** `entrenar.sh`
> es reanudable: **salta toda versión que ya tenga fila en `resultados.csv`**, y la entrega
> incluye ese CSV y los modelos ya calculados, así que sin vaciarlos el comando no reentrena
> nada (marca cada versión como `[HECHA]`). Antes de reproducir:
>
> ```bash
> mv resultados/resultados.csv resultados/resultados.csv.orig   # o bórralo
> mv models/best models/best.orig                               # o bórralo
> ```
>
> Reentrenar regenera (sobrescribe) esos mismos `models/best/` y `resultados/resultados.csv`.

---

## Paso 2 · Pasar a producción y cargar el SaaS

El controlador se lleva a producción exportándolo al formato ligero **ONNX** (sin PyTorch), que
ejecuta el edge. La plataforma completa (PostgreSQL + web SaaS + edge) se levanta con Docker.

```bash
# (Solo si reentrenaste en el Paso 1) regenerar el ONNX desde el nuevo campeón:
python -m src.production.export_onnx \
  --model    models/best/residual_sac_SAC-C.zip \
  --vec-norm models/best/residual_sac_SAC-C_vecnorm.pkl

# Levantar la plataforma (el ONNX de producción ya viene incluido):
cp .env.example .env
docker compose up -d --build
```

El edge decide cada hora con el modelo y publica en el SaaS; la web se siembra sola con datos
de ejemplo (simula un año en ~6-7 min).

- **Web:** <http://localhost:5001>

  | Rol | Email | Contraseña |
  |---|---|---|
  | Superadmin | `roberto@lealink.es` | `roberto1234` |
  | Admin de comunidad | `carmen.vidal@vecinos.es` | `carmen1234` |
  | Vecino | `ana.garcia@vecinos.es` | `ana1234` |

  Incluye 2 comunidades de ejemplo (27 viviendas, 40 usuarios): baterías con SoC en vivo,
  operaciones horarias, cierres mensuales con ahorro por vivienda, notificaciones, incidencias.

- **Decisiones del edge en vivo:** `docker compose logs -f edge` (un JSON por hora)
- **Parar:** `docker compose down` (añade `-v` para borrar también la base de datos)

> Raspberry Pi 4/5 (edge ARM64): `docker buildx build --platform linux/arm64 -f docker/Dockerfile.edge -t energycomm-edge:arm64 .`

---

## Ayuda (opcional)

Comprobaciones y utilidades que no forman parte del hilo principal.

### Tests

```bash
pytest tests/ -v                          # núcleo: física, MPC, entornos, ONNX, golden (~10 min)

pip install -r src/saas/requirements.txt
pytest src/saas/tests/ -v                 # SaaS: 212 tests (SQLite en memoria, ~1-2 min)
```

### Otras formas de evaluar (sin reentrenar)

`entrenar.sh` ya evalúa, pero también puedes evaluar los modelos incluidos por separado:

```bash
# Rápida, sin escribir nada (reproduce el resultado central ~44 €/sem):
python experimentos/eval_rapida.py --familia residual_sac --version SAC-C

# Formal, escribe en resultados/resultados.csv (⚠️ añade filas al CSV incluido):
python src/evaluation/suite.py --baselines      # baselines G1-G4
python src/evaluation/suite.py --todas          # las 31 versiones + baselines
```

### Regenerar el dataset

```bash
python experimentos/build_dataset.py --semilla 42   # raw ESIOS/PVGIS → dataset_final.csv (reproducible)
```

---

## Configuración

- [`config/system.yaml`](config/system.yaml) — batería (80 kWh, η=0,95, SoC 10-90 %),
  comunidad, split train/eval (semilla 42, 50 semanas), ruido de pronóstico, MPC
  (horizonte 24 h), hiperparámetros del residual, rutas ONNX de producción.
- [`config/experimentos.yaml`](config/experimentos.yaml) — registry de las familias RL
  (`dqn`, `ppo`, `ppo_continuo`, `td3_residual`, `ddpg_residual`, `residual_sac`, `sac_puro`)
  más los baselines `G`; cada versión declara solo sus *overrides*.
- [`.env.example`](.env.example) — plantilla del `.env` (`SECRET_KEY`). La `DATABASE_URL`
  la fija `docker-compose.yml` automáticamente.

## Solución de problemas

| Problema | Solución |
|---|---|
| `docker compose up`: "env file .env not found" | `cp .env.example .env` |
| Puerto 5001 ocupado | Cambiar `5001:5000` en `docker-compose.yml` |
| `Permission denied` con `entrenar.sh` | `chmod +x experimentos/entrenar.sh` |
| `ModuleNotFoundError: src` | Ejecutar siempre desde la raíz del proyecto |
