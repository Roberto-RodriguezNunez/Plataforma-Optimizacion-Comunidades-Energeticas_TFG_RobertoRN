# Instrucciones de implementación: capa SaaS + ONNX + conectores API

> Documento de handoff para una IA/desarrollador que continúe el proyecto.
> Repo: plataforma de optimización de comunidades energéticas (RL/MPC).
> Lee primero `config/system.yaml` y los ficheros listados en "Qué reutilizar".

## Contexto

El repositorio actual contiene el **motor** (Residual SAC + MPC) + ingeniería de datos +
tests, pero la memoria (`doc/memoria_latex/memoria_tipo2 (3).tex`) describe una plataforma de
5 capas. Faltan tres bloques que la memoria da por implementados y que hay que construir:

1. **Producción**: export del actor SAC a **ONNX** + runtime de inferencia *sin SB3 ni
   Gymnasium*, desplegable en **Docker** (edge).
2. **Conectores de datos** (`src/api/data_feed.py`): ESIOS / Open-Meteo / datadis.
3. **SaaS**: dashboards **Streamlit** (rol Administrador y Vecino) + **PostgreSQL**.

### Decisiones de alcance (fijadas por el autor)
- Se implementan los **tres** bloques.
- BD = **PostgreSQL** (SQLAlchemy + psycopg2).
- **No se hacen llamadas HTTP reales**: las APIs se **emulan con datos simulados** (sin tokens).
  El contenedor **ONNX en Docker** consume ese feed emulado *como si* viniera de las APIs
  reales. La *interfaz* de los conectores debe quedar lista para pasar a HTTP real sin tocar
  el resto del sistema.

**Objetivo**: que el código respalde lo que afirma la memoria, manteniendo el matiz del Alcance
("APIs implementadas pero no activadas con datos reales").

---

## Qué REUTILIZAR (no reimplementar)

- `src/controllers/base.py` → `BaseController` (`solve(state, forecast) -> dict`).
- `src/controllers/residual_sac_controller.py` → `ResidualSACController`: **ruta de inferencia
  de referencia**. Carga SAC + VecNormalize, construye la obs de 112 dims, normaliza a mano y
  aplica el residual multiplicativo `max(0, mpc_flow*(1+delta*delta_max))`. El runtime ONNX debe
  **replicar su salida**. Ojo: su `delta_max` por defecto es **0.30**; instánciarlo con **0.15**
  (valor v5, leer de `system.yaml`).
- `src/benchmarks/mpc_benchmark.py` → `LinearMPC` (HiGHS/scipy), `aplicar_ruido_ar1`,
  `aplicar_ruido_precio_3capas`, `simular_hora_mpc`.
- `src/core/simulador.py` → `ComunidadSimulador` (física, split semanal, `get_data_window`).
- `src/eval_unificada.py` → `evaluar_controlador(ctrl, forecast_mode)` → dict con
  `bens_marg/bens_abs/bens_idle` (50 semanas). Para las comparativas del dashboard Admin.
- Modelos: `models/best_model_v5_1M.zip` + `models/vec_normalize_v5_1M.pkl` (v5 final, 1M pasos).

---

## Bloque A — Producción: ONNX + runtime edge (Docker)

### A1. `src/production/export_onnx.py`
- `SAC.load('models/best_model_v5_1M.zip', device='cpu')`. Actor en `model.policy.actor`.
- Wrapper `nn.Module`: `obs_112 (float32) -> delta_4d ∈ [-1,1]` = `tanh(mean)` determinista.
  Usar `actor.get_action_dist_params(obs) -> mean_actions`, luego `torch.tanh(mean_actions)`.
- `torch.onnx.export` → `models/residual_sac_actor.onnx` (opset ≥ 17, input `obs [1,112]`,
  output `delta [1,4]`, nombres explícitos).
- **Crítico**: volcar stats de VecNormalize a formato sin SB3. Abrir el `.pkl`, extraer
  `obs_rms.mean`, `obs_rms.var`, `clip_obs` → `models/vec_normalize_v5_1M.npz`.

### A2. `src/production/onnx_inference.py`
- `OnnxResidualController(BaseController)` que replica `ResidualSACController` pero:
  - actor → `onnxruntime.InferenceSession`.
  - normalización → lee el `.npz` (misma fórmula que `ResidualSACController.solve` paso 3).
  - MPC → sigue siendo `LinearMPC` (NO se exporta a ONNX).
  - obs 112 → extraer `ResidualSACController._build_obs` a `src/production/obs_builder.py` e
    importarla desde ambos para evitar divergencia.
- Deps del edge: `numpy, scipy, onnxruntime, pyyaml`. **Sin** SB3 / Gymnasium / torch.

### A3. `tests/test_onnx_equivalence.py`
- ~200 obs representativas (recorrer semanas con el MPC). Comparar `ResidualSACController.solve`
  vs `OnnxResidualController.solve`. Aserción: máx |Δflujo| < 1e-3 (o |Δdelta| < 1e-4).
- Bonus: €/sem agregado de ambos difiere < 0.1.

### A4. `docker/Dockerfile.edge` + `src/production/edge_loop.py`
- `edge_loop.py`: por cada hora simulada → (1) pide ventana al DataFeed emulado *como si fuera
  la API*, (2) resuelve `OnnxResidualController`, (3) emite acción, (4) persiste en PostgreSQL.
- `Dockerfile.edge`: imagen slim con solo deps de A2 + `.onnx` + `.npz` + `dataset_final.csv`.
- Env: `DATABASE_URL`, `FEED_MODE=simulado`, rutas de modelo.

---

## Bloque B — Conectores emulados — `src/api/data_feed.py`

- ABC `DataFeed` con la misma semántica que `ComunidadSimulador.get_data_window`:
  - `get_window(step|ts, horizon=24) -> np.ndarray (H,4)` cols
    `[consumo_total, generacion_total, precio_kwh, precio_excedente]`.
  - `get_current(step|ts) -> dict/row`.
- `HistoricalDataFeed`: envuelve el dataset/`ComunidadSimulador` actual (default).
- `SimulatedLiveFeed`: emula ESIOS (1001/1739/1006), Open-Meteo (irradiancia→solar), datadis
  (consumo) con métodos `_fetch_esios()/_fetch_open_meteo()/_fetch_datadis()` que hoy devuelven
  datos simulados (reusar `dataset_final.csv` / `src/utils/generador_datos.py`) con la firma y
  el shape de una integración real → pasar a HTTP real = solo rellenar esos 3 métodos. **No**
  importar `requests` ni requerir tokens. Opcional: aplicar `aplicar_ruido_precio_3capas` al
  horizonte futuro para realismo. Selección por `FEED_MODE` (default `historico`).

---

## Bloque C — SaaS: PostgreSQL + Streamlit

### C1. `src/saas/db/` (SQLAlchemy + psycopg2, `DATABASE_URL`)
- Modelos: `Comunidad`, `Vivienda(coef_reparto)`, `Usuario(username, password_hash, rol:
  admin|vecino, vivienda_id)`, `RegistroHorario(ts, consumo, generacion, precio_kwh,
  precio_excedente, soc, cs/cm/dc/dr, beneficio_marginal, controlador)`,
  `RepartoVivienda(registro_id, vivienda_id, consumo, generacion_asignada, ahorro)`.
- `init_db.py`: crea esquema + siembra 1 comunidad de 15 viviendas (coherente con
  `system.yaml`) + usuarios de prueba (1 admin, 1 vecino). `repository.py` con CRUD.

### C2. `src/saas/run_operacion.py`
- Rango temporal → `ResidualSACController` (u `OnnxResidualController`) + `ComunidadSimulador`,
  recorre el periodo, calcula reparto por vivienda con `coef_reparto` (RD 244/2019: balance
  horario agregado → reparto por coeficiente), persiste `RegistroHorario` + `RepartoVivienda`.
  Reusar `evaluar_controlador` para las métricas MPC/SAC/IDLE del Admin.

### C3. `src/saas/app.py` (Streamlit)
- Login contra `Usuario` (hash `passlib[bcrypt]`), gating por rol.
- **Admin** (comunidad): SoC, flujos, KPIs €/sem vs IDLE, comparativa de controladores
  (MPC realista +48,73 / Residual SAC v5 / oráculo +57,71), curvas de entrenamiento.
- **Vecino** (vivienda): su consumo, cuota solar, ahorro vs sin-batería, coef. de reparto.
- Los dashboards **leen de PostgreSQL** (lo que escribió el runner/edge), no recalculan.

---

## Bloque D — Plumbing

- `requirements.txt`: añadir `scipy, pyyaml, pytest, onnx, onnxruntime, torch (pin), streamlit,
  sqlalchemy, psycopg2-binary, passlib[bcrypt]`. (La memoria menciona **OmegaConf** pero el
  código usa `yaml.safe_load`: o se adopta OmegaConf o se corrige esa fila del stack en el `.tex`.)
- `docker-compose.yml`: servicios `postgres`, `edge` (Dockerfile.edge), `streamlit`. Red común,
  `DATABASE_URL` compartido, volúmenes para modelos + dataset.
- `config/system.yaml`: bloques `produccion` (rutas .onnx/.npz), `api` (feed_mode), `saas`
  (roles, semilla usuarios). Sin magic numbers.
- **No romper** los 47 tests, ni `main_sac.py`, ni `eval_unificada.py`.

---

## Verificación end-to-end

1. `pytest tests/ -q` → 47 tests + `test_onnx_equivalence` en verde.
2. `python -m src.production.export_onnx` → genera `.onnx` + `.npz`.
3. `OnnxResidualController` vs SB3 en `eval_unificada`: Δ€/sem < 0.1.
4. `docker compose up` → postgres + edge + streamlit; `edge_loop` puebla `RegistroHorario`.
5. Streamlit: login `admin` (comunidad + comparativa) y `vecino` (solo su vivienda); datos desde
   PostgreSQL.
6. El edge corre **sin** SB3/Gymnasium instalados.

## Fidelidad con la memoria
Al completar A–C se vuelven ciertas las afirmaciones de `data_feed.py` (líneas ~1337-1339),
capa de producción/ONNX y SaaS/Streamlit/PostgreSQL (líneas ~491-495 y ~651-668). Mantener el
matiz "APIs implementadas pero no activadas con datos reales". Revisar la fila OmegaConf del stack.
