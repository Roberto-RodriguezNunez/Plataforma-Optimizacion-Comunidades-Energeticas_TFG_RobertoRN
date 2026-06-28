# Cambios de la sesión — Suite experimental + preparación del reentreno

Documento exhaustivo de **todos los cambios realizados** en esta sesión, con sus
respectivos cambios en el código. Al inicio, el **plan seguido**.

---

# 0. PLAN SEGUIDO

## Context
La memoria es un TFG de **investigación** y necesita documentar muchos modelos/versiones
con sus hiperparámetros y resultados (anexos F.1–F.6 + baselines G). El entorno cambió
(batería 80 kWh desde config, convención sin-lookahead con ruido desde k=0, física de 4
flujos unificada, fuente única de forecast, MPC recalibrado) → **todos los números viejos
quedan invalidados**. Hay que: (1) montar infraestructura para entrenar TODAS las versiones
con 3 semillas de entreno + 10 de eval, (2) crear los entrenadores de los algoritmos que aún
no tienen código (PPO continuo, SAC puro, TD3 residual, DDPG residual), reutilizando el código
base ya unificado, y (3) guardar resultados bien estructurados (CSV/JSON) por versión.
**Objetivo: dejarlo listo para lanzar los entrenos repartidos en varios ordenadores.**

## Decisiones (cerradas con el usuario)
- **Las 6 familias completas**: G (baselines), A (DQN), B (PPO discreto), C (PPO continuo),
  D (TD3 residual), E (DDPG residual), F (Residual SAC).
- **Semillas: 3 de entreno `[42,1337,2024]` + 10 de eval de ruido** para TODAS (incluida F1).
- **El Residual SAC 80 kWh en curso = SAC-C** (se deja terminar y se mapea su artefacto).
- **Documentación: solo CSV/JSON estructurado** (el usuario rellena las plantillas LaTeX).
- **Frontera**: se construye y smoke-verifica la maquinaria; el usuario lanza la matriz larga.

## Fases del plan
- **Fase 0** — `doc/CAMBIOS_ENTORNO.md` (justificación) + baselines G.
- **Fase 1** — Registry `config/experimentos.yaml` + loader `src/training/registro.py`.
- **Fase 2** — Refactor de los 3 mains existentes (registry-driven + N-step DQN + aditivo SAC).
- **Fase 3** — 4 entrenadores nuevos (ppo_continuo, sac_puro, td3_residual, ddpg_residual).
- **Fase 4** — `residual_mode` mult/add en env+controller + `ContinuousController`.
- **Fase 5** — `src/eval_suite.py` (dispatcher 10 semillas + IC95 + Wilcoxon) + estadística + CSV/JSON.
- **Fase 6** — `experimentos/manifiesto.md` + `experimentos/run_familia.sh`.

---

# 1. RESUMEN DE COMMITS (orden cronológico)

| Commit | Descripción |
|---|---|
| `e85f51f` | fix(simulador): leer batería del config (80 kWh) en vez de hardcodear 100 |
| `8b8b365` | docs: limpiar referencia a diagnostico_recompensa en el árbol |
| `e4ada33` | refactor(controllers): unificar Residual SAC en base común (torch/onnx) |
| `beec978` | rename: main_sac.py → main_residual_sac.py |
| `edcac5b` | feat(suite): registry de experimentos + loader (Fase 1) |
| `8f0610d` | feat(suite): residual_mode mult/add + controller residual genérico + continuo (Fase 4) |
| `25422a5` | refactor(suite): mains DQN/PPO/residual_sac registry-driven (Fase 2) |
| `e8cf07a` | feat(suite): 4 entrenadores nuevos (Fase 3) |
| `638b300` | feat(suite): eval_suite + estadística + salida CSV/JSON (Fase 5) |
| `c67dd6b` | docs(suite): reparto multi-PC + doc de cambios del entorno (Fase 6 + 0) |
| `e8266e1` | refactor: mover dawn_warmup a src/training/ (compartido) |

Verificación final: **suite de tests 51 passed, 2 skipped** (sin romper nada).

---

# 2. CAMBIOS PREVIOS A LA SUITE (misma sesión)

## 2.1 Batería 80 kWh desde config — `e85f51f`
**Problema:** `simulador.py` hardcodeaba `BATERIA_CAPACIDAD = 100` mientras config/SaaS/memoria
decían 80 → el entreno corría con 100 kWh. **Fix:** leer toda la batería de `config['bateria']`.

`src/core/simulador.py` (cabecera del módulo + `__init__`):
```python
# top del módulo
try:
    with open(config_path) as _f:
        _BAT = yaml.safe_load(_f)['bateria']
except Exception:
    _BAT = {}

# en __init__:
self.BATERIA_CAPACIDAD   = _BAT.get('capacidad_kwh', 80.0)         # antes: 100.0 hardcoded
self.POTENCIA_INVERSOR   = _BAT.get('potencia_inversor_kw', 50.0)
self.SOC_MIN             = _BAT.get('soc_min', 0.10)
self.SOC_MAX             = _BAT.get('soc_max', 0.90)
self.EFICIENCIA_CARGA    = _BAT.get('eficiencia_carga', 0.95)
self.EFICIENCIA_DESCARGA = _BAT.get('eficiencia_descarga', 0.95)
self.AUTODESCARGA_POR_HORA = _BAT.get('autodescarga_por_hora', 0.00004)
```
El MPC (`mpc_benchmark.py`) ya leía `_CAP = sim.BATERIA_CAPACIDAD` → se ajustó solo.
Borrados además 2 ficheros muertos: `src/utils/generador_datos.py` (vacío) y
`src/benchmarks/diagnostico_recompensa.py` (script de debug). Test
`test_energy_env_continuo.py` actualizado 47.71 → **43.30** (delta=0 MPC en 80 kWh).

## 2.2 Limpieza del árbol de directorios — `8b8b365`
`doc/especificacion_sistema_sgec.md`: eliminada la línea de `diagnostico_recompensa.py`.

## 2.3 Unificación de los controllers Residual SAC — `e4ada33`
`ResidualSACController` (eval, torch) y `OnnxResidualController` (producción, onnx) eran ~90%
idénticos. Se extrajo toda la lógica a una **base común** `ResidualControllerBase`; las
subclases solo implementan `_infer()`.

`src/controllers/residual_base.py` (NUEVO) — el `solve()` común:
```python
class ResidualControllerBase(BaseController):
    def solve(self, state, forecast):
        mpc_action = self._mpc.solve(state, forecast)            # 1. MPC base
        cs_mpc, cm_mpc, dc_mpc, dr_mpc = (mpc_action['P_carga_solar'], ...)
        obs_108 = build_obs(state, forecast, self._sim)          # 2. obs (fuente única)
        mpc_feat = np.array([cs_mpc/P, cm_mpc/P, dc_mpc/P, dr_mpc/P], np.float32)
        obs_112 = np.append(obs_108, mpc_feat).astype(np.float32)  # 3. +4 features MPC
        obs_norm = self._normalizar(obs_112)                     # 4. VecNormalize
        delta = self._infer(obs_norm)                            # 5. backend (torch/onnx)
        return { ... max(0, mpc * (1 + delta*dm)) ... }          # 6. residual

    def _infer(self, obs_norm):  # lo implementa la subclase
        raise NotImplementedError
```
- `ResidualSACController._infer` → `SAC.predict` (torch).
- `OnnxResidualController._infer` → `session.run` (onnx).
- La base **no importa torch ni onnxruntime** → edge-safe.

## 2.4 Rename `main_sac.py` → `main_residual_sac.py` — `beec978`
El nombre genérico inducía a error: entrena el SAC **residual** (ResidualEnv + warmup MPC).
Actualizados docstring y `README.md`.

---

# 3. SUITE EXPERIMENTAL

## Fase 1 — Registry + loader (`edcac5b`)

### `config/experimentos.yaml` (NUEVO) — fuente única de hiperparámetros
Estructura por familia: `algo`, `env`, `base` (común) + `versiones` (overrides + `nota`) + `tesis`.
Bases = configs ya ajustados (DQN_13, PPO_4, residual_sac). 31 versiones en 7 familias.
```yaml
comun:
  train_seeds: [42, 1337, 2024]
  eval_seeds: 10
  eval_episodes: 50

dqn:
  algo: dqn
  env: discreto
  base: { total_timesteps: 3_000_000, gamma: 0.995, n_step: 1, net_arch: [256,128], ... }
  versiones:
    DQN-1: {n_step: 1,  gamma: 0.995, learning_rate: 1.0e-4, nota: "Línea base"}
    DQN-2: {n_step: 8,  ..., nota: "Propagación media (mejor familia esperada)"}
    DQN-3: {n_step: 8,  gamma: 0.99, ...}
    DQN-4: {n_step: 24, ...}
    DQN-5: {n_step: 8, learning_rate: 5.0e-4, target_update_interval: 2_000}
    DQN-6: {n_step: 4, ...}
# ... ppo (PPO-D1..5), ppo_continuo (PPO-C1..3), td3_residual (TD3-1..4),
#     ddpg_residual (DDPG-1..2), residual_sac (SAC-A/B/C, F2-d05/10/20/30,
#     F4-nodawn, F6-s05/s25), sac_puro (F5)
```
Notas de diseño codificadas:
- `learning_rate`/`ent_coef` admiten **escalar (const)** o `{init, final}` (**decay lineal**).
- `residual_sac` lleva `residual_mode: mult|add`, `delta_max`, `dawn_warmup_steps`, `ent_coef`.
- F6 lleva `sigma_consumo` (override del ruido de pronóstico).

### `src/training/registro.py` (NUEVO) — loader
```python
def cargar_version(familia, version):      # base ⊕ override de primer nivel
    cfg = dict(fam.get("base", {})); cfg.update(versiones[version] or {})
    cfg["algo"], cfg["env"], cfg["familia"], cfg["version"] = ...
    return cfg
def listar_versiones(familia): ...
def seeds_comunes():  return train_seeds, eval_seeds, eval_episodes
def es_decay(v):      return isinstance(v, dict) and "init" in v and "final" in v
def schedule_lineal(v):  # escalar→const ; {init,final}→callable decay de SB3
    if es_decay(v): return lambda p: fin + p*(ini-fin)
    return lambda p: float(v)
```

## Fase 4 — envs/controllers (`8f0610d`)

### `src/envs/residual_env.py` — `residual_mode` mult/add
```python
def __init__(self, env, mpc, delta_max=0.30, residual_mode='mult'):
    ...
    self.residual_mode = residual_mode

def _residual(self, mpc_flow, delta_i):
    dm = self.delta_max
    if self.residual_mode == 'add':
        return max(0.0, mpc_flow + delta_i * dm * self._P_MAX)   # aditivo (SAC-A/F3)
    return max(0.0, mpc_flow * (1.0 + delta_i * dm))             # multiplicativo
# step() ahora usa self._residual(cs_mpc, delta[0]), etc.
```

### `src/controllers/residual_base.py` — mismo `residual_mode` en eval
```python
def __init__(self, mpc, sim, delta_max, residual_mode='mult'): ...
def _combinar(self, mpc_flow, delta_i):
    dm = self._delta_max
    if self._residual_mode == 'add':
        return max(0.0, mpc_flow + float(delta_i)*dm*self._P_MAX)
    return max(0.0, mpc_flow * (1.0 + float(delta_i)*dm))
```

### `src/controllers/residual_sac_controller.py` — genérico SAC/TD3/DDPG
```python
_ALGOS_TORCH = ('SAC', 'TD3', 'DDPG')
def __init__(self, model_path, mpc, sim, delta_max, vec_normalize_path=None,
             algo='SAC', residual_mode='mult'):
    import stable_baselines3 as sb3
    super().__init__(mpc, sim, delta_max, residual_mode=residual_mode)
    self._model = getattr(sb3, algo.upper()).load(model_path, device='cpu')
# _infer() = self._model.predict(obs, deterministic=True)[0]  (igual para los 3)
```

### `src/controllers/continuous_controller.py` (NUEVO) — continuo puro (sin MPC)
```python
class ContinuousController(BaseController):     # PPO continuo y SAC puro
    def __init__(self, model_path, sim, vec_normalize_path=None, algo='SAC'):
        self._model = getattr(sb3, algo.upper()).load(model_path, device='cpu')
        # carga mean/var de VecNormalize si hay
    def solve(self, state, forecast):
        obs = build_obs(state, forecast, self._sim)              # 108 dims, SIN +4 MPC
        if self._mean is not None: obs = np.clip((obs-mean)/sqrt(var+1e-8), ±clip)
        a = np.clip(self._model.predict(obs, deterministic=True)[0], 0.0, self._P_MAX)
        return {'P_carga_solar': float(a[0]), ...}              # 4 flujos directos
```

## Fase 2 — refactor mains existentes (`25422a5`)

### `src/main_dqn.py` — registry + **N-step buffer**
```python
def make_model(train_env, seed):
    c = _CFG; base_gamma = float(c["gamma"]); n_step = int(c.get("n_step", 1))
    kwargs = dict(policy="MlpPolicy", env=train_env, gamma=base_gamma, ... )
    if n_step > 1:
        from src.buffers.nstep_replay_buffer import NStepReplayBuffer
        kwargs["gamma"] = base_gamma ** n_step                  # γ^n para el bootstrap
        kwargs["replay_buffer_class"]  = NStepReplayBuffer
        kwargs["replay_buffer_kwargs"] = {"n_steps": n_step, "base_gamma": base_gamma}
    return DQN(**kwargs)

def main(version="DQN-2", seeds=None, total_timesteps=None):
    global _CFG; _CFG = cargar_version("dqn", version)
    train_seeds, _es, eval_episodes = seeds_comunes()
    entrenar_multiseed(algo="dqn", version=version, seeds=seeds or train_seeds, ...)
```
Verificado: DQN-2 → `model.gamma == 0.995**8` y buffer `NStepReplayBuffer`.

### `src/main_ppo.py` — lr/ent const o decay condicional
```python
class EntCoefScheduler(BaseCallback):
    def __init__(self, init, final, total): ...
    def _on_step(self):
        frac = min(1.0, self.num_timesteps/self._total)
        self.model.ent_coef = self._init + frac*(self._final-self._init); return True

def make_model(train_env, seed):
    c = _CFG; ent = c["ent_coef"]
    ent_init = float(ent["init"]) if es_decay(ent) else float(ent)
    return PPO(..., learning_rate=schedule_lineal(c["learning_rate"]), ent_coef=ent_init, ...)

def _extra_callbacks(model):
    cbs = [DiscreteMetricasCallback()]
    if es_decay(_CFG["ent_coef"]): cbs.append(EntCoefScheduler(init, final, _TOTAL))
    return cbs
```

### `src/main_residual_sac.py` — registry + aditivo + DAWN off + σ F6
```python
def _make_residual_env(mpc, mode='train'):
    renv = ResidualEnv(inner, mpc, delta_max=_VCFG['delta_max'],
                       residual_mode=_VCFG.get('residual_mode', 'mult'))  # mult|add

def make_model(train_env, seed):
    c = _VCFG
    learning_starts = 0 if int(c['dawn_warmup_steps']) > 0 else 10_000   # sin DAWN→explora
    return SAC(..., ent_coef=c['ent_coef'], learning_starts=learning_starts, ...)

def main(version="SAC-C", seeds=None, total_timesteps=None):
    global _VCFG; _VCFG = cargar_version("residual_sac", version)
    if 'sigma_consumo' in _VCFG:                                # F6: σ del entorno por-run
        import src.core.forecast as _fc; _fc.SIGMA_CONS_BASE = float(_VCFG['sigma_consumo'])
    usar_warmup = warmup if int(_VCFG['dawn_warmup_steps']) > 0 else None   # F4-nodawn
    entrenar_multiseed(algo="residual_sac", version=version, warmup=usar_warmup, ...)
```

## Fase 3 — 4 entrenadores nuevos (`e8cf07a`)

### `src/training/action_noise.py` (NUEVO) — ruido TD3/DDPG
```python
def construir_action_noise(spec, n_actions=4):    # {tipo: normal|ou|none, sigma}
    if tipo == "none": return None
    if tipo == "ou":   return OrnsteinUhlenbeckActionNoise(mean, sig)
    return NormalActionNoise(mean, sig)
```

### `src/main_td3_residual.py` y `src/main_ddpg_residual.py` (NUEVOS)
Mismo entorno residual que el SAC (ResidualEnv + `dawn_warmup` **reutilizado**):
```python
from src.main_residual_sac import dawn_warmup            # reutiliza el warmup MPC
from src.training.action_noise import construir_action_noise

def make_model(train_env, seed):                          # TD3 (DDPG análogo sin policy_delay)
    c = _VCFG
    return TD3(policy='MlpPolicy', env=train_env, ...,
               policy_delay=int(c['policy_delay']),
               target_policy_noise=float(c['target_policy_noise']),
               target_noise_clip=float(c['target_noise_clip']),
               action_noise=construir_action_noise(c.get('action_noise'), 4),
               learning_starts=0)
def warmup(model, train_env, seed):
    dawn_warmup(model, _get_mpc(), train_env, warmup_steps=int(_VCFG['dawn_warmup_steps']), seed=seed)
# entrenar_multiseed(algo="td3_residual"|"ddpg_residual", ...)
```

### `src/main_sac_puro.py` (NUEVO) — F5, SAC continuo SIN MPC
```python
def make_envs(seed):
    # EnergyEnvContinuo DIRECTO (Box-4, obs 108, sin ResidualEnv)
    return VecNormalize(DummyVecEnv([lambda: Monitor(EnergyEnvContinuo(...,'train'))]), ...), ...
def make_model(train_env, seed):
    c = _VCFG
    return SAC(..., learning_starts=int(c['learning_starts']), ...)   # 10k, sin DAWN
# entrenar_multiseed(algo="sac_puro", ...)  — sin warmup, sin extra_callbacks residuales
```

### `src/main_ppo_continuo.py` (NUEVO) — familia C, PPO continuo
```python
def make_model(train_env, seed):
    c = _VCFG
    log_std_init = math.log(float(c["action_std_init"])) if "action_std_init" in c else c["log_std_init"]
    return PPO(..., policy_kwargs={"net_arch": {"pi": c["net_arch_pi"], "vf": c["net_arch_vf"]},
                                   "log_std_init": log_std_init})
# reutiliza EntCoefScheduler de main_ppo para el decay
```

### Post-Fase 3 — `dawn_warmup` movido a `src/training/` (`e8266e1`)
Estaba dentro de `main_residual_sac.py` y TD3/DDPG lo importaban **desde ese main**
(pequeño *smell*: importar lógica de un `main_`). Movido a `src/training/dawn_warmup.py`
y parametrizado (`delta_max`, `residual_mode`); los 3 mains residuales lo importan ahora
del módulo compartido. Verificado: `residual_sac.dawn_warmup is td3.dawn_warmup is ddpg.dawn_warmup`.
```python
# src/training/dawn_warmup.py (NUEVO) — pre-llena el buffer con MPC puro (delta=0)
def dawn_warmup(model, mpc, train_env, warmup_steps, seed=42,
                delta_max=0.15, residual_mode='mult'):
    renv = ResidualEnv(EnergyEnvContinuo(...), mpc, delta_max=delta_max, residual_mode=residual_mode)
    for _ in range(warmup_steps):
        next_obs, r, term, trunc, info = renv.step([0,0,0,0])   # delta=0 → acción MPC
        model.replay_buffer.add(obs, next_obs, action, r, term, [info])
    train_env.obs_rms.mean/var = ...   # calibra VecNormalize con las obs del warmup
```

## Fase 5 — eval_suite + estadística (`638b300`)

### `src/reporting/estadistica.py` (NUEVO)
```python
def ic95_bootstrap(x, n_boot=10000, seed=0):    # IC95% de la media por bootstrap
    idx = rng.integers(0, x.size, size=(n_boot, x.size)); means = x[idx].mean(1)
    return percentile(means, 2.5), percentile(means, 97.5)
def wilcoxon_vs(x, base):                        # Wilcoxon pareado (mismo (seed,semana))
    if np.allclose(x-base, 0): return 1.0
    return stats.wilcoxon(x, base).pvalue
```

### `src/eval_suite.py` (NUEVO) — dispatcher por familia
```python
_ALGO_SB3 = {"residual_sac":"SAC","td3_residual":"TD3","ddpg_residual":"DDPG",
             "sac_puro":"SAC","ppo_continuo":"PPO","dqn":"DQN","ppo":"PPO"}

def _construir_controller(familia, cfg, model_path, vecnorm, sim, mpc):
    if cfg["env"]=="discreto": return DiscreteRLController(model_path, sim, vecnorm, algo=_ALGO_SB3[familia])
    if cfg["env"]=="continuo": return ContinuousController(model_path, sim, vecnorm, algo=_ALGO_SB3[familia])
    if cfg["env"]=="residual": return ResidualSACController(model_path, mpc, sim, cfg["delta_max"],
                                       vecnorm, algo=_ALGO_SB3[familia], residual_mode=cfg.get("residual_mode","mult"))

def evaluar_version(familia, version, eval_seeds):
    # localiza models/best/{algo}_{version}.zip + _vecnorm.pkl ; F6 fija σ
    res = evaluar_multiseed(ctrl, "realista", eval_seeds)        # reusa eval_unificada
    lo, hi = ic95_bootstrap(res["bens_marg"]); media = res["bens_marg"].mean()
    base = _baseline_mpc_pooled(eval_seeds)                       # MPC realista (cacheado)
    # escribe fila CSV: media, std_seeds, ic95, vs_mpc, wilcoxon_p, mejor_seed, hiperparametros...

def evaluar_baselines(eval_seeds):    # G1 MPC oráculo, G2 MPC realista, G3 heurístico, G4 IDLE
```
Salida: `resultados/resultados.csv` (1 fila/versión, 15 columnas) + `resultados/{algo}_{version}.json`.
`resultados/.gitignore` ignora csv/json/npy (regenerables). Verificado end-to-end con artefacto real.

## Fase 6 + Fase 0 — reparto + doc (`c67dd6b`)

### `experimentos/run_familia.sh` (NUEVO) — una orden por PC
```bash
PYTHON=.venv/bin/python ./experimentos/run_familia.sh <familia>   # o 'G' para baselines
# itera las versiones de la familia: entrena (3 semillas) + evalúa (10) + añade fila al CSV
# soporta --solo-eval (no reentrena)
```

### `experimentos/manifiesto.md` (NUEVO)
Orden (G→A,B,C,F en paralelo→D,F5→E), coste/seed por familia, **mapeo del run 80kwh en curso →
SAC-C** (renombrar `residual_sac_80kwh.*` → `residual_sac_SAC-C.*`, sin reentrenar), y cómo unir
los `resultados.csv` de varias máquinas.

### `doc/CAMBIOS_ENTORNO.md` (NUEVO)
Justifica invalidar los resultados previos: 80 kWh desde config, 42 kWp, ruido sin lookahead
(k=0), fuente única de forecast/obs, física de 4 flujos unificada, MPC recalibrado. + protocolo
de eval definitivo (50 semanas held-out, 3 semillas entreno + 10 ruido, IC95 + Wilcoxon).

---

# 4. CÓMO LANZAR LOS ENTRENOS

```bash
# 1) baselines (primero, bloquean las métricas relativas)
PYTHON=.venv/bin/python ./experimentos/run_familia.sh G
# 2) una familia por ordenador
PYTHON=.venv/bin/python ./experimentos/run_familia.sh dqn
PYTHON=.venv/bin/python ./experimentos/run_familia.sh ppo
PYTHON=.venv/bin/python ./experimentos/run_familia.sh ppo_continuo
PYTHON=.venv/bin/python ./experimentos/run_familia.sh td3_residual
PYTHON=.venv/bin/python ./experimentos/run_familia.sh ddpg_residual
PYTHON=.venv/bin/python ./experimentos/run_familia.sh residual_sac
PYTHON=.venv/bin/python ./experimentos/run_familia.sh sac_puro
```
> `python` pelado no está en PATH en este entorno → usar `.venv/bin/python`.

El **Residual SAC 80 kWh** ya en curso = SAC-C: al terminar, renómbralo (manifiesto) y NO lo reentrenes.

---

# 5. VERIFICACIÓN

- **Smokes por fase** (todos verdes): registry resuelve base⊕override; δ=0 ⇒ passthrough MPC en
  mult y add; DQN-2 γ=0.995⁸ + NStep; PPO decay vs const; los 4 mains nuevos construyen modelo;
  eval_suite end-to-end (artefacto real → controller → fila CSV completa); `run_familia.sh` lista
  y salta versiones sin artefacto.
- **Suite de tests completa: 51 passed, 2 skipped** (los 2 skip = ONNX-equivalence, faltan los
  artefactos; se reactivan al exportar el SAC 80 kWh). Ningún refactor rompió imports.

---

# 6. INVENTARIO DE FICHEROS

**Nuevos:**
`config/experimentos.yaml`, `src/training/registro.py`, `src/training/action_noise.py`,
`src/training/dawn_warmup.py`,
`src/controllers/residual_base.py`, `src/controllers/continuous_controller.py`,
`src/main_ppo_continuo.py`, `src/main_sac_puro.py`, `src/main_td3_residual.py`,
`src/main_ddpg_residual.py`, `src/eval_suite.py`, `src/reporting/__init__.py`,
`src/reporting/estadistica.py`, `experimentos/run_familia.sh`, `experimentos/manifiesto.md`,
`doc/CAMBIOS_ENTORNO.md`, `resultados/.gitignore`.

**Modificados:**
`src/core/simulador.py` (80 kWh), `src/main_dqn.py`, `src/main_ppo.py`,
`src/main_residual_sac.py` (renombrado de `main_sac.py`), `src/envs/residual_env.py`,
`src/controllers/residual_sac_controller.py`, `src/production/onnx_inference.py`,
`tests/test_energy_env_continuo.py`, `README.md`, `doc/especificacion_sistema_sgec.md`.

**Eliminados:**
`src/utils/generador_datos.py`, `src/benchmarks/diagnostico_recompensa.py`.

---

# 7. LAS UNIFICACIONES — QUÉ SON Y PARA QUÉ SIRVEN

A lo largo del proyecto se fue **sacando a un solo sitio** todo lo que varios componentes
hacían por separado. Cada pieza compartida es una **fuente única**: imposible que dos caminos
diverjan, menos código que mantener y comparaciones justas entre algoritmos. Con la suite,
ahora las usan **más modelos**.

## 7.1 Tabla (con cuántos componentes la usan ahora)

| Unificación | Fichero | La usan ahora | Para qué |
|---|---|---|---|
| **Registry de experimentos** | `config/experimentos.yaml` + `training/registro.py` | **7 mains** | fuente única de hiperparámetros de las 31 versiones; alimenta entreno y tablas |
| **Runner de entreno** | `training/multiseed.py` (`entrenar_multiseed`) | **7 mains** | el bucle de 3 semillas + elegir mejor + guardar artefacto, idéntico para todo algoritmo |
| **Fuente de pronóstico** | `core/forecast.py` (`ventana_observada` + ruido) | envs, eval, MPC, producción | la ventana y el ruido en 1 sitio → todos ven lo mismo, sin fuga |
| **Observación** | `production/obs_builder.py` (`build_obs`) | **5 componentes** | el vector de 108 dims en 1 función → entreno = eval = producción |
| **Física de batería** | `core/simulador.py` (`aplicar_fisica_4flujos`) | env continuo + MPC | SAC y MPC con la MISMA física (verificado byte-idéntico) |
| **DAWN warmup** | `training/dawn_warmup.py` | **3 mains** (SAC, TD3, DDPG) | pre-llenar el buffer con MPC puro, idéntico para los off-policy |
| **Ruido de exploración** | `training/action_noise.py` | **2 mains** (TD3, DDPG) | construir el ruido OU/Normal/none sin duplicar |
| **Callbacks de métricas** | `training/callbacks.py` (`DiscreteMetricasCallback`) | **2 mains** (DQN, PPO) | loguear métricas sin reescribir el bucle |
| **Callback de evaluación** | `training/multiseed.py` (`SeededEvalCallback`) | **7 mains** (vía el runner) | evaluar reproducible + guardar el mejor, igual para todos |
| **Controller residual** | `controllers/residual_base.py` (`ResidualControllerBase`) | base + 2 subclases (torch/onnx) | eval == producción por construcción |

## 7.2 ¿Qué es un *Callback*? (para qué sirve)

En Stable-Baselines3 (la librería de entreno), el bucle de entrenamiento está cerrado: tú no lo
escribes. Un **callback es un pequeño objeto que ese bucle llama automáticamente cada cierto
número de pasos** para hacer tareas laterales **sin tocar el núcleo del entreno**. Es un
"gancho": en vez de reescribir el bucle, enchufas comportamientos.

En el proyecto hay tres, y la gracia de unificarlos es que su lógica vive **una sola vez**:
- **`SeededEvalCallback`** (en `multiseed.py`): cada `eval_freq` pasos, re-fija la semilla y
  evalúa el modelo sobre las **mismas 50 semanas**, guardando el mejor. Lo usan **todos** los
  algoritmos a través del runner → todos se evalúan y se seleccionan igual.
- **`DiscreteMetricasCallback`** (en `callbacks.py`): loguea métricas propias de DQN/PPO
  (epsilon, prints periódicos) a TensorBoard. **Compartido por DQN y PPO** (antes cada uno tenía el suyo).
- **`EntCoefScheduler`**: cada paso baja el coeficiente de entropía (decay). Se enchufa solo si
  la versión pide decay.

Sin esto, cada algoritmo tendría que reimplementar "evalúa cada N pasos y guarda el mejor" y
"loguea estas métricas" → código repetido que se desincroniza. Unificado, **el cómo se evalúa y
se mide es el mismo para todos**.

## 7.3 ¿Qué es `ControllerBase` / `ResidualControllerBase`? (para qué sirve)

Un **controller** es "el que decide": tiene un método `solve(estado, pronóstico) → acción` y se
usa en **evaluación y producción** (no en el entreno, que lo lleva SB3). Todos los controllers
(MPC, SAC, heurístico…) exponen ese mismo `solve` para poder compararlos en igualdad.

El SAC residual existe en **dos versiones que hacen exactamente los mismos pasos** y solo cambian
en el "motor" que ejecuta la red:
- **torch** (`ResidualSACController`) → para evaluar el `.zip` en el PC,
- **onnx** (`OnnxResidualController`) → para el aparato real (sin torch).

Esos pasos son: resolver el MPC → construir la observación → normalizar → pedir el `delta` a la
red → combinar `flujo = max(0, mpc·(1+δ·δmax))`. **`ResidualControllerBase` es la clase padre que
contiene TODOS esos pasos en su `solve()`**; cada hija solo rellena el hueco `_infer()` (su motor).
Antes esos pasos estaban **copiados en los dos ficheros** (coincidían solo porque un test lo
comprobaba); ahora viven **una vez** → es **imposible que evaluación y producción difieran**. Con
la suite, esa misma base sirve además para **TD3 y DDPG residuales** (solo cambia la clase SB3 que
carga el `.zip`), así que un único `solve()` cubre las tres familias residuales.

En una frase: **el Callback evita reescribir el bucle de entreno; el ControllerBase evita
reescribir la lógica de decisión** — los dos son "escríbelo una vez, úsalo en muchos sitios".
