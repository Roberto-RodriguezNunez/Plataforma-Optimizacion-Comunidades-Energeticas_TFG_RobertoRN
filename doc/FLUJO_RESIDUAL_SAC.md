# Flujo completo del Residual SAC — diagramas UML (PlantUML)

Recorrido de **todo el ciclo de vida** del Residual SAC y de cómo va reutilizando el **código
unificado (núcleo)**, desde los datasets crudos hasta la facturación en el SaaS. Para evitar
líneas cruzadas, el ciclo se parte en **7 diagramas** (uno por fase + vista general + núcleo).

> **Cómo verlos:** son bloques PlantUML. Pégalos en [planttext.com](https://www.planttext.com),
> o usa la extensión *PlantUML* de VSCode (Alt+D), o `plantuml.jar`.

## Leyenda (estereotipos / colores)
- 🟡 `<<unificado>>` — módulo **reutilizado por varias fases** (el foco: una pieza, muchos usuarios).
- 🔵 `<<dato>>` — ficheros de datos (csv/json/yaml). 🟢 `<<artefacto>>` — modelos (zip/pkl/onnx/npz).
- 🟠 `<<entrada>>` — scripts de entrada (mains/eval/edge). 🟣 `<<saas>>` — piezas Flask del SaaS.

---

## D0 — Vista general (las 6 fases + el núcleo)

```plantuml
@startuml D0_overview
skinparam packageStyle rectangle
skinparam package {
  BackgroundColor<<nucleo>> #FFF3B0
}
package "1 · Datos / ETL"            as P1
package "2 · Entreno"                as P2
package "3 · Evaluación"             as P3
package "4 · Exportación ONNX"       as P4
package "5 · Producción / Edge"      as P5
package "6 · SaaS (facturación)"     as P6
package "Núcleo unificado"           as NU <<nucleo>>

P1 --> P2 : dataset_final.csv
P2 --> P3 : models/best/*.zip
P2 --> P4 : best_model.zip
P4 --> P5 : .onnx + .npz
P5 --> P6 : HTTP POST
P2 ..> NU : usa
P3 ..> NU : usa
P5 ..> NU : usa
@enduml
```

---

## D1 — Datos / ETL

```plantuml
@startuml D1_datos
skinparam componentStyle rectangle
skinparam component {
  BackgroundColor<<dato>> #CDE7FF
  BackgroundColor<<entrada>> #FFE0B2
}
component "consumo_esios_2021_2023.csv"               as C   <<dato>>
component "precios_esios_2021_2023.csv"               as PR  <<dato>>
component "compensacion_autoconsumo_..._2023.csv"     as CO  <<dato>>
component "solar_pvgis_2021_2023.csv"                 as SO  <<dato>>
component "config/system.yaml"                        as CFG <<dato>>
component "utils/generar_dataset_final.py"            as ETL <<entrada>>
component "data/processed/dataset_final.csv"          as DS  <<dato>>
component "data/processed/metadata.json"              as MD  <<dato>>

C  --> ETL
PR --> ETL
CO --> ETL
SO --> ETL
CFG --> ETL
ETL --> DS  : escribe
ETL --> MD  : hashes/params
@enduml
```

---

## D2 — Entreno del Residual SAC

```plantuml
@startuml D2_entreno
skinparam componentStyle rectangle
skinparam component {
  BackgroundColor<<unificado>> #FFF3B0
  BackgroundColor<<dato>> #CDE7FF
  BackgroundColor<<artefacto>> #C8E6C9
  BackgroundColor<<entrada>> #FFE0B2
}
component "main_residual_sac.py\n(make_envs/make_model/warmup)" as MRS <<entrada>>

package "Registry" {
  component "config/experimentos.yaml" as EXP <<dato>>
  component "training/registro.py"     as REG <<unificado>>
}
package "Runner" {
  component "training/multiseed.py\n(entrenar_multiseed + SeededEvalCallback)" as MS <<unificado>>
  component "training/callbacks.py"    as CB  <<unificado>>
  component "training/dawn_warmup.py"  as DW  <<unificado>>
}
package "Entornos" {
  component "envs/residual_env.py"          as RE  <<unificado>>
  component "envs/energy_env_continuo.py"   as EEC <<unificado>>
  component "envs/energy_env.py"            as EE  <<unificado>>
}
package "MPC + Núcleo" {
  component "benchmarks/mpc_benchmark.py" as MPC <<unificado>>
  component "core/simulador.py"           as SIM <<unificado>>
  component "core/forecast.py"            as FC  <<unificado>>
  component "production/obs_builder.py"   as OB  <<unificado>>
}
component "data/processed/dataset_final.csv" as DS <<dato>>
component "config/system.yaml"               as CFG <<dato>>
component "models/best/residual_sac_SAC-C.zip\n+ _vecnorm.pkl + _seeds.json" as ART <<artefacto>>
component "logs/ (TensorBoard)"              as LOG <<artefacto>>

EXP --> REG
MRS --> REG : cargar_version
MRS --> MS  : entrenar_multiseed
MRS --> CB
MRS --> DW
MRS --> RE
MRS --> MPC : crear_mpc
RE  --> EEC
EEC --> EE
DW  --> RE  : transiciones MPC puro
EE  --> SIM
EE  --> FC
EE  --> OB
MPC --> SIM
MPC --> FC
SIM --> DS  : lee
SIM --> CFG : batería 80 kWh
MS  --> ART : guarda el mejor
MS  --> LOG
@enduml
```

---

## D3 — Evaluación (eval_suite)

```plantuml
@startuml D3_eval
skinparam componentStyle rectangle
skinparam component {
  BackgroundColor<<unificado>> #FFF3B0
  BackgroundColor<<dato>> #CDE7FF
  BackgroundColor<<artefacto>> #C8E6C9
  BackgroundColor<<entrada>> #FFE0B2
}
component "eval_suite.py"            as ES  <<entrada>>
component "eval_unificada.py\n(evaluar_multiseed)" as EU <<unificado>>
component "reporting/estadistica.py\n(IC95 + Wilcoxon)" as ST <<unificado>>
component "training/registro.py"     as REG <<unificado>>

package "Controllers" {
  component "controllers/residual_sac_controller.py" as RSC <<unificado>>
  component "controllers/residual_base.py"           as RB  <<unificado>>
  component "controllers/base.py"                    as BC  <<unificado>>
}
package "Núcleo" {
  component "production/obs_builder.py"   as OB  <<unificado>>
  component "benchmarks/mpc_benchmark.py" as MPC <<unificado>>
  component "core/simulador.py"           as SIM <<unificado>>
  component "core/forecast.py"            as FC  <<unificado>>
}
component "models/best/residual_sac_SAC-C.zip\n+ _vecnorm.pkl" as ART <<artefacto>>
component "resultados/resultados.csv\n+ {algo}_{version}.json" as RES <<dato>>

ES --> EU
ES --> RSC
ES --> REG
ES --> ST
RSC --> RB
RB  --> BC
RB  --> OB
RB  --> MPC : resuelve MPC base
RSC --> ART : carga modelo
EU  --> MPC
EU  --> SIM
EU  --> FC
ES  --> RES : escribe fila
@enduml
```

---

## D4 — Exportación a ONNX

```plantuml
@startuml D4_onnx
skinparam componentStyle rectangle
skinparam component {
  BackgroundColor<<artefacto>> #C8E6C9
  BackgroundColor<<entrada>> #FFE0B2
}
component "models/best_model.zip"          as ZIP <<artefacto>>
component "models/best_vecnormalize.pkl"   as PKL <<artefacto>>
component "production/export_onnx.py"      as EXO <<entrada>>
component "models/residual_sac_actor.onnx" as ONNX <<artefacto>>
component "models/vec_normalize_v5_1M.npz" as NPZ  <<artefacto>>

ZIP --> EXO
PKL --> EXO
EXO --> ONNX : actor tanh(mean), opset 17
EXO --> NPZ  : mean/var/clip
@enduml
```

---

## D5 — Producción / Edge (inferencia)

```plantuml
@startuml D5_edge
skinparam componentStyle rectangle
skinparam component {
  BackgroundColor<<unificado>> #FFF3B0
  BackgroundColor<<dato>> #CDE7FF
  BackgroundColor<<artefacto>> #C8E6C9
  BackgroundColor<<entrada>> #FFE0B2
  BackgroundColor<<saas>> #E1D5F5
}
component "production/edge_loop.py"        as EL  <<entrada>>
component "api/data_feed.py"               as DF
component "production/onnx_inference.py\n(OnnxResidualController)" as ONI
component "controllers/residual_base.py"   as RB  <<unificado>>
component "benchmarks/mpc_benchmark.py"    as MPC <<unificado>>
component "production/obs_builder.py"      as OB  <<unificado>>
component "core/simulador.py"              as SIM <<unificado>>
component "core/forecast.py"               as FC  <<unificado>>
component "models/residual_sac_actor.onnx" as ONNX <<artefacto>>
component "models/vec_normalize_v5_1M.npz" as NPZ  <<artefacto>>
component "data/processed/dataset_final.csv" as DS <<dato>>
component "POST /api/edge/decision"        as EPD <<saas>>
component "POST /api/edge/generar-cierre"  as EPC <<saas>>

EL  --> DF  : get_feed
EL  --> ONI
EL  --> MPC : crear_mpc
ONI --> RB
RB  --> OB
ONI --> ONNX : carga
ONI --> NPZ
DF  --> SIM
DF  --> FC
DF  --> DS
EL  --> EPD : flujos por hora
EL  --> EPC : cierre mensual
@enduml
```

---

## D6 — SaaS (solo el camino del SAC: edge → operación → cierre → display)

```plantuml
@startuml D6_saas
skinparam componentStyle rectangle
skinparam component {
  BackgroundColor<<saas>> #E1D5F5
  BackgroundColor<<dato>> #CDE7FF
}
component "POST /api/edge/decision"        as EPD <<saas>>
component "POST /api/edge/generar-cierre"  as EPC <<saas>>

package "App factory" {
  component "app/__init__.py"     as APP <<saas>>
  component "config.py\n(src/saas/config.py)" as CFG <<saas>>
  component "app/extensions.py\n(db)" as DB <<saas>>
}
package "Edge" {
  component "modules/edge/routes.py\n(recibir_decision / generar_cierre_edge)" as ER <<saas>>
}
package "Cierre" {
  component "modules/cierres/routes.py\n(_calcular_cierre_desde_operaciones)" as CR <<saas>>
}
package "Modelos (ORM)" {
  component "models/operacion.py\n(OperacionHoraria)" as MOP <<saas>>
  component "models/bateria.py"   as MBA <<saas>>
  component "models/cierre.py\n(CierreMensual)" as MCI <<saas>>
  component "models/vivienda.py"  as MVI <<saas>>
  component "models/comunidad.py" as MCO <<saas>>
  component "models/acceso.py"    as MAC <<saas>>
}
package "Display" {
  component "modules/comunidades/routes.py" as RCO <<saas>>
  component "modules/viviendas/routes.py"   as RVI <<saas>>
  component "modules/baterias/routes.py"    as RBA <<saas>>
  component "templates/*/detalle.html\n(+ base.html)" as TPL <<saas>>
  component "static/js/{charts,ajax}.js"    as JS  <<saas>>
}

APP --> ER : registra blueprint
APP --> CR
DB  --> MOP
EPD --> ER
ER  --> MOP : upsert OperacionHoraria
ER  --> MBA : actualiza SoC
EPC --> ER
ER  --> CR  : genera cierre del mes
CR  --> MOP : lee operaciones
CR  --> MCI : escribe CierreMensual
CR  --> MVI
CR  --> MCO
CR  --> MAC
RCO --> MCI
RVI --> MCI
RBA --> MBA
RCO --> TPL
RVI --> TPL
RBA --> TPL
TPL --> JS
@enduml
```

---

## D7 — El núcleo unificado (una pieza, muchos usuarios)

Muestra de un vistazo cómo **las distintas fases entran a los mismos módulos** → entreno = evaluación
= producción ven exactamente lo mismo.

```plantuml
@startuml D7_nucleo
skinparam componentStyle rectangle
skinparam component {
  BackgroundColor<<unificado>> #FFF3B0
  BackgroundColor<<fase>> #FFE0B2
}
component "FASE Entreno\n(main_residual_sac)"      as F2 <<fase>>
component "FASE Evaluación\n(eval_suite/eval_unificada)" as F3 <<fase>>
component "FASE Producción\n(edge_loop/onnx_inference)"  as F5 <<fase>>

package "NÚCLEO UNIFICADO" {
  component "core/forecast.py\n(ventana + ruido)"       as FC  <<unificado>>
  component "production/obs_builder.py\n(build_obs 108)" as OB  <<unificado>>
  component "core/simulador.py\n(aplicar_fisica_4flujos)" as SIM <<unificado>>
  component "benchmarks/mpc_benchmark.py\n(crear_mpc)"   as MPC <<unificado>>
  component "controllers/residual_base.py\n(solve)"     as RB  <<unificado>>
  component "training/registro.py"                      as REG <<unificado>>
  component "training/multiseed.py"                     as MS  <<unificado>>
  component "training/callbacks.py"                     as CB  <<unificado>>
  component "training/dawn_warmup.py"                   as DW  <<unificado>>
}

F2 --> REG
F2 --> MS
F2 --> CB
F2 --> DW
F2 --> SIM
F2 --> FC
F2 --> OB
F2 --> MPC
F3 --> REG
F3 --> RB
F3 --> SIM
F3 --> FC
F3 --> OB
F3 --> MPC
F5 --> RB
F5 --> SIM
F5 --> FC
F5 --> OB
F5 --> MPC
@enduml
```

---

## D8 — TODO JUNTO (diagrama único, fin a fin)

El ciclo completo en un solo diagrama **legible**: cada fase es **una caja** con sus ficheros
listados dentro, encadenadas de arriba abajo (1 → 2 → … → 6). El **Núcleo unificado** (amarillo)
es una caja aparte a la que bajan Entreno, Evaluación y Producción (una sola flecha cada una). El
detalle fichero-a-fichero (con todas las flechas internas) está en D1–D7.

```plantuml
@startuml D8_todo_junto
skinparam shadowing false
skinparam ranksep 28
skinparam nodesep 22
skinparam rectangle {
  BackgroundColor<<unificado>> #FFF3B0
  BackgroundColor<<dato>> #CDE7FF
  BackgroundColor<<art>> #C8E6C9
  BackgroundColor<<entrada>> #FFE0B2
  BackgroundColor<<saas>> #E1D5F5
}

rectangle "1 · DATOS / ETL\l--------\ldata/raw/* (4 csv ESIOS/PVGIS)\lconfig/system.yaml\lutils/generar_dataset_final.py\l=> data/processed/dataset_final.csv\l" as P1 <<dato>>

rectangle "2 · ENTRENO\l--------\lmain_residual_sac.py   ·   config/experimentos.yaml\lenvs: residual_env -> energy_env_continuo -> energy_env\l=> models/best/residual_sac_SAC-C.zip (+_vecnorm.pkl +_seeds.json)\l" as P2 <<entrada>>

rectangle "3 · EVALUACION\l--------\leval_suite.py -> eval_unificada.py\lcontrollers/residual_sac_controller.py\lreporting/estadistica.py\l=> resultados/resultados.csv\l" as P3 <<entrada>>

rectangle "4 · EXPORT ONNX\l--------\lproduction/export_onnx.py\l=> residual_sac_actor.onnx + vec_normalize_v5_1M.npz\l" as P4 <<art>>

rectangle "5 · PRODUCCION / EDGE\l--------\lproduction/edge_loop.py\lapi/data_feed.py\lproduction/onnx_inference.py\l" as P5 <<entrada>>

rectangle "6 · SaaS (camino SAC)\l--------\lmodules/edge/routes.py -> operacion.py / bateria.py\lmodules/cierres/routes.py -> cierre.py / vivienda / comunidad / acceso\lmodules display (comunidades/viviendas/baterias) -> templates + static/js\l" as P6 <<saas>>

rectangle "NUCLEO UNIFICADO  (lo usan Entreno, Evaluacion y Produccion)\l--------\lcore/simulador.py · core/forecast.py · production/obs_builder.py\lbenchmarks/mpc_benchmark.py · controllers/residual_base.py\ltraining/registro.py · multiseed.py · callbacks.py · dawn_warmup.py\l" as NU <<unificado>>

P1 --> P2 : dataset_final.csv
P2 --> P3 : models/best/*.zip
P2 --> P4 : best_model.zip + .pkl
P4 --> P5 : .onnx + .npz
P5 --> P6 : HTTP POST
P2 --> NU
P3 --> NU
P5 --> NU
@enduml
```

---

## Tabla índice — todas las piezas que toca el ciclo

`U` = unificado (reutilizado por varias fases).

### Datos / config (7)
| Fichero | Fase | Rol |
|---|---|---|
| `data/raw/consumo_esios_2021_2023.csv` | Datos | demanda horaria cruda (ESIOS) |
| `data/raw/precios_esios_2021_2023.csv` | Datos | PVPC crudo |
| `data/raw/compensacion_autoconsumo_esios_2021_2023.csv` | Datos | precio excedente |
| `data/raw/solar_pvgis_2021_2023.csv` | Datos | generación solar (PVGIS) |
| `data/processed/dataset_final.csv` | Datos→Entreno/Eval/Prod | dataset final (entrada del simulador) |
| `data/processed/metadata.json` | Datos | parámetros + hashes |
| `config/system.yaml` | **U** todas | config (batería 80 kWh, ruido, MPC, split) |

### Núcleo unificado (9)
| Fichero | Fases | Rol |
|---|---|---|
| `src/core/simulador.py` | **U** Entreno/Eval/Prod | física de batería (`aplicar_fisica_4flujos`) |
| `src/core/forecast.py` | **U** Entreno/Eval/Prod | ventana + ruido (`ventana_observada`) |
| `src/production/obs_builder.py` | **U** Entreno/Eval/Prod | observación de 108 dims (`build_obs`) |
| `src/benchmarks/mpc_benchmark.py` | **U** Entreno/Eval/Prod | MPC LP (`crear_mpc`, `LinearMPC`) |
| `src/controllers/residual_base.py` | **U** Eval/Prod | `solve()` residual común (torch/onnx) |
| `src/training/registro.py` | **U** Entreno/Eval | loader de la registry |
| `src/training/multiseed.py` | **U** Entreno | runner + `SeededEvalCallback` |
| `src/training/callbacks.py` | **U** Entreno | callbacks de métricas + `EntCoefScheduler` |
| `src/training/dawn_warmup.py` | **U** Entreno | warmup MPC del buffer |

### Entornos + buffers (4)
| Fichero | Fase | Rol |
|---|---|---|
| `src/envs/residual_env.py` | Entreno | wrapper residual (mult/add) sobre el MPC |
| `src/envs/energy_env_continuo.py` | Entreno | entorno continuo Box-4 (hereda del discreto) |
| `src/envs/energy_env.py` | Entreno | entorno base (obs, reset, step) |
| `src/buffers/nstep_replay_buffer.py` | Entreno (DQN) | buffer N-step (no lo usa el SAC; del registry general) |

### Entreno (entrada + registry) (3)
| Fichero | Fase | Rol |
|---|---|---|
| `src/main_residual_sac.py` | Entreno | entrada: factorías + lanza `entrenar_multiseed` |
| `config/experimentos.yaml` | Entreno/Eval | hiperparámetros de cada versión (SAC-C…) |
| `src/training/action_noise.py` | Entreno (TD3/DDPG) | ruido exploración (no SAC; del núcleo de la suite) |

### Controllers (6)
| Fichero | Fase | Rol |
|---|---|---|
| `src/controllers/base.py` | **U** Eval/Prod | interfaz `solve()` |
| `src/controllers/residual_sac_controller.py` | Eval | carga el `.zip` torch (SAC/TD3/DDPG) |
| `src/controllers/discrete_rl_controller.py` | Eval (DQN/PPO) | controller discreto |
| `src/controllers/continuous_controller.py` | Eval (PPO-cont/SAC-puro) | continuo sin MPC |
| `src/controllers/heuristic_controller.py` | Eval (baseline) | reglas if-then |
| `src/controllers/residual_base.py` | (ver núcleo) | base común |

### Evaluación (4)
| Fichero | Fase | Rol |
|---|---|---|
| `src/eval_suite.py` | Eval | dispatcher por familia + estadística → CSV |
| `src/eval_unificada.py` | **U** Eval | motor de evaluación (50 semanas, multiseed) |
| `src/reporting/estadistica.py` | Eval | IC95 bootstrap + Wilcoxon |
| `resultados/resultados.csv` | Eval (salida) | una fila por versión (→ tablas LaTeX) |

### Producción / Edge (4 + artefactos)
| Fichero | Fase | Rol |
|---|---|---|
| `src/production/export_onnx.py` | Export | `.zip`+`.pkl` → `.onnx`+`.npz` |
| `src/production/edge_loop.py` | Prod | bucle edge: decide y postea al SaaS |
| `src/production/onnx_inference.py` | Prod | `OnnxResidualController` (ONNX) |
| `src/api/data_feed.py` | Prod | feed de datos (histórico/simulado) |
| `models/best_model.zip` · `best_vecnormalize.pkl` | artefactos | modelo torch + normalización |
| `models/residual_sac_actor.onnx` · `vec_normalize_v5_1M.npz` | artefactos | modelo edge + stats |

### SaaS — solo el camino del SAC (~22)
> Rutas relativas a `src/saas/` (el paquete Flask se llama `app`; `config.py` está en `src/saas/config.py`).

| Fichero | Rol |
|---|---|
| `app/__init__.py` · `config.py` · `app/extensions.py` | app factory + `db` |
| `src/saas/run.py` · `seed.py` | arranque + carga inicial de datos |
| `app/modules/edge/routes.py` | recibe `/decision` y `/generar-cierre` del edge |
| `app/modules/cierres/routes.py` (+ `forms.py`) | `_calcular_cierre_desde_operaciones` |
| `app/modules/comunidades/routes.py` · `viviendas/routes.py` · `baterias/routes.py` | display/facturación |
| `app/models/operacion.py` (`OperacionHoraria`) | guarda cada hora del edge |
| `app/models/bateria.py` · `cierre.py` (`CierreMensual`) | estado batería + cierre mensual |
| `app/models/vivienda.py` · `comunidad.py` · `acceso.py` | reparto por vivienda |
| `app/helpers.py` · `decorators.py` | utilidades/permisos |
| `app/templates/{cierres,comunidades,baterias}/detalle.html` · `base.html` | vistas |
| `app/static/js/charts.js` · `ajax.js` | gráficas + interacción |

> **Lectura del ciclo en una línea:** datos crudos → `dataset_final.csv` → **entreno** (main_residual_sac
> usa todo el núcleo + `multiseed`) → `models/best/*.zip` → **eval** (eval_suite, mismo núcleo) →
> **export** a `.onnx` → **edge** (onnx_inference, mismo núcleo) → **POST** al **SaaS** → operación →
> cierre mensual → facturación por vivienda. El **núcleo unificado** (D7) es el que atraviesa entreno,
> eval y producción garantizando que ven lo mismo.
