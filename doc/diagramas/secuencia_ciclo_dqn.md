# Explicación Detallada — Diagrama de Secuencia del Ciclo DQN

Este documento describe paso a paso el diagrama de secuencia `secuencia_ciclo_dqn.puml`, que modela el flujo completo de entrenamiento del agente DQN dentro del sistema SGEC.

---

## 1. Participantes

El diagrama define cinco líneas de vida (lifelines), cada una representando un componente del sistema:

| Participante | Componente real | Responsabilidad |
|---|---|---|
| **main.py (Orquestador)** | `src/main.py` | Punto de entrada. Instancia el entorno y el agente, lanza el entrenamiento y exporta el modelo final. |
| **DQN Agent (Stable Baselines3)** | Clase `DQN` de la librería `stable-baselines3` | Implementa el algoritmo Deep Q-Network: red neuronal Q, política ε-greedy, replay buffer y actualización de pesos. |
| **EnergyEnv (gymnasium.Env)** | `src/envs/energy_env.py` | Entorno Gymnasium que envuelve al simulador. Traduce acciones discretas del agente en llamadas al motor físico y construye las observaciones de 76 variables. |
| **ComunidadSimulador (Motor Físico)** | `src/core/simulador.py` | Gemelo Digital de la comunidad energética. Ejecuta la física de la batería (flujos energéticos, SoC, degradación) sin conocer al agente. |
| **dataset_final.csv** | `data/processed/dataset_final.csv` | Fuente de datos estática. Matriz de 8760 filas × 3 columnas: `[consumo_total, generacion_total, precio_kwh]` para todo el año 2023. |

---

## 2. Fase de Inicialización

```
MAIN -> ENV : EnergyEnv()
  ENV -> SIM : ComunidadSimulador('data/processed/dataset_final.csv')
    SIM -> DATA : pd.read_csv()
    DATA --> SIM : DataFrame (8760 × 3)
  SIM --> ENV : instancia lista
ENV --> MAIN : entorno instanciado

MAIN -> DQN : DQN('MlpPolicy', env)
DQN --> MAIN : agente configurado
```

### Qué ocurre:

1. **`main.py` crea el entorno** llamando a `EnergyEnv()`. Dentro del constructor de `EnergyEnv`, se instancia `ComunidadSimulador` pasándole la ruta al CSV.

2. **El simulador carga los datos**: `ComunidadSimulador.__init__()` ejecuta `pd.read_csv(data_path)` y almacena el DataFrame completo en `self.df`. Esto ocurre una sola vez; el simulador mantiene los datos en memoria durante todo el entrenamiento.

3. **`main.py` crea el agente DQN** de Stable Baselines3 con una política `MlpPolicy` (red neuronal fully-connected). El agente recibe la referencia al entorno para poder interactuar con él.

**Nota importante sobre la lectura del CSV:** Aunque en el diagrama de arquitectura de capas la flecha va de `dataset_final.csv` hacia `EnergyEnv`, en realidad es `ComunidadSimulador` quien ejecuta el `pd.read_csv()`. Ambas representaciones son correctas a distintos niveles de abstracción: `EnergyEnv` es quien decide qué archivo usar (pasa la ruta), pero la lectura física la hace el simulador.

---

## 3. Fase de Reset (Inicio de Episodio)

```
DQN -> ENV : env.reset()
  ENV -> SIM : current_step = random_start, soc = 0.5
  ENV -> ENV : _get_obs()
    ENV -> SIM : get_data_window(t, horizon=1)
      SIM -> DATA : iloc[t:t+1]
    ENV -> SIM : get_data_window(t+1, horizon=24)
      SIM -> DATA : iloc[t+1:t+25]
    ENV --> ENV : obs[76] = [SoC, precio, exc, def] + forecast.flatten()
  ENV --> DQN : obs_0 (76,), info={}
```

### Qué ocurre:

1. **`model.learn()` inicia el entrenamiento** y lo primero que hace SB3 es llamar a `env.reset()`.

2. **`EnergyEnv.reset()` configura el estado inicial:**
   - Elige un paso de tiempo aleatorio (`random_start`) dentro del rango válido del dataset (dejando margen de 168 + 25 pasos para el episodio completo y el horizonte de pronóstico).
   - Establece `self.simulador.soc = 0.5` (la batería empieza al 50% de carga).
   - Establece `self.simulador.current_step = random_start`.

3. **Construcción de la observación inicial `_get_obs()`:**
   - **Primera llamada a `get_data_window(t, horizon=1)`**: obtiene consumo, generación y precio del instante actual `t`. Con estos valores calcula:
     - `Excedente_t = max(0, generacion - consumo)`
     - `Deficit_t = abs(min(0, generacion - consumo))`
   - **Segunda llamada a `get_data_window(t+1, horizon=24)`**: obtiene las próximas 24 horas de datos para el pronóstico.
   - **Ensamblaje del vector de observación** de 76 elementos:
     - Posición 0: `SoC_t` (estado de carga actual = 0.5)
     - Posición 1: `Precio_t`
     - Posición 2: `Excedente_t`
     - Posición 3: `Deficit_t`
     - Posiciones 4–75: pronóstico de 24h aplanado (24 × 3 = 72 valores: consumo, generación, precio por cada hora futura)

4. **Devuelve `obs_0` al agente DQN**, que usará esta observación para tomar su primera decisión.

**Sobre las barras de activación superpuestas:** En el diagrama, `_get_obs()` aparece como una auto-llamada de `EnergyEnv` (una barra de activación que se superpone a otra en la misma línea de vida). Esto es notación UML estándar para representar que un método del mismo objeto llama a otro método propio (o a sí mismo). No es un error: indica ejecución anidada dentro del mismo componente.

---

## 4. Bucle de Pasos (168 pasos = 1 semana)

Este es el núcleo del entrenamiento. Cada episodio consta de 168 pasos (24 horas × 7 días). El bucle se repite miles de veces a lo largo del entrenamiento completo (`total_timesteps`).

### 4.1. Predicción de la acción

```
DQN -> DQN : predict(obs_t) → ε-greedy sobre Q(s,a)
```

El agente DQN recibe la observación `obs_t` y decide una acción mediante **política ε-greedy**:
- Con probabilidad **ε** (exploración): elige una acción aleatoria entre las 13 disponibles.
- Con probabilidad **1 - ε** (explotación): pasa `obs_t` por la red neuronal Q, obtiene `Q(s, a)` para las 13 acciones y elige la de mayor valor Q.

El valor de ε decrece gradualmente durante el entrenamiento (de 1.0 a ~0.05), pasando de exploración pura a explotación casi total.

### 4.2. Ejecución del paso en el entorno

```
DQN -> ENV : env.step(action)
  ENV -> SIM : ejecutar_accion_fisica(action, t)
```

`EnergyEnv.step()` recibe el índice de acción (0–12) y lo delega al simulador.

### 4.3. Ejecución en el motor físico

```
SIM -> DATA : iloc[t] → gen, cons, precio
SIM -> SIM : decodificar action_map[action] → (estrategia, nivel_potencia)
SIM -> SIM : calcular flujos energéticos (comprado, vendido, cargado, descargado)
SIM -> SIM : actualizar SoC (soc_antes → soc_después)
SIM -> SIM : calcular_degradacion_no_lineal(energia_movida, soc_medio)
SIM -> SIM : beneficio = ingresos - gastos - coste_deg
SIM --> ENV : {beneficio, soc, comprado}
```

Dentro de `ComunidadSimulador.ejecutar_accion_fisica()` ocurre toda la física del sistema:

1. **Lectura de datos del instante `t`**: obtiene del DataFrame los valores de generación, consumo y precio de la hora actual.

2. **Decodificación de la acción**: el `action_map` traduce el índice (0–12) a una tupla `(estrategia, nivel_potencia)`:
   - **Estrategia**: IDLE, CARGAR_SOLAR, CARGAR_MIXTA, DESCARGAR_CASA o DESCARGAR_RED.
   - **Nivel de potencia**: 33%, 66% o 100% de los 50 kW del inversor (= 16.5, 33 o 50 kW).

3. **Cálculo de flujos energéticos**: dependiendo de la estrategia y el balance de la comunidad (excedente/déficit), se calculan los kWh comprados a red, vendidos a red, cargados en batería y descargados de batería.

4. **Actualización del SoC (State of Charge)**: el nivel de carga de la batería se actualiza según la energía cargada/descargada, respetando los límites [0, 1] de la batería de 100 kWh.

5. **Degradación no lineal de la batería**: se calcula el coste de degradación usando:
   - `soc_medio = (soc_antes + soc_despues) / 2` — refleja el estrés real durante la operación.
   - Factor I² — penaliza más el ciclado a potencias altas.
   - Factor SoC⁴ — penaliza operar en extremos de carga (cerca de 0% o 100%).
   - `coste_deg = coste_base × energia_movida × factor_I² × factor_SoC⁴`

6. **Cálculo del beneficio neto (recompensa)**:
   ```
   beneficio = vendido × precio_venta_excedente - comprado × precio_kwh - coste_degradacion
   ```
   Este valor será directamente la recompensa `R_t` que recibe el agente.

7. **Retorno al entorno**: el simulador devuelve un diccionario con `{beneficio, soc, comprado}`. No sabe qué se hará con estos datos; su trabajo termina aquí.

### 4.4. Post-procesado en el entorno

```
ENV -> SIM : current_step += 1
ENV -> ENV : _get_obs() → obs_t+1
ENV -> ENV : terminated = (steps >= 168)
ENV --> DQN : obs_t+1, reward, terminated, truncated, info
```

Después de recibir el resultado del simulador:

1. **Avanza el paso de tiempo**: `current_step` se incrementa en 1.
2. **Construye la nueva observación** `obs_t+1` llamando a `_get_obs()` (mismo proceso que en el reset: datos actuales + pronóstico 24h).
3. **Comprueba terminación**: si se han completado 168 pasos, el episodio ha terminado (`terminated = True`).
4. **Devuelve la tupla estándar de Gymnasium** al agente: `(obs_t+1, reward, terminated, truncated, info)`.

### 4.5. Aprendizaje del agente

```
DQN -> DQN : almacenar transición (obs_t, action, reward, obs_t+1) en Replay Buffer
DQN -> DQN : actualizar red Q (si buffer suficientemente lleno)
```

1. **Almacenamiento en Replay Buffer**: la transición completa `(s_t, a_t, r_t, s_{t+1})` se guarda en un buffer circular de experiencias pasadas.

2. **Actualización de la red Q**: si el buffer tiene suficientes transiciones (supera `learning_starts`), se muestrea un mini-batch aleatorio y se actualiza la red neuronal minimizando la pérdida de Bellman:
   ```
   L = E[(r + γ · max_a' Q_target(s', a') - Q(s, a))²]
   ```
   Donde `γ` es el factor de descuento y `Q_target` es la red objetivo (copia periódica de Q que estabiliza el entrenamiento).

### 4.6. Fin de episodio

```
alt episodio terminado
    DQN -> ENV : env.reset() → nuevo episodio
end
```

Si `terminated = True`, SB3 automáticamente llama a `env.reset()` para iniciar un nuevo episodio con un punto de inicio aleatorio diferente. El entrenamiento continúa hasta agotar `total_timesteps`.

---

## 5. Fase de Exportación (Post-Entrenamiento)

```
MAIN -> DQN : model.save('models/dqn_sgec')
MAIN -> DQN : exportar a ONNX → 'models/dqn_sgec.onnx'
```

Una vez completado el entrenamiento:

1. **Guardado del modelo SB3**: se serializa el modelo completo (pesos, hiperparámetros, replay buffer) en formato nativo de Stable Baselines3. Útil para continuar entrenamiento o inspección.

2. **Exportación a ONNX**: se exporta únicamente la red de política (la red Q) al formato ONNX. Este archivo es el que se despliega en producción (Docker / Raspberry Pi) y solo necesita `onnxruntime` + `numpy` para ejecutarse — sin dependencia de PyTorch ni SB3.

---

## 6. Notación UML utilizada

| Elemento | Significado |
|---|---|
| `->` (flecha sólida) | Mensaje síncrono: el emisor espera la respuesta |
| `-->` (flecha discontinua) | Mensaje de retorno: respuesta al mensaje anterior |
| Barra de activación (rectángulo vertical) | Período durante el cual un participante está ejecutando código |
| Barras superpuestas en la misma línea de vida | Auto-llamada: un método del objeto invoca otro método del mismo objeto (ej: `_get_obs()` dentro de `EnergyEnv`) |
| `== Texto ==` | Separador de fase: divide el diagrama en secciones lógicas |
| `loop` | Fragmento combinado de repetición: el contenido se ejecuta múltiples veces |
| `alt` | Fragmento combinado condicional: el contenido se ejecuta solo si la condición se cumple |
| `activate` / `deactivate` | Inicio y fin de una barra de activación |
| `database` | Estereotipo de participante que representa un almacén de datos persistente |

---

## 7. Correspondencia con el código fuente

| Mensaje en el diagrama | Archivo | Línea / Método |
|---|---|---|
| `EnergyEnv()` | `src/envs/energy_env.py` | `__init__()` |
| `ComunidadSimulador(path)` | `src/core/simulador.py` | `__init__(self, data_path)` |
| `pd.read_csv()` | `src/core/simulador.py` | Dentro de `__init__()` |
| `env.reset()` | `src/envs/energy_env.py` | `reset()` |
| `current_step = random_start` | `src/envs/energy_env.py` | `reset()`, asignación directa a `self.simulador.current_step` |
| `_get_obs()` | `src/envs/energy_env.py` | `_get_obs()` |
| `get_data_window(t, horizon)` | `src/core/simulador.py` | `get_data_window(self, step, horizon=24)` |
| `env.step(action)` | `src/envs/energy_env.py` | `step(self, action)` |
| `ejecutar_accion_fisica(action, t)` | `src/core/simulador.py` | `ejecutar_accion_fisica(self, action_idx, step)` |
| `model.save()` | Stable Baselines3 | `DQN.save()` |
