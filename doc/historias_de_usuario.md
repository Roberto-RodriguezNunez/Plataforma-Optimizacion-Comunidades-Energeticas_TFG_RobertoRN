# Historias de Usuario — SGEC

> Derivadas del Diagrama de Casos de Uso y la arquitectura completa del Sistema de Gestión Energética Comunitaria.
> Cada historia incluye: objetivo, criterios de aceptación detallados y desglose en tareas.

---

## Priorización

Este proyecto tiene una **dependencia en cadena dura**: sin un modelo DQN entrenado no hay nada que desplegar, demostrar ni analizar. La priorización refleja esa realidad, no la importancia abstracta de cada feature.

```
Bloque 1 (núcleo RL)  →  Bloque 2 (demo técnico)  →  Bloque 3 (cierre de ciclo SaaS)
```

---

### Bloque 1 — Núcleo del TFG (desbloquea todo lo demás)

> Sin este bloque no existe proyecto. Es el valor técnico central.

| HU | Título | Puntos | Estado |
|---|---|---|---|
| HU-01 | Obtener datos de consumo de ESIOS | 2 | ✅ Hecho |
| HU-02 | Obtener datos de generación solar de PVGIS | 2 | ✅ Hecho |
| HU-03 | Obtener datos de precios de electricidad de ESIOS | 2 | ✅ Hecho |
| HU-04 | Ejecutar pipeline ETL de consumo | 3 | ✅ Hecho |
| HU-05 | Ejecutar pipeline ETL de generación solar | 2 | ✅ Hecho |
| HU-06 | Ejecutar pipeline ETL de precios | 1 | ✅ Hecho |
| HU-07 | Generar dataset unificado | 2 | ✅ Hecho |
| HU-08 | Implementar el Gemelo Digital (motor físico) | 8 | ✅ Hecho |
| HU-09 | Implementar degradación no lineal de la batería | 5 | ✅ Hecho |
| HU-10 | Implementar entorno Gymnasium | 5 | ✅ Hecho |
| HU-11 | Construir vector de observación con pronóstico 24h | 3 | ✅ Hecho |
| **HU-12** | **Entrenar agente DQN** | **5** | **⏳ Siguiente** |
| **HU-13** | **Monitorizar el progreso del entrenamiento** | **3** | **⏳ Junto con HU-12** |
| **HU-28** | **Analizar resultados del entrenamiento** | **5** | **⏳ Tras entrenar** |

---

### Bloque 2 — Demo técnico para el tribunal

> Demuestra que el modelo entrenado es desplegable en producción (Edge / Raspberry Pi).
> Depende de tener el modelo del Bloque 1.

| HU | Título | Puntos | Estado |
|---|---|---|---|
| **HU-14** | Exportar modelo a formato ONNX | 3 | ⏳ Pendiente |
| **HU-18** | Ejecutar inferencia en tiempo real | 5 | ⏳ Pendiente |
| **HU-19** | Gestionar carga/descarga de la batería (Edge) | 3 | ⏳ Pendiente |
| **HU-15** | Contenerizar el sistema Edge con Docker | 5 | ⏳ Pendiente |

---

### Bloque 3 — Cierre de ciclo SaaS (componente secundario)

> Cierra el ciclo de vida del dato ante el tribunal (Edge → BD → Dashboard).
> Depende de tener el sistema Edge del Bloque 2 generando datos.

| HU | Título | Puntos | Estado |
|---|---|---|---|
| HU-21 | Diseñar esquema de base de datos PostgreSQL | 2 | ⏳ Pendiente |
| HU-22 | Dar de alta vecinos en la comunidad | 2 | ⏳ Pendiente |
| HU-20 | Enviar métricas de inferencia a PostgreSQL | 3 | ⏳ Pendiente |
| HU-23 | Calcular reparto de ahorro individual | 3 | ⏳ Pendiente |
| HU-24 | Consultar métricas globales (Dashboard Admin) | 5 | ⏳ Pendiente |
| HU-25 | Consultar métricas globales (Vista Vecino) | 2 | ⏳ Pendiente |
| HU-26 | Consultar ahorro individual (Vecino) | 3 | ⏳ Pendiente |
| HU-27 | Consultar ahorro individual (Admin) | 2 | ⏳ Pendiente |

---

### Bloque 4 — Configuración avanzada (opcional / mejora)

> Parametrización del sistema. Útil para experimentar con distintas configuraciones
> pero no bloquea ningún otro bloque.

| HU | Título | Puntos | Estado |
|---|---|---|---|
| HU-16 | Configurar parámetros de la batería | 3 | ⏳ Pendiente |
| HU-17 | Configurar la comunidad de vecinos | 2 | ⏳ Pendiente |

---

## HU-01 — Obtener datos de consumo de ESIOS

**Actor:** Red Eléctrica (ESIOS)

**Como** fuente de datos externa (ESIOS),
**quiero** proporcionar los datos horarios de consumo eléctrico,
**para que** el sistema disponga de los perfiles de demanda reales de la comunidad.

### Objetivo

Disponer de una serie temporal de consumo eléctrico nacional con resolución horaria para el año 2023, que servirá como base para modelar el consumo individual de cada vivienda de la comunidad energética.

### Criterios de aceptación

- CA-01.1: Los datos de consumo se obtienen de la API REST de ESIOS (indicador de demanda real) en formato CSV.
- CA-01.2: El fichero resultante contiene exactamente 8760 filas (1 por hora del año 2023).
- CA-01.3: Cada fila incluye al menos: fecha/hora y valor de consumo en kWh.
- CA-01.4: El fichero se almacena en `data/raw/consumo_esios_2023.csv` como dato inmutable.
- CA-01.5: No se realizan transformaciones sobre los datos descargados; se preserva el formato original de ESIOS.
- CA-01.6: Si existen horas faltantes (cambio horario, errores API), se documentan y se rellenan por interpolación lineal.

### Tareas

- [ ] T-01.1: Investigar el indicador correcto de demanda en la API de ESIOS y obtener el token de acceso.
- [ ] T-01.2: Implementar script de descarga que consulte la API de ESIOS para el rango 01/01/2023 – 31/12/2023.
- [ ] T-01.3: Parsear la respuesta JSON/CSV y generar el fichero `consumo_esios_2023.csv`.
- [ ] T-01.4: Validar que el fichero contiene 8760 filas y sin valores nulos.
- [ ] T-01.5: Documentar el indicador ESIOS utilizado y el proceso de descarga en el README del directorio `data/raw/`.

### Prioridad: Alta
### Estimación: 2 puntos

---

## HU-02 — Obtener datos de generación solar de PVGIS

**Actor:** Red Eléctrica (ESIOS)

**Como** fuente de datos externa (PVGIS),
**quiero** proporcionar los datos horarios de producción fotovoltaica para la localización de la comunidad,
**para que** el sistema conozca la generación solar disponible hora a hora.

### Objetivo

Disponer de una serie temporal de irradiancia/producción solar con resolución horaria para la ubicación geográfica de la comunidad, que se usará para calcular la generación fotovoltaica total de la instalación.

### Criterios de aceptación

- CA-02.1: Los datos se obtienen de la herramienta PVGIS de la Comisión Europea (API o descarga manual).
- CA-02.2: Se configura la localización geográfica correspondiente a la comunidad energética.
- CA-02.3: El fichero resultante contiene 8760 filas con producción unitaria (kWh por kWp instalado).
- CA-02.4: El fichero se almacena en `data/raw/solar_pvgis_2023.csv` como dato inmutable.
- CA-02.5: Los datos reflejan estacionalidad real (mayor producción en verano, menor en invierno).
- CA-02.6: Se incluyen las horas nocturnas con generación = 0.

### Tareas

- [ ] T-02.1: Acceder a PVGIS y configurar la localización, inclinación y orientación de los paneles.
- [ ] T-02.2: Descargar los datos de producción horaria para un año tipo (TMY) o para 2023 si está disponible.
- [ ] T-02.3: Convertir el formato de salida de PVGIS a CSV limpio con columnas estandarizadas.
- [ ] T-02.4: Validar que el fichero contiene 8760 filas, que las horas nocturnas tienen valor 0 y que no hay valores negativos.
- [ ] T-02.5: Documentar los parámetros de configuración PVGIS (lat, lon, inclinación, azimut) en `data/raw/`.

### Prioridad: Alta
### Estimación: 2 puntos

---

## HU-03 — Obtener datos de precios de electricidad de ESIOS

**Actor:** Red Eléctrica (ESIOS)

**Como** fuente de datos externa (ESIOS),
**quiero** proporcionar los precios horarios del mercado eléctrico mayorista,
**para que** el sistema pueda evaluar el coste y beneficio económico de cada decisión de la batería.

### Objetivo

Disponer de los precios reales del mercado spot (PVPC o mercado diario) con resolución horaria para 2023, que la función de recompensa del agente DQN usará para valorar cada maniobra de la batería.

### Criterios de aceptación

- CA-03.1: Los datos de precios se obtienen de la API de ESIOS (indicador de precio del mercado diario) en formato CSV.
- CA-03.2: El fichero resultante contiene 8760 filas con precio en €/MWh.
- CA-03.3: El fichero se almacena en `data/raw/precios_esios_2023.csv` como dato inmutable.
- CA-03.4: Los precios reflejan la variabilidad real del mercado (picos, valles, estacionalidad).
- CA-03.5: No se realizan conversiones de unidades en esta fase (se mantienen en €/MWh).

### Tareas

- [ ] T-03.1: Identificar el indicador ESIOS correcto para precios del mercado diario (PVPC o pool).
- [ ] T-03.2: Implementar script de descarga para el rango 01/01/2023 – 31/12/2023.
- [ ] T-03.3: Parsear la respuesta y generar el fichero `precios_esios_2023.csv`.
- [ ] T-03.4: Validar 8760 filas, ausencia de nulos y que los precios son positivos.
- [ ] T-03.5: Documentar el indicador ESIOS utilizado y las unidades (€/MWh).

### Prioridad: Alta
### Estimación: 2 puntos

---

## HU-04 — Ejecutar pipeline ETL de consumo

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** transformar los datos brutos de consumo en el consumo agregado de las 15 viviendas de la comunidad,
**para que** representen el consumo total horario con variabilidad realista entre viviendas.

### Objetivo

Transformar el perfil de consumo nacional en 15 perfiles individuales de vivienda con variabilidad gaussiana, y agregarlos en una columna `consumo_total` que represente la demanda horaria completa de la comunidad.

### Criterios de aceptación

- CA-04.1: Se lee `data/raw/consumo_esios_2023.csv` como entrada.
- CA-04.2: Se aplican coeficientes de perfil horarios para distribuir el consumo anual a lo largo del año.
- CA-04.3: El consumo base por vivienda es ~3500 kWh/año.
- CA-04.4: Se generan 15 perfiles individuales, cada uno con ruido gaussiano (σ configurable) para simular diferencias entre viviendas.
- CA-04.5: La columna `consumo_total` es la suma de los 15 perfiles individuales en cada hora.
- CA-04.6: Los valores resultantes son siempre positivos (consumo ≥ 0).
- CA-04.7: El consumo total anual de la comunidad es aproximadamente 15 × 3500 = 52500 kWh (±10%).

### Tareas

- [ ] T-04.1: Leer el fichero CSV de consumo bruto y normalizar el perfil horario.
- [ ] T-04.2: Implementar la función de distribución de consumo anual por coeficientes de perfil.
- [ ] T-04.3: Implementar la generación de 15 perfiles con ruido gaussiano (semilla configurable para reproducibilidad).
- [ ] T-04.4: Agregar los 15 perfiles en `consumo_total` (suma por hora).
- [ ] T-04.5: Validar que no hay valores negativos y que el total anual está en el rango esperado.

### Prioridad: Alta
### Estimación: 3 puntos

---

## HU-05 — Ejecutar pipeline ETL de generación solar

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** transformar los datos brutos de PVGIS en la generación real de la instalación fotovoltaica,
**para que** la producción refleje la potencia pico instalada de 42 kWp de la comunidad.

### Objetivo

Escalar la producción unitaria de PVGIS (kWh/kWp) a la potencia pico real de la instalación para obtener la generación fotovoltaica horaria total en kWh.

### Criterios de aceptación

- CA-05.1: Se lee `data/raw/solar_pvgis_2023.csv` como entrada.
- CA-05.2: La producción unitaria se multiplica por la potencia pico instalada (42 kWp por defecto).
- CA-05.3: La columna resultante `generacion_total` contiene la generación horaria en kWh.
- CA-05.4: Los valores son ≥ 0 (no hay generación negativa).
- CA-05.5: Las horas nocturnas tienen generación = 0.
- CA-05.6: La producción anual total es coherente con la irradiancia de la zona (~1300–1700 kWh/kWp/año × 42 kWp).

### Tareas

- [ ] T-05.1: Leer el fichero CSV de PVGIS y extraer la columna de producción unitaria.
- [ ] T-05.2: Multiplicar por la potencia pico instalada (parámetro configurable, defecto 42 kWp).
- [ ] T-05.3: Generar la columna `generacion_total` en kWh.
- [ ] T-05.4: Validar valores no negativos y producción anual total en rango esperado.

### Prioridad: Alta
### Estimación: 2 puntos

---

## HU-06 — Ejecutar pipeline ETL de precios

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** normalizar los precios del mercado eléctrico a la unidad €/kWh,
**para que** sean directamente utilizables por el simulador y la función de recompensa sin conversiones adicionales.

### Objetivo

Convertir los precios de €/MWh (formato ESIOS) a €/kWh (formato interno del sistema) para que la función de recompensa del agente pueda multiplicar directamente precio × kWh.

### Criterios de aceptación

- CA-06.1: Se lee `data/raw/precios_esios_2023.csv` como entrada.
- CA-06.2: Los precios se dividen entre 1000 para convertir de €/MWh a €/kWh.
- CA-06.3: La columna resultante `precio_kwh` contiene el precio horario en €/kWh.
- CA-06.4: Los valores son positivos (precio > 0).
- CA-06.5: El rango de precios es coherente con el mercado español 2023 (~0.05–0.30 €/kWh).

### Tareas

- [ ] T-06.1: Leer el fichero CSV de precios brutos.
- [ ] T-06.2: Aplicar la conversión €/MWh → €/kWh (÷ 1000).
- [ ] T-06.3: Generar la columna `precio_kwh`.
- [ ] T-06.4: Validar rango de precios y ausencia de valores negativos o atípicos.

### Prioridad: Alta
### Estimación: 1 punto

---

## HU-07 — Generar dataset unificado

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** unificar las tres transformaciones ETL en un único dataset final alineado temporalmente,
**para que** el agente DQN disponga de un solo fichero limpio con las tres variables de entrada.

### Objetivo

Producir el fichero `dataset_final.csv` que será el único input del entorno de entrenamiento y del sistema de inferencia, conteniendo consumo, generación y precio alineados hora a hora.

### Criterios de aceptación

- CA-07.1: El script `generar_dataset_final.py` orquesta las tres transformaciones (HU-04, HU-05, HU-06) y genera el fichero de salida.
- CA-07.2: El dataset final tiene exactamente 8760 filas y 3 columnas: `[consumo_total, generacion_total, precio_kwh]`.
- CA-07.3: Las tres columnas están alineadas temporalmente (misma hora en cada fila).
- CA-07.4: El fichero se genera en `data/processed/dataset_final.csv`.
- CA-07.5: El script es idempotente: ejecutarlo múltiples veces con la misma semilla produce el mismo resultado.
- CA-07.6: No se modifican los ficheros originales en `data/raw/`.
- CA-07.7: El script se puede ejecutar desde línea de comandos: `python src/utils/generar_dataset_final.py`.

### Tareas

- [ ] T-07.1: Implementar la función principal del script que orquesta las tres transformaciones.
- [ ] T-07.2: Alinear temporalmente las tres columnas en un único DataFrame.
- [ ] T-07.3: Exportar el DataFrame a `data/processed/dataset_final.csv`.
- [ ] T-07.4: Añadir validación final: 8760 filas, 3 columnas, sin nulos.
- [ ] T-07.5: Verificar idempotencia ejecutando el script dos veces y comparando resultados.
- [ ] T-07.6: Crear el directorio `data/processed/` si no existe.

### Prioridad: Alta
### Estimación: 2 puntos

---

## HU-08 — Implementar el Gemelo Digital (motor físico)

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** disponer de un simulador físico de la comunidad energética con batería comunitaria,
**para que** el agente DQN pueda entrenarse en un entorno que replica fielmente los flujos energéticos y económicos reales.

### Objetivo

Construir la clase `ComunidadSimulador` que modela la física de la batería, los flujos de energía entre la comunidad, la batería y la red, y el cálculo de beneficio económico por cada paso temporal. Este componente es el corazón del Gemelo Digital y debe ser totalmente independiente del agente.

### Criterios de aceptación

- CA-08.1: La clase `ComunidadSimulador` se instancia recibiendo la ruta al CSV y carga los datos en memoria con `pd.read_csv()`.
- CA-08.2: El método `ejecutar_accion_fisica(action_idx, step)` recibe un índice de acción (0–12) y el paso temporal, y devuelve `{beneficio, soc, comprado}`.
- CA-08.3: Soporta 13 acciones discretas organizadas como IDLE + 4 estrategias × 3 niveles de potencia.
- CA-08.4: Las 5 estrategias son: IDLE, CARGAR_SOLAR, CARGAR_MIXTA, DESCARGAR_CASA, DESCARGAR_RED.
- CA-08.5: Los 3 niveles de potencia son 33% (16.5 kW), 66% (33 kW) y 100% (50 kW) de la potencia máxima del inversor.
- CA-08.6: El SoC (State of Charge) se actualiza tras cada acción respetando los límites [0, 1].
- CA-08.7: El método `get_data_window(step, horizon)` devuelve un slice del DataFrame para la construcción de observaciones.
- CA-08.8: El simulador NO importa ni referencia al agente; opera como una caja negra de física.
- CA-08.9: El beneficio neto se calcula como: `ingresos_venta - coste_compra - coste_degradacion`.

### Tareas

- [ ] T-08.1: Crear la clase `ComunidadSimulador` en `src/core/simulador.py` con `__init__(self, data_path)`.
- [ ] T-08.2: Implementar la carga del CSV con `pd.read_csv()` y almacenamiento en `self.df`.
- [ ] T-08.3: Definir el `action_map`: diccionario que mapea índice (0–12) → `(estrategia, nivel_potencia)`.
- [ ] T-08.4: Implementar `ejecutar_accion_fisica()` con la lógica de cada estrategia:
  - T-08.4.1: IDLE — vender excedente a red, comprar déficit de red.
  - T-08.4.2: CARGAR_SOLAR — almacenar excedente solar en batería (sin usar red).
  - T-08.4.3: CARGAR_MIXTA — cargar batería usando solar + red si es necesario.
  - T-08.4.4: DESCARGAR_CASA — cubrir déficit de la comunidad con batería.
  - T-08.4.5: DESCARGAR_RED — vender energía de batería a red (cascada: primero cubre casa, luego vende excedente a red).
- [ ] T-08.5: Implementar el cálculo de flujos energéticos (comprado, vendido, cargado, descargado) en cada estrategia.
- [ ] T-08.6: Implementar la actualización del SoC con clipping a [0, 1]: `soc_nuevo = clip(soc + energia/capacidad, 0, 1)`.
- [ ] T-08.7: Implementar `get_data_window(step, horizon)` que retorna `self.df.iloc[step:step+horizon]`.
- [ ] T-08.8: Implementar el cálculo del beneficio neto: `vendido × precio_venta_excedente - comprado × precio_kwh - coste_degradacion`.
- [ ] T-08.9: Definir los parámetros físicos como atributos de instancia con valores por defecto: capacidad=100, potencia_max=50, soc_inicial=0.5, precio_venta=0.05.
- [ ] T-08.10: Escribir tests unitarios para cada estrategia verificando flujos y SoC resultante.

### Prioridad: Alta
### Estimación: 8 puntos

---

## HU-09 — Implementar degradación no lineal de la batería

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** que el simulador penalice el coste de degradación de la batería de forma no lineal,
**para que** el agente aprenda a evitar maniobras agresivas (alta potencia, SoC extremo) que acorten la vida útil de la batería.

### Objetivo

Modelar el desgaste real de una batería de litio, donde la degradación no es proporcional a la energía movida sino que depende cuadráticamente de la corriente (potencia) y del punto de operación del SoC, incentivando al agente a usar potencias moderadas y mantener el SoC en la zona media.

### Criterios de aceptación

- CA-09.1: El coste de degradación se calcula usando el SoC medio del ciclo: `soc_medio = (soc_antes + soc_despues) / 2`.
- CA-09.2: El factor I² penaliza cuadráticamente la potencia: `factor_I2 = (potencia_usada / potencia_max)²`.
- CA-09.3: El factor SoC⁴ penaliza operar en extremos: `factor_SoC4 = 1 + k × (soc_medio - 0.5)⁴`, donde operar cerca de 0% o 100% es más caro que cerca de 50%.
- CA-09.4: La fórmula completa es: `coste_deg = coste_base × energia_movida × factor_I2 × factor_SoC4`.
- CA-09.5: El coste base por defecto es 0.005 €/kWh.
- CA-09.6: Si la acción es IDLE (energía movida = 0), el coste de degradación es 0.
- CA-09.7: La degradación se resta del beneficio neto y se incluye en la recompensa del agente.
- CA-09.8: Los factores son siempre ≥ 0 (la degradación nunca es negativa).

### Tareas

- [ ] T-09.1: Implementar el cálculo del SoC medio: `(soc_antes + soc_despues) / 2`.
- [ ] T-09.2: Implementar el factor I²: `(potencia_real / potencia_max) ** 2`.
- [ ] T-09.3: Implementar el factor SoC⁴: penalización por operar fuera de la zona media [0.3–0.7].
- [ ] T-09.4: Integrar la fórmula de degradación en `ejecutar_accion_fisica()`.
- [ ] T-09.5: Restar `coste_degradacion` del beneficio neto.
- [ ] T-09.6: Verificar que IDLE no genera degradación.
- [ ] T-09.7: Escribir tests unitarios: degradación a SoC=0.5 < degradación a SoC=0.95, degradación a 33% potencia < degradación a 100%.
- [ ] T-09.8: Calibrar el coste base para que la degradación sea significativa pero no domine la recompensa.

### Prioridad: Alta
### Estimación: 5 puntos

---

## HU-10 — Implementar entorno Gymnasium

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** un entorno compatible con la API estándar de Gymnasium,
**para que** el agente DQN de Stable Baselines3 pueda interactuar con el Gemelo Digital mediante la interfaz `reset()` / `step()`.

### Objetivo

Construir la clase `EnergyEnv` que envuelve al simulador y lo expone como un entorno Gymnasium estándar, permitiendo que cualquier algoritmo de RL de SB3 pueda entrenarse sobre él sin modificaciones.

### Criterios de aceptación

- CA-10.1: La clase `EnergyEnv` hereda de `gymnasium.Env`.
- CA-10.2: Define `observation_space = Box(low=-inf, high=inf, shape=(76,), dtype=float32)`.
- CA-10.3: Define `action_space = Discrete(13)`.
- CA-10.4: El constructor instancia `ComunidadSimulador` pasando la ruta al CSV.
- CA-10.5: `reset(seed)` elige un paso aleatorio dentro del rango válido del dataset y establece SoC = 0.5.
- CA-10.6: `step(action)` devuelve la tupla `(obs, reward, terminated, truncated, info)` conforme a la API Gymnasium.
- CA-10.7: El episodio termina (`terminated=True`) cuando se completan 168 pasos (1 semana).
- CA-10.8: El entorno pasa `gymnasium.utils.env_checker.check_env()` sin errores ni warnings.

### Tareas

- [ ] T-10.1: Crear la clase `EnergyEnv` en `src/envs/energy_env.py` heredando de `gymnasium.Env`.
- [ ] T-10.2: Definir `observation_space` y `action_space` en `__init__()`.
- [ ] T-10.3: Instanciar `ComunidadSimulador` en `__init__()` con la ruta al CSV.
- [ ] T-10.4: Implementar `reset(seed, options)`:
  - T-10.4.1: Calcular el rango válido de inicio: `[0, len(data) - 168 - 25]`.
  - T-10.4.2: Elegir `random_start` aleatorio dentro del rango.
  - T-10.4.3: Asignar `self.simulador.current_step = random_start`.
  - T-10.4.4: Asignar `self.simulador.soc = 0.5`.
  - T-10.4.5: Retornar `(_get_obs(), {})`.
- [ ] T-10.5: Implementar `step(action)`:
  - T-10.5.1: Llamar a `self.simulador.ejecutar_accion_fisica(action, current_step)`.
  - T-10.5.2: Incrementar `self.simulador.current_step += 1`.
  - T-10.5.3: Construir `obs_t+1` con `_get_obs()`.
  - T-10.5.4: Calcular `terminated = (pasos_transcurridos >= 168)`.
  - T-10.5.5: Retornar `(obs, reward, terminated, False, info)`.
- [ ] T-10.6: Ejecutar `env_checker.check_env(EnergyEnv())` y corregir cualquier error.

### Prioridad: Alta
### Estimación: 5 puntos

---

## HU-11 — Construir vector de observación con pronóstico 24h

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** que la observación del agente incluya el estado actual y una ventana de pronóstico de 24 horas,
**para que** el DQN pueda anticipar la evolución futura de consumo, generación y precios al tomar decisiones.

### Objetivo

Diseñar y construir el vector de 76 variables que el agente recibe en cada paso: 4 variables de estado actual + 72 variables de pronóstico (24 horas × 3 magnitudes), proporcionando al agente información suficiente para tomar decisiones informadas sobre arbitraje temporal.

### Criterios de aceptación

- CA-11.1: Las posiciones 0–3 contienen el estado actual: `[SoC_t, Precio_t, Excedente_t, Deficit_t]`.
- CA-11.2: `Excedente_t = max(0, generacion_t - consumo_t)` en kWh.
- CA-11.3: `Deficit_t = abs(min(0, generacion_t - consumo_t))` en kWh.
- CA-11.4: Las posiciones 4–75 contienen el pronóstico de 24h: `[cons_1, gen_1, precio_1, cons_2, gen_2, precio_2, ..., cons_24, gen_24, precio_24]` (24 × 3 = 72 valores aplanados).
- CA-11.5: Se realizan dos llamadas a `get_data_window()`: `horizon=1` para el instante actual, `horizon=24` para el pronóstico.
- CA-11.6: El vector tiene exactamente 76 elementos de tipo `float32`.
- CA-11.7: El pronóstico mira hacia adelante desde `t+1` hasta `t+24`, no incluye el paso actual.
- CA-11.8: Si el paso está cerca del final del dataset, el pronóstico se recorta (el rango de reset lo evita).

### Tareas

- [ ] T-11.1: Implementar `_get_obs()` en `EnergyEnv`.
- [ ] T-11.2: Obtener datos del instante actual con `get_data_window(t, horizon=1)`.
- [ ] T-11.3: Calcular `Excedente_t` y `Deficit_t` a partir de generación y consumo actuales.
- [ ] T-11.4: Obtener pronóstico con `get_data_window(t+1, horizon=24)`.
- [ ] T-11.5: Aplanar la matriz 24×3 del pronóstico con `.values.flatten()`.
- [ ] T-11.6: Concatenar `[SoC, precio, excedente, deficit]` + `forecast_flat` en un array numpy de 76 elementos.
- [ ] T-11.7: Castear a `float32` y retornar.
- [ ] T-11.8: Validar con test unitario que el shape es `(76,)` y el dtype es `float32`.

### Prioridad: Alta
### Estimación: 3 puntos

---

## HU-12 — Entrenar agente DQN

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** entrenar un agente DQN sobre el Gemelo Digital usando Stable Baselines3,
**para que** el modelo aprenda una política óptima de gestión de la batería que maximice el beneficio económico colectivo a lo largo del tiempo.

### Objetivo

Configurar y ejecutar el entrenamiento del agente DQN con los hiperparámetros adecuados, usando el entorno `EnergyEnv` como interfaz con el Gemelo Digital, produciendo un modelo capaz de tomar decisiones rentables de gestión de batería.

### Criterios de aceptación

- CA-12.1: El script `main.py` instancia `EnergyEnv` y crea un agente `DQN('MlpPolicy', env, ...)`.
- CA-12.2: Se configuran los hiperparámetros: learning_rate, buffer_size, learning_starts, batch_size, gamma, exploration_fraction, exploration_final_eps.
- CA-12.3: La política utiliza una red neuronal MLP (fully-connected).
- CA-12.4: El agente utiliza política ε-greedy con ε decreciente durante el entrenamiento.
- CA-12.5: Se entrena con `model.learn(total_timesteps=N)` con N suficiente para convergencia.
- CA-12.6: El modelo entrenado se guarda con `model.save('models/dqn_sgec')`.
- CA-12.7: La recompensa media por episodio muestra tendencia creciente durante el entrenamiento.
- CA-12.8: El modelo final obtiene beneficio positivo consistente en episodios de evaluación.

### Tareas

- [ ] T-12.1: Crear `src/main.py` como punto de entrada del entrenamiento.
- [ ] T-12.2: Instanciar `EnergyEnv` en `main.py`.
- [ ] T-12.3: Configurar los hiperparámetros del DQN (investigar valores razonables para el problema).
- [ ] T-12.4: Instanciar `DQN('MlpPolicy', env, **hiperparams)`.
- [ ] T-12.5: Ejecutar `model.learn(total_timesteps=N)`.
- [ ] T-12.6: Guardar el modelo con `model.save('models/dqn_sgec')`.
- [ ] T-12.7: Ejecutar varios entrenamientos cortos para ajustar hiperparámetros (grid search manual o informal).
- [ ] T-12.8: Evaluar el modelo final en episodios de test no vistos durante el entrenamiento.
- [ ] T-12.9: Documentar los hiperparámetros finales y la razón de su elección.

### Prioridad: Alta
### Estimación: 5 puntos

---

## HU-13 — Monitorizar el progreso del entrenamiento

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** observar la evolución de las métricas durante el entrenamiento del agente,
**para que** pueda detectar problemas (no convergencia, inestabilidad, overfitting) y ajustar hiperparámetros a tiempo.

### Objetivo

Configurar callbacks y logging para que el administrador pueda seguir en tiempo real el progreso del entrenamiento: recompensa media, pérdida del Q-network, tasa de exploración y otros indicadores relevantes.

### Criterios de aceptación

- CA-13.1: Se registra la recompensa media por episodio a lo largo del entrenamiento.
- CA-13.2: Se registra la longitud media de episodio.
- CA-13.3: Se registra la pérdida (loss) de la red Q.
- CA-13.4: Se registra el valor actual de ε (tasa de exploración).
- CA-13.5: Se puede visualizar la curva de aprendizaje (recompensa vs. timesteps) durante o después del entrenamiento.
- CA-13.6: Los logs se guardan en un directorio `logs/` compatible con TensorBoard o CSV.
- CA-13.7: El administrador puede consultar las métricas sin interrumpir el entrenamiento.

### Tareas

- [ ] T-13.1: Configurar el callback `EvalCallback` de SB3 para evaluar periódicamente en un entorno separado.
- [ ] T-13.2: Configurar `tensorboard_log` en el constructor del DQN para guardar logs en `logs/`.
- [ ] T-13.3: Implementar un callback personalizado que registre métricas adicionales (SoC medio, acción más frecuente).
- [ ] T-13.4: Crear script o notebook para visualizar las curvas de entrenamiento (matplotlib o TensorBoard).
- [ ] T-13.5: Verificar que los logs se generan correctamente y son legibles.

### Prioridad: Media
### Estimación: 3 puntos

---

## HU-14 — Exportar modelo a formato ONNX

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** exportar la red de política del modelo entrenado a formato ONNX,
**para que** la inferencia pueda ejecutarse en producción sin dependencias de PyTorch ni Stable Baselines3.

### Objetivo

Extraer la red neuronal Q del modelo DQN entrenado y exportarla al formato ONNX (Open Neural Network Exchange), que permite ejecutar la inferencia con `onnxruntime` en cualquier plataforma, incluyendo dispositivos de borde con recursos limitados.

### Criterios de aceptación

- CA-14.1: El fichero ONNX se genera en `models/dqn_sgec.onnx` tras el entrenamiento.
- CA-14.2: Contiene solo la red Q (forward pass), no el replay buffer, optimizador ni hiperparámetros.
- CA-14.3: La entrada del modelo ONNX tiene shape `(1, 76)` tipo `float32`.
- CA-14.4: La salida del modelo son los 13 Q-values, uno por acción.
- CA-14.5: La inferencia con ONNX produce exactamente las mismas acciones que el modelo SB3 original (equivalencia funcional verificada).
- CA-14.6: El fichero ONNX es ligero (orden de KB, no MB).

### Tareas

- [ ] T-14.1: Extraer la red de política (q_net) del modelo DQN de SB3.
- [ ] T-14.2: Crear un tensor dummy de entrada con shape `(1, 76)` para la exportación.
- [ ] T-14.3: Exportar con `torch.onnx.export()` al fichero `models/dqn_sgec.onnx`.
- [ ] T-14.4: Cargar el modelo ONNX con `onnxruntime.InferenceSession` y ejecutar inferencia de prueba.
- [ ] T-14.5: Comparar las acciones del modelo ONNX con las del modelo SB3 en 100 observaciones aleatorias, verificando equivalencia al 100%.
- [ ] T-14.6: Añadir la exportación ONNX al final de `main.py` (tras `model.save()`).

### Prioridad: Alta
### Estimación: 3 puntos

---

## HU-15 — Contenerizar el sistema Edge con Docker

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** empaquetar el script de inferencia en un contenedor Docker,
**para que** simule el despliegue en una Raspberry Pi como controlador de borde (Edge) y demuestre la viabilidad de la producción.

### Objetivo

Crear un contenedor Docker mínimo que ejecute el ciclo de inferencia usando únicamente `onnxruntime` y `numpy`, demostrando la separación dura entre entrenamiento e inferencia y la viabilidad de despliegue en hardware de borde.

### Criterios de aceptación

- CA-15.1: El `Dockerfile` usa una imagen base ligera (python:3.x-slim o similar).
- CA-15.2: Las únicas dependencias instaladas son `onnxruntime` y `numpy` (NO SB3, gymnasium ni pandas).
- CA-15.3: El contenedor incluye el modelo ONNX (`models/dqn_sgec.onnx`) y el script de inferencia.
- CA-15.4: El contenedor se construye con `docker build` sin errores.
- CA-15.5: El contenedor se ejecuta de forma autónoma simulando el ciclo horario de decisión.
- CA-15.6: La imagen resultante pesa menos de 500 MB.
- CA-15.7: Se puede construir para arquitectura ARM (target Raspberry Pi) mediante multi-arch build o QEMU.

### Tareas

- [ ] T-15.1: Crear el `Dockerfile` con imagen base Python slim.
- [ ] T-15.2: Copiar el modelo ONNX y el script de inferencia al contenedor.
- [ ] T-15.3: Instalar solo `onnxruntime` y `numpy` vía `pip`.
- [ ] T-15.4: Definir el `CMD` o `ENTRYPOINT` para ejecutar el script de inferencia.
- [ ] T-15.5: Crear `docker-compose.yml` para facilitar el despliegue (opcional: incluir PostgreSQL).
- [ ] T-15.6: Construir y probar la imagen localmente con `docker build -t sgec-edge .` y `docker run`.
- [ ] T-15.7: Verificar que `import stable_baselines3` falla dentro del contenedor (confirmar separación).
- [ ] T-15.8: Documentar las instrucciones de build y despliegue.

### Prioridad: Alta
### Estimación: 5 puntos

---

## HU-16 — Configurar parámetros de la batería

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** poder ajustar los parámetros físicos de la batería sin modificar código fuente,
**para que** el Gemelo Digital pueda adaptarse a diferentes especificaciones técnicas de baterías.

### Objetivo

Externalizar los parámetros físicos de la batería (capacidad, potencia, SoC inicial, coste de degradación, precio de venta) a un fichero de configuración o argumentos de línea de comandos, permitiendo experimentar con distintas configuraciones sin tocar el código del simulador.

### Criterios de aceptación

- CA-16.1: Se puede configurar la capacidad de la batería (por defecto: 100 kWh).
- CA-16.2: Se puede configurar la potencia máxima del inversor (por defecto: 50 kW).
- CA-16.3: Se puede configurar el SoC inicial (por defecto: 0.5).
- CA-16.4: Se puede configurar el precio de venta de excedentes a red (por defecto: 0.05 €/kWh).
- CA-16.5: Se puede configurar el coste de degradación base (por defecto: 0.005).
- CA-16.6: Los parámetros se pasan al constructor de `ComunidadSimulador` como argumentos.
- CA-16.7: El simulador funciona correctamente con valores distintos a los por defecto.

### Tareas

- [ ] T-16.1: Añadir parámetros opcionales al constructor de `ComunidadSimulador`: `capacidad`, `potencia_max`, `soc_inicial`, `precio_venta_excedente`, `coste_degradacion_base`.
- [ ] T-16.2: Sustituir constantes hardcodeadas por los atributos de instancia.
- [ ] T-16.3: Propagar los parámetros desde `EnergyEnv` hasta `ComunidadSimulador`.
- [ ] T-16.4: Crear un fichero de configuración (YAML o JSON) con los valores por defecto.
- [ ] T-16.5: Validar que los parámetros recibidos son coherentes (capacidad > 0, SoC ∈ [0,1], etc.).
- [ ] T-16.6: Ejecutar tests con configuraciones extremas (batería muy grande, muy pequeña) para verificar robustez.

### Prioridad: Media
### Estimación: 3 puntos

---

## HU-17 — Configurar la comunidad de vecinos

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** poder definir el número de viviendas y la potencia solar instalada de la comunidad,
**para que** el pipeline ETL y el simulador se adapten a comunidades de diferentes tamaños.

### Objetivo

Parametrizar las variables que definen el tamaño y composición de la comunidad energética, permitiendo reutilizar el sistema para comunidades con distinto número de viviendas, potencia solar o consumo base.

### Criterios de aceptación

- CA-17.1: Se puede configurar el número de viviendas (por defecto: 15).
- CA-17.2: Se puede configurar la potencia pico instalada en kWp (por defecto: 42 kWp).
- CA-17.3: Se puede configurar el consumo base anual por vivienda en kWh (por defecto: 3500 kWh).
- CA-17.4: Los cambios se reflejan en la generación del `dataset_final.csv`.
- CA-17.5: El simulador se adapta automáticamente al dataset generado (no asume 15 viviendas fijas).

### Tareas

- [ ] T-17.1: Parametrizar `generar_dataset_final.py` con argumentos: `num_viviendas`, `potencia_kwp`, `consumo_base_kwh`.
- [ ] T-17.2: Modificar la transformación de consumo para usar `num_viviendas` y `consumo_base_kwh` dinámicamente.
- [ ] T-17.3: Modificar la transformación solar para usar `potencia_kwp` como factor de escala.
- [ ] T-17.4: Regenerar `dataset_final.csv` con la nueva parametrización y verificar coherencia.
- [ ] T-17.5: Documentar los parámetros configurables y su impacto.

### Prioridad: Media
### Estimación: 2 puntos

---

## HU-18 — Ejecutar inferencia en tiempo real

**Actor:** Sistema Edge (Automático)

**Como** sistema Edge desplegado en un dispositivo de borde,
**quiero** ejecutar inferencia usando el modelo ONNX cada hora de forma automática,
**para que** la batería comunitaria se gestione sin intervención humana en un ciclo continuo.

### Objetivo

Implementar el script de inferencia que, ejecutándose dentro del contenedor Docker, lee datos actuales, construye la observación, pasa el vector por el modelo ONNX y devuelve la acción óptima, simulando el comportamiento real de un controlador Edge en producción.

### Criterios de aceptación

- CA-18.1: El script lee los datos actuales (consumo, generación, precio) de la fuente de datos disponible.
- CA-18.2: Construye el vector de observación de 76 variables con el mismo formato que durante el entrenamiento.
- CA-18.3: Carga el modelo ONNX con `onnxruntime.InferenceSession`.
- CA-18.4: Ejecuta la red Q y obtiene los 13 Q-values.
- CA-18.5: Selecciona la acción con mayor Q-value (argmax, política greedy sin exploración).
- CA-18.6: El ciclo se ejecuta cada hora de forma automática (ej: bucle con sleep o cron).
- CA-18.7: Solo utiliza `onnxruntime` y `numpy` como dependencias.
- CA-18.8: El script no importa `stable_baselines3`, `gymnasium` ni `pandas`.

### Tareas

- [ ] T-18.1: Crear el script de inferencia `src/inference/run_edge.py`.
- [ ] T-18.2: Implementar la carga del modelo ONNX con `onnxruntime.InferenceSession('dqn_sgec.onnx')`.
- [ ] T-18.3: Implementar la lectura de datos actuales (en el alcance del TFG: leer del CSV de prueba).
- [ ] T-18.4: Implementar la construcción del vector de observación de 76 variables usando solo numpy.
- [ ] T-18.5: Implementar la ejecución de la red Q: `session.run(None, {'input': obs})`.
- [ ] T-18.6: Implementar la selección de acción: `action = np.argmax(q_values)`.
- [ ] T-18.7: Implementar el bucle horario (simulado) con la lógica de avance temporal.
- [ ] T-18.8: Loggear cada decisión: timestamp, observación, acción seleccionada, Q-values.
- [ ] T-18.9: Verificar que el script funciona dentro del contenedor Docker.

### Prioridad: Alta
### Estimación: 5 puntos

---

## HU-19 — Gestionar carga/descarga de la batería

**Actor:** Sistema Edge (Automático) — incluido en HU-18

**Como** sistema de control de la batería,
**quiero** ejecutar la maniobra física de carga o descarga decidida por el modelo ONNX,
**para que** la batería opere según la estrategia óptima calculada por el agente DQN.

### Objetivo

Traducir la acción seleccionada por el modelo (índice 0–12) en una orden concreta de operación de la batería dentro del script de inferencia, replicando la lógica de `ejecutar_accion_fisica()` del simulador pero adaptada al entorno de producción.

### Criterios de aceptación

- CA-19.1: Se decodifica la acción (0–12) en una tupla `(estrategia, nivel_potencia)` usando el mismo `action_map` del entrenamiento.
- CA-19.2: Se calculan los flujos energéticos: kWh comprados, vendidos, cargados y descargados.
- CA-19.3: Se actualiza el SoC de la batería respetando los límites [0, 1].
- CA-19.4: Se calcula la degradación no lineal con la misma fórmula del entrenamiento.
- CA-19.5: Se calcula el beneficio neto: `ingresos - gastos - degradación`.
- CA-19.6: Se registran las métricas del paso: SoC, acción, beneficio, flujos energéticos.
- CA-19.7: La lógica de gestión es idéntica a la del simulador de entrenamiento (determinismo garantizado).

### Tareas

- [ ] T-19.1: Extraer el `action_map` del simulador de entrenamiento y reutilizarlo en el script de inferencia.
- [ ] T-19.2: Implementar la lógica de ejecución de acción en el script Edge (sin pandas, solo numpy).
- [ ] T-19.3: Implementar el cálculo de flujos energéticos para las 5 estrategias.
- [ ] T-19.4: Implementar la actualización de SoC.
- [ ] T-19.5: Implementar la degradación no lineal.
- [ ] T-19.6: Implementar el cálculo de beneficio neto.
- [ ] T-19.7: Comparar resultados del script Edge con los del simulador en 100 pasos idénticos para verificar equivalencia.

### Prioridad: Alta
### Estimación: 3 puntos

---

## HU-20 — Enviar métricas de inferencia a PostgreSQL

**Actor:** Sistema Edge (Automático)

**Como** sistema Edge,
**quiero** enviar los resultados de cada paso de inferencia a la base de datos PostgreSQL,
**para que** el dashboard SaaS pueda mostrar el estado operativo en tiempo real y el ahorro acumulado.

### Objetivo

Establecer la conexión entre el sistema Edge y la Capa 5 SaaS, enviando los resultados de cada decisión del agente a PostgreSQL para que estén disponibles en los dashboards de Streamlit.

### Criterios de aceptación

- CA-20.1: Tras cada paso de inferencia se inserta un registro en la tabla `ResultadoGlobal`.
- CA-20.2: El registro incluye: `timestamp`, `ahorro_global`, `soc_final`, `accion_tomada`, `beneficio_neto`.
- CA-20.3: La conexión a PostgreSQL es configurable mediante variables de entorno (host, puerto, usuario, contraseña, base de datos).
- CA-20.4: Se usa `psycopg2` o `sqlalchemy` como driver de PostgreSQL.
- CA-20.5: Si la conexión a PostgreSQL falla, el sistema Edge sigue operando normalmente (la inferencia no depende de la BD).
- CA-20.6: Se loggea un warning cuando la conexión falla, indicando que los datos no se persisten.
- CA-20.7: Los timestamps son UTC y se generan automáticamente en el momento de la inserción.

### Tareas

- [ ] T-20.1: Añadir `psycopg2-binary` como dependencia del contenedor Docker Edge.
- [ ] T-20.2: Implementar la función de conexión a PostgreSQL con configuración por variables de entorno.
- [ ] T-20.3: Implementar la función `insertar_resultado_global(timestamp, ahorro_global, soc_final, accion_tomada, beneficio_neto)`.
- [ ] T-20.4: Integrar la inserción en el bucle de inferencia (tras cada paso).
- [ ] T-20.5: Implementar manejo de errores de conexión (try/except) sin detener la inferencia.
- [ ] T-20.6: Verificar que los datos insertados son correctos consultando la BD manualmente.
- [ ] T-20.7: Probar el comportamiento cuando PostgreSQL no está disponible (verificar que la inferencia continúa).

### Prioridad: Media
### Estimación: 3 puntos

---

## HU-21 — Diseñar esquema de base de datos PostgreSQL

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** crear el esquema de tablas en PostgreSQL para la Capa 5 SaaS,
**para que** los resultados del agente y los perfiles de vecinos se persistan de forma estructurada y normalizada.

### Objetivo

Diseñar e implementar las tablas de la base de datos PostgreSQL conforme al MER (Modelo Entidad-Relación) en notación Chen del proyecto, estableciendo las entidades, atributos, claves primarias, claves foráneas y cardinalidades que soportarán el dashboard SaaS.

### Criterios de aceptación

- CA-21.1: Tabla `vecino` con columnas: `id_vecino` (PK, SERIAL), `nombre` (VARCHAR NOT NULL), `rol` (VARCHAR NOT NULL), `cuota_participacion` (DECIMAL(5,4) NOT NULL).
- CA-21.2: Tabla `resultado_global` con columnas: `id_resultado` (PK, SERIAL), `timestamp` (TIMESTAMP NOT NULL), `ahorro_global` (DECIMAL(10,4)), `soc_final` (DECIMAL(5,4)), `accion_tomada` (INTEGER NOT NULL), `beneficio_neto` (DECIMAL(10,4)).
- CA-21.3: Tabla `ahorro_vecino` con columnas: `id_ahorro` (PK, SERIAL), `id_vecino` (FK → vecino), `id_resultado` (FK → resultado_global), `ahorro_individual` (DECIMAL(10,4)).
- CA-21.4: Cardinalidades: Un vecino tiene 0 a N registros de ahorro. Un resultado global tiene 0 a N registros de ahorro (uno por vecino). Cada ahorro pertenece a exactamente 1 vecino y 1 resultado.
- CA-21.5: Se crean índices sobre `ahorro_vecino(id_vecino)` y `ahorro_vecino(id_resultado)` para consultas eficientes.
- CA-21.6: El esquema se implementa mediante un script SQL de migración.
- CA-21.7: El esquema coincide con el MER en notación Chen del proyecto (`mer_saas.drawio`).

### Tareas

- [ ] T-21.1: Escribir el script SQL `schema.sql` con las sentencias `CREATE TABLE` para las tres tablas.
- [ ] T-21.2: Definir las claves primarias `SERIAL` (auto-incrementales) para cada tabla.
- [ ] T-21.3: Definir las claves foráneas en `ahorro_vecino` apuntando a `vecino` y `resultado_global`.
- [ ] T-21.4: Definir los constraints `NOT NULL` en las columnas obligatorias.
- [ ] T-21.5: Crear los índices sobre `id_vecino` e `id_resultado` en `ahorro_vecino`.
- [ ] T-21.6: Añadir un constraint CHECK sobre `cuota_participacion`: valor entre 0 y 1.
- [ ] T-21.7: Añadir un constraint CHECK sobre `soc_final`: valor entre 0 y 1.
- [ ] T-21.8: Añadir un constraint CHECK sobre `accion_tomada`: valor entre 0 y 12.
- [ ] T-21.9: Ejecutar el script contra una instancia PostgreSQL de desarrollo y verificar que se crean las tablas correctamente.
- [ ] T-21.10: Insertar datos de prueba y verificar que las FK se respetan.

### Prioridad: Media
### Estimación: 2 puntos

---

## HU-22 — Dar de alta vecinos en la comunidad

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** registrar los vecinos de la comunidad con su nombre, rol y cuota de participación,
**para que** el motor de reparto pueda calcular el ahorro individual de cada uno.

### Objetivo

Poblar la tabla `vecino` con los 15 perfiles de la comunidad energética, asignando a cada uno un rol (propietario, inquilino, etc.) y una cuota de participación que determine su porcentaje del ahorro global.

### Criterios de aceptación

- CA-22.1: Se pueden insertar los 15 perfiles de vecinos en la tabla `vecino`.
- CA-22.2: Cada vecino tiene un nombre único, un rol (ej: propietario, inquilino, comunidad) y una cuota de participación entre 0 y 1.
- CA-22.3: La suma de todas las `cuota_participacion` es exactamente 1.0 (100%).
- CA-22.4: Se pueden modificar las cuotas si cambia la composición de la comunidad, siempre manteniendo la suma = 1.
- CA-22.5: Se pueden añadir o eliminar vecinos (operaciones CRUD básicas).
- CA-22.6: No se puede eliminar un vecino que tenga registros asociados en `ahorro_vecino` (integridad referencial).

### Tareas

- [ ] T-22.1: Crear un script de seed `seed_vecinos.py` que inserte los 15 vecinos iniciales con cuotas iguales (1/15 cada uno).
- [ ] T-22.2: Implementar función para insertar un nuevo vecino.
- [ ] T-22.3: Implementar función para modificar la cuota de participación de un vecino.
- [ ] T-22.4: Implementar validación de que la suma de cuotas = 1.0 tras cualquier modificación.
- [ ] T-22.5: Implementar función para listar todos los vecinos con sus cuotas.
- [ ] T-22.6: Verificar que la integridad referencial impide borrar vecinos con datos asociados.

### Prioridad: Media
### Estimación: 2 puntos

---

## HU-23 — Calcular reparto de ahorro individual

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** que el motor de post-procesado calcule automáticamente el ahorro de cada vecino tras cada resultado global,
**para que** el beneficio se distribuya proporcionalmente según las cuotas de participación.

### Objetivo

Implementar el motor de reparto determinista que, por cada registro insertado en `ResultadoGlobal`, genera 15 registros en `AhorroVecino` multiplicando el ahorro global por la cuota de cada vecino. Este cálculo es posterior e independiente del agente DQN.

### Criterios de aceptación

- CA-23.1: El cálculo es determinista: `ahorro_individual = ahorro_global × cuota_participacion`.
- CA-23.2: El DQN no interviene en el reparto; es un post-procesado puro.
- CA-23.3: Por cada registro en `ResultadoGlobal` se generan exactamente N registros en `AhorroVecino` (uno por vecino activo, N=15 por defecto).
- CA-23.4: La suma de todos los `ahorro_individual` de un `id_resultado` es igual al `ahorro_global` de ese resultado (conservación del dinero).
- CA-23.5: Si se añade un vecino nuevo, los futuros repartos lo incluyen automáticamente.
- CA-23.6: El motor se ejecuta automáticamente cada vez que llega un nuevo resultado global (trigger o proceso batch).

### Tareas

- [ ] T-23.1: Implementar la función `calcular_reparto(id_resultado)` que lee el `ahorro_global` y las cuotas de todos los vecinos.
- [ ] T-23.2: Calcular `ahorro_individual = ahorro_global × cuota` para cada vecino.
- [ ] T-23.3: Insertar los N registros en `ahorro_vecino` con los `id_vecino` e `id_resultado` correspondientes.
- [ ] T-23.4: Implementar validación: la suma de ahorros individuales debe igualar al global (con tolerancia por redondeo DECIMAL).
- [ ] T-23.5: Decidir e implementar el mecanismo de disparo: trigger PostgreSQL, o integración en el script de inserción de resultados (HU-20).
- [ ] T-23.6: Escribir test que inserta un resultado con ahorro_global=100€ y 15 vecinos con cuotas iguales, y verifica que cada uno recibe 6.67€ (redondeado).
- [ ] T-23.7: Verificar que el reparto funciona con cuotas desiguales (ej: un vecino con cuota 0.2, otro con 0.05).

### Prioridad: Media
### Estimación: 3 puntos

---

## HU-24 — Consultar métricas globales (Dashboard Admin)

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** acceder a un dashboard con las métricas globales de rendimiento del sistema,
**para que** pueda supervisar el estado operativo de la batería, verificar el beneficio económico acumulado y detectar anomalías.

### Objetivo

Construir la vista de administrador en Streamlit que muestre las métricas globales del sistema: ahorro acumulado, evolución del SoC, historial de acciones y beneficio neto, consultando los datos desde PostgreSQL.

### Criterios de aceptación

- CA-24.1: El dashboard muestra el ahorro global acumulado de la comunidad en € (sum de beneficio_neto).
- CA-24.2: Se muestra el beneficio neto desglosado por período: último hora, último día, última semana.
- CA-24.3: Se muestra el estado de carga actual de la batería (SoC del último registro).
- CA-24.4: Se muestra la última acción tomada por el agente con su nombre legible (no solo el índice).
- CA-24.5: Se muestra una gráfica temporal de evolución del SoC (eje X: tiempo, eje Y: SoC 0–1).
- CA-24.6: Se muestra una gráfica temporal de beneficio neto acumulado.
- CA-24.7: Se muestra un histograma de distribución de acciones seleccionadas por el agente.
- CA-24.8: El dashboard se implementa en Streamlit y consulta datos desde PostgreSQL.
- CA-24.9: El dashboard se actualiza automáticamente (auto-refresh configurable).

### Tareas

- [ ] T-24.1: Crear la app Streamlit principal `app.py` con navegación multi-página (admin / vecino).
- [ ] T-24.2: Implementar la conexión a PostgreSQL desde Streamlit (usando `psycopg2` o `sqlalchemy`).
- [ ] T-24.3: Implementar la consulta SQL para obtener el ahorro global acumulado.
- [ ] T-24.4: Implementar la consulta SQL para desglose por período (hora/día/semana).
- [ ] T-24.5: Implementar el widget KPI con SoC actual y última acción.
- [ ] T-24.6: Implementar la gráfica de evolución del SoC con `st.line_chart` o matplotlib.
- [ ] T-24.7: Implementar la gráfica de beneficio acumulado.
- [ ] T-24.8: Implementar el histograma de distribución de acciones.
- [ ] T-24.9: Implementar el mapeo de índice de acción a nombre legible (ej: 3 → "CARGAR_SOLAR 100%").
- [ ] T-24.10: Añadir auto-refresh al dashboard (ej: `st.rerun()` con timer).
- [ ] T-24.11: Estilizar el dashboard con colores y layout coherentes con el proyecto.

### Prioridad: Media
### Estimación: 5 puntos

---

## HU-25 — Consultar métricas globales (Vista Vecino)

**Actor:** Vecino

**Como** vecino de la comunidad energética,
**quiero** ver las métricas globales de rendimiento de la comunidad,
**para que** pueda entender cómo se está gestionando la batería comunitaria y confiar en el sistema.

### Objetivo

Construir una vista simplificada en Streamlit accesible por los vecinos, que muestre las métricas globales de la comunidad sin revelar información de administración ni datos individuales de otros vecinos.

### Criterios de aceptación

- CA-25.1: El vecino puede ver el ahorro global acumulado de la comunidad (€).
- CA-25.2: El vecino puede ver el SoC actual de la batería.
- CA-25.3: El vecino puede ver una gráfica simplificada de evolución del ahorro global.
- CA-25.4: El vecino NO puede ver acciones individuales del agente ni detalles técnicos (Q-values, acciones, etc.).
- CA-25.5: El vecino NO puede ver datos individuales de otros vecinos.
- CA-25.6: La vista es de solo lectura (sin botones de acción ni configuración).
- CA-25.7: Se accede desde el dashboard Streamlit seleccionando el rol de vecino.

### Tareas

- [ ] T-25.1: Crear la página de vecino en la app Streamlit.
- [ ] T-25.2: Implementar el widget KPI con ahorro global y SoC actual (reutilizar consultas de HU-24).
- [ ] T-25.3: Implementar la gráfica de evolución del ahorro global (versión simplificada).
- [ ] T-25.4: Ocultar los elementos de administración (distribución de acciones, detalles técnicos).
- [ ] T-25.5: Implementar la selección de rol en la barra lateral (admin / vecino).
- [ ] T-25.6: Verificar que desde la vista vecino no se puede acceder a datos de otros vecinos.

### Prioridad: Media
### Estimación: 2 puntos

---

## HU-26 — Consultar ahorro individual (Vecino)

**Actor:** Vecino

**Como** vecino de la comunidad energética,
**quiero** consultar mi ahorro individual acumulado y su desglose temporal,
**para que** pueda ver cuánto dinero me ahorro gracias a la gestión inteligente de la batería comunitaria.

### Objetivo

Construir la vista personalizada de ahorro individual en Streamlit, donde cada vecino puede ver su ahorro acumulado, su desglose por período y su proporción respecto al ahorro global, accediendo solo a sus propios datos.

### Criterios de aceptación

- CA-26.1: El vecino se identifica seleccionando su nombre de una lista desplegable en Streamlit.
- CA-26.2: Se muestra el ahorro individual acumulado del vecino seleccionado (€).
- CA-26.3: Se muestra el desglose del ahorro por período: última hora, último día, última semana.
- CA-26.4: Se muestra una gráfica temporal de evolución del ahorro individual.
- CA-26.5: Se muestra la cuota de participación del vecino y su proporción sobre el total.
- CA-26.6: Cada vecino solo ve sus propios datos (filtrado por `id_vecino`).
- CA-26.7: Los datos se consultan desde PostgreSQL (tabla `ahorro_vecino` JOIN `vecino`).

### Tareas

- [ ] T-26.1: Implementar el desplegable de selección de vecino (`st.selectbox` con nombres de la tabla `vecino`).
- [ ] T-26.2: Implementar la consulta SQL para el ahorro acumulado del vecino seleccionado.
- [ ] T-26.3: Implementar la consulta SQL para el desglose por período.
- [ ] T-26.4: Implementar la gráfica temporal de evolución del ahorro individual.
- [ ] T-26.5: Mostrar la cuota de participación y su porcentaje.
- [ ] T-26.6: Verificar que cambiar el vecino seleccionado actualiza todos los widgets.

### Prioridad: Media
### Estimación: 3 puntos

---

## HU-27 — Consultar ahorro individual (Admin)

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** consultar el ahorro individual de cualquier vecino y comparar entre todos,
**para que** pueda verificar que el reparto es correcto y atender consultas de los vecinos.

### Objetivo

Añadir a la vista de administrador una sección de desglose por vecino, con una tabla comparativa de todos los ahorros individuales y la posibilidad de seleccionar cualquier vecino para ver su detalle.

### Criterios de aceptación

- CA-27.1: El administrador puede seleccionar cualquier vecino de la comunidad y ver su ahorro individual.
- CA-27.2: Se muestra una tabla comparativa con el ahorro acumulado de todos los vecinos, ordenada por cuota.
- CA-27.3: Se muestra una columna con la cuota de participación de cada vecino.
- CA-27.4: Se puede verificar que la suma de ahorros individuales coincide con el ahorro global (se muestra la diferencia, que debe ser ~0).
- CA-27.5: Se puede exportar la tabla comparativa a CSV.

### Tareas

- [ ] T-27.1: Implementar la consulta SQL que obtiene el ahorro acumulado de todos los vecinos con sus cuotas.
- [ ] T-27.2: Implementar la tabla comparativa en Streamlit (`st.dataframe` o `st.table`).
- [ ] T-27.3: Añadir fila de totales: suma de ahorros individuales vs. ahorro global (verificación).
- [ ] T-27.4: Implementar el selector de vecino individual para ver detalle (reutilizar lógica de HU-26).
- [ ] T-27.5: Implementar botón de exportación a CSV (`st.download_button`).
- [ ] T-27.6: Integrar la sección en la página de administrador.

### Prioridad: Media
### Estimación: 2 puntos

---

## HU-28 — Analizar resultados del entrenamiento

**Actor:** Administrador

**Como** administrador del sistema,
**quiero** analizar los resultados del modelo entrenado mediante notebooks y gráficas detalladas,
**para que** pueda evaluar la calidad de la política aprendida, compararla con baselines y documentar los resultados en la memoria del TFG.

### Objetivo

Producir un análisis riguroso del rendimiento del agente DQN entrenado, comparándolo con estrategias baseline y visualizando su comportamiento en episodios representativos, generando las gráficas y métricas que se incluirán en la memoria del TFG.

### Criterios de aceptación

- CA-28.1: Se crea un notebook Jupyter (`notebooks/analisis_resultados.ipynb`) con el análisis completo.
- CA-28.2: Se visualiza la curva de recompensa acumulada durante el entrenamiento (convergencia).
- CA-28.3: Se compara el beneficio del agente DQN con al menos 2 estrategias baseline:
  - Baseline 1: Sin batería (comprar/vender directamente a red).
  - Baseline 2: Regla fija (cargar de día con solar, descargar de noche).
- CA-28.4: Se visualiza la evolución del SoC a lo largo de episodios representativos (verano e invierno).
- CA-28.5: Se muestra la distribución de acciones seleccionadas por el agente (histograma de 13 barras).
- CA-28.6: Se calculan métricas de rendimiento: ahorro total anual (€), número de ciclos de batería, beneficio medio por paso, % de decisiones IDLE.
- CA-28.7: Se muestra un heatmap de acción vs. hora del día (¿el agente aprende patrones diarios?).
- CA-28.8: Se incluye texto explicativo en celdas Markdown del notebook.
- CA-28.9: Las gráficas son exportables a PNG para incluir en la memoria del TFG.

### Tareas

- [ ] T-28.1: Crear el notebook `notebooks/analisis_resultados.ipynb`.
- [ ] T-28.2: Cargar el modelo entrenado y ejecutar evaluación en múltiples episodios.
- [ ] T-28.3: Implementar la estrategia baseline "sin batería" y ejecutarla en los mismos episodios.
- [ ] T-28.4: Implementar la estrategia baseline "regla fija" (solar→batería de día, batería→casa de noche).
- [ ] T-28.5: Generar la tabla comparativa: DQN vs. Baseline 1 vs. Baseline 2 (ahorro, ciclos, beneficio).
- [ ] T-28.6: Graficar la curva de aprendizaje (recompensa media vs. timesteps).
- [ ] T-28.7: Graficar la evolución del SoC en un episodio de verano y otro de invierno.
- [ ] T-28.8: Graficar el histograma de distribución de acciones.
- [ ] T-28.9: Graficar el heatmap de acción vs. hora del día.
- [ ] T-28.10: Calcular y mostrar métricas finales: ahorro anual total, beneficio medio/paso, ciclos, % IDLE.
- [ ] T-28.11: Añadir celdas Markdown con explicaciones de cada gráfica y conclusiones.
- [ ] T-28.12: Exportar las gráficas principales a PNG en un directorio `doc/figuras/`.

### Prioridad: Media
### Estimación: 5 puntos

---

## Resumen

| ID | Historia de Usuario | Actor(es) | Prioridad | Puntos | Tareas |
|---|---|---|---|---|---|
| HU-01 | Obtener datos de consumo de ESIOS | Red Eléctrica | Alta | 2 | 5 |
| HU-02 | Obtener datos de generación solar de PVGIS | Red Eléctrica | Alta | 2 | 5 |
| HU-03 | Obtener datos de precios de ESIOS | Red Eléctrica | Alta | 2 | 5 |
| HU-04 | Ejecutar pipeline ETL de consumo | Administrador | Alta | 3 | 5 |
| HU-05 | Ejecutar pipeline ETL de generación solar | Administrador | Alta | 2 | 4 |
| HU-06 | Ejecutar pipeline ETL de precios | Administrador | Alta | 1 | 4 |
| HU-07 | Generar dataset unificado | Administrador | Alta | 2 | 6 |
| HU-08 | Implementar el Gemelo Digital (motor físico) | Administrador | Alta | 8 | 10 |
| HU-09 | Implementar degradación no lineal de la batería | Administrador | Alta | 5 | 8 |
| HU-10 | Implementar entorno Gymnasium | Administrador | Alta | 5 | 6 |
| HU-11 | Construir vector de observación con pronóstico 24h | Administrador | Alta | 3 | 8 |
| HU-12 | Entrenar agente DQN | Administrador | Alta | 5 | 9 |
| HU-13 | Monitorizar el progreso del entrenamiento | Administrador | Media | 3 | 5 |
| HU-14 | Exportar modelo a formato ONNX | Administrador | Alta | 3 | 6 |
| HU-15 | Contenerizar el sistema Edge con Docker | Administrador | Alta | 5 | 8 |
| HU-16 | Configurar parámetros de la batería | Administrador | Media | 3 | 6 |
| HU-17 | Configurar la comunidad de vecinos | Administrador | Media | 2 | 5 |
| HU-18 | Ejecutar inferencia en tiempo real | Sistema Edge | Alta | 5 | 9 |
| HU-19 | Gestionar carga/descarga de la batería | Sistema Edge | Alta | 3 | 7 |
| HU-20 | Enviar métricas de inferencia a PostgreSQL | Sistema Edge | Media | 3 | 7 |
| HU-21 | Diseñar esquema de base de datos PostgreSQL | Administrador | Media | 2 | 10 |
| HU-22 | Dar de alta vecinos en la comunidad | Administrador | Media | 2 | 6 |
| HU-23 | Calcular reparto de ahorro individual | Administrador | Media | 3 | 7 |
| HU-24 | Consultar métricas globales (Dashboard Admin) | Administrador | Media | 5 | 11 |
| HU-25 | Consultar métricas globales (Vista Vecino) | Vecino | Media | 2 | 6 |
| HU-26 | Consultar ahorro individual (Vecino) | Vecino | Media | 3 | 6 |
| HU-27 | Consultar ahorro individual (Admin) | Administrador | Media | 2 | 6 |
| HU-28 | Analizar resultados del entrenamiento | Administrador | Media | 5 | 12 |
| | | | **Total** | **91** | **190** |

---

## Trazabilidad con Casos de Uso

| Caso de Uso | Historias de Usuario |
|---|---|
| UC1 — Proporcionar datos de consumo y precios | HU-01, HU-02, HU-03 |
| UC2 — Ejecutar pipeline ETL | HU-04, HU-05, HU-06, HU-07 |
| UC3 — Entrenar agente DQN | HU-08, HU-09, HU-10, HU-11, HU-12, HU-13 |
| UC4 — Exportar y desplegar modelo (ONNX / Edge) | HU-14, HU-15 |
| UC5 — Configurar batería y comunidad | HU-16, HU-17, HU-22 |
| UC6 — Ejecutar inferencia en tiempo real | HU-18, HU-20 |
| UC7 — Gestionar carga/descarga batería (<<include>> UC6) | HU-19 |
| UC8 — Consultar métricas globales | HU-24, HU-25 |
| UC9 — Consultar ahorro individual | HU-23, HU-26, HU-27 |
| — (transversal) | HU-21 (BD), HU-28 (análisis) |
