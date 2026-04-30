# Historias de Usuario — SGEC

## Actores del sistema

| Actor | Descripción |
|---|---|
| **Administrador** | Gestor técnico de la comunidad energética. Configura, entrena y monitoriza el sistema. |
| **Vecino** | Miembro de la comunidad. Consume los beneficios del sistema pero no interactúa con él directamente. |
| **Sistema DQN** | Actor automatizado. Toma decisiones de gestión energética de forma autónoma en cada ciclo. |

---

## Épica 1 — Gestión del Dataset de Entrenamiento

### HU-01
> **Como** administrador,  
> **quiero** ejecutar el pipeline ETL con un solo comando,  
> **para** generar el dataset de entrenamiento a partir de las fuentes oficiales (ESIOS y PVGIS) sin manipulación manual.

**Criterios de aceptación:**
- El script procesa los tres ficheros raw y genera `dataset_final.csv`.
- El dataset contiene exactamente 8760 filas (una por hora del año 2023).
- Las columnas son `consumo_total` (kWh), `generacion_total` (kWh) y `precio_kwh` (€/kWh).
- Si falta algún fichero raw, el script informa del error sin crashear.

---

### HU-02
> **Como** administrador,  
> **quiero** que el consumo simulado represente la diversidad real de 15 viviendas,  
> **para** que el agente aprenda con patrones de demanda estocásticos y no con clones idénticos.

**Criterios de aceptación:**
- Cada vecino tiene un factor de escala constante `α ∈ U(0.8, 1.2)`.
- Se aplica ruido gaussiano horario `β ∈ N(1.0, 0.1)`.
- El consumo agregado resulta siempre positivo (sin valores negativos).

---

## Épica 2 — Simulación del Entorno Físico

### HU-03
> **Como** administrador,  
> **quiero** que el simulador modele correctamente la batería comunitaria,  
> **para** que las decisiones del agente tengan consecuencias físicas realistas.

**Criterios de aceptación:**
- El SoC está siempre acotado en `[0, 1]`.
- La potencia de carga/descarga no supera los 50 kW del inversor.
- La capacidad máxima de la batería es 100 kWh.
- El coste de degradación penaliza operaciones en extremos del SoC y a alta potencia.

---

### HU-04
> **Como** administrador,  
> **quiero** disponer de 13 modos de operación diferenciados,  
> **para** que el agente pueda seleccionar estrategias con granularidad de potencia (33% / 66% / 100%).

**Criterios de aceptación:**
- Existen 4 estrategias: IDLE, CARGAR_SOLAR, CARGAR_MIXTA, DESCARGAR_CASA, DESCARGAR_RED.
- Cada estrategia (excepto IDLE) tiene 3 niveles de potencia.
- Total de 13 acciones válidas en `Discrete(13)`.

---

### HU-05
> **Como** administrador,  
> **quiero** que la función de recompensa refleje el beneficio económico neto real,  
> **para** que el agente aprenda a maximizar el ahorro de la comunidad, no una métrica ficticia.

**Criterios de aceptación:**
- Recompensa = `ingresos_venta - gastos_compra - coste_degradacion`.
- Recompensa positiva cuando se vende excedente o se evita comprar a red.
- Recompensa negativa cuando se compra a la red o se degrada la batería innecesariamente.

---

## Épica 3 — Entrenamiento del Agente DQN

### HU-06
> **Como** administrador,  
> **quiero** lanzar el entrenamiento del agente con un solo script (`main.py`),  
> **para** no tener que configurar manualmente el bucle de aprendizaje.

**Criterios de aceptación:**
- `main.py` instancia `EnergyEnv` y configura el agente DQN de Stable Baselines3.
- El entrenamiento guarda el mejor modelo automáticamente mediante `EvalCallback`.
- El modelo se guarda en `models/`.

---

### HU-07
> **Como** administrador,  
> **quiero** que el entorno sea compatible con el estándar de Gymnasium,  
> **para** poder cambiar el algoritmo de entrenamiento (DQN → PPO, A2C...) sin reescribir el entorno.

**Criterios de aceptación:**
- `check_env(EnergyEnv())` pasa sin errores ni advertencias.
- Los espacios `observation_space` y `action_space` están correctamente definidos y acotados.

---

## Épica 4 — Despliegue en Producción (Edge)

### HU-08
> **Como** administrador,  
> **quiero** exportar el modelo entrenado a formato ONNX,  
> **para** ejecutar la inferencia en la Raspberry Pi sin depender de Stable Baselines3 o PyTorch.

**Criterios de aceptación:**
- El modelo exportado carga correctamente con `onnxruntime`.
- La inferencia solo requiere `onnxruntime` y `numpy`.
- El resultado de `argmax()` sobre las 13 salidas da la misma acción que el modelo original.

---

### HU-09
> **Como** administrador,  
> **quiero** que el sistema funcione dentro de un contenedor Docker,  
> **para** garantizar que el entorno de producción en la Raspberry Pi es reproducible y aislado.

**Criterios de aceptación:**
- Existe un `Dockerfile` con la imagen base ligera (Python slim).
- El contenedor ejecuta inferencias leyendo del `dataset_final.csv`.
- No se incluyen librerías de entrenamiento (SB3, gymnasium) en la imagen de producción.

---

## Épica 5 — Visibilidad para los Vecinos

### HU-10
> **Como** vecino de la comunidad,  
> **quiero** conocer cuánto ahorro ha generado el sistema en mi factura,  
> **para** entender el beneficio económico que obtengo por participar en la comunidad energética.

**Criterios de aceptación:**
- Existe algún mecanismo (log, dashboard básico o informe) que muestre el beneficio acumulado.
- Los datos son comprensibles sin conocimientos técnicos.

---

### HU-11
> **Como** vecino,  
> **quiero** saber el estado actual de la batería comunitaria,  
> **para** tener visibilidad sobre el recurso compartido.

**Criterios de aceptación:**
- Se puede consultar el SoC actual en tiempo real o con un retraso máximo de 1 ciclo.
