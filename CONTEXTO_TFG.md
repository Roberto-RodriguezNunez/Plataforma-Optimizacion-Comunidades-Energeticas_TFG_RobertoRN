# CONTEXTO_TFG.md — Fuente de Verdad del Proyecto
> Leer este archivo al inicio de cada sesión antes de tocar cualquier código.

---

## 1. Identidad del Proyecto

**Título:** Diseño y Desarrollo de una Plataforma Software para la Optimización de Comunidades Energéticas  
**Autor:** Roberto Rodríguez Núñez  
**Universidad:** Universidade de Vigo — ESEI (Grado en Enxeñaría Informática)  
**Tutora:** Eva Mª Lorenzo Iglesias  
**Co-tutor:** Pedro Celard Pérez  

---

## 2. Resumen Ejecutivo

El proyecto implementa un **Gemelo Digital (Digital Twin)** de una comunidad energética de **15 viviendas** con una **batería comunitaria compartida**. Un agente de Inteligencia Artificial (**DQN**) aprende a gestionar la batería para maximizar el beneficio económico colectivo, tomando decisiones **macro** sobre el balance total de energía de la comunidad (excedente/déficit agregado), nunca sobre casas individuales.

El sistema es una **Central Eléctrica Virtual (VPP)** pensada para ejecutarse en una **Raspberry Pi** como controlador de borde (Edge), pero el alcance actual del TFG es 100% **Software-in-the-Loop** (sin hardware físico real).

---

## 3. Reglas Arquitectónicas Inquebrantables

### REGLA 1 — Sin frameworks web pesados
- **PROHIBIDO:** Django, React, Vue, FastAPI con frontend complejo.
- **PERMITIDO:** Scripts Python planos o Flask muy básico solo para monitorización del entrenamiento.

### REGLA 2 — Sin hardware real en el alcance actual
- No hay Modbus, MQTT real, ni sensores físicos en el código.
- Todo se lee de `data/processed/dataset_final.csv`.
- Las pruebas de producción simulan el entorno Edge con **Docker**.

### REGLA 3 — El DQN decide en MACRO, no en micro
- El agente gestiona el **excedente total** y el **déficit total** de la comunidad.
- Las 13 acciones son maniobras sobre la batería comunitaria.
- El reparto de ahorro entre los 15 vecinos es un cálculo determinista **posterior** a la decisión del agente.

### REGLA 4 — Separación dura: Entrenamiento vs. Inferencia
| Fase | Librerías permitidas |
|---|---|
| Entrenamiento | `stable-baselines3`, `gymnasium`, `pandas`, `numpy` |
| Producción/Docker/Edge | `onnxruntime`, `numpy` ÚNICAMENTE |

El modelo entrenado se exporta a **ONNX** para inferencia ligera. Nunca importar SB3 ni Gymnasium en los scripts de despliegue.

---

## 4. El Agente DQN

### Espacio de Observación: 76 variables (shape `(76,)`, dtype `float32`)

| Índice | Variable | Descripción |
|---|---|---|
| 0 | `SoC_t` | Estado de carga actual de la batería [0,1] |
| 1 | `Precio_t` | Precio de la electricidad en €/kWh |
| 2 | `Excedente_t` | `max(0, generacion - consumo)` en kWh |
| 3 | `Deficit_t` | `abs(min(0, generacion - consumo))` en kWh |
| 4–75 | Pronóstico 24h | Ventana deslizante: 24 × [consumo, generacion, precio] aplanada |

### Espacio de Acción: `Discrete(13)`

| ID | Estrategia | Nivel de Potencia |
|---|---|---|
| 0 | IDLE | — (vende excedente, compra déficit) |
| 1 | CARGAR_SOLAR | 33% (16.5 kW) |
| 2 | CARGAR_SOLAR | 66% (33 kW) |
| 3 | CARGAR_SOLAR | 100% (50 kW) |
| 4 | CARGAR_MIXTA | 33% (usa red si hace falta) |
| 5 | CARGAR_MIXTA | 66% |
| 6 | CARGAR_MIXTA | 100% |
| 7 | DESCARGAR_CASA | 33% (cubre déficit de la comunidad) |
| 8 | DESCARGAR_CASA | 66% |
| 9 | DESCARGAR_CASA | 100% |
| 10 | DESCARGAR_RED | 33% (vende a red, cascada casa→red) |
| 11 | DESCARGAR_RED | 66% |
| 12 | DESCARGAR_RED | 100% |

### Función de Recompensa
```
R_t = Ingresos_venta - Coste_compra - Coste_degradacion
    = vendido × precio_venta_excedente - comprado × precio_kwh - coste_deg
```
Cash flow neto por paso de tiempo. Positivo = beneficio, negativo = coste.

El término de degradación es **no lineal**: penaliza más el ciclado a potencias altas y con SoC en extremos (factor I² × factor SoC⁴). Se calcula con el **SoC medio** del ciclo `(soc_antes + soc_después) / 2` para reflejar el stress real durante la operación.

> **Nota para la memoria:** Los PDFs de `doc/foundation/` documentan la fórmula antigua con término `ahorro`. Actualizar en la redacción final de la memoria.

### Normalización de Observaciones (VecNormalize)

Las 76 variables del vector de observación tienen escalas muy diferentes:
- SoC: 0–1, Precio: 0.05–0.30 €/kWh, Consumo/Generación: 0–200 kWh

`VecNormalize` aplica normalización en línea (running mean/std) para que todas
tengan media ~0 y desviación ~1. Esto permite que la red aprenda qué variable es
importante en vez de dejarse llevar por la magnitud numérica.

Las estadísticas de normalización se guardan en `models/vec_normalize.pkl` y son
**imprescindibles para inferencia**: el modelo ONNX espera la entrada normalizada.

### Episodio
- Duración: 24 × 7 = 168 pasos (1 semana)
- Inicio: aleatorio con margen suficiente para el horizonte de pronóstico (+25 pasos)

---

## 5. El Motor Físico (Gemelo Digital)

**Archivo:** `src/core/simulador.py` — Clase `ComunidadSimulador`

Parámetros físicos fijos:
- Capacidad batería: **100 kWh**
- Potencia máxima inversor: **50 kW**
- SoC inicial por defecto: **0.5**
- Precio venta excedente a red: **0.05 €/kWh**
- Coste degradación base: **0.005**

El simulador NO conoce al agente. Solo recibe un `action_idx` y devuelve `{beneficio, soc, comprado}`.

---

## 6. Entorno RL

**Archivo:** `src/envs/energy_env.py` — Clase `EnergyEnv(gym.Env)`

- Implementa la API estándar Gymnasium (compatible con SB3).
- Instancia `ComunidadSimulador` internamente.
- Datos leídos desde: `data/processed/dataset_final.csv`

---

## 7. Pipeline de Datos (ETL)

**Archivo:** `src/utils/generar_dataset_final.py`

| Fuente | Archivo raw | Transformación |
|---|---|---|
| ESIOS (consumo) | `data/raw/consumo_esios_2023.csv` | Coeficientes × 3500 kWh × 15 vecinos con ruido gaussiano |
| PVGIS (solar) | `data/raw/solar_pvgis_2023.csv` | Producción unitaria × 50 kWp |
| ESIOS (precios) | `data/raw/precios_esios_2023.csv` | €/MWh ÷ 1000 → €/kWh |

**Salida:** `data/processed/dataset_final.csv` — Matriz 8760 × 3: `[consumo_total, generacion_total, precio_kwh]`  
**Año de datos:** 2023 (estacionalidad completa invierno/verano)

---

## 8. Estructura del Proyecto

```
TFG/
├── data/
│   ├── raw/          # INMUTABLE — fuentes oficiales ESIOS y PVGIS
│   └── processed/    # dataset_final.csv — único input del agente
├── src/
│   ├── core/
│   │   └── simulador.py      # Gemelo Digital / Motor físico
│   ├── envs/
│   │   └── energy_env.py     # Entorno Gymnasium para SB3
│   ├── utils/
│   │   └── generar_dataset_final.py  # ETL pipeline
│   └── main.py               # Orquestador del entrenamiento (PENDIENTE)
├── doc/
│   └── foundation/   # Documentación técnica de referencia
├── notebooks/
│   └── analisis_datos.ipynb
├── requirements.txt          # pandas, numpy, matplotlib, gymnasium, stable-baselines3
├── CONTEXTO_TFG.md           # ESTE ARCHIVO
└── CHANGELOG.md              # Registro de progreso
```

---

## 9. Stack Tecnológico

- **Lenguaje:** Python 3
- **Entrenamiento:** `stable-baselines3` (DQN), `gymnasium`, `pandas`, `numpy`
- **Análisis:** `matplotlib`, Jupyter notebooks
- **Producción:** `onnxruntime` + `numpy` en Docker (Raspberry Pi target)
- **NO usar:** Django, React, Vue, PyTorch directo, TensorFlow

---

## 10. Estado Actual del Proyecto (actualizar al avanzar)

| Componente | Estado |
|---|---|
| ETL pipeline (`generar_dataset_final.py`) | ✅ Completo y validado |
| Motor físico (`simulador.py`) | ✅ Completo — 13 acciones, degradación no lineal |
| Entorno RL (`energy_env.py`) | ✅ Completo — validado con `env_checker` |
| Script entrenamiento (`main.py`) | ✅ v3 — red 64x64, VecNormalize (solo obs), 500k pasos, EvalCallback, MetricasCallback |
| Exportación a ONNX | ⏳ PENDIENTE |
| Script de inferencia (Docker/Edge) | ⏳ PENDIENTE |
| SaaS — Streamlit + PostgreSQL (secundario) | ⏳ PENDIENTE (componente secundario para cierre de ciclo ante el tribunal) |
| Dashboard de monitorización | ⏳ PENDIENTE (opcional, Flask básico) |
| Análisis de resultados (notebook) | ⏳ PENDIENTE |
