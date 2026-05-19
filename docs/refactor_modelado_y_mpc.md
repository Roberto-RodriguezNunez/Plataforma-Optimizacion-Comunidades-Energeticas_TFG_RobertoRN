# Refactorizacion del modelado de datos y controlador MPC

## 1. Resumen de cambios

### 1.1 Pipeline ETL (`src/utils/generar_dataset_final.py`)

| Aspecto | Antes | Despues |
|---------|-------|---------|
| Columna fecha | No existia | Datetime local (Europe/Madrid) |
| Ruido demanda | i.i.d. N(1, 0.1) por vivienda | AR(1) sobre demanda agregada (rho=0.50, sigma=0.05) |
| Precios excedente negativos | Sin sanear (~214 horas) | Clamped a >= 0 (RD 244/2019) |
| Metadata | No existia | `data/processed/metadata.json` con SHA-256, params, split |
| Configuracion | Hardcoded | Centralizada en `config/system.yaml` |

### 1.2 Simulador (`src/core/simulador.py`)

- Parametro `mode='all'|'train'|'eval'` para split por semanas.
- El dataset se divide en ~134 semanas completas de 168h.
- Con semilla 42, se muestrean 50 semanas al azar como pool eval.
- Las 84 restantes forman el pool train.
- Ambos pools cubren todo el rango temporal (crisis 2022 + post-crisis 2023),
  eliminando el sesgo de distribucion del split temporal anterior.

### 1.3 Entorno RL (`src/envs/energy_env.py`)

- **Bug corregido**: timestamps siempre None → mes=0 en todas las horas.
- Ahora lee columna `fecha` con `parse_dates=['fecha']`.
- Parametro `mode` propagado al simulador.

### 1.4 Controlador MPC (`src/benchmarks/mpc_benchmark.py`)

| Aspecto | Antes | Despues |
|---------|-------|---------|
| Interfaz | Funcion `solve_lp()` | Clase `LinearMPC(BaseController)` |
| Valor terminal | No existia | V_T = lambda * p_T * eta_d * E[H] |
| Autodescarga en LP | Omitida | alpha^(h-k) en restricciones de SoC |
| K_DEG_LIN | Fijo 0.005 | Calibrable (optimo: 0.010) |
| Infactibilidad | Fallback silencioso a IDLE | RuntimeError explicito |

### 1.5 Interfaz abstracta (`src/controllers/base.py`)

`BaseController(ABC)` con metodos `solve(state, forecast) -> dict` y `nombre() -> str`.
Permite sustituir LinearMPC por cualquier controlador (DQN, PPO, reglas) con la misma interfaz.

---

## 2. Formulacion matematica del MPC LP

### 2.1 Variables de decision (4 x H = 96 por horizonte)

- `cs[h]`: kWh excedente solar cargados (lado input inversor)
- `cm[h]`: kWh comprados de red cargados (lado input inversor)
- `dc[h]`: kWh descargados a casas (lado bateria)
- `dr[h]`: kWh descargados a red (lado bateria)

### 2.2 Funcion objetivo (minimizar)

```
min sum_{h=0}^{H-1} [
    (pv[h] + K_DEG) * cs[h]           // pierde venta solar + degradacion
  + (pc[h] + K_DEG) * cm[h]           // paga compra red + degradacion
  - (pc[h] * eta_d - K_DEG) * dc[h]   // ahorra compra x eta_d - degradacion
  - (pv[h] * eta_d - K_DEG) * dr[h]   // ingresa venta x eta_d - degradacion
]
```

**Valor terminal** (si lambda > 0):
```
- lambda * p_T * eta_d * E[H]

donde E[H] = alpha^H * E0 + sum_{k=0}^{H-1} alpha^(H-1-k) * [(cs[k]+cm[k])*eta_c - dc[k] - dr[k]]
      p_T  = pc[H-1]   (modo "ultimo", coherente con energy_env terminal correction)
```

### 2.3 Restricciones

- **Potencia carga**: cs[h] + cm[h] <= P_MAX
- **Potencia descarga**: dc[h] + dr[h] <= P_MAX
- **SoC superior**: E[h+1] <= SOC_MAX * CAP
- **SoC inferior**: E[h+1] >= SOC_MIN * CAP
- **Autodescarga**: E[h+1] = alpha^(h+1) * E0 + sum_{k=0}^{h} alpha^(h-k) * [carga*eta_c - descarga]
- **Bounds**: cs[h] in [0, exc[h]], cm[h] in [0, P_MAX], dc[h] in [0, dfc[h]/eta_d], dr[h] in [0, P_MAX]

### 2.4 Parametros fisicos

| Parametro | Valor | Fuente |
|-----------|-------|--------|
| Capacidad bateria (CAP) | 100 kWh | config/system.yaml |
| Potencia inversor (P_MAX) | 50 kW | config/system.yaml |
| Eficiencia carga (eta_c) | 0.95 | config/system.yaml |
| Eficiencia descarga (eta_d) | 0.95 | config/system.yaml |
| Autodescarga (alpha) | 1 - 0.00004 = 0.99996/h | config/system.yaml |
| SOC_MIN | 0.10 | config/system.yaml |
| SOC_MAX | 0.90 | config/system.yaml |
| K_DEG_LIN | 0.005 EUR/kWh | K=0.005 y K=0.010 equivalentes |
| lambda (TV) | 0.0 | TV sin efecto con H=24 y episodios 168h |

---

## 3. Estrategia de split train/eval

### 3.1 Problema del split temporal

Un split temporal por bloques (train hasta jun-2023, test desde sept-2023)
genera distribuciones heterogeneas: la crisis energetica de 2022 (precios PVPC
hasta 0.50 EUR/kWh) queda solo en train, mientras que el test post-crisis tiene
precios ~50% menores. Resultado: MPC oraculo da 36.80 EUR/sem en test vs
~50 EUR/sem en train — caida del 24% atribuible al shift de distribucion,
no al rendimiento del controlador.

### 3.2 Solucion: muestreo aleatorio por semanas

- Se divide el dataset en 134 semanas completas de 168h.
- Con `np.random.RandomState(42)` se eligen 50 semanas como pool eval.
- Las 84 restantes forman el pool train.
- Ambos pools cubren aleatoriamente todo el rango temporal.
- En `EnergyEnv.reset()`:
  - `mode='train'`: muestrea aleatoriamente una semana del pool train.
  - `mode='eval'`: recorre deterministicamente las 50 semanas eval.

### 3.3 Verificacion de no-leakage

El test `test_split_no_solapado` verifica que los indices horarios de train
y eval son completamente disjuntos (0 horas compartidas).

---

## 4. Calibracion de hiperparametros

### 4.1 K_DEG_LIN (coste linealizado de degradacion)

Calibrado sobre 20 semanas del pool train (MPC oraculo sin TV, semilla=42):

| K_DEG | Beneficio marginal (EUR/sem) |
|-------|------------------------------|
| 0.002 | +50.49 |
| 0.003 | +50.57 |
| 0.004 | +50.70 |
| 0.005 | +50.76 |
| 0.006 | +50.75 |
| 0.008 | +50.76 |
| **0.010** | **+50.88** |
| 0.015 | +50.48 |

La curva es muy plana entre K=0.005 y K=0.010 (diferencia < 0.12 EUR/sem).
Se mantiene K=0.005 (valor nominal de degradacion) por simplicidad.

### 4.2 LAMBDA_TERM (peso del valor terminal)

Calibrado sobre 20 semanas del pool train (K_DEG=0.010):

| lambda | Beneficio marginal (EUR/sem) |
|--------|------------------------------|
| **0.0** | **+50.88** |
| 0.3 | +50.88 |
| 0.5 | +50.72 |
| 0.7 | +50.69 |
| 1.0 | +50.28 |

**Optimo: lambda = 0.0** (valor terminal sin efecto con H=24 y episodios 168h).
El LP ya anticipa suficiente futuro para mantener el SoC razonablemente.

---

## 5. Resultados empiricos (eval pool)

Evaluacion sobre las 50 semanas eval (muestreo aleatorio, cubren todo el rango temporal).

### 5.1 Tabla resumen

| Variante | Media (EUR/sem) | Std (EUR/sem) |
|----------|-----------------|---------------|
| MPC oraculo sin TV | **+57.55** | 34.49 |
| MPC realista sin TV (K=0.010) | +51.63 | 30.30 |
| MPC realista sin TV (K=0.005) | +51.69 | 30.21 |
| IDLE (sin bateria) | 0.00 | -- |

**IDLE absoluto medio: -34.96 EUR/sem** (la comunidad paga ~35 EUR/sem sin bateria).

### 5.2 Analisis

1. **Coste del ruido de pronostico**: 57.55 - 51.63 = **5.92 EUR/sem** (10.3%).
   El MPC con pronostico imperfecto pierde ~6 EUR/sem respecto al oraculo.

2. **Efecto de K_DEG**: K=0.005 vs K=0.010 son equivalentes (-0.06 EUR/sem).
   La degradacion linealizada es una buena aproximacion para el rango operativo
   tipico del controlador.

3. **Coherencia con baseline historico**: el MPC realista da ~51.7 EUR/sem,
   coherente con el baseline previo de 48.46 EUR/sem (diferencia explicable
   por la formalizacion del LP con autodescarga y horizonte completo).

4. **Objetivo para el agente RL**: acercarse a **~52 EUR/sem** (MPC realista).
   Este es el benchmark justo: mismo ruido de pronostico, mismo pool de
   semanas eval.

---

## 6. Tests

### 5.1 Pipeline ETL (`tests/test_dataset.py`) — 12 tests

| Test | Descripcion |
|------|------------|
| test_dataset_dimensiones | ~22.646 filas, 5 columnas |
| test_columnas_esperadas | fecha + 4 datos |
| test_demanda_anualizada_en_rango | 42.500-69.000 kWh/anio |
| test_generacion_anualizada_en_rango | 60-90 MWh/anio |
| test_precios_excedente_no_negativos | 0 precios < 0 |
| test_precios_kwh_no_negativos | 0 precios PVPC < 0 |
| test_precios_en_unidades_kwh | max < 2.0 EUR/kWh |
| test_determinismo | Dos ejecuciones con semilla 42 son identicas |
| test_autocorrelacion_ruido | lag-1 autocorr ~0.50 (atol=0.1) |
| test_split_no_solapado | pools train/eval disjuntos a nivel hora |
| test_timestamps_cargables | Primer mes = junio, datetime valido |
| test_timestamps_monotonos | No-decrecientes, <= 5 duplicados DST |

### 5.2 Controlador MPC (`tests/test_mpc.py`) — 7 tests

| Test | Descripcion |
|------|------------|
| test_idle_precio_constante | Accion neta ~0 con gen=cons y TV activado |
| test_arbitraje_oraculo | Carga >30 kWh en 6h baratas (horizonte rodante) |
| test_no_descarga_en_soc_min | dc~0, dr~0 en SOC_MIN |
| test_no_carga_en_soc_max | cs~0, cm~0 en SOC_MAX |
| test_factibilidad_1000_escenarios | 1000 escenarios aleatorios sin infactibilidad |
| test_coherencia_lp_simulador | 4 semanas: beneficio marginal MPC > 0 |
| test_terminal_value_no_empeora | TV no empeora significativamente (atol=3 EUR) |

**Resultado: 19/19 tests pasan.**

---

## 7. Archivos modificados/creados

| Archivo | Accion |
|---------|--------|
| `config/system.yaml` | Creado |
| `src/utils/generar_dataset_final.py` | Reescrito |
| `src/core/simulador.py` | Modificado (parametro mode) |
| `src/envs/energy_env.py` | Modificado (fix timestamps + mode) |
| `src/controllers/__init__.py` | Creado |
| `src/controllers/base.py` | Creado |
| `src/benchmarks/mpc_benchmark.py` | Reescrito |
| `tests/__init__.py` | Creado |
| `tests/test_dataset.py` | Creado |
| `tests/test_mpc.py` | Creado |
| `data/processed/dataset_final.csv` | Regenerado |
| `data/processed/metadata.json` | Creado |

---

## 8. Limitaciones conocidas

1. **Calibracion K_DEG y lambda**: realizada sobre 20 semanas del pool train.
   El resultado es robusto (curva muy plana), pero una validacion cruzada temporal
   seria mas rigurosa.

2. **Valor terminal**: sin efecto con H=24 y episodios de 168h.
   Podria ser significativo con horizontes mas cortos (e.g., H=6 o H=12).

3. **Varianza alta en eval** (std ~30 EUR/sem): las 50 semanas eval incluyen
   semanas de crisis 2022 (alto beneficio) y post-crisis 2023 (bajo beneficio).
   Esto es correcto — refleja la variabilidad real del mercado. Para comparacion
   con agentes RL, usar las mismas 50 semanas garantiza comparabilidad.

4. **Ruido AR(1)**: modelo simplificado de error de pronostico. En produccion,
   se usarian pronosticos reales de REE con errores calibrados empiricamente.
