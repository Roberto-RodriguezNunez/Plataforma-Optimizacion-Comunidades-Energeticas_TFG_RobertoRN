# Especificación Técnica del Sistema SGEC
## Gemelo Digital + Agente DQN para Gestión de Comunidad Energética

---

## 0. CONTEXTO Y MARCO LEGAL

### 0.1 Qué es una Comunidad Energética

Una comunidad energética es una agrupación de consumidores (viviendas, comercios, edificios) que comparten una instalación de generación renovable y opcionalmente un sistema de almacenamiento. El objetivo es maximizar el autoconsumo colectivo: que la energía generada se consuma internamente antes de exportarla a la red.

En este proyecto la comunidad está formada por **15 viviendas** con una instalación fotovoltaica compartida de **42 kWp** y una batería de almacenamiento compartida de **80 kWh**.

### 0.2 Marco Legal Español

**Real Decreto 244/2019** (15 de abril de 2019):
- Regula las condiciones administrativas, técnicas y económicas del autoconsumo de energía eléctrica en España
- Legaliza el **autoconsumo colectivo**: varios consumidores pueden compartir una instalación de generación sin que sea su propietario
- Establece el mecanismo de **compensación simplificada de excedentes** (modalidad acogida en este proyecto): los excedentes vertidos a red se compensan económicamente en la factura mensual
- La compensación se valora al precio del mercado horario (Pmh) menos los costes de desvío (CDSVh), lo que corresponde al indicador ESIOS 1739
- **Limitación clave**: la compensación nunca puede superar el coste de los consumos en el mismo periodo de facturación — no se puede "ganar dinero" vendiendo, solo reducir la factura

**Real Decreto-Ley 15/2018** (5 de octubre de 2018):
- Medidas urgentes para la transición energética y la protección de los consumidores
- Elimina el "impuesto al sol" (cargos a la energía autoconsumida)
- Simplifica la tramitación de instalaciones de pequeña potencia

**Ley 24/2013 del Sector Eléctrico**:
- Marco general del sistema eléctrico español
- Define los roles de generadores, distribuidores, comercializadores y consumidores
- Base sobre la que se desarrollan los decretos de autoconsumo

**Directiva Europea 2018/2001 (RED II)**:
- Fomento de energías renovables en la UE
- Reconoce el derecho de los ciudadanos a producir, consumir, almacenar y vender energía renovable
- Introduce el concepto de "comunidades de energía renovable" con derechos explícitos

**Precio Voluntario al Pequeño Consumidor (PVPC)**:
- Tarifa regulada por el Gobierno para consumidores con potencia contratada ≤ 10 kW
- El precio varía cada hora y se publica por REE/ESIOS con un día de antelación (indicador 1001)
- Es el precio que pagan los vecinos de la comunidad cuando compran de la red
- Incluye todos los cargos: precio de mercado, peajes de acceso, cargos del sistema, impuesto eléctrico (5.11%) y IVA (21%)

### 0.3 Mecanismo de Reparto de Excedentes (RD 244/2019, Art. 6)

En autoconsumo colectivo, la energía generada se reparte entre los participantes según coeficientes de reparto fijos pactados. En este proyecto se asume que los 15 vecinos tienen coeficientes iguales (1/15 cada uno), lo que simplifica el sistema a trabajar con los totales agregados.

### 0.4 Rol del Almacenamiento

La batería no está explícitamente regulada en el RD 244/2019 para autoconsumo colectivo, pero es técnicamente viable como elemento de la instalación. Su función en este proyecto:
- Almacenar excedentes solares que no pueden consumirse inmediatamente
- Suministrar energía almacenada cuando no hay generación solar y hay déficit
- Realizar arbitraje de precios: cargar en horas baratas y descargar en horas caras

El modelo económico implementado es conservador: la batería opera bajo el mismo marco de compensación simplificada. Los ingresos por descarga a red se contabilizan al precio excedentaria (ind. 1739), no al PVPC.

---

## 1. DATASET

### 1.1 Origen y periodo
- **Periodo**: junio 2021 – diciembre 2023 (~22.656 horas, ~31 meses)
- **Resolución**: horaria
- **Ruta**: `data/processed/dataset_final.csv`
- **Motivo del periodo**: cubre la crisis energética europea (2021-2022) con precios extremos y el periodo posterior de normalización, lo que proporciona alta varianza de precios para que el agente aprenda a responder a distintos contextos económicos.

### 1.2 Variables (columnas en orden del CSV)
| Columna | Unidad | Descripción |
|---|---|---|
| `consumo_total` | kWh | Consumo agregado de 15 viviendas |
| `generacion_total` | kWh | Generación solar fotovoltaica agregada |
| `precio_kwh` | EUR/kWh | Precio de compra de red (PVPC, ESIOS ind. 1001) |
| `precio_excedente` | EUR/kWh | Precio de venta de excedentes (ESIOS ind. 1739) |

### 1.3 Fuentes de datos originales

**ESIOS (Sistema de Información del Operador del Sistema)**:
- Portal público de Red Eléctrica de España (REE): `www.esios.ree.es`
- Indicador 1001 (PVPC): precio horario publicado el día anterior para consumidores acogidos a tarifa regulada
- Indicador 1739 (Compensación autoconsumo): precio al que se compensan los excedentes según RD 244/2019
- Indicador de coeficiente de perfil PVPC: distribución horaria del consumo eléctrico residencial español normalizada a 1 kWh/año
- Datos descargados en formato CSV con separador `;` y valores en EUR/MWh

**PVGIS (Photovoltaic Geographical Information System)**:
- Portal de la Comisión Europea: `re.jrc.ec.europa.eu/pvg_tools`
- Proporciona series temporales horarias de generación fotovoltaica para cualquier ubicación de Europa
- Datos para localización representativa de Galicia (noroeste de España), tecnología Si-cristalino, 1 kWp de potencia instalada, inclinación óptima
- Unidades originales: W/hora → convertido a kWh dividiendo entre 1.000
- El perfil de 1 kWp se escala luego a 42 kWp con la distribución entre viviendas

### 1.4 Sistema eléctrico español — actores relevantes

**REE (Red Eléctrica de España)**: Operador del sistema de transporte. Publica los precios PVPC e índices de compensación con 24h de antelación. Garantiza el equilibrio entre generación y consumo en tiempo real.

**OMIE (Operador del Mercado Ibérico de Electricidad)**: Gestiona el mercado spot (Day-Ahead y Intraday). El precio del mercado diario (Pmh, €/MWh) es la base sobre la que se construye tanto el PVPC como el precio de compensación de excedentes.

**Comercializadora de referencia**: Empresa que ofrece la tarifa PVPC. En España, las principales son Endesa, Iberdrola y Naturgy. El PVPC incluye el precio de mercado (Pmh), los peajes de acceso a la red, los cargos del sistema eléctrico (financiación de renovables, déficit tarifario, etc.), el impuesto eléctrico (5,11%) y el IVA (21%).

**Desglose aproximado del PVPC** (precio 0.22 EUR/kWh en hora media):
- Precio de mercado (Pmh): ~0.13 EUR/kWh (60%)
- Peajes de acceso: ~0.04 EUR/kWh (18%)
- Cargos del sistema: ~0.02 EUR/kWh (9%)
- Impuestos (eléctrico + IVA): ~0.03 EUR/kWh (13%)

El precio de compensación de excedentes (ind. 1739) equivale aproximadamente al Pmh menos los costes de desvío, es decir, la parte mayorista sin cargos ni impuestos. Por eso siempre es inferior al PVPC.

### 1.5 Patrones temporales de precios

**Patrón horario del PVPC**:
- Horas valle (00:00–08:00): precios bajos, demanda baja, alta proporción de renovables
- Horas llano (08:00–10:00, 14:00–18:00, 22:00–00:00): precios intermedios
- Horas punta (10:00–14:00, 18:00–22:00): precios altos, máxima demanda residencial e industrial
- La diferencia punta/valle puede ser de 2-5× en días normales y >10× en días de alta demanda invernal

**Patrón estacional**:
- Verano: precios moderados (alta generación solar en toda la península), consumo de calefacción nulo
- Invierno: precios altos (menor generación solar, mayor consumo de calefacción), episodios de precios extremos
- Crisis energética 2021-2022: precios >0.40 EUR/kWh frecuentes, máximo 0.954 EUR/kWh

**Implicación para el agente**: el patrón horario (punta/valle) crea oportunidades de arbitraje diarias. El patrón estacional explica la distribución bimodal de resultados (verano vs invierno).

### 1.6 Generación del dataset (`src/utils/generar_dataset_final.py`)

**Consumo**: El coeficiente de perfil PVPC es una serie normalizada (suma = 1 anual) que describe cómo se distribuye el consumo residencial español a lo largo del año. Se multiplica por el consumo anual por vivienda (3.500 kWh/año) para obtener kWh/hora. Se generan 15 perfiles individuales con variabilidad realista:
- Factor de escala por vecino: `uniform(0.8, 1.2)` → simula vecinos con distinto nivel de consumo (familias grandes vs pequeñas)
- Ruido horario multiplicativo: `normal(1.0, 0.1)` → variabilidad aleatoria hora a hora (electrodomésticos, presencia/ausencia)
- Con 15 vecinos, el ruido individual (~10%) se atenúa al agregarse: std_agregado ≈ 10%/√15 ≈ 2.6%
- Resultado: suma de los 15 perfiles, recortada a ≥ 0

**Solar**: PVGIS proporciona generación para 1 kWp en la ubicación. Se escala a 42 kWp distribuyendo entre 15 instalaciones individuales para modelar la heterogeneidad real de una comunidad:
- Capacidad por vivienda: valores fijos por casa (seed.py), suma exactamente 42 kWp
- Factor de rendimiento por vivienda: `uniform(0.80, 1.0)` → representa diferencias de orientación (sur perfecto ~1.0, sureste/suroeste ~0.90), inclinación subóptima y sombras parciales
- Ruido horario: `normal(1.0, 0.03)` → suciedad puntual, sombras de nubes locales
- Con 15 instalaciones el ruido se atenúa considerablemente (3%/√15 ≈ 0.8% a nivel agregado)
- Resultado: suma de las 15 instalaciones, recortada a ≥ 0
- Justificación 42 kWp: 15 × 3.500 kWh/año ÷ 1.250 h_eq/año (Galicia) = 42 kWp → generación ≈ consumo anual, maximizando autoconsumo bajo compensación simplificada (RD 244/2019)

**Precios**: Descargados directamente de ESIOS sin modificación. Conversión: EUR/MWh ÷ 1.000 = EUR/kWh. No se aplica ningún ruido ni transformación (son datos reales históricos).

**Fusión**: Se toman las primeras `min(len_consumo, len_solar, len_precios_compra, len_precios_venta)` horas de cada fuente y se combinan en un único DataFrame de 4 columnas.

**Semilla fija**: `SEED=42` para reproducibilidad completa de los perfiles de vecinos y la distribución de capacidades solares.

---

## 2. SIMULADOR FÍSICO (`src/core/simulador.py`)

### 2.0 Tecnología de batería modelada

El simulador modela una batería de **iones de litio (Li-ion)**, la tecnología dominante para almacenamiento estacionario a escala comunitaria. Características físicas reales que justifican los parámetros elegidos:

**Eficiencia**: Las baterías Li-ion comerciales tienen eficiencia de carga/descarga entre 92% y 98% por dirección. El valor elegido (95% por dirección, round-trip 90.25%) es representativo de sistemas de almacenamiento residencial/comunitario modernos (ej. Tesla Powerwall: ~90% round-trip, LG RESU: ~95%).

**Autodescarga**: Las baterías Li-ion tienen autodescarga de 1-3% mensual en condiciones normales de temperatura. El valor 0.004%/hora = 2.9% mensual es representativo del extremo superior para modelar condiciones conservadoras.

**SOC operativo**: Operar entre 10% y 90% (en lugar de 0-100%) es estándar en sistemas de almacenamiento para prolongar la vida útil. Los fabricantes (LG, Samsung, BYD) típicamente limitan el rango operativo a 80% de la capacidad nominal.

**Degradación**: Las baterías Li-ion se degradan por:
1. **Estrés por corriente alta** (factor I²R): corrientes elevadas generan calor que acelera la degradación química. Modelado como `1 + (P/P_max)²`.
2. **Estrés por SoC extremo**: operar cerca del 0% (litio metálico) o del 100% (tensión alta) acelera la degradación de los electrodos. Modelado como `1 + 15×(|SoC-0.5|)⁴`.
3. **Coste base**: 0.005 EUR/kWh movido. Para una batería de 100 kWh con 4.000 ciclos completos a 0.5 EUR/kWh de coste de reposición: coste/kWh = 0.5/(4000×100×2) ≈ 0.000625 EUR/kWh. El valor 0.005 EUR/kWh es conservador (8× ese mínimo) para incluir costes de instalación, mantenimiento y reposición.

### Parámetros de la batería
| Parámetro | Valor | Descripción |
|---|---|---|
| `BATERIA_CAPACIDAD` | 100 kWh | Capacidad nominal total |
| `POTENCIA_INVERSOR` | 50 kW | Potencia máxima de carga/descarga |
| `SOC_MIN` | 0.10 (10%) | Límite operativo inferior (protección ciclos profundos) |
| `SOC_MAX` | 0.90 (90%) | Límite operativo superior (protección sobrecargas) |
| `EFICIENCIA_CARGA` | 0.95 (95%) | Pérdidas AC→DC al cargar |
| `EFICIENCIA_DESCARGA` | 0.95 (95%) | Pérdidas DC→AC al descargar |
| `AUTODESCARGA_POR_HORA` | 0.00004 | ~3% mensual, típico de baterías Li-ion |
| `COSTE_DEGRADACION_BASE` | 0.005 EUR/kWh | Coste base de degradación por kWh movido |

**Round-trip efficiency**: 0.95 × 0.95 = 90.25%  
**Rango operativo efectivo**: 10% a 90% → 80 kWh utilizables de los 100 kWh nominales

### Flujo de ejecución por hora (`ejecutar_accion_fisica`)

**Paso 1 — Leer datos del dataset**:
```
gen, cons = generacion_total, consumo_total
precio_compra = precio_kwh        # PVPC (ind. 1001)
precio_venta  = precio_excedente  # Compensación (ind. 1739)
balance       = gen - cons
exc_disp      = max(0, balance)   # kWh solares disponibles para batería o red
def_cub       = max(0, -balance)  # kWh que las casas necesitan y el solar no cubre
```

**Paso 2 — Autodescarga**:
```
soc = soc * (1 - 0.00004)   # se aplica ANTES de cualquier acción
bateria_kwh    = soc * 100
espacio_libre  = max(0, 90 - bateria_kwh)          # hasta SOC_MAX
bat_disponible = max(0, bateria_kwh - 10)          # desde SOC_MIN
```

**Paso 3 — Ejecutar acción** (ver sección de acciones)

**Paso 4 — Economía**:
```
ingresos       = vendido * precio_venta
gastos         = comprado * precio_compra
energia_movida = cargado + descargado
soc_medio      = (soc_antes + soc_nuevo) / 2
coste_deg      = calcular_degradacion_no_lineal(energia_movida, soc_medio)
beneficio      = ingresos - gastos - coste_deg
beneficio_idle = exc_disp * precio_venta - def_cub * precio_compra
beneficio_marginal = beneficio - beneficio_idle
```

### Degradación no lineal

```python
def calcular_degradacion_no_lineal(energia_kwh, soc_actual):
    potencia = energia_kwh                          # proxy de potencia (kWh ≈ kW a 1h)
    ratio    = potencia / 50.0                      # normalizado al inversor
    factor_potencia = 1.0 + ratio**2               # estrés por corriente alta (I²R)
    desviacion      = abs(soc_actual - 0.5)
    factor_soc      = 1.0 + 15.0 * desviacion**4   # estrés en extremos (SoC<0.25 o >0.75)
    return 0.005 * factor_potencia * factor_soc * energia_kwh
```

**Comportamiento**:
- SoC=0.5 (centro): `factor_soc=1.0` → degradación mínima
- SoC=0.1 o 0.9 (extremos): `factor_soc=1+15×0.4⁴=1+0.384=1.384` → +38% degradación
- Potencia baja (10 kWh): `factor_potencia=1+0.04=1.04` → casi sin penalización
- Potencia alta (50 kWh): `factor_potencia=1+1.0=2.0` → el doble de degradación

---

## 3. ESPACIO DE ACCIONES (9 acciones)

### Mapa de acciones
| Índice | Estrategia | Nivel | Potencia objetivo |
|---|---|---|---|
| 0 | IDLE | — | 0 kW |
| 1 | CARGAR_SOLAR | único | limitado por exc_disp |
| 2 | CARGAR_MIXTA | 33% | 16.5 kW |
| 3 | CARGAR_MIXTA | 66% | 33.0 kW |
| 4 | CARGAR_MIXTA | 100% | 50.0 kW |
| 5 | DESCARGAR_CASA | único | limitado por def_cub |
| 6 | DESCARGAR_RED | 33% | 16.5 kW |
| 7 | DESCARGAR_RED | 66% | 33.0 kW |
| 8 | DESCARGAR_RED | 100% | 50.0 kW |

### Lógica detallada por estrategia

**IDLE (acción 0)**:
- Vende todo el excedente solar a red: `vendido = exc_disp`
- Compra todo el déficit de red: `comprado = def_cub`
- La batería no se toca (solo sufre autodescarga)

**CARGAR_SOLAR (acción 1)**:
- Carga en batería hasta `min(exc_disp, espacio_libre / EFF_C)` kWh
- El excedente restante (si la batería se llena) se vende a red
- Sigue comprando el déficit de las casas
- No compra de red para cargar la batería
- El nivel de potencia no se aplica porque el cuello de botella es siempre exc_disp (raramente supera 16.5 kW en promedio), no el inversor

**CARGAR_MIXTA (acciones 2, 3, 4)**:
- Carga hasta `min(espacio_libre / EFF_C, potencia_obj)` kWh
- Usa primero excedente solar; si no es suficiente, compra de red para completar
- Sigue comprando el déficit de las casas
- El excedente solar sobrante (si potencia_obj < exc_disp) se vende

**DESCARGAR_CASA (acción 5)**:
- Descarga de batería para cubrir déficit: `descarga = min(def_cub / EFF_D, bat_disponible)`
- Reduce la compra de red en `descarga × EFF_D` kWh
- Vende todo el excedente solar a red
- No vende a red desde la batería
- El nivel de potencia no se aplica (mismo razonamiento que CARGAR_SOLAR)

**DESCARGAR_RED (acciones 6, 7, 8)**:
- Descarga hasta `min(bat_disponible, potencia_obj)` kWh de la batería
- Flujo en cascada: primero cubre déficit de casas, el resto va a red
- `energia_util = descarga_total × EFF_D`
- `para_casa = min(energia_util, def_cub)`
- `para_red  = energia_util - para_casa`
- Vende: `para_red + exc_disp`
- Compra: `def_cub - para_casa`

---

## 4. MODELO DE PRECIOS

### Asimetría compra/venta (regulación española RD 244/2019)

**Compra de red** — precio PVPC (indicador ESIOS 1001):
- Precio retail con todos los cargos (peajes, impuestos, margen comercializadora)
- Varía cada hora, publicado por REE el día anterior
- Rango típico en el dataset: 0.05 – 0.45 EUR/kWh según hora y temporada
- Media aproximada en el dataset: ~0.217 EUR/kWh

**Venta de excedentes** — compensación simplificada (indicador ESIOS 1739):
- Precio mayorista OMIE menos costes de desvío (Pmh - CDSVh)
- Siempre inferior al precio de compra (precio de mercado sin márgenes retail)
- Media aproximada en el dataset: ~0.132 EUR/kWh
- **La asimetría implica que comprar de red para vender siempre pierde dinero antes de contar degradación**

### Consecuencia para la estrategia óptima
El precio de compra es ~1.64× el precio de venta en media. Por tanto:
- Cargar desde red (CARGAR_MIXTA) solo es rentable si el precio en el momento de descarga es suficientemente alto para recuperar la asimetría y la degradación
- El autoconsumo (cargar solar → descargar en casa) evita tanto la compra como la pérdida de venta → doble beneficio marginal

---

## 5. ENTORNO GYMNASIUM (`src/envs/energy_env.py`)

### Espacio de observación (101 dimensiones)

**Primeras 5 — estado actual exacto (sin ruido)**:
| Índice | Variable | Descripción |
|---|---|---|
| 0 | `soc` | Estado de carga actual [0, 1] |
| 1 | `precio_compra` | PVPC actual en EUR/kWh |
| 2 | `precio_venta` | Excedentaria actual en EUR/kWh |
| 3 | `exc` | Excedente solar disponible ahora (kWh) |
| 4 | `def_` | Déficit de consumo ahora (kWh) |

**Siguientes 96 — pronóstico 24h futuras (con ruido)**:
- 24 horas × 4 variables = 96 valores
- Orden por hora: `[consumo_t+1, generacion_t+1, precio_compra_t+1, precio_venta_t+1, consumo_t+2, ...]`
- Los precios (índices 2 y 3 de cada hora) son exactos (sin ruido)
- Consumo y generación llevan ruido AR(1) (ver sección 6)

### Espacio de acciones
`Discrete(9)` — índices 0 a 8 según la tabla de acciones

### Recompensa

**Recompensa base** (por paso):
```
reward = beneficio_marginal = beneficio_accion - beneficio_idle
```
Donde `beneficio_idle = exc_disp × precio_venta - def_cub × precio_compra` (lo que ganaría sin batería)

**Corrección en step 0** (simetría de valor):
```
reward -= energia_inicial × precio_compra × EFF_D
```
El agente "paga" el valor de la energía con la que empieza para evitar que distintos SoC iniciales creen ventajas artificiales.

**Corrección en step final** (step 167):
```
reward += energia_restante × precio_compra_actual × EFF_D
```
La energía que queda en la batería al terminar el episodio tiene valor — el agente lo recupera.

**Limitación conocida**: con γ=0.995, el terminal bonus vale γ^167=43.2% en términos descontados. El agente paga el coste inicial al 100% pero solo recupera el 43.2% del terminal. Esto crea un sesgo leve hacia vaciar la batería al final.

### Estructura del episodio
- **Longitud**: 168 pasos (1 semana = 24h × 7 días)
- **Inicio**: posición aleatoria en el dataset con margen de 25h extra al final
- **SoC inicial**: aleatorio `uniform(SOC_MIN + 0.05, SOC_MAX - 0.05)` = uniform(0.15, 0.85)
- **`terminated`**: True en step 168
- **`truncated`**: siempre False

---

## 6. MODELO DE RUIDO EN EL PRONÓSTICO (AR(1))

### Proceso
El error de pronóstico sigue un proceso autorregresivo de orden 1 con varianza estacionaria unitaria:
```
ε_t = ρ × ε_{t-1} + √(1 - ρ²) × N(0,1)
```
Esto garantiza que la varianza de ε se mantiene en 1 en estado estacionario.

### Parámetros
| Variable | ρ (autocorrelación) | σ (escala) | Justificación |
|---|---|---|---|
| Solar (hora h) | 0.7 | 0.05 + h×(0.20/23) → 5% a 25% | Las nubes persisten (alta ρ); error crece con el horizonte |
| Consumo | 0.3 | 10% fijo | Patrón horario domina sobre inercia; 15 hogares atenúan varianza |
| Precios | — | 0% (sin ruido) | PVPC e ind.1739 publicados por REE el día anterior |

### Aplicación al pronóstico
```python
for h in range(24):
    sigma_sol = 0.05 + h × (0.20 / 23)   # crece de 5% a 25%
    generacion[h] = max(0, generacion[h] × (1 + error_solar × sigma_sol))
    consumo[h]    = max(0, consumo[h] × (1 + error_cons × 0.10))
```

### Ciclo de vida del error AR(1)
- **En `reset()`**: `error_solar = 0.0`, `error_cons = 0.0` (se reinicia cada episodio)
- **En `step()` al inicio**: se actualiza el error ANTES de construir la observación
- El mismo estado AR(1) se usa para todas las 24h del pronóstico del step actual

---

## 7. ENTRENAMIENTO DQN (`src/main.py`)

### Versión actual: DQN_12 (v10)

### Hiperparámetros
| Parámetro | Valor | Justificación |
|---|---|---|
| `TOTAL_TIMESTEPS` | 3.000.000 | γ=0.995 necesita más explotación para converger |
| `LEARNING_RATE` | 1e-4 | Validado vs 5e-5 (peor) |
| `BUFFER_SIZE` | 200.000 | ~1.190 episodios completos en buffer |
| `LEARNING_STARTS` | 10.000 | Exploración inicial antes de entrenar |
| `BATCH_SIZE` | 64 | Validado vs 128 (más actualizaciones con 64) |
| `GAMMA` | 0.995 | H_eff≈200h; cubre episodio 168h; credito 48h: 78.6% |
| `EXPLORATION_FRAC` | 0.5 | Exploración termina en 1.5M; explotación = 1.5M |
| `EXPLORATION_FINAL` | 0.05 | ε mínimo tras exploración |
| `TARGET_UPDATE` | 1.000 | Pasos entre sincronización de target network |
| `TRAIN_FREQ` | 4 | Actualizar red cada 4 pasos de entorno |
| `NET_ARCH` | [64, 64] | Red 101→64→64→9, ~11.273 parámetros |

### Arquitectura de la red neuronal
```
Input (101) → Linear(101→64) → ReLU → Linear(64→64) → ReLU → Linear(64→9)
```
~11.273 parámetros entrenables. Implementación: MlpPolicy de SB3.

### Exploración ε-greedy lineal
```
ε(t) = 1.0  si t < LEARNING_STARTS
ε(t) = 1.0 - (0.95 / 1.500.000) × t   para t entre 10k y 1.5M
ε(t) = 0.05 para t > 1.5M
```

### Normalización de observaciones (VecNormalize)
- `norm_obs=True`: normaliza cada dimensión del vector de observación con media y std rolling
- `norm_reward=False`: la recompensa NO se normaliza (preserva la señal económica real)
- `clip_obs=10.0`: corta en ±10 sigmas (en la práctica nunca corta)
- Las estadísticas de normalización se sincronizan entre entorno de entrenamiento y evaluación

### Callbacks
**EvalCallback**:
- Frecuencia: cada 20.000 pasos
- Episodios por evaluación: 50 (determinista, ε=0)
- Guarda el mejor modelo en `models/best_model.zip`
- Guarda evaluaciones en `logs/evaluations.npz`

**MetricasCallback** (personalizado):
- Cada 1.000 pasos loguea en TensorBoard:
  - SoC medio, mínimo y máximo
  - Energía media cargada y descargada
  - Energía media comprada a red
  - % de cada acción por grupo (IDLE, CARGAR_SOLAR, CARGAR_MIXTA, DESCARGAR_CASA, DESCARGAR_RED)
  - % de cada acción individual (0-8)
  - ε actual

---

## 8. BENCHMARK MPC (`src/benchmarks/mpc_benchmark.py`)

### Descripción
Model Predictive Control con horizonte deslizante de 24h. Resuelve un LP en cada hora y ejecuta solo la primera acción. Dos variantes:

**MPC previsión perfecta**: cota superior teórica (conoce el futuro exacto)
**MPC con ruido AR(1)**: comparación justa con el DQN (mismo modelo de error)

### Variables del LP (por hora h, total 96)
- `cs[h]`: kWh de excedente solar → batería (entrada al inversor)
- `cm[h]`: kWh comprados de red → batería (entrada al inversor)
- `dc[h]`: kWh extraídos batería → casas (salida de la batería)
- `dr[h]`: kWh extraídos batería → red (salida de la batería)

### Simplificaciones del LP vs simulador
- Degradación linealizada (usa `COSTE_DEGRADACION_BASE` sin factores de estrés)
- Autodescarga ignorada (<0.07% semanal, despreciable)
- En la ejecución real se aplica la física completa del simulador

### Resultados obtenidos (100 episodios de 1 semana, seed=42)
| Variante | Beneficio marginal (DQN − IDLE) |
|---|---|
| MPC previsión perfecta | +54.38 ± 26.94 EUR/semana |
| MPC con ruido AR(1) | +48.46 ± 26.87 EUR/semana |
| Coste del ruido | 5.92 EUR/semana |

**Objetivo del DQN**: acercarse a 48.46 EUR/semana (MPC con mismo ruido)

---

## 9. ALGORITMO DQN — FUNDAMENTOS

### Deep Q-Network (Mnih et al., 2015)

DQN aprende una función Q(s, a) que estima el retorno esperado descontado al tomar acción `a` en estado `s` y seguir la política óptima a partir de ahí:

```
Q*(s, a) = E[ r_t + γ·r_{t+1} + γ²·r_{t+2} + ... ]
```

La red neuronal aproxima Q*(s, a) con parámetros θ. Se entrena minimizando el error de Bellman:

```
L(θ) = E[ (y - Q(s, a; θ))² ]
donde y = r + γ · max_{a'} Q(s', a'; θ⁻)
```

`θ⁻` son los parámetros de la **target network**: una copia de la red que se actualiza cada `TARGET_UPDATE=1.000` pasos para estabilizar el entrenamiento.

### Double DQN (Van Hasselt et al., 2016)
SB3 implementa Double DQN por defecto. La diferencia respecto al DQN clásico:

```
# DQN clásico (sobreestima Q):
y = r + γ · Q(s', argmax_{a'} Q(s', a'; θ⁻); θ⁻)

# Double DQN (desacopla selección y evaluación):
y = r + γ · Q(s', argmax_{a'} Q(s', a'; θ); θ⁻)
```

La red online selecciona la acción, la red target evalúa su valor. Reduce la sobreestimación sistemática de Q-values.

### Experience Replay
Las transiciones `(s, a, r, s', done)` se almacenan en un buffer circular de 200.000 entradas. En cada actualización se muestrea un minibatch aleatorio de 64 transiciones. Ventajas:
- Rompe la correlación temporal entre transiciones consecutivas
- Permite reusar cada transición múltiples veces
- Estabiliza el aprendizaje

### Exploración ε-greedy
```
P(acción aleatoria) = ε(t)    → exploración
P(acción óptima)   = 1 - ε(t) → explotación
```
ε decrece linealmente desde 1.0 hasta 0.05 en los primeros 1.5M pasos, luego se mantiene en 0.05.

### Horizonte efectivo
Con factor de descuento γ, el horizonte efectivo de planificación es aproximadamente:
```
H_eff ≈ 1 / (1 - γ)
```
- γ=0.99: H_eff ≈ 100 pasos (100 horas)
- γ=0.995: H_eff ≈ 200 pasos (200 horas) → bien calibrado para episodios de 168h

### Problema de crédito tardío en este dominio
El DQN usa actualizaciones Bellman de 1 paso. Para un ciclo de arbitraje:
```
CARGAR_MIXTA (hora 0): reward = -2 EUR  ← penalización inmediata
...
DESCARGAR_CASA (hora 48): reward = +4 EUR ← beneficio tardío
```
El Q-value de CARGAR_MIXTA se actualiza a partir del Q-value del estado siguiente, no directamente del reward de hora 48. Con γ=0.99, ese reward vale 0.617 en el presente → incentivo neto de solo +0.47 EUR. Con γ=0.995 vale 0.786 → incentivo neto de +1.14 EUR (2.4× más señal).

---

## 10. ESTADÍSTICAS DEL DATASET

### Valores reales del dataset (22.656 horas, jun 2021 – dic 2023)

| Variable | Min | Media | Mediana | Max | Std |
|---|---|---|---|---|---|
| consumo_total (kWh) | 2.09 | 5.95 | 5.96 | 12.04 | 1.65 |
| generacion_total (kWh) | 0.00 | 8.06 | 0.00 | 45.69 | 12.22 |
| precio_kwh EUR/kWh | 0.012 | 0.2172 | 0.2024 | 0.954 | 0.105 |
| precio_excedente EUR/kWh | -0.003 | 0.1324 | 0.1204 | 0.703 | 0.071 |

### Observaciones clave
- **Generación solar**: mediana = 0 (más del 50% de horas son nocturnas o muy nubladas)
- **Horas con excedente solar** (gen > cons): 7.596 horas = **33.5%** del total
- **Horas con déficit** (gen < cons): 15.050 horas = **66.5%** del total
- **Excedente medio** cuando hay: 16.68 kWh (puede llenar 16.7% de la batería en 1h)
- **Déficit medio** cuando hay: 5.24 kWh (la batería puede cubrirlo completamente)
- **Ratio precio compra/venta**: 1.641× en media (comprar es 64% más caro que vender)
- **Precio compra máximo**: 0.954 EUR/kWh (crisis energética 2021-2022)
- **Precio venta negativo**: -0.003 EUR/kWh en horas de exceso renovable en la red (el sistema eléctrico "cobra" por aceptar energía)

### Implicaciones para la estrategia óptima
- En el 66.5% de horas el sistema tiene déficit → la batería puede ahorrar compras de red
- En el 33.5% de horas hay excedente → la batería puede almacenar solar para usarlo después
- El precio de venta negativo (aunque raro) implica que en esas horas **no** se debe vender excedente — lo óptimo es almacenarlo aunque sea a coste de degradación
- La alta varianza del precio de compra (std=0.105) crea oportunidades de arbitraje entre horas baratas (<0.14 EUR) y caras (>0.28 EUR)

---

## 11. HISTORIAL DE VERSIONES DQN

| Versión | Cambios principales | Resultado |
|---|---|---|
| DQN_1–4 | Exploraciones iniciales de arquitectura y datos | No documentado formalmente |
| DQN_5 | Primera versión con datos multi-año (jun21–dic23), 13 acciones | Baseline inicial |
| DQN_6 | Ajuste de learning rate: 5e-5 → 1e-4 | Mejora significativa |
| DQN_7 | Buffer 100k → 200k; batch 128 → 64 | Mejora por más diversidad |
| DQN_8 | 13 acciones → 9 (eliminadas degeneradas); EXPLORATION_FRAC=0.6 | Red 64×64 validada |
| DQN_9 | Ruido AR(1) en pronóstico; SoC inicial aleatorio; corrección terminal | Realismo aumentado |
| DQN_10 | TOTAL_TIMESTEPS: 1.0M → 1.5M | Seguía mejorando al acabar |
| DQN_11 | TOTAL_TIMESTEPS: 1.5M → 2.0M; EVAL_FREQ: 15k → 20k | Plateau ~39 EUR/sem; γ identificado como cuello de botella |
| **DQN_12** | **γ: 0.99 → 0.995; 3M pasos; EXPLORATION_FRAC=0.5; 50 eps eval** | **En curso** |

---

## 12. MÉTRICAS Y COMPARACIÓN

### Métrica principal
**Beneficio marginal vs IDLE** (EUR/semana):
```
beneficio_marginal = beneficio_con_bateria - beneficio_sin_bateria
```
Es la misma métrica que la recompensa del DQN acumulada por episodio (sin descuento).

### Escala de referencia completa
| Referencia | EUR/semana | Notas |
|---|---|---|
| IDLE (sin batería) | 0.00 | Baseline — vende excedentes y compra déficit sin batería |
| DQN_11 (γ=0.99, 2M) | ~39.38 | Explotación media; plateau sin mejora |
| DQN_12 (γ=0.995, 3M) | — | En curso (680k/3M pasos a fecha de escritura) |
| MPC con ruido AR(1) | 48.46 | Objetivo del DQN — comparación justa |
| MPC previsión perfecta | 54.38 | Cota superior teórica |

### Distribución de episodios DQN_11 (fase explotación)
- < 20 EUR: 16.9% (episodios de invierno, sin solar, arbitraje difícil)
- 20–40 EUR: 52.4% (episodios moderados)
- 40–60 EUR: 16.0% (episodios buenos)
- \> 60 EUR: 14.7% (episodios de verano con alto solar)

La distribución bimodal (invierno vs verano) explica la alta varianza (std ~20-40 EUR/sem).

---

## 13. JUSTIFICACIÓN DEL DISEÑO DEL ESPACIO DE OBSERVACIÓN

### 13.1 Por qué 101 dimensiones

El vector de observación combina estado actual (5 variables) + pronóstico 24h (96 variables). Alternativas consideradas y descartadas:

**Solo estado actual (5 variables)**: el agente no podría planificar ciclos de carga/descarga porque no sabe si en las próximas horas habrá solar o precios altos. Conduciría a política reactiva, no proactiva.

**Pronóstico de 48h (197 variables)**: el error del pronóstico solar a h=48 supera el 45% (extrapolando el modelo lineal de σ). La señal adicional estaría muy contaminada por ruido, y la red necesitaría procesar 96 entradas adicionales de baja calidad informacional. La ventana de 24h coincide además con el horizonte de publicación real del PVPC (REE publica precios con 24h de antelación).

**Incluir hora del día / mes**: la información temporal ya está implícita en los valores de las variables (precios altos = horas punta, solar alto = verano/mediodía). Añadir variables temporales redundantes podría introducir sobreajuste a patrones calendáricos específicos del dataset en lugar de aprender la física subyacente.

### 13.2 Orden de las variables en el pronóstico

Las 96 variables del pronóstico están ordenadas por hora primero, luego por variable dentro de cada hora:
```
[h+1_consumo, h+1_generacion, h+1_precio_compra, h+1_precio_venta,
 h+2_consumo, h+2_generacion, h+2_precio_compra, h+2_precio_venta,
 ...
 h+24_consumo, h+24_generacion, h+24_precio_compra, h+24_precio_venta]
```
Este orden (hora-mayor) es natural para que la red pueda aprender dependencias temporales entre horas consecutivas.

### 13.3 Por qué los precios actuales están en las primeras 5

Los precios del pronóstico (h+1 a h+24) no incluyen la hora actual (h=0). Los precios actuales están en las primeras 5 variables (posiciones 1 y 2). Esto evita duplicar información y garantiza que el estado actual se representa con precisión absoluta (sin ruido).

---

## 14. JUSTIFICACIÓN DEL DISEÑO DE LA RECOMPENSA

### 14.1 Beneficio marginal vs beneficio absoluto

La recompensa es `beneficio_marginal = beneficio_accion - beneficio_idle`, no el beneficio absoluto.

**Problema del beneficio absoluto**: en horas de alto excedente solar (verano, mediodía), el beneficio absoluto de IDLE ya es alto (ej. +3 EUR/h por vender excedente). El agente recibiría recompensas positivas grandes incluso sin hacer nada útil con la batería. La varianza entre episodios de verano e invierno sería enorme, dificultando el aprendizaje.

**Ventaja del beneficio marginal**: el baseline IDLE ya descuenta lo que el sistema ganaría sin batería. El agente solo recibe recompensa positiva cuando su acción *mejora* el resultado respecto a no hacer nada. Reduce la varianza de la señal de aprendizaje, acelerando la convergencia. Esta técnica se conoce en RL como *advantage-based reward shaping*.

### 14.2 Corrección inicial/terminal — motivación

**Problema sin corrección**: si el agente empieza con SoC=0.80 y el episodio dura 168h, tiene ~56 kWh "gratis" en la batería que puede vender para obtener ingresos. Un agente que empieza con SoC=0.20 tiene solo ~7 kWh. La diferencia de recompensa acumulada no refleja la calidad de la política, sino la lotería del SoC inicial. Esto añade ruido al entrenamiento.

**Corrección inicial**: al inicio del episodio, se descuenta del reward del primer paso el valor de la energía inicial:
```python
energia_inicial = max(0, soc_inicial - SOC_MIN) * BATERIA_CAPACIDAD
coste_inicial   = energia_inicial * precio_compra * EFF_D
reward[0] -= coste_inicial
```
El agente "paga" la energía con la que empieza al precio de mercado actual ajustado por eficiencia de descarga. Esto iguala el punto de partida entre episodios.

**Corrección terminal**: al final del episodio, se suma el valor de la energía restante:
```python
energia_final = max(0, soc_final - SOC_MIN) * BATERIA_CAPACIDAD
valor_final   = energia_final * precio_compra_actual * EFF_D
reward[167] += valor_final
```
Sin esta corrección, el agente aprendería a vaciar la batería al final del episodio (para no "desperdiciar" el valor terminal). Con la corrección, mantener energía en la batería al final es tan bueno como haberla descargado.

**Limitación conocida con γ=0.995**: el coste inicial se paga al 100% (en step 0, sin descuento). El valor terminal se recupera descontado: `γ^167 = 0.995^167 ≈ 0.432`. Esto crea un sesgo residual de `coste_inicial × (1 - 0.432)` contra mantener energía en la batería. Para un SoC inicial de 0.5 con precio 0.22 EUR/kWh:
```
sesgo = (0.5 - 0.10) × 100 × 0.22 × 0.95 × (1 - 0.432) ≈ -4.74 EUR/episodio
```
Este sesgo es aceptable y se reduciría con γ más alto (pero γ=0.999 crearía inestabilidad en el entrenamiento).

### 14.3 Por qué no normalizar la recompensa

`norm_reward=False` en VecNormalize. Si se normalizara la recompensa con media y std rolling:
- La escala de la señal económica (EUR/hora) se perdería
- Comparar diferentes fases del entrenamiento (exploración vs explotación) sería imposible
- Los callbacks de evaluación mostrarían valores normalizados sin unidades económicas reales
- Dificultaría la comparación directa con el benchmark MPC

### 14.4 Diagnóstico de recompensas por estrategia

Ejecutado con `src/benchmarks/diagnostico_recompensa.py` sobre el dataset completo (resultados discounted con γ=0.99 y episodio de 168h):

| Estrategia | Reward marginal medio por episodio |
|---|---|
| DESCARGAR_RED (siempre) | -6.25 EUR/sem |
| IDLE (siempre) | -7.82 EUR/sem |
| DESCARGAR_CASA (siempre) | +2.10 EUR/sem |
| CARGAR_SOLAR (siempre) | +4.80 EUR/sem |
| Autoconsumo (CARGAR_SOLAR→DESCARGAR_CASA) | +5.61 EUR/sem |

**Interpretación**: IDLE aparece peor que DESCARGAR_RED porque el sesgo del terminal discount penaliza la acumulación de energía. La estrategia óptima en términos descontados es el autoconsumo (cargar solar durante el día, descargar en casa por la noche).

---

## 15. STABLE BASELINES3 — DETALLES DE IMPLEMENTACIÓN

### 15.1 Stack de wrappers del entorno

```python
# Orden de wrappers (de interior a exterior):
EnergyEnv()                    # Entorno base (Gymnasium)
  → Monitor(EnergyEnv())       # Registra episodios en CSV (monitor.csv)
    → DummyVecEnv([...])       # Vectoriza 1 entorno (necesario para SB3)
      → VecNormalize(...)      # Normaliza observaciones con stats rolling
```

**Monitor**: registra `ep_rew_mean` (recompensa media por episodio) y `ep_len_mean` (longitud media) en `train.monitor.csv`. Estos son los valores que aparecen en los logs de SB3 durante el entrenamiento.

**DummyVecEnv**: SB3 requiere entornos vectorizados. DummyVecEnv ejecuta los entornos en el mismo proceso (no en paralelo). Con N=1 entorno no hay paralelismo real, pero sí compatibilidad con la API de SB3.

**VecNormalize**: mantiene una media y varianza running para cada una de las 101 dimensiones de observación. Normaliza: `obs_norm = clip((obs - mean) / sqrt(var + eps), -10, 10)`. Las estadísticas se actualizan solo durante el entrenamiento (no durante la evaluación).

### 15.2 Sincronización de VecNormalize en evaluación

El EvalCallback sincroniza automáticamente las estadísticas del VecNormalize del entorno de entrenamiento al entorno de evaluación antes de cada evaluación. Esto garantiza que el agente ve las mismas observaciones normalizadas en evaluación que en entrenamiento.

Si no se hiciera esta sincronización, las observaciones de evaluación estarían normalizadas con estadísticas distintas (el entorno de evaluación tiene sus propias stats), lo que haría que el agente viera estados "diferentes" durante la evaluación, degradando artificialmente los resultados.

### 15.3 check_env

Al inicio de `main.py` se llama a `check_env(EnergyEnv())`. Este verificador de SB3 comprueba:
- Que `observation_space` y `action_space` están correctamente definidos
- Que las observaciones devueltas por `reset()` y `step()` tienen el dtype y shape correcto
- Que `terminated` y `truncated` son booleanos
- Que la recompensa es un escalar
- Que el entorno es reproducible con `seed`

### 15.4 Inferencia del modelo entrenado

Para cargar y usar el modelo entrenado son necesarios **dos archivos**:
```python
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Cargar entorno con las estadísticas de normalización del entrenamiento
env = VecNormalize.load("models/vec_normalize.pkl",
                        DummyVecEnv([lambda: Monitor(EnergyEnv())]))
env.training = False      # No actualizar las stats durante inferencia
env.norm_reward = False

# Cargar el mejor modelo (o el final)
model = DQN.load("models/best_model.zip", env=env)

# Inferencia determinista
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

Si se carga `best_model.zip` sin `vec_normalize.pkl`, las observaciones no estarán normalizadas correctamente y la política producirá acciones incorrectas (el modelo vería valores en escalas completamente distintas a las del entrenamiento).

### 15.5 Logs de TensorBoard

SB3 crea una subcarpeta por cada run con nombre incremental (`DQN_1`, `DQN_2`, ..., `DQN_12`). Para visualizar:
```bash
tensorboard --logdir logs/
```

Métricas disponibles por defecto en SB3:
- `train/loss`: pérdida TD del DQN
- `train/n_updates`: número de actualizaciones de la red
- `train/exploration_rate`: ε actual
- `eval/mean_reward`: recompensa media en evaluación
- `eval/mean_ep_length`: longitud media de episodio en evaluación

Métricas adicionales del MetricasCallback:
- `custom/soc_medio`, `custom/soc_minimo`, `custom/soc_maximo`
- `custom/cargado_medio`, `custom/descargado_medio`, `custom/comprado_medio`
- `custom/epsilon`
- `acciones/IDLE_pct`, `acciones/CARGAR_SOLAR_pct`, etc.
- `acciones/accion_00_pct` a `acciones/accion_08_pct`
- `acciones/dominante_idx`, `acciones/dominante_pct`

---

## 16. ESTRATEGIAS ÓPTIMAS — REGLAS HEURÍSTICAS

Esta sección describe cuándo debería elegir cada acción el agente óptimo, en base a la física y la economía del sistema. Útil como referencia para auditar el comportamiento del DQN entrenado.

### Condiciones por estrategia

**IDLE (acción 0)** — óptimo cuando:
- Hay excedente solar y la batería está llena (no cabe más)
- Hay déficit y la batería está vacía (no hay nada que descargar)
- Los precios futuros son similares a los actuales (no hay oportunidad de arbitraje)
- La degradación de mover energía supera el beneficio esperado

**CARGAR_SOLAR (acción 1)** — óptimo cuando:
- Hay excedente solar disponible (`exc_disp > 0`)
- La batería tiene espacio libre (`soc < SOC_MAX`)
- El precio de venta actual es inferior al precio de compra futuro ajustado por eficiencia: `precio_venta × EFF_C × EFF_D < precio_compra_futuro`
- En la práctica, dada la asimetría de precios (compra ~1.64× venta), casi siempre es mejor almacenar el excedente solar que venderlo, salvo que la batería esté llena

**CARGAR_MIXTA (acciones 2-4)** — óptimo cuando:
- El precio de compra actual es bajo (hora valle, <25% percentil ~0.14 EUR/kWh)
- Se prevé un precio de descarga significativamente más alto en las próximas horas (arbitraje)
- La condición de rentabilidad: `precio_compra_actual × (1/EFF_C) × (1/EFF_D) + coste_degradacion < precio_descarga_futuro`
- Con los parámetros del sistema: `precio_compra_actual / (0.95×0.95) + ~0.005×factor_stress < precio_futuro`
- Aproximadamente: precio actual < precio futuro × 0.90 − 0.005
- La diferencia punta/valle necesaria para justificar CARGAR_MIXTA es ≥ 10-15%

**DESCARGAR_CASA (acción 5)** — óptimo cuando:
- Hay déficit de consumo (`def_cub > 0`)
- La batería tiene energía disponible
- El precio de compra actual es alto (mejor usar la batería que comprar a red cara)
- Siempre preferible a DESCARGAR_RED cuando hay déficit, porque cubre la necesidad directamente sin pérdidas de venta

**DESCARGAR_RED (acciones 6-8)** — óptimo cuando:
- Hay batería disponible
- El precio de venta actual es alto
- No hay déficit de casa que cubrir (o es pequeño comparado con la batería disponible)
- La condición de rentabilidad: `precio_venta_actual × EFF_D − coste_degradacion > 0`
- Con degradación base: precio_venta > 0.005/0.95 ≈ 0.00526 EUR/kWh → casi siempre rentable en términos puros
- Pero la asimetría con el precio de compra hace que DESCARGAR_RED solo tenga sentido en horas de precio de venta alto (>0.20 EUR/kWh)

### Tabla de decisión simplificada

| Situación | Acción recomendada |
|---|---|
| exc_disp>0, batería con espacio, precio_futuro>precio_venta_actual | CARGAR_SOLAR |
| exc_disp>0, batería llena | IDLE |
| def_cub>0, batería disponible, precio_compra alto | DESCARGAR_CASA |
| precio_compra bajo + precio_futuro alto (arbitraje) | CARGAR_MIXTA |
| precio_venta alto + batería disponible + sin déficit | DESCARGAR_RED |
| Sin oportunidades claras | IDLE |

---

## 17. LIMITACIONES Y SIMPLIFICACIONES DEL MODELO

### 17.1 Lo que el simulador NO modela

**Restricciones de red**: en la realidad, el transformador de la comunidad tiene capacidad limitada. Inyectar o absorber potencia muy alta puede crear problemas en la red de distribución. El simulador asume que siempre se puede operar a plena potencia del inversor (50 kW).

**Temperatura de la batería**: la eficiencia y degradación de las baterías Li-ion dependen fuertemente de la temperatura. A temperaturas bajas (<10°C) la eficiencia puede caer al 80-85%. El modelo usa eficiencias constantes (95%).

**Vida útil de la batería**: el coste de degradación se modela como coste por kWh movido, sin acumulación de daño total ni reemplazo. En la realidad, después de ~4.000 ciclos completos la capacidad cae al 80% y requiere sustitución.

**Restricciones de potencia diferenciadas carga/descarga**: el simulador usa la misma potencia máxima (50 kW) para cargar y descargar. En la práctica, los inversores pueden tener límites asimétricos.

**Curva C-rate real**: la degradación por corriente alta es más compleja que `1 + (P/P_max)²`. En la realidad sigue una curva de Arrhenius no lineal.

**Pérdidas en el inversor a potencia baja**: los inversores tienen menor eficiencia a potencias muy bajas (ej. 30% del nominal). El modelo usa eficiencia constante del 95%.

**Coste de la instalación solar y la batería**: el simulador solo modela los flujos económicos hora a hora. No incluye la amortización del CAPEX (coste de la instalación), que en la realidad puede ser de 1.000-1.500 EUR/kWp para solar y 400-800 EUR/kWh para batería.

**Variación de precio en tiempo real (intraday)**: el PVPC puede tener correcciones intradiarias. El simulador usa el precio del mercado day-ahead como definitivo.

### 17.2 Simplificaciones justificadas

**Autodescarga constante**: la autodescarga real varía con la temperatura y el SoC. La constante 0.00004/h es una aproximación razonable para modelar pérdidas de autoconsumo sin añadir complejidad.

**Precios sin incertidumbre**: el PVPC real se publica el día anterior pero puede tener pequeñas desviaciones por costes adicionales (regulación, etc.). Para este proyecto se asume que el precio publicado es el definitivo.

**Reparto de generación equitativo**: se asume que los 15 vecinos tienen coeficientes de reparto iguales (1/15). El RD 244/2019 permite coeficientes variables, pero la simplificación es coherente con el objetivo del proyecto (gestión agregada de la comunidad).

**Un único punto de conexión a red**: se modela la comunidad como un único consumidor/generador ante la red. En la realidad, cada vivienda tiene su propio contador y conexión, aunque el autoconsumo colectivo virtualiza el reparto.

---

## 18. ARCHIVOS DEL PROYECTO

```
TFG/
├── data/
│   ├── raw/                          # Datos originales ESIOS y PVGIS
│   └── processed/dataset_final.csv  # Dataset generado (22.656 horas × 4 variables)
├── src/
│   ├── core/simulador.py             # Motor físico de la comunidad
│   ├── envs/energy_env.py            # Entorno Gymnasium para SB3
│   ├── main.py                       # Orquestador de entrenamiento DQN
│   ├── benchmarks/
│   │   └── mpc_benchmark.py          # Benchmark MPC (LP con scipy HiGHS)
│   └── utils/
│       └── generar_dataset_final.py  # Preprocesado y fusión del dataset
├── models/
│   ├── best_model.zip                # Mejor checkpoint durante entrenamiento
│   ├── dqn_sgec.zip                  # Modelo final al acabar entrenamiento
│   └── vec_normalize.pkl             # Estadísticas de normalización (necesarias para inferencia)
└── logs/
    ├── evaluations.npz               # Array numpy con todas las evaluaciones
    ├── DQN_1/ … DQN_12/             # Logs TensorBoard por entrenamiento
    ├── eval.monitor.csv              # Monitor del entorno de evaluación
    └── train.monitor.csv             # Monitor del entorno de entrenamiento
```
