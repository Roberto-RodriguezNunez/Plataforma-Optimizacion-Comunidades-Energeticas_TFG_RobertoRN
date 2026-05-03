# Cambios pendientes en documentación PDF

Registro de actualizaciones necesarias en los documentos de `doc/foundation/`
tras correcciones aplicadas al código. Estos PDFs no se pueden editar directamente;
los cambios se aplican en la fuente original (Notion/Word/LaTeX) al redactar la memoria.

---

## Doc_Motor_de_Simulacin_y_Entorno_RL.pdf

### 1. Espacio de acción — `Discrete(7)` → `Discrete(13)`
- **Ubicación:** Sección "2. Espacio de Acción"
- **Texto actual:** "Se define un espacio discreto Discrete(7)..."
- **Corrección:** El espacio real es `Discrete(13)`: 1 acción IDLE + 4 estrategias × 3 niveles de potencia (33%/66%/100%).
- **Tabla correcta:**

  | ID | Estrategia | Nivel |
  |---|---|---|
  | 0 | IDLE | — |
  | 1-3 | CARGAR_SOLAR | 33% / 66% / 100% |
  | 4-6 | CARGAR_MIXTA | 33% / 66% / 100% |
  | 7-9 | DESCARGAR_CASA | 33% / 66% / 100% |
  | 10-12 | DESCARGAR_RED | 33% / 66% / 100% |

### 2. Función de recompensa — fórmula incorrecta
- **Ubicación:** Sección "3. Función de Recompensa"
- **Texto actual:** `R_t = Ahorro_evitado + Ingresos_venta - Coste_compra - Coste_degradación`
- **Problema:** La fórmula restaba el coste de compra dos veces (una en `Ahorro_evitado` y otra en `Coste_compra`).
- **Fórmula correcta (actualizada 2026-05-03):**
  ```
  R_t = Ingresos_venta - Coste_compra - Coste_degradación
      = vendido × precio_excedente - comprado × precio_kwh - coste_deg
  ```
- **Modelo de precios asimétrico real:**
  - Compra: **precio PVPC completo** (ind. ESIOS 1001) — incluye energía + peajes + cargos
  - Venta: **precio compensación simplificada** (ind. ESIOS 1739, RD 244/2019 Art.14) — equivale al precio mayorista OMIE menos costes de desvío (~60% del PVPC)
- **Justificación:** Cash flow neto por paso de tiempo. El spread compra-venta refleja los peajes y cargos regulados que el consumidor paga pero que el productor de excedentes no recibe.

### 3. Cálculo de degradación — SoC utilizado
- **Ubicación:** Sección "A. Módulo Físico" (descripción del modelo de batería)
- **Corrección:** El coste de degradación se calcula con el **SoC medio del ciclo** `(SoC_antes + SoC_después) / 2`, no con el SoC final, para reflejar el stress real durante la operación.

### 3b. Física realista de batería — nueva sección a añadir (v7, 2026-05-03)
- **Ubicación:** Sección "A. Módulo Físico" (junto a la descripción de la batería)
- **Contenido a añadir:**
  - **Eficiencia de carga/descarga**: 95% en cada dirección (round-trip 90.25%). Al cargar,
    de cada 10 kWh de entrada solo 9.5 kWh se almacenan. Al descargar, la batería pierde
    10 kWh pero solo entrega 9.5 kWh útiles. Valores típicos de baterías Li-ion.
  - **Límites operativos de SoC**: mínimo 10%, máximo 90%. La batería de 100 kWh tiene
    80 kWh útiles. Protege la vida útil evitando estados extremos de carga.
  - **Autodescarga**: ~3% mensual (0.004%/hora). Se aplica cada paso antes de la acción.
    Efecto pequeño (~0.67% por semana) pero realista para Li-ion.
- **Justificación:** Sin estos parámetros, el simulador sobreestimaba el valor de la batería
  al no penalizar las pérdidas de conversión ni los límites físicos reales.

---

## Doc_Motor_de_Simulacin_y_Entorno_RL.pdf (adicional)

### 4. VecNormalize — nueva sección a añadir
- **Ubicación:** Sección "2. Espacio de Observación" (tras la descripción del vector de 101 variables)
- **Contenido a añadir:** Documentar que las observaciones se normalizan en línea con `VecNormalize`
  (running mean/std) antes de ser procesadas por la red Q. Las estadísticas se guardan en
  `vec_normalize.pkl` y son necesarias para reproducir el comportamiento en inferencia.
- **Justificación:** Las 101 variables tienen escalas muy diferentes (SoC: 0-1, precios: 0.01-0.95 €/kWh).
  Sin normalización la red da más peso a variables de mayor magnitud.

### 5. Arquitectura de la red — actualizar
- **Ubicación:** Sección "4. Hiperparámetros del DQN" (si existe) o nueva sección
- **Corrección:** La red final es `101 -> 64 -> 64 -> 13`.
  Se probó 256x256 (v2) pero overfitteaba con el dataset. 64x64 validada como óptima
  tras análisis científico de 4 runs.

### 5b. Espacio de observación — 76 -> 101 variables
- **Ubicación:** Sección "2. Espacio de Observación"
- **Corrección:** El vector pasó de 76 a **101 dimensiones** al añadir el precio de venta
  (precio excedentaria) como variable separada del precio de compra (PVPC):
  - 5 actuales: SoC, precio_compra, precio_venta, excedente, déficit
  - 96 pronóstico: 24h × 4 variables (consumo, generación, precio_compra, precio_venta)

### 6. Dataset y periodo de datos — actualizar
- **Ubicación:** Sección sobre datos de entrada
- **Texto actual:** Puede referir a datos de solo 2023 (8.760 horas)
- **Corrección:** Dataset multi-año jun-2021 a dic-2023 (22.646 horas, ~31 meses).
  Datos de ESIOS (consumo PVPC 2.0TD, precios Península) y PVGIS (solar 1kWp, Vigo).
  **Nuevo dato raw:** `compensacion_autoconsumo_esios_2021_2023.csv` (ind. ESIOS 1739, precio excedentaria).
  Dataset ahora tiene 4 columnas: `[consumo_total, generacion_total, precio_kwh, precio_excedente]`.

### 7. Precio de venta — actualizar en TODOS los documentos
- **Ubicación:** Cualquier referencia a "precio_venta_excedente = 0.05 EUR/kWh" o "modelo simétrico"
- **Corrección:** Modelo **asimétrico real** con dos precios:
  - Compra de red: precio PVPC completo (ind. ESIOS 1001, media 0.217 EUR/kWh)
  - Venta de excedentes: precio compensación simplificada (ind. ESIOS 1739, media 0.132 EUR/kWh)
  - RD 244/2019 Art.14: excedentes valorados a Pmh - CDSVh (precio mayorista OMIE - costes desvío)
- Esto aplica a:
  - Doc_Motor_de_Simulacion: sección de función de recompensa
  - Cualquier diagrama que muestre flujos económicos
  - Descripción del pipeline de datos (nueva fuente ESIOS)

### 8. Simplificación documentada — tope mensual de compensación
- **Ubicación:** Sección de limitaciones / simplificaciones del modelo
- **Contenido:** El RD 244/2019 establece que en compensación simplificada el valor económico
  de los excedentes no puede superar el coste de la energía consumida en el periodo de facturación
  (máx. 1 mes). Este tope **no se implementa** en el simulador porque:
  1. La comunidad de 15 viviendas consume más de lo que genera en cómputo mensual
  2. Los episodios de entrenamiento son de 1 semana (independientes entre sí)
  3. Implementarlo requeriría episodios de 1 mes o estado acumulado entre episodios,
     complicando significativamente el entrenamiento RL sin aportar valor práctico

---

## Otros documentos — sin cambios requeridos

| Documento | Estado |
|---|---|
| Doc_Ingeniera_de_Datos_y_Generacin_del_Entorno.pdf | **Actualizar**: nueva fuente ESIOS (ind. 1739), dataset 4 cols |
| Doc_Justificacin_de_la_Estructura_y_Datos.pdf | Sin cambios |
| Arquitectura SGEC_ Física y Funcionamiento.pdf | Sin cambios |
| REGLAS ARQUITECTÓNICAS Y CONTEXTO OCULTO.pdf | Sin cambios |
| DQN_funcionamiento.pdf | Sin cambios |
| Dqn_vs_PPO.pdf | Sin cambios |
| Acta_de_Constitucion_TFG_.pdf | Sin cambios |
| Enunciado_de_Alcance_TFG_.pdf | Sin cambios |
