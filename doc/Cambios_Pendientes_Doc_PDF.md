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
- **Fórmula correcta:**
  ```
  R_t = Ingresos_venta - Coste_compra - Coste_degradación
      = vendido × precio_venta_excedente - comprado × precio_kwh - coste_deg
  ```
- **Justificación:** Cash flow neto por paso de tiempo. El término `ahorro` era redundante con `gastos` y causaba que el incentivo a reducir compras de red fuese el doble del correcto.

### 3. Cálculo de degradación — SoC utilizado
- **Ubicación:** Sección "A. Módulo Físico" (descripción del modelo de batería)
- **Corrección:** El coste de degradación se calcula con el **SoC medio del ciclo** `(SoC_antes + SoC_después) / 2`, no con el SoC final, para reflejar el stress real durante la operación.

---

## Otros documentos — sin cambios requeridos

| Documento | Estado |
|---|---|
| Doc_Ingeniera_de_Datos_y_Generacin_del_Entorno.pdf | Sin cambios |
| Doc_Justificacin_de_la_Estructura_y_Datos.pdf | Sin cambios |
| Arquitectura SGEC_ Física y Funcionamiento.pdf | Sin cambios |
| REGLAS ARQUITECTÓNICAS Y CONTEXTO OCULTO.pdf | Sin cambios |
| DQN_funcionamiento.pdf | Sin cambios |
| Dqn_vs_PPO.pdf | Sin cambios |
| Acta_de_Constitucion_TFG_.pdf | Sin cambios |
| Enunciado_de_Alcance_TFG_.pdf | Sin cambios |
