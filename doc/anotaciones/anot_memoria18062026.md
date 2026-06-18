# Anotaciones para la memoria — 2026-06-18

Cambios pendientes de incorporar al documento LaTeX `doc/memoria_latex/memoria_tipo2 (3).tex`.
Cada sección indica qué cambiar, en qué línea aproximada, y el texto exacto.

---

## 1. Actualización de parámetros del sistema (50 kWp → 42 kWp, 100 kWh → 80 kWh)

### 1.1 Todas las menciones de 50 kWp

Buscar en el .tex: `\SI{50}{kWp}` y reemplazar por `\SI{42}{kWp}`.

También buscar texto plano: `50 kWp`, `50~kWp`, `50\,kWp` y sustituir según contexto.

Ocurrencias conocidas:
- Capítulo 4 (descripción del sistema): "instalación fotovoltaica de \SI{50}{kWp}"
- Capítulo 6 (Ingeniería de Datos), línea ~686: "Se escala a los \SI{50}{kWp}"

### 1.2 Todas las menciones de 100 kWh (batería)

Buscar: `\SI{100}{kWh}`, `100 kWh` en contexto de batería → reemplazar por `\SI{80}{kWh}` / `80 kWh`.

Ocurrencias conocidas:
- Capítulo 4 (descripción hardware): "batería de \SI{100}{kWh}"
- Capítulo 6 (parámetros del entorno): `capacidad_kwh: 100.0`

---

## 2. Párrafo de justificación del dimensionado (añadir en Cap. 6 — Ingeniería de Datos)

Insertar tras la descripción de la generación solar en el ETL (después de explicar el perfil PVGIS):

```latex
\subsubsection*{Justificación del dimensionado de la instalación}

La instalación fotovoltaica se dimensionó en \SI{42}{kWp} para equilibrar generación y
consumo a nivel anual. Con un consumo medio de \SI{3.500}{kWh/año} por vivienda y
15~viviendas, el consumo agregado asciende a \SI{52.500}{kWh/año}. Aplicando las
\SI{1.250}{h} equivalentes de irradiancia anual en Galicia (zona climática H5 según
PVGIS), la potencia necesaria es:
%
\begin{equation}
  P_\text{solar} = \frac{52{.}500\;\text{kWh/año}}{1{.}250\;\text{h/año}} = \SI{42}{kWp}
\end{equation}
%
Este dimensionado maximiza la tasa de autoconsumo bajo el régimen de compensación
simplificada (RD~244/2019): toda la energía generada puede ser compensada en factura,
sin excedentes irrecuperables. Los \SI{50}{kWp} originales generaban un~18\,\% de
excedente estructural que nunca podía compensarse al superar el coste de los consumos
mensuales (art.\,14 RD~244/2019).

La batería comunitaria se dimensionó en \SI{80}{kWh} nominales
(\SI{64}{kWh} útiles con SoC\,\(\in\)[10\,\%,\,90\,\%]).
El déficit energético nocturno (20:00–08:00~h, ventana de~12~h con potencia media de
\SI{5{,}72}{kW}) asciende a~\SI{68{,}6}{kWh/noche}.
Los \SI{64}{kWh} útiles cubren el~93\,\% de ese déficit, con una reducción de coste de
capital del~20\,\% respecto a una unidad de~\SI{100}{kWh} cuya capacidad útil
(\SI{80}{kWh}) excedía en un~14\,\% el ciclo diario real.
```

---

## 3. Corrección de la definición de factura\_base / ahorro (Cap. 7 o donde esté la descripción económica)

### Texto antiguo (incorrecto)

La `factura_base` se definía como el coste sin ningún sistema solar: `consumo × precio_compra`. El `ahorro` comparaba la factura real contra ese escenario sin paneles.

### Texto nuevo (correcto)

```latex
El escenario de referencia (\textit{factura\_base}) corresponde al coste que tendría la
vivienda con sus propios paneles fotovoltaicos pero \emph{sin} batería comunitaria ni
mecanismo de reparto colectivo. Formalmente, para cada hora~$h$:
%
\begin{align}
  \text{compra\_base}_h  &= \max(0,\; c_h - g_h) \cdot p^c_h \\
  \text{comp\_base}_h    &= \max(0,\; g_h - c_h) \cdot p^e_h
\end{align}
%
\noindent donde $c_h$ es el consumo individual, $g_h$ la generación individual
proporcional al kWp de la vivienda, $p^c_h$ el precio de compra (PVPC) y $p^e_h$ el
precio de excedente (ind.\,1739).
La \textit{factura\_base} mensual es:
%
\begin{equation}
  \text{factura\_base} = \max\!\left(0,\;\sum_h \text{compra\_base}_h - \sum_h \text{comp\_base}_h\right)
\end{equation}
%
El \textbf{ahorro} mide así el valor añadido de la comunidad (batería + reparto colectivo)
sobre el autoconsumo individual con paneles propios:
%
\begin{equation}
  \text{ahorro} = \max(0,\;\text{factura\_base} - \text{factura\_real})
\end{equation}
```

### Por qué cambió

La definición anterior (`consumo × precio`) inflaba el ahorro: cualquier vecino con paneles ya tiene ahorro antes de que exista la comunidad. La nueva definición aísla el valor que aporta la comunidad por encima de lo que ya tendría con sus propios paneles. Es el indicador correcto para justificar la inversión en batería y red de reparto.

---

## 4. Tabla de parámetros del sistema (actualizar donde aparezca)

| Parámetro | Valor anterior | Valor nuevo | Justificación |
|---|---|---|---|
| Potencia solar total | 50 kWp | 42 kWp | gen = consumo anual (RD 244/2019) |
| Capacidad batería nominal | 100 kWh | 80 kWh | 64 kWh útiles = 93% ciclo nocturno |
| Capacidad batería útil (SoC operativo) | 80 kWh | 64 kWh | SoC [10%–90%] × 80 kWh |
| Capacidad útil actual Vilarín (degradación) | 18,4 kWh | 73,6 kWh | 487 ciclos → 8% degradación |
| Capacidad útil actual Brañas (degradación) | 29,1 kWh | 77,6 kWh | 203 ciclos → 3% degradación |
| Ciclo nocturno cubierto | ~93% (80/86 kWh) | 93% (64/68,6 kWh) | Igual cobertura, 20% menos capital |
| kWp por vivienda (ETL) | aleatorio [2,5] | fijo según seed | Coherencia SaaS ↔ training |

---

## 5. Descripción del ETL solar en Cap. 6 (actualizar)

Sustituir el párrafo que describe el muestreo aleatorio de kWp:

**Anterior:**
> "La capacidad instalada de cada vivienda se muestrea con `uniform(2.0, 5.0)` kWp y se normaliza para que la suma sea exactamente 50 kWp."

**Nuevo:**
> "La capacidad instalada de cada vivienda toma los valores reales de la comunidad Vilarín registrados en la plataforma SaaS (seed.py), garantizando coherencia entre los datos de entrenamiento y los atributos del gemelo digital. Los valores suman exactamente \SI{42}{kWp} (véase tabla de kWp por vivienda en el Apéndice A)."

---

## 6. Apéndice A — Tabla de kWp por vivienda (añadir o actualizar)

| Vivienda | kWp | Paneles (400 Wp) |
|---|---|---|
| I001 — Rúa da Igrexa 1  | 2,4 |  6 |
| I002 — Rúa da Igrexa 3  | 2,4 |  6 |
| I003 — Rúa da Igrexa 5  | 2,0 |  5 |
| I004 — Rúa da Igrexa 7  | 2,0 |  5 |
| I005 — Rúa da Igrexa 9  | 3,2 |  8 |
| I006 — Rúa da Igrexa 11 | 2,0 |  5 |
| R001 — Camiño do Río 2  | 4,4 | 11 |
| R002 — Camiño do Río 4  | 2,8 |  7 |
| R003 — Camiño do Río 6  | 2,0 |  5 |
| R004 — Camiño do Río 8  | 2,4 |  6 |
| R005 — Camiño do Río 10 | 5,2 | 13 |
| R006 — Camiño do Río 12 | 2,8 |  7 |
| O001 — Lugar de Outeiro 1 | 2,4 |  6 |
| O002 — Lugar de Outeiro 3 | 2,4 |  6 |
| O003 — Lugar de Outeiro 5 | 3,6 |  9 |
| **Total** | **42,0** | **105** |

Paneles de 400 Wp cada uno (tecnología Si-cristalino monocristalino, orientación sur, inclinación óptima).

---

## 7. Sección SaaS — desarrollo de la plataforma web (Cap. 8 nuevo o en Cap. 7)

La memoria debe incluir una sección que describa el desarrollo de la plataforma SaaS (LeaLink). Puntos a cubrir:

- **Arquitectura**: Flask + SQLAlchemy + PostgreSQL + Docker Compose. Blueprint structure. CSRF protection con exención para la API edge.
- **Módulos principales**: comunidades, viviendas, cierres mensuales, baterías, notificaciones, incidencias, accesos.
- **API edge** (`/api/edge/decision`, `/api/edge/generar-cierre`): autenticación Bearer token, exención CSRF, cálculo de cierres mensuales bottom-up.
- **Healthcheck**: endpoint `/health` para Docker Compose `depends_on: service_healthy`.
- **Despliegue**: `docker-compose.yml`, `Dockerfile`, `entrypoint.sh`, seed de datos.
- Incluir **diagrama de secuencia** del flujo edge → SaaS → cierre mensual.
- Incluir **casos de uso** de los roles: superadmin, titular, convivente, solo_lectura.

---

## 8. Sección ONNX en Cap. 7 (despliegue en producción)

Añadir subsección en el capítulo de entrenamiento/evaluación del agente:

- Exportación del actor SAC a ONNX con `torch.onnx.export`.
- Exportación de los parámetros de normalización (VecNormalize) a `.npz`.
- Runtime en el edge: `onnxruntime`, inferencia < 5 ms, sin dependencia PyTorch.
- `delta_max = 0.15` multiplicativo: `flow_final = max(0, mpc_flow × (1 + delta × 0.15))`.
- Citar que `residual_sac_actor.onnx` y `vec_normalize_v5_1M.npz` son los artefactos de producción.

---

## 9. Parámetros `system.yaml` a actualizar en el apéndice de configuración

Si hay un apéndice con el `system.yaml` completo, los valores a actualizar son:

```yaml
# línea 4
# Comunidad de autoconsumo fotovoltaico colectivo (15 viviendas, 42 kWp, 80 kWh)

# batería
capacidad_kwh: 80.0

# comunidad
potencia_solar_total_kwp: 42.0
```
