# Justificación del espacio de acciones: de 13 a 9 acciones

**Fecha:** 2026-05-17  
**Afecta a:** `simulador.py`, `energy_env.py`, `main.py`  
**DQN anterior:** DQN_9 (13 acciones) — **DQN desde ahora:** DQN_10+ (9 acciones)

---

## 1. Diseño original (13 acciones)

El espacio de acciones era una cuadrícula de 4 estrategias × 3 niveles de potencia + IDLE:

| ID | Estrategia | Nivel | Potencia objetivo |
|---|---|---|---|
| 0 | IDLE | — | — |
| 1 | CARGAR_SOLAR | 33% | 16.5 kW |
| 2 | CARGAR_SOLAR | 66% | 33.0 kW |
| 3 | CARGAR_SOLAR | 100% | 50.0 kW |
| 4 | CARGAR_MIXTA | 33% | 16.5 kW |
| 5 | CARGAR_MIXTA | 66% | 33.0 kW |
| 6 | CARGAR_MIXTA | 100% | 50.0 kW |
| 7 | DESCARGAR_CASA | 33% | 16.5 kW |
| 8 | DESCARGAR_CASA | 66% | 33.0 kW |
| 9 | DESCARGAR_CASA | 100% | 50.0 kW |
| 10 | DESCARGAR_RED | 33% | 16.5 kW |
| 11 | DESCARGAR_RED | 66% | 33.0 kW |
| 12 | DESCARGAR_RED | 100% | 50.0 kW |

La justificación de los niveles era la **degradación no lineal**: el coste de degradación
crece cuadráticamente con la potencia (factor I²), por lo que cargar/descargar al 33%
cuesta menos que al 100%. Esto es correcto físicamente.

---

## 2. El problema: degeneración estructural

Una acción es **degenerada** cuando produce el mismo resultado físico que otra acción
en el mismo estado. Esto es un problema para el DQN porque:
- Recibe el mismo reward para acciones con distinto ID.
- Sus Q-values para esas acciones deben converger al mismo valor.
- Malgasta exploración (epsilon-greedy) probando acciones que sabemos que son equivalentes.
- El gradiente de actualización es ruidoso: mezcla señales de acciones idénticas.

### 2.1 Degeneración en CARGAR_SOLAR

La lógica de CARGAR_SOLAR es:
```
carga = min(exc_disp, espacio_libre/0.95, potencia_obj)
```

El cuello de botella es `min(exc_disp, potencia_obj)`. Los tres niveles solo producen
resultados distintos cuando `exc_disp > potencia_obj_nivel_1 = 16.5 kWh`.

**Análisis de frecuencia:**
- En horas nocturnas: `exc_disp = 0` → las tres acciones cargan 0 kWh (= IDLE).
- En horas con poco sol (amanecer, atardecer, nublado): `exc_disp < 16.5 kWh`
  → las tres acciones cargan `exc_disp` (el mismo valor). **Idénticas.**
- Solo en horas de sol pleno de verano con `exc_disp > 16.5 kWh` se diferencian.

En la práctica, las tres acciones de CARGAR_SOLAR son idénticas en la **gran mayoría**
de las horas en las que tiene sentido usarlas.

### 2.2 Degeneración en DESCARGAR_CASA

La lógica de DESCARGAR_CASA es:
```
descarga = min(def_cub / 0.95, bateria_disponible, potencia_obj)
```

El cuello de botella es `min(def_cub/0.95, potencia_obj)`.

- Si `def_cub = 0` (horas con excedente solar): descarga = 0 → **idéntica a IDLE**.
- Si `def_cub < 15.7 kWh`: las tres acciones cubren el déficit completo. **Idénticas.**
- Si `def_cub < 31.4 kWh`: acciones 66% y 100% cubren el déficit completo. **Idénticas.**
- Solo con `def_cub > 31.4 kWh` (déficit muy alto, poco frecuente) se diferencian.

Para una comunidad de 15 viviendas, el déficit horario típico es 5-20 kWh.
Las tres acciones de DESCARGAR_CASA son casi siempre idénticas entre sí (y a veces = IDLE).

### 2.3 Por qué los niveles SÍ funcionan en CARGAR_MIXTA y DESCARGAR_RED

- **CARGAR_MIXTA**: carga hasta `potencia_obj` comprando red si hace falta.
  El cuello de botella es siempre el inversor (la red es ilimitada).
  → Los tres niveles **siempre producen resultados distintos** (salvo batería llena).

- **DESCARGAR_RED**: descarga hasta `potencia_obj` y vende el resto.
  El cuello de botella es siempre el inversor (mientras haya batería).
  → Los tres niveles **siempre producen resultados distintos** (salvo batería vacía).

---

## 3. La solución: quitar niveles donde el inversor no es el cuello de botella

**Regla:** los niveles de potencia tienen sentido cuando el **inversor** es el factor
limitante. Cuando el factor limitante es la **energía disponible** (sol o déficit),
los niveles son redundantes.

| Estrategia | Factor limitante real | ¿Niveles útiles? |
|---|---|---|
| CARGAR_SOLAR | exc_disp (el sol manda) | **No** |
| CARGAR_MIXTA | Inversor (red ilimitada) | **Sí** |
| DESCARGAR_CASA | def_cub (el déficit manda) | **No** |
| DESCARGAR_RED | Inversor (batería lo permite) | **Sí** |

**Nueva semántica:**

- **CARGAR_SOLAR (acción única):** "Carga todo el excedente solar disponible,
  sin comprar de red." La restricción de no usar red se conserva. La potencia
  la dicta el sol, no el inversor.

- **DESCARGAR_CASA (acción única):** "Cubre todo el déficit actual desde la batería,
  sin vender a red." La restricción de no vender se conserva. La potencia
  la dicta el déficit, no el inversor.

---

## 4. Nuevo espacio de acciones (9 acciones)

| ID | Estrategia | Nivel | Cuello de botella |
|---|---|---|---|
| 0 | IDLE | — | — |
| 1 | CARGAR_SOLAR | único | exc_disp / espacio batería |
| 2 | CARGAR_MIXTA | 33% (16.5 kW) | Inversor |
| 3 | CARGAR_MIXTA | 66% (33.0 kW) | Inversor |
| 4 | CARGAR_MIXTA | 100% (50.0 kW) | Inversor |
| 5 | DESCARGAR_CASA | único | def_cub / batería disponible |
| 6 | DESCARGAR_RED | 33% (16.5 kW) | Inversor |
| 7 | DESCARGAR_RED | 66% (33.0 kW) | Inversor |
| 8 | DESCARGAR_RED | 100% (50.0 kW) | Inversor |

**Garantía:** las 9 acciones producen resultados distintos en casi todos los estados.
Las únicas degeneraciones residuales son:
- DESCARGAR_RED cuando `bateria_disponible = 0` (SoC en el límite inferior).
- CARGAR_MIXTA cuando `espacio_libre = 0` (batería llena al 90%).
Ambas son condiciones extremas poco frecuentes y manejables.

---

## 5. ¿Se pierde control sobre la degradación?

No. La degradación no lineal sigue intacta:

- **CARGAR_SOLAR:** carga toda la energía disponible de golpe. Si el sol da 5 kWh,
  se cargan 5 kWh con su factor de degradación correspondiente. No hay alternativa
  de "cargar más lento desde el sol" que tuviera sentido (ya es el mínimo físico
  dado lo que da el sol).

- **DESCARGAR_CASA:** descarga solo lo que piden las casas. Si el déficit es 8 kWh,
  se descargan 8 kWh. No hay alternativa de "cubrir el déficit más despacio".

- **CARGAR_MIXTA y DESCARGAR_RED:** mantienen los 3 niveles. El agente puede elegir
  carga/descarga lenta (33%, menos degradación) o rápida (100%, más degradación).
  Esto es exactamente donde la degradación no lineal más importa: cuando decides
  cuánta energía comprar/vender de forma activa.

---

## 6. Impacto esperado en el aprendizaje

El DQN tiene ahora 9 acciones en lugar de 13. Ventajas:

1. **Menos exploración malgastada:** epsilon-greedy no probará acciones que sabemos
   que son equivalentes.
2. **Q-values más limpios:** cada acción tiene una identidad única → gradientes menos
   ruidosos.
3. **Red más pequeña:** la capa de salida tiene 9 neuronas en lugar de 13 (~30% menos
   parámetros en la última capa). Con los mismos 1.5M pasos, el ratio datos/parámetros
   mejora ligeramente.
4. **Misma capacidad de control real:** las decisiones que importan (cuánto cargar de
   red, cuánto descargar para vender) siguen siendo controlables con 3 niveles cada una.
