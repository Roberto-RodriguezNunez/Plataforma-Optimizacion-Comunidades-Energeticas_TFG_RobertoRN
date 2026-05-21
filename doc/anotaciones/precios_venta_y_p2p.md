# Precios de venta de excedentes y P2P en comunidades energeticas

> Anotacion personal — NO subir al repo. Investigacion hecha el 2026-05-02.

---

## 1. El problema detectado

El simulador (`src/core/simulador.py`) vende TODA la energia excedente a un precio fijo
de 0.05 EUR/kWh (`PRECIO_VENTA_EXCEDENTE = 0.05`). Esto es **incorrecto** para cualquier
escenario actual espanol:

- No refleja la compensacion simplificada (que usa precio horario OMIE)
- No refleja la venta a mercado (precio pool)
- Elimina completamente el incentivo de arbitraje temporal (cargar barato, vender caro)
- Hace que DESCARGAR_RED sea casi siempre inutil (vendes a 0.05 lo que cuesta 0.15)

---

## 2. Que dice la regulacion espanola (RD 244/2019)

### Compensacion simplificada (instalaciones <= 100 kW)

- Los excedentes se compensan al **precio horario de mercado OMIE**, NO a precio fijo.
- En PVPC: cada kWh vertido se valora al "precio medio horario" del mercado mayorista.
- En mercado libre: al precio pactado con la comercializadora (Repsol 0.10, Naturgy 0.07,
  Iberdrola 0.04...).
- **Limitacion**: la compensacion no puede superar el coste del consumo en el periodo de
  facturacion (max 1 mes). La factura baja a 0 como maximo.

### Venta a mercado (sin limite)

- Te registras como productor y vendes al precio pool de OMIE.
- Sin limite de compensacion — puedes tener beneficio neto.

### Comunidades energeticas

- El RDL de junio 2025 introduce la figura del gestor de autoconsumo.
- Distancia generacion-consumo ampliada a 5 km.
- Coeficientes de reparto flexibles (mensuales, variables, por acuerdo unanime).

---

## 3. Precio P2P entre vecinos

### El razonamiento

Si la red vende a 0.15 EUR/kWh y compra a ~0.05-0.10 EUR/kWh, un vecino con excedente
puede venderle a otro vecino a un precio intermedio (ej: 0.10). Ambos ganan:
- El comprador paga menos que a la red (0.10 < 0.15)
- El vendedor cobra mas que vendiendo a red (0.10 > 0.05)

### En mi arquitectura (DQN macro)

El P2P ya esta **implicito** en el modelo agregado:
- El dataset tiene consumo_total y generacion_total de las 15 casas
- Si Casa A tiene excedente y Casa B deficit, el intercambio ya ocurrio en el agregado
- El DQN solo ve el excedente/deficit NETO de la comunidad
- El DQN gestiona la bateria sobre ese neto

### Donde entra el precio P2P

NO entra en el DQN. Entra en el **post-procesado de reparto** (Capa 5 SaaS):

```
precio_p2p = (precio_compra_red + precio_venta_red) / 2   # Mid-Market Rate (MMR)

ahorro_vecino_comprador = energia_recibida * (precio_compra_red - precio_p2p)
ingreso_vecino_productor = energia_cedida * (precio_p2p - precio_venta_red)
```

El Mid-Market Rate es el modelo mas usado en la literatura academica (Zhou et al., 2023).

---

## 4. Los 3 niveles de precio

| Flujo                          | Precio                    | Quien decide         |
|--------------------------------|---------------------------|----------------------|
| Vecino -> Vecino (P2P)         | Pactado (Mid-Market Rate) | Los vecinos / estatutos |
| Comunidad -> Red (vender)      | Precio horario OMIE       | El mercado           |
| Red -> Comunidad (comprar)     | Precio horario PVPC       | El mercado           |

---

## 5. Lo que hay que arreglar en el simulador

### Cambio minimo (una linea)

```python
# ANTES (incorrecto):
ingresos = vendido * self.PRECIO_VENTA_EXCEDENTE   # siempre 0.05

# DESPUES (correcto):
ingresos = vendido * precio   # precio horario de mercado de esa hora
```

### Modelo simetrico vs asimetrico

- **Simetrico** (compra = venta = precio_kwh): simplificacion estandar en papers academicos
  de DQN+bateria. Perfectamente defendible para un TFG.
- **Asimetrico** (con spread): mas realista. Venta = precio * 0.9 (peajes).
  Pero anade complejidad sin cambiar fundamentalmente el problema.

**Decision: usar modelo simetrico.** Es lo que hacen los papers de referencia y simplifica
la interpretacion de resultados.

### Consecuencias del cambio

- DESCARGAR_RED pasa a ser rentable cuando precio alto (0.20-0.35 EUR/kWh)
- CARGAR_MIXTA pasa a ser rentable cuando precio bajo (0.02-0.06 EUR/kWh)
- El agente tiene que aprender CUANDO actuar, no solo QUE hacer
- Las 13 acciones se vuelven todas potencialmente utiles
- **Hay que reentrenar desde cero** (todo lo aprendido con 0.05 es invalido)

---

## 6. Analisis del DQN_4 (ANTES del fix de precios)

### Rendimiento comparativo

| Metrica              | IDLE (sin bateria) | Oracle (heuristico) | DQN_4 (500k)  |
|----------------------|--------------------|---------------------|----------------|
| Media semanal        | -44.12 EUR         | +1.51 EUR           | +0.24 EUR      |
| % semanas positivas  | 10%                | 65%                 | 64%            |
| Mejora vs IDLE       | ---                | +45.62 EUR          | +44.36 EUR     |

El agente alcanzaba el 97% del Oracle, pero con el modelo de precios incorrecto.

### Evolucion del aprendizaje

| Bloque       | ep_rew_mean | Mejor  | % positivos (eval) |
|--------------|-------------|--------|---------------------|
| Q1 (0-125k)  | -86.86      | -52.95 | 53%                 |
| Q2 (125-250k) | -22.77     | -4.39  | 57%                 |
| Q3 (250-375k) | -7.86      | -0.58  | 58%                 |
| Q4 (375-500k) | -5.41      | +2.29  | 64%                 |

### Tendencia

- Pendiente 2a mitad: +3.87 reward/100k (bootstrap IC95% [-0.80, +9.01])
- 94.7% de probabilidad de ser positiva
- El modelo seguia mejorando a 500k pasos

### Acciones dominantes

| Bloque        | Accion dominante             |
|---------------|------------------------------|
| Q1 (0-125k)   | DESCARGAR_CASA 33% (46%) + DESCARGAR_RED 100% (31%) |
| Q2 (125-250k)  | DESCARGAR_CASA 66% (70%)    |
| Q3 (250-375k)  | DESCARGAR_CASA 33% (88%)    |
| Q4 (375-500k)  | DESCARGAR_CASA 100% (63%)   |

Con precio fijo a 0.05, el agente solo aprendio a descargar a casas (evitar compras).
Con precio de mercado deberia aprender tambien arbitraje temporal.

### Metricas operativas

- SoC medio: 0.505 -> 0.472 (tendencia descendente, descarga mas de lo que carga)
- Energia comprada: 5.21 -> 1.52 kWh/paso (reduccion 71%)
- Loss: 0.41 -> 0.82 (sube proporcionalmente a los Q-values, normal)

---

## 7. Justificacion de cada hiperparametro (config v3/v4)

| Parametro         | Valor   | Justificacion                                           |
|-------------------|---------|---------------------------------------------------------|
| TOTAL_TIMESTEPS   | 1M      | DQN_4 seguia mejorando a 500k (+3.87/100k)             |
| LEARNING_RATE     | 1e-4    | Estandar DQN. v2 probo 5e-5 y fue peor                 |
| NET_ARCH          | [64,64] | 5k params. Con 8760h, ratio datos/params ~1.7k. 256x256 (88k params) overfitteo |
| BUFFER_SIZE       | 100k    | ~595 episodios. Suficiente diversidad estacional        |
| BATCH_SIZE        | 64      | Estandar. v2 probo 128 sin mejora                      |
| GAMMA             | 0.99    | Horizonte 100 pasos ~ 4 dias. Correcto para episodios de 168 |
| EXPLORATION_FRAC  | 0.4     | Epsilon decae en 40% del training. v2 uso 0.5, peor    |
| EXPLORATION_FINAL | 0.05    | 5% residual. Estandar DQN                              |
| TARGET_UPDATE     | 1000    | Estabiliza sin ser conservador                          |
| TRAIN_FREQ        | 4       | Balanceo eficiencia/coste                               |
| VecNorm obs       | True    | 76 vars con escalas 0-1 a 0-200. Necesario             |
| VecNorm reward    | False   | v2 lo activo y fue 2x peor. Distorsiona senal economica |
| EVAL_EPISODES     | 20      | 52 tipos de semana, 20 ep reduce error estandar        |

**NOTA**: estos hiperparametros se validaron con precio fijo 0.05. Tras el fix de precios
podrian necesitar reajuste (especialmente exploration_frac y gamma, porque ahora el agente
necesita aprender patrones de precios que antes no existian).

---

## 8. Proximos pasos (en orden)

1. Cambiar simulador: `ingresos = vendido * precio` (una linea)
2. Eliminar constante `PRECIO_VENTA_EXCEDENTE` o dejarla documentada como legacy
3. Reentrenar desde cero (los modelos anteriores son invalidos)
4. Evaluar si los hiperparametros necesitan ajuste con el nuevo modelo de precios
5. Datos multi-anio 2020-2023 (4x datos)
6. P2P pricing en Capa 5 SaaS (post-procesado de reparto)

---

## 9. Fuentes

- [BOE RD 244/2019](https://www.boe.es/buscar/doc.php?id=BOE-A-2019-5089)
- [RD 244/2019 novedades 2025](https://puentesdemuras.com/es/rd-244-2019-normativa-fotovoltaica-espana/)
- [Venta excedentes autoconsumo | SunFields](https://www.sfe-solar.com/autoconsumo/excedentes/)
- [Precio excedentes PVPC | Selectra](https://selectra.es/autoconsumo/tarifas/pvpc)
- [Compensacion simplificada | Garriga](https://www.garriga-enginyers.com/es/noticies/compensacio-dexcedents-dautoconsum-segons-rd-2442019)
- [Energy Storage Management via DQN](https://arxiv.org/pdf/1903.11107)
- [DRL-Based Energy Storage Arbitrage](https://www.researchgate.net/publication/340535096)
- [Community Battery for Self-Consumption and Arbitrage](https://www.mdpi.com/2071-1050/16/8/3111)
- [P2P Energy Trading with DRL](https://pubs.aip.org/aip/jrse/article/15/6/065501/2930371)
- [P2P with MARL](https://arxiv.org/html/2511.23148v1)
- [RL Based P2P with Community Energy Storage](https://www.mdpi.com/1996-1073/14/14/4131)
- [Coeficientes de reparto variables](https://laadministracionaldia.inap.es/noticia.asp?id=1216845)
