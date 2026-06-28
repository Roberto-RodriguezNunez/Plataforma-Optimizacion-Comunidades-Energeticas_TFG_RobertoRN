# Guía para entrenar la familia PPO (Windows) — para papá

Esta guía explica, paso a paso y sin dar nada por sabido, cómo:
1. Descargar el proyecto desde GitHub.
2. Instalarlo.
3. Entrenar la familia de algoritmos **PPO**.
4. Enviarme los resultados.

No hace falta saber programar. Solo hay que copiar y pegar comandos. Cada bloque
gris es **un comando**: cópialo entero, pégalo en la ventana negra (Git Bash) y
pulsa Enter. Espera a que termine antes de pegar el siguiente.

---

## ⚠️ ANTES DE EMPEZAR (esto lo hace Roberto, no papá)

La rama con los datos y el código actualizado tiene que estar subida a GitHub:

```bash
git push origin feature/entreno-residual-sac
```

Si esto no se hace, papá descargará una versión vieja y sin datos. **Imprescindible.**

---

## PARTE 1 — Instalar dos programas

### 1.1 Instalar Python 3.12

1. Entra en: https://www.python.org/downloads/release/python-3127/
2. Baja hasta "Files" y descarga **"Windows installer (64-bit)"**.
3. Abre el archivo descargado.
4. **MUY IMPORTANTE:** en la primera pantalla del instalador, marca la casilla de
   abajo **"Add python.exe to PATH"** antes de pulsar nada.
5. Pulsa **"Install Now"** y espera a que termine. Cierra el instalador.

### 1.2 Instalar Git (incluye "Git Bash", la ventana donde escribiremos todo)

1. Entra en: https://git-scm.com/download/win
2. Se descarga solo. Abre el archivo.
3. Pulsa **"Next" / "Siguiente"** en todas las pantallas (las opciones por defecto
   están bien) y al final **"Install"**.

### 1.3 Abrir Git Bash

- Pulsa la tecla **Windows**, escribe **"Git Bash"** y ábrelo.
- Aparece una ventana negra con texto. **Todos los comandos de esta guía se escriben
  aquí.** Para pegar en esta ventana: clic derecho → *Paste* (o `Shift+Insert`).

---

## PARTE 2 — Descargar el proyecto desde GitHub

Pega estos comandos **uno a uno** en Git Bash:

```bash
cd ~
```

```bash
git clone https://github.com/Roberto-RodriguezNunez/Plataforma-Optimizacion-Comunidades-Energeticas_TFG_RobertoRN.git TFG
```

(Esto tarda un poco; descarga el proyecto y los datos. Crea una carpeta llamada
`TFG` dentro de tu carpeta de usuario.)

```bash
cd TFG
```

```bash
git checkout feature/entreno-residual-sac
```

> Este último paso es **obligatorio**: cambia a la versión del proyecto que tiene los
> datos y el código de entrenamiento.

---

## PARTE 3 — Instalar el proyecto

### 3.1 Crear el entorno

```bash
python -m venv .venv
```

> Si al ejecutarlo se abre la **Microsoft Store** en lugar de funcionar, usa este
> comando en su lugar: `py -3.12 -m venv .venv`

### 3.2 Instalar las librerías (tarda varios minutos; descarga ~1 GB)

```bash
.venv/Scripts/python -m pip install --upgrade pip
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

### 3.3 Comprobar que ha ido bien

```bash
.venv/Scripts/python -c "import stable_baselines3, torch; print('INSTALACION OK')"
```

Si la última línea que aparece es **`INSTALACION OK`**, todo está listo. Si sale un
error en rojo, hazme una foto de la pantalla y me la mandas.

---

## PARTE 4 — Preparar el ordenador para una tanda larga

El entrenamiento de PPO tarda **bastante** (puede ser un día o más con el ordenador
encendido). Para que no se interrumpa:

1. Conecta el ordenador a la corriente (no con batería).
2. Configura que **no se suspenda**: *Configuración de Windows → Sistema → Inicio/apagado
   → "Suspender" = Nunca* (al menos mientras dure el entrenamiento).
3. **No cierres la ventana de Git Bash** mientras entrena. Puedes minimizarla.

---

## PARTE 5 — Entrenar

> Los comandos empiezan por `PYTHON=.venv/Scripts/python` — eso le dice al programa
> qué Python usar. Cópialos enteros.

### 5.1 (Obligatorio primero) Calcular las referencias base — rápido, ~10 min

```bash
PYTHON=.venv/Scripts/python ./experimentos/run_familia.sh G
```

Espera a que termine y veas otra vez la línea para escribir.

### 5.2 (Prueba opcional, ~5 min) Comprobar que un entreno arranca

```bash
.venv/Scripts/python src/main_ppo.py --version PPO-D1 --seeds 42 --timesteps 25000
```

Si termina sin error en rojo, perfecto. (Genera un modelo de prueba que el
entrenamiento real de abajo sobrescribe; no pasa nada.)

### 5.3 Entrenar la familia PPO entera (esto es lo importante; tarda muchas horas)

```bash
PYTHON=.venv/Scripts/python ./experimentos/run_familia.sh ppo 2>&1 | tee resultados/log_ppo.txt
```

Qué hace este comando:
- Entrena las **5 versiones** de PPO (PPO-D1 a PPO-D5), cada una con 3 semillas.
- Evalúa cada una y va escribiendo los resultados en la carpeta `resultados`.
- El `tee resultados/log_ppo.txt` guarda además todo el texto en un archivo, por si
  hay que revisar algo.

Verás ir apareciendo texto con números (recompensas de evaluación). Eso es normal:
significa que está entrenando. **Déjalo hasta que veas la línea final que dice algo
como `Familia 'ppo' COMPLETA`.**

> Si por lo que sea se corta a mitad (apagón, etc.), no pasa nada: vuelve a abrir Git
> Bash, ejecuta `cd ~/TFG` y lanza otra vez el comando del paso 5.3.

---

## PARTE 6 — Enviarme los resultados

Cuando termine (línea `Familia 'ppo' COMPLETA`), todo lo que necesito está en la
carpeta **`resultados`** dentro de `TFG`.

### Forma fácil (con el ratón)

1. Abre el **Explorador de archivos** de Windows.
2. Ve a tu carpeta de usuario → carpeta **`TFG`** → carpeta **`resultados`**.
3. Clic derecho sobre la carpeta `resultados` → **"Enviar a" → "Carpeta comprimida
   (en zip)"**. Se crea un archivo `resultados.zip`.
4. Mándame ese `resultados.zip` por correo o por WeTransfer (https://wetransfer.com)
   si pesa mucho.

### O por comando (crea el zip automáticamente)

```bash
cd ~/TFG && powershell -Command "Compress-Archive -Path resultados\* -DestinationPath resultados_ppo.zip -Force"
```

Eso crea `resultados_ppo.zip` dentro de la carpeta `TFG`. Ese es el que me mandas.

> Dentro van el `resultados.csv` (la tabla principal), los archivos `ppo_PPO-D*.json`
> (detalle de cada versión) y el `log_ppo.txt`. Con eso me basta.

---

## Resumen de comandos (chuleta)

```bash
# 1. Descargar
cd ~
git clone https://github.com/Roberto-RodriguezNunez/Plataforma-Optimizacion-Comunidades-Energeticas_TFG_RobertoRN.git TFG
cd TFG
git checkout feature/entreno-residual-sac

# 2. Instalar
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -c "import stable_baselines3, torch; print('INSTALACION OK')"

# 3. Entrenar
PYTHON=.venv/Scripts/python ./experimentos/run_familia.sh G
PYTHON=.venv/Scripts/python ./experimentos/run_familia.sh ppo 2>&1 | tee resultados/log_ppo.txt

# 4. Empaquetar resultados
cd ~/TFG && powershell -Command "Compress-Archive -Path resultados\* -DestinationPath resultados_ppo.zip -Force"
```

## Si algo falla
Haz una foto de la ventana de Git Bash (que se vea el error en rojo) y mándamela.
</content>
