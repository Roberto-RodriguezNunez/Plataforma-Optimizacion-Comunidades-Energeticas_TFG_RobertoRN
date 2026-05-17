import pandas as pd
import numpy as np
import os

# --- CONFIGURACION ---
# Datos multi-anio: jun-2021 a dic-2023 (~22.656 horas, ~31 meses)
RUTA_CONSUMO    = 'data/raw/consumo_esios_2021_2023.csv'
RUTA_PRECIOS    = 'data/raw/precios_esios_2021_2023.csv'
RUTA_EXCEDENTE  = 'data/raw/compensacion_autoconsumo_esios_2021_2023.csv'
RUTA_SOLAR      = 'data/raw/solar_pvgis_2021_2023.csv'
RUTA_SALIDA     = 'data/processed/dataset_final.csv'

NUM_VECINOS = 15
POTENCIA_SOLAR_TOTAL = 50.0   # kWp
CONSUMO_ANUAL_CASA   = 3500.0 # kWh

# Semilla fija para reproducibilidad de los perfiles de vecinos
SEED = 42


def generar():
    print("INICIANDO GENERACION DEL DATASET MULTI-ANIO ...")
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 1. PROCESAR CONSUMO
    print("   -> 1. Procesando Consumo (ESIOS)...")
    try:
        df_cons = pd.read_csv(RUTA_CONSUMO, sep=';', encoding='utf-8')
        perfil_base = df_cons['value'].values
        n_horas = len(perfil_base)

        # Convertir coeficiente PVPC a kWh
        perfil_kwh = perfil_base * CONSUMO_ANUAL_CASA

        # Generar 15 vecinos con perturbacion estocastica
        consumo_total = np.zeros(n_horas)
        for i in range(NUM_VECINOS):
            factor = rng.uniform(0.8, 1.2)
            ruido = rng.normal(1.0, 0.1, n_horas)
            vecino = perfil_kwh * factor * ruido
            consumo_total += np.maximum(vecino, 0)

        print(f"      - {n_horas} horas, {NUM_VECINOS} vecinos agregados.")

    except Exception as e:
        print(f"ERROR EN CONSUMO: {e}")
        return

    # 2. PROCESAR SOLAR
    print("   -> 2. Procesando Solar (PVGIS) — instalaciones individuales por vivienda...")
    try:
        df_sol = pd.read_csv(RUTA_SOLAR, skiprows=10, skipfooter=10, engine='python')
        solar_unitario = df_sol['P'].values / 1000.0  # W -> kW (perfil para 1 kWp)
        n_horas_solar = len(solar_unitario)

        # Distribuir la potencia total entre las 15 viviendas de forma aleatoria.
        # Cada casa tiene su propia instalación con capacidad y rendimiento distintos.
        caps = rng.uniform(2.0, 5.0, NUM_VECINOS)          # kWp por vivienda (2-5 kWp)
        caps = caps / caps.sum() * POTENCIA_SOLAR_TOTAL    # normalizar a 50 kWp total

        generacion_total = np.zeros(n_horas_solar)
        for i in range(NUM_VECINOS):
            # Factor de rendimiento: orientación, inclinación, sombras parciales
            # Valores reales: sur perfecto ~1.0, sureste/suroeste ~0.90, sombras ~0.80
            perf = rng.uniform(0.80, 1.0)
            # Ruido horario pequeño: suciedad puntual, sombras de nubes locales
            # Con 15 casas el ruido se atenúa por agregación (~3% individual)
            ruido = rng.normal(1.0, 0.03, n_horas_solar)
            casa_solar = solar_unitario * caps[i] * perf * ruido
            generacion_total += np.maximum(casa_solar, 0)

        print(f"      - {n_horas_solar} horas, {NUM_VECINOS} instalaciones individuales.")
        print(f"      - Capacidades (kWp): min={caps.min():.2f}, max={caps.max():.2f}, "
              f"total={caps.sum():.2f}")

    except Exception as e:
        print(f"ERROR EN SOLAR: {e}")
        return

    # 3. PROCESAR PRECIOS PVPC (compra de red)
    print("   -> 3. Procesando Precios PVPC (ESIOS ind. 1001)...")
    try:
        df_prec = pd.read_csv(RUTA_PRECIOS, sep=';', encoding='utf-8')
        precios = df_prec['value'].values
        precios_kwh = precios / 1000.0  # EUR/MWh -> EUR/kWh
        print(f"      - {len(precios_kwh)} horas, convertidos a EUR/kWh.")

    except Exception as e:
        print(f"ERROR EN PRECIOS: {e}")
        return

    # 4. PROCESAR PRECIO EXCEDENTARIA (venta de excedentes, RD 244/2019)
    print("   -> 4. Procesando Precio Excedentaria (ESIOS ind. 1739)...")
    try:
        df_exc = pd.read_csv(RUTA_EXCEDENTE, sep=';', encoding='utf-8')
        precio_exc = df_exc['value'].values
        precio_exc_kwh = precio_exc / 1000.0  # EUR/MWh -> EUR/kWh
        print(f"      - {len(precio_exc_kwh)} horas, convertidos a EUR/kWh.")

    except Exception as e:
        print(f"ERROR EN PRECIO EXCEDENTARIA: {e}")
        return

    # 5. FUSION Y GUARDADO
    min_len = min(len(consumo_total), len(solar_total), len(precios_kwh), len(precio_exc_kwh))
    print(f"   -> 5. Fusionando ({min_len} horas)...")

    df_final = pd.DataFrame({
        'consumo_total': consumo_total[:min_len],
        'generacion_total': solar_total[:min_len],
        'precio_kwh': precios_kwh[:min_len],
        'precio_excedente': precio_exc_kwh[:min_len]
    })

    df_final.to_csv(RUTA_SALIDA, index=False)
    print(f"\nDataset creado: {RUTA_SALIDA}")
    print(f"Shape: {df_final.shape}")
    print(f"\nEstadisticas:")
    print(df_final.describe().round(4))


if __name__ == "__main__":
    generar()