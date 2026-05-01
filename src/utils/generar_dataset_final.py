import pandas as pd
import numpy as np
import os

# --- CONFIGURACIÓN ---
RUTA_CONSUMO = 'data/raw/consumo_esios_2023.csv'
RUTA_PRECIOS = 'data/raw/precios_esios_2023.csv'
RUTA_SOLAR = 'data/raw/solar_pvgis_2023.csv'
RUTA_SALIDA = 'data/processed/dataset_final.csv'

NUM_VECINOS = 15
POTENCIA_SOLAR_TOTAL = 50.0  # kWp
CONSUMO_ANUAL_CASA = 3500.0  # kWh

def generar():
    print("🚀 INICIANDO GENERACIÓN DEL DATASET MAESTRO ...")
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)

    # 1. PROCESAR CONSUMO
    print("   -> 1. Procesando Consumo (ESIOS)...")
    try:
        # CORRECCIÓN: Quitamos decimal=',' porque tu archivo usa puntos.
        df_cons = pd.read_csv(RUTA_CONSUMO, sep=';', encoding='utf-8')
        
        # Tu columna de datos es la 'value' (índice 4)
        perfil_base = df_cons['value'].values 
        
        # Convertir Coeficiente a kWh
        perfil_kwh = perfil_base * CONSUMO_ANUAL_CASA
        perfil_kwh = perfil_kwh[:8760] # Recortar a 1 año
        
        # Generar 15 vecinos
        consumo_total = np.zeros(8760)
        for i in range(NUM_VECINOS):
            factor = np.random.uniform(0.8, 1.2)
            ruido = np.random.normal(1.0, 0.1, 8760)
            vecino = perfil_kwh * factor * ruido
            consumo_total += np.maximum(vecino, 0)
            
        print(f"      - Generados {NUM_VECINOS} vecinos agregados.")

    except Exception as e:
        print(f"❌ ERROR EN CONSUMO: {e}")
        return

    # 2. PROCESAR SOLAR
    print("   -> 2. Procesando Solar (PVGIS)...")
    try:
        df_sol = pd.read_csv(RUTA_SOLAR, skiprows=10, skipfooter=10, engine='python')
        solar_unitario = df_sol['P'].values / 1000.0
        solar_total = solar_unitario[:8760] * POTENCIA_SOLAR_TOTAL
        print(f"      - Solar escalada a {POTENCIA_SOLAR_TOTAL} kWp.")

    except Exception as e:
        print(f"❌ ERROR EN SOLAR: {e}")
        return

    # 3. PROCESAR PRECIOS
    print("   -> 3. Procesando Precios (ESIOS)...")
    try:
        # CORRECCIÓN: Quitamos decimal=','
        df_prec = pd.read_csv(RUTA_PRECIOS, sep=';', encoding='utf-8')
        
        # Tu columna de precio también se llama 'value'
        precios = df_prec['value'].values
        
        # Convertir €/MWh a €/kWh
        precios_kwh = precios[:8760] / 1000.0
        print(f"      - Precios convertidos a €/kWh.")

    except Exception as e:
        print(f"❌ ERROR EN PRECIOS: {e}")
        return

    # 4. FUSIÓN Y GUARDADO
    print("   -> 4. Fusionando...")
    min_len = min(len(consumo_total), len(solar_total), len(precios_kwh))
    
    df_final = pd.DataFrame({
        'consumo_total': consumo_total[:min_len],
        'generacion_total': solar_total[:min_len],
        'precio_kwh': precios_kwh[:min_len]
    })
    
    df_final.to_csv(RUTA_SALIDA, index=False)
    print(f"\n✅ ¡ÉXITO TOTAL! Dataset creado en: {RUTA_SALIDA}")
    print(df_final.head(24))

if __name__ == "__main__":
    generar()