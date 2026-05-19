"""
generar_dataset_final.py — Pipeline ETL para el dataset de la comunidad energética
==================================================================================
Genera data/processed/dataset_final.csv a partir de fuentes ESIOS y PVGIS.

Cambios principales respecto a la versión anterior:
  - Columna 'fecha' explícita (datetime, primera columna) para corregir el bug
    de timestamps en energy_env.py (mes=0 siempre).
  - Ruido de consumo: AR(1) sobre la demanda agregada (antes: i.i.d. por vivienda).
  - Saneamiento: precio_excedente clamped a ≥ 0 (RD 244/2019 Art.14).
  - Metadata: data/processed/metadata.json con parámetros y hashes.
  - Semilla configurable por argumento (default: 42).

Ejecución desde la raíz del proyecto (carpeta TFG/):
    python src/utils/generar_dataset_final.py [--semilla 42]
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# --- Cargar configuración ---
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_PATH = os.path.join(_ROOT, 'config', 'system.yaml')

with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
    _CFG = yaml.safe_load(f)

# Parámetros desde config
RUTA_CONSUMO   = _CFG['rutas']['consumo_raw']
RUTA_PRECIOS   = _CFG['rutas']['precios_raw']
RUTA_EXCEDENTE = _CFG['rutas']['compensacion_raw']
RUTA_SOLAR     = _CFG['rutas']['solar_raw']
RUTA_SALIDA    = _CFG['rutas']['dataset_final']
RUTA_METADATA  = _CFG['rutas']['metadata']

NUM_VECINOS            = _CFG['comunidad']['num_viviendas']
POTENCIA_SOLAR_TOTAL   = _CFG['comunidad']['potencia_solar_total_kwp']
CONSUMO_ANUAL_CASA     = _CFG['comunidad']['consumo_anual_vivienda_kwh']

RHO_CONSUMO    = _CFG['etl']['ruido_demanda']['rho']
SIGMA_CONSUMO  = _CFG['etl']['ruido_demanda']['sigma']
CLIP_MIN       = _CFG['etl']['ruido_demanda']['clip_min']
CLIP_MAX       = _CFG['etl']['ruido_demanda']['clip_max']
PRECIO_EXC_MIN = _CFG['etl']['precio_excedente_min']


def _sha256(filepath: str) -> str:
    """Calcula el hash SHA-256 de un archivo."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for bloque in iter(lambda: f.read(8192), b''):
            h.update(bloque)
    return h.hexdigest()


def generar(semilla: int = 42):
    """
    Genera el dataset final de la comunidad energética.

    Args:
        semilla: Semilla para reproducibilidad de todos los procesos estocásticos.
    """
    print("=" * 62)
    print("GENERACIÓN DEL DATASET MULTI-AÑO")
    print("=" * 62)
    print(f"  Semilla: {semilla}")
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    rng = np.random.default_rng(semilla)

    # =================================================================
    # 1. PROCESAR CONSUMO (ESIOS ind. 1006, perfil 2.0TD)
    # =================================================================
    print("\n  1. Procesando Consumo (ESIOS ind. 1006)...")
    df_cons = pd.read_csv(RUTA_CONSUMO, sep=';', encoding='utf-8')
    perfil_base = df_cons['value'].values
    n_horas = len(perfil_base)

    # Extraer timestamps del campo 'datetime' de ESIOS.
    # Se convierten a hora local peninsular (Europe/Madrid) para que las features
    # temporales (hora, día, mes) reflejen los patrones reales de consumo/solar/precio.
    # Nota: las transiciones DST (último domingo de octubre) producen 1 hora con
    # timestamp local duplicado (~2-3 veces en 2.5 años). Es despreciable y no
    # afecta al entrenamiento del RL.
    fechas = pd.to_datetime(df_cons['datetime'], utc=True)
    fechas = fechas.dt.tz_convert('Europe/Madrid').dt.tz_localize(None)

    # Convertir coeficiente PVPC (normalizado) a kWh por vivienda
    perfil_kwh = perfil_base * CONSUMO_ANUAL_CASA

    # Consumo agregado de las 15 viviendas
    consumo_base = perfil_kwh * NUM_VECINOS

    # Ruido AR(1) multiplicativo sobre la demanda agregada.
    # ε[h] = ρ × ε[h-1] + √(1 - ρ²) × N(0,1)  →  Var(ε) = 1 (estacionaria)
    # factor[h] = clip(1 + σ × ε[h], clip_min, clip_max)
    #
    # Justificación:
    #   - ρ=0.50: persistencia horaria moderada (condiciones meteorológicas y
    #     patrones de ocupación introducen correlación temporal) [VERIFICAR CITA].
    #   - σ=0.05: ~5% de variabilidad, coherente con errores de pronóstico de
    #     demanda residencial agregada para comunidades de ~15 viviendas.
    #   - Este modelo es coherente con el ruido AR(1) del pronóstico del sistema
    #     (ρ_consumo_pronóstico=0.3, σ=10%), cerrando el círculo metodológico.
    epsilon = np.zeros(n_horas)
    innovacion = rng.standard_normal(n_horas)
    coef_innovacion = np.sqrt(1 - RHO_CONSUMO ** 2)
    for t in range(1, n_horas):
        epsilon[t] = RHO_CONSUMO * epsilon[t - 1] + coef_innovacion * innovacion[t]

    factor = np.clip(1.0 + SIGMA_CONSUMO * epsilon, CLIP_MIN, CLIP_MAX)
    consumo_total = np.maximum(consumo_base * factor, 0.0)

    print(f"     {n_horas} horas, {NUM_VECINOS} viviendas (demanda agregada con AR(1)).")
    print(f"     rho={RHO_CONSUMO}, sigma={SIGMA_CONSUMO}, rango factor=[{CLIP_MIN}, {CLIP_MAX}]")

    # =================================================================
    # 2. PROCESAR SOLAR (PVGIS — instalaciones individuales por vivienda)
    # =================================================================
    print("\n  2. Procesando Solar (PVGIS) — instalaciones individuales...")
    df_sol = pd.read_csv(RUTA_SOLAR, skiprows=10, skipfooter=10, engine='python')
    solar_unitario = df_sol['P'].values / 1000.0  # W → kW (perfil para 1 kWp)
    n_horas_solar = len(solar_unitario)

    # Distribuir la potencia total entre las 15 viviendas de forma aleatoria.
    # Cada casa tiene su propia instalación con capacidad y rendimiento distintos.
    caps = rng.uniform(2.0, 5.0, NUM_VECINOS)          # kWp por vivienda
    caps = caps / caps.sum() * POTENCIA_SOLAR_TOTAL     # normalizar a 50 kWp total

    generacion_total = np.zeros(n_horas_solar)
    perfs = np.zeros(NUM_VECINOS)
    for i in range(NUM_VECINOS):
        # Factor de rendimiento: orientación, inclinación, sombras parciales
        # Sur perfecto ~1.0, sureste/suroeste ~0.90, sombras ~0.80
        perf = rng.uniform(0.80, 1.0)
        perfs[i] = perf
        # Ruido horario pequeño: suciedad puntual, sombras de nubes locales
        # Con 15 casas el ruido se atenúa por agregación (~3% individual)
        ruido = rng.normal(1.0, 0.03, n_horas_solar)
        casa_solar = solar_unitario * caps[i] * perf * ruido
        generacion_total += np.maximum(casa_solar, 0)

    print(f"     {n_horas_solar} horas, {NUM_VECINOS} instalaciones individuales.")
    print(f"     Capacidades (kWp): min={caps.min():.2f}, max={caps.max():.2f}, "
          f"total={caps.sum():.2f}")
    print(f"     Rendimientos: min={perfs.min():.3f}, max={perfs.max():.3f}, "
          f"media={perfs.mean():.3f}")

    # =================================================================
    # 3. PROCESAR PRECIOS PVPC (compra de red, ESIOS ind. 1001)
    # =================================================================
    print("\n  3. Procesando Precios PVPC (ESIOS ind. 1001)...")
    df_prec = pd.read_csv(RUTA_PRECIOS, sep=';', encoding='utf-8')
    precios = df_prec['value'].values
    precios_kwh = precios / 1000.0  # EUR/MWh → EUR/kWh
    print(f"     {len(precios_kwh)} horas, convertidos a EUR/kWh.")

    # =================================================================
    # 4. PROCESAR PRECIO EXCEDENTARIA (venta de excedentes, ESIOS ind. 1739)
    # =================================================================
    print("\n  4. Procesando Precio Excedentaria (ESIOS ind. 1739)...")
    df_exc = pd.read_csv(RUTA_EXCEDENTE, sep=';', encoding='utf-8')
    precio_exc = df_exc['value'].values
    precio_exc_kwh = precio_exc / 1000.0  # EUR/MWh → EUR/kWh

    # Saneamiento: clamp a ≥ 0.
    # RD 244/2019 Art.14: bajo compensación simplificada, el productor nunca
    # recibe remuneración negativa por excedentes [VERIFICAR ARTÍCULO EXACTO].
    n_negativos_raw = int((precio_exc_kwh < PRECIO_EXC_MIN).sum())
    precio_exc_kwh = np.maximum(precio_exc_kwh, PRECIO_EXC_MIN)
    print(f"     {len(precio_exc_kwh)} horas, convertidos a EUR/kWh.")
    print(f"     Precios negativos saneados (clamped a {PRECIO_EXC_MIN}): {n_negativos_raw}")

    # =================================================================
    # 5. FUSIÓN Y GUARDADO
    # =================================================================
    min_len = min(len(consumo_total), len(generacion_total),
                  len(precios_kwh), len(precio_exc_kwh))
    print(f"\n  5. Fusionando ({min_len} horas)...")

    df_final = pd.DataFrame({
        'fecha':            fechas.values[:min_len],
        'consumo_total':    consumo_total[:min_len],
        'generacion_total': generacion_total[:min_len],
        'precio_kwh':       precios_kwh[:min_len],
        'precio_excedente': precio_exc_kwh[:min_len],
    })

    df_final.to_csv(RUTA_SALIDA, index=False)
    print(f"\n  Dataset creado: {RUTA_SALIDA}")
    print(f"  Shape: {df_final.shape}")
    print(f"\n  Estadísticas:")
    print(df_final.describe().round(4))

    # =================================================================
    # 6. METADATA
    # =================================================================
    hashes_fuente = {
        'consumo_raw':      _sha256(RUTA_CONSUMO),
        'precios_raw':      _sha256(RUTA_PRECIOS),
        'compensacion_raw': _sha256(RUTA_EXCEDENTE),
        'solar_raw':        _sha256(RUTA_SOLAR),
    }

    metadata = {
        'version_script': '2.0.0',
        'timestamp_generacion': datetime.now().isoformat(),
        'semilla': semilla,
        'n_horas': min_len,
        'rango_fechas': {
            'inicio': str(df_final['fecha'].iloc[0]),
            'fin':    str(df_final['fecha'].iloc[-1]),
        },
        'parametros': {
            'num_viviendas': NUM_VECINOS,
            'potencia_solar_kwp': POTENCIA_SOLAR_TOTAL,
            'consumo_anual_vivienda_kwh': CONSUMO_ANUAL_CASA,
            'ruido_demanda': {
                'tipo': 'AR(1)',
                'rho': RHO_CONSUMO,
                'sigma': SIGMA_CONSUMO,
                'clip': [CLIP_MIN, CLIP_MAX],
            },
            'ruido_solar': {
                'tipo': 'iid_por_vivienda',
                'sigma': 0.03,
            },
            'precio_excedente_min': PRECIO_EXC_MIN,
            'n_precios_negativos_saneados': n_negativos_raw,
        },
        'hashes_sha256_fuentes': hashes_fuente,
        'hash_sha256_salida': _sha256(RUTA_SALIDA),
        'split': {
            'tipo': 'muestreo_aleatorio_semanal',
            'semilla_split': _CFG['split']['semilla_split'],
            'n_semanas_eval': _CFG['split']['n_semanas_eval'],
            'horas_por_semana': _CFG['split']['horas_por_semana'],
        },
    }

    os.makedirs(os.path.dirname(RUTA_METADATA), exist_ok=True)
    with open(RUTA_METADATA, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Metadata: {RUTA_METADATA}")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Genera dataset_final.csv para la comunidad energética.')
    parser.add_argument('--semilla', type=int,
                        default=_CFG['etl']['semilla'],
                        help='Semilla para reproducibilidad (default: config)')
    args = parser.parse_args()
    generar(semilla=args.semilla)
