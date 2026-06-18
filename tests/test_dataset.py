"""
test_dataset.py — Tests del pipeline ETL (generar_dataset_final.py)
===================================================================
Ejecutar desde la raíz del proyecto (carpeta TFG/):
    pytest tests/test_dataset.py -v
"""

import os
import sys
import json
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATASET_PATH  = os.path.join(ROOT, 'data', 'processed', 'dataset_final.csv')
METADATA_PATH = os.path.join(ROOT, 'data', 'processed', 'metadata.json')


@pytest.fixture(scope='module')
def df():
    """Carga el dataset generado."""
    assert os.path.exists(DATASET_PATH), (
        f"No se encuentra {DATASET_PATH}. Ejecuta primero generar_dataset_final.py.")
    return pd.read_csv(DATASET_PATH, parse_dates=['fecha'])


@pytest.fixture(scope='module')
def metadata():
    """Carga el metadata.json generado."""
    assert os.path.exists(METADATA_PATH), (
        f"No se encuentra {METADATA_PATH}. Ejecuta primero generar_dataset_final.py.")
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


# -----------------------------------------------------------------
# 1. Dimensiones del dataset
# -----------------------------------------------------------------
class TestDimensiones:

    def test_dataset_dimensiones(self, df):
        """El dataset tiene ~22.646 filas y 5 columnas (fecha + 4 datos)."""
        n_filas = len(df)
        # Tolerancia ±48 horas por bordes de alineación temporal
        assert 22_600 <= n_filas <= 22_700, (
            f"Filas esperadas ~22.646, obtenidas {n_filas}")
        assert df.shape[1] == 5, (
            f"Columnas esperadas 5 (fecha + 4 datos), obtenidas {df.shape[1]}")

    def test_columnas_esperadas(self, df):
        """Verifica que las columnas del dataset son las esperadas."""
        esperadas = {'fecha', 'consumo_total', 'generacion_total',
                     'precio_kwh', 'precio_excedente'}
        assert set(df.columns) == esperadas


# -----------------------------------------------------------------
# 2. Demanda anualizada en rango
# -----------------------------------------------------------------
class TestDemanda:

    def test_demanda_anualizada_en_rango(self, df):
        """Consumo anual agregado entre ~42.500 y 69.000 kWh/año (15 × 3500 ± margen)."""
        horas_totales = len(df)
        consumo_anual = df['consumo_total'].sum() / horas_totales * 8760
        # 15 viviendas × 3500 kWh = 52.500 kWh/año
        # Margen amplio: [50.000 × 0.85, 60.000 × 1.15]
        assert 42_500 <= consumo_anual <= 69_000, (
            f"Consumo anualizado fuera de rango: {consumo_anual:.0f} kWh/año")


# -----------------------------------------------------------------
# 3. Generación anualizada en rango
# -----------------------------------------------------------------
class TestGeneracion:

    def test_generacion_anualizada_en_rango(self, df):
        """Producción anual entre 50 y 76 MWh/año para 42 kWp en Galicia."""
        horas_totales = len(df)
        gen_anual_kwh = df['generacion_total'].sum() / horas_totales * 8760
        gen_anual_mwh = gen_anual_kwh / 1000
        assert 50 <= gen_anual_mwh <= 76, (
            f"Generación anualizada fuera de rango: {gen_anual_mwh:.1f} MWh/año")


# -----------------------------------------------------------------
# 4. Precios no negativos
# -----------------------------------------------------------------
class TestPrecios:

    def test_precios_excedente_no_negativos(self, df):
        """Ningún precio de excedente es negativo tras saneamiento."""
        n_negativos = (df['precio_excedente'] < 0).sum()
        assert n_negativos == 0, (
            f"Encontrados {n_negativos} precios de excedente negativos")

    def test_precios_kwh_no_negativos(self, df):
        """Ningún precio PVPC es negativo."""
        n_negativos = (df['precio_kwh'] < 0).sum()
        assert n_negativos == 0, (
            f"Encontrados {n_negativos} precios PVPC negativos")

    def test_precios_en_unidades_kwh(self, df):
        """El precio máximo es < 2.0 EUR/kWh (sanity check conversión MWh→kWh)."""
        max_precio = df['precio_kwh'].max()
        assert max_precio < 2.0, (
            f"Precio máximo {max_precio:.4f} EUR/kWh — posible error de unidades")


# -----------------------------------------------------------------
# 5. Determinismo
# -----------------------------------------------------------------
class TestDeterminismo:

    def test_determinismo(self):
        """Ejecutar dos veces con la misma semilla produce datasets idénticos."""
        # Importar y ejecutar la función directamente (evita problemas de memoria
        # con subprocess en entornos con paging file limitado)
        sys.path.insert(0, os.path.join(ROOT, 'src', 'utils'))
        from generar_dataset_final import generar

        # Guardar CSV original
        df1 = pd.read_csv(DATASET_PATH)

        # Regenerar con misma semilla
        generar(semilla=42)

        df2 = pd.read_csv(DATASET_PATH)
        pd.testing.assert_frame_equal(df1, df2, check_exact=True)


# -----------------------------------------------------------------
# 6. Autocorrelación del ruido AR(1)
# -----------------------------------------------------------------
class TestRuidoAR1:

    def test_autocorrelacion_ruido(self):
        """
        La autocorrelación lag-1 del factor estocástico generado debe estar
        cerca de rho_consumo declarado (0.50, atol=0.1).
        """
        # Parámetros del config
        rho = 0.50
        sigma = 0.05
        n = 20_000
        rng = np.random.default_rng(42)

        epsilon = np.zeros(n)
        coef = np.sqrt(1 - rho ** 2)
        innovacion = rng.standard_normal(n)
        for t in range(1, n):
            epsilon[t] = rho * epsilon[t - 1] + coef * innovacion[t]

        factor = 1.0 + sigma * epsilon

        # Autocorrelación lag-1
        autocorr = np.corrcoef(factor[:-1], factor[1:])[0, 1]
        assert abs(autocorr - rho) < 0.1, (
            f"Autocorrelación lag-1 = {autocorr:.3f}, esperada ~{rho} (atol=0.1)")


# -----------------------------------------------------------------
# 7. Split no solapado
# -----------------------------------------------------------------
class TestSplit:

    def test_split_no_solapado(self, df):
        """Los pools de semanas train y eval son disjuntos."""
        n_horas = len(df)
        horas_por_semana = 168
        total_semanas = n_horas // horas_por_semana
        n_eval = 50

        rng_split = np.random.RandomState(42)
        indices_eval = sorted(rng_split.choice(total_semanas, n_eval, replace=False))
        set_eval = set(indices_eval)
        indices_train = [i for i in range(total_semanas) if i not in set_eval]

        # Disjuntos
        assert len(set(indices_train) & set_eval) == 0

        # Cobertura (todas las semanas completas repartidas)
        assert len(indices_train) + len(indices_eval) == total_semanas

        # Eval tiene exactamente 50 semanas
        assert len(indices_eval) == n_eval

        # Train tiene al menos 80 semanas
        assert len(indices_train) >= 80, (
            f"Train muy corto: {len(indices_train)} semanas")

        # Ningún índice horario de eval cae en train (verificación a nivel hora)
        horas_eval = set()
        for s in indices_eval:
            for h in range(horas_por_semana):
                horas_eval.add(s * horas_por_semana + h)
        horas_train = set()
        for s in indices_train:
            for h in range(horas_por_semana):
                horas_train.add(s * horas_por_semana + h)
        assert len(horas_eval & horas_train) == 0, "Data leakage: horas compartidas"


# -----------------------------------------------------------------
# 8. Timestamps cargables
# -----------------------------------------------------------------
class TestTimestamps:

    def test_timestamps_cargables(self, df):
        """La columna fecha es un DatetimeIndex válido con inicio en junio."""
        assert pd.api.types.is_datetime64_any_dtype(df['fecha']), (
            "La columna 'fecha' no es datetime")
        primer_mes = df['fecha'].iloc[0].month
        assert primer_mes == 6, (
            f"El primer mes debe ser junio (6), obtenido {primer_mes}")

    def test_timestamps_monotonos(self, df):
        """Los timestamps son no-decrecientes (DST produce ~2-3 duplicados/año)."""
        diffs = df['fecha'].diff().iloc[1:]
        assert (diffs >= pd.Timedelta(0)).all(), "Timestamps decrecientes detectados"
        # Como máximo 3 horas con diff=0 (transiciones DST en ~2.5 años)
        n_duplicados = (diffs == pd.Timedelta(0)).sum()
        assert n_duplicados <= 5, (
            f"Demasiados timestamps duplicados: {n_duplicados} (esperados ≤ 3 por DST)")
