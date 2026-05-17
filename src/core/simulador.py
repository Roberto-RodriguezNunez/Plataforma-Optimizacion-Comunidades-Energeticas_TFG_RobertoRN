import numpy as np
import pandas as pd

class ComunidadSimulador:
    """
    Motor físico de la comunidad energética con 13 ACCIONES (3 Niveles de Potencia).
    """
    def __init__(self, data_path):
        # 1. Cargar datos
        try:
            self.df = pd.read_csv(data_path)
            self.df.columns = self.df.columns.str.strip()
        except FileNotFoundError:
            raise Exception(f"❌ ERROR: No se encuentra {data_path}.")

        self.max_steps = len(self.df)
        
        # 2. Configuración Física
        self.BATERIA_CAPACIDAD = 100.0    # kWh
        self.POTENCIA_INVERSOR = 50.0     # kW
        self.SOC_INICIAL = 0.5
        self.SOC_MIN = 0.10               # Límite operativo inferior (protección batería)
        self.SOC_MAX = 0.90               # Límite operativo superior (protección batería)
        self.EFICIENCIA_CARGA = 0.95      # 95% — pérdidas AC→DC al cargar
        self.EFICIENCIA_DESCARGA = 0.95   # 95% — pérdidas DC→AC al descargar (round-trip ~90%)
        self.AUTODESCARGA_POR_HORA = 0.00004  # ~3% mensual, típico Li-ion

        # Configuración Económica
        # Modelo asimétrico real (regulación española):
        # - Compra de red: precio PVPC completo (ind. ESIOS 1001) → precio_kwh
        # - Venta excedentes: precio compensación simplificada (ind. ESIOS 1739) → precio_excedente
        # RD 244/2019 Art.14: excedentes se valoran a Pmh - CDSVh (≈ precio mayorista OMIE)
        self.COSTE_DEGRADACION_BASE = 0.005

        # Estado interno
        self.soc = self.SOC_INICIAL
        self.current_step = 0

        # --- DEFINICIÓN DE ACCIONES (9 acciones) ---
        #
        # Diseño v2 (2026-05-17): eliminados los niveles de potencia de
        # CARGAR_SOLAR y DESCARGAR_CASA por degeneración estructural.
        #
        # Justificación completa: anotaciones_para_mi/justificacion_9_acciones.md
        #
        # 0        : IDLE
        # 1        : CARGAR_SOLAR  (nivel único — limitada por exc_disp, no por inversor)
        # 2, 3, 4  : CARGAR_MIXTA  33% / 66% / 100%  (inversor = cuello de botella)
        # 5        : DESCARGAR_CASA (nivel único — limitada por def_cub, no por inversor)
        # 6, 7, 8  : DESCARGAR_RED  33% / 66% / 100%  (inversor = cuello de botella)

        self.action_map = {
            0: ("IDLE",           0.0),
            1: ("CARGAR_SOLAR",   1.0),   # nivel ignorado en la lógica
            2: ("CARGAR_MIXTA",   0.33),
            3: ("CARGAR_MIXTA",   0.66),
            4: ("CARGAR_MIXTA",   1.0),
            5: ("DESCARGAR_CASA", 1.0),   # nivel ignorado en la lógica
            6: ("DESCARGAR_RED",  0.33),
            7: ("DESCARGAR_RED",  0.66),
            8: ("DESCARGAR_RED",  1.0),
        }
        # Total acciones: 9

    def get_data_window(self, step, horizon=24):
        cols = ['consumo_total', 'generacion_total', 'precio_kwh', 'precio_excedente']
        end = step + horizon
        if end > self.max_steps:
            padding = end - self.max_steps
            real_data = self.df.iloc[step : self.max_steps][cols].values
            return np.pad(real_data, ((0, padding), (0, 0)), mode='constant')
        return self.df.iloc[step : end][cols].values

    def calcular_degradacion_no_lineal(self, energia_kwh, soc_actual):
        if energia_kwh == 0: return 0.0
        potencia = energia_kwh 
        
        # Factor Potencia (I^2)
        ratio_potencia = (potencia / self.POTENCIA_INVERSOR)
        factor_stress_potencia = 1.0 + (ratio_potencia ** 2)
        
        # Factor SoC (Extremos)
        desviacion = abs(soc_actual - 0.5)
        factor_stress_soc = 1.0 + 15.0 * (desviacion ** 4)
        
        return self.COSTE_DEGRADACION_BASE * factor_stress_potencia * factor_stress_soc * energia_kwh

    def ejecutar_accion_fisica(self, action_idx, step):
        """
        Ejecuta la lógica usando el Mapa de Acciones.
        """
        # 1. Leer datos
        row = self.df.iloc[step]
        gen, cons = row['generacion_total'], row['consumo_total']
        precio_compra = row['precio_kwh']           # PVPC (ind. 1001)
        precio_venta = row['precio_excedente']       # Compensación simplificada (ind. 1739)
        
        balance = gen - cons
        exc_disp = max(0, balance)
        def_cub = abs(min(0, balance))

        # Autodescarga (pérdida natural de la batería, ~3% mensual)
        self.soc *= (1 - self.AUTODESCARGA_POR_HORA)

        bateria_kwh = self.soc * self.BATERIA_CAPACIDAD
        # Límites operativos: solo operar entre SOC_MIN y SOC_MAX
        espacio_libre = max(0, self.SOC_MAX * self.BATERIA_CAPACIDAD - bateria_kwh)
        bateria_disponible = max(0, bateria_kwh - self.SOC_MIN * self.BATERIA_CAPACIDAD)

        # 2. DECODIFICAR ACCIÓN
        estrategia, nivel_potencia = self.action_map[action_idx]

        # Calcular potencia objetivo en kW (ej: 0.33 * 50 = 16.5 kW)
        potencia_obj = nivel_potencia * self.POTENCIA_INVERSOR

        # Flujos
        comprado, vendido, cargado, descargado = 0.0, 0.0, 0.0, 0.0

        # --- LÓGICA GENERALIZADA ---
        # Eficiencia: cargar pierde 5% (AC→DC), descargar pierde 5% (DC→AC)
        # Round-trip: 0.95 × 0.95 = 90.25%

        if estrategia == "IDLE":
            vendido += exc_disp
            comprado += def_cub

        elif estrategia == "CARGAR_SOLAR":
            # Carga TODO el excedente solar disponible (sin comprar red).
            # No se aplica potencia_obj: el cuello de botella es siempre exc_disp
            # o el espacio libre de la batería, nunca el inversor.
            # Con niveles de potencia, las 3 acciones eran idénticas cuando
            # exc_disp < 16.5 kWh (la mayoría de horas) → degeneración eliminada.
            max_entrada = espacio_libre / self.EFICIENCIA_CARGA
            carga = min(exc_disp, max_entrada)
            bateria_kwh += carga * self.EFICIENCIA_CARGA
            cargado += carga
            vendido += (exc_disp - carga)
            comprado += def_cub

        elif estrategia == "CARGAR_MIXTA":
            # Cargar hasta el objetivo SÍ O SÍ (usando red si hace falta)
            max_entrada = espacio_libre / self.EFICIENCIA_CARGA
            carga_total = min(max_entrada, potencia_obj)

            de_sol = min(exc_disp, carga_total)
            de_red = carga_total - de_sol

            bateria_kwh += carga_total * self.EFICIENCIA_CARGA
            cargado += carga_total
            vendido += (exc_disp - de_sol)
            comprado += (def_cub + de_red)

        elif estrategia == "DESCARGAR_CASA":
            # Cubre TODO el déficit de las casas desde la batería (sin vender a red).
            # No se aplica potencia_obj: el cuello de botella es siempre def_cub
            # o la batería disponible, nunca el inversor.
            # Con niveles de potencia, las 3 acciones eran idénticas cuando
            # def_cub < 15.7 kWh (la mayoría de horas) → degeneración eliminada.
            descarga = min(def_cub / self.EFICIENCIA_DESCARGA, bateria_disponible)
            energia_util = descarga * self.EFICIENCIA_DESCARGA
            bateria_kwh -= descarga
            descargado += descarga
            comprado += (def_cub - energia_util)
            vendido += exc_disp

        elif estrategia == "DESCARGAR_RED":
            # Descargar a tope (Cascada: Casa → Red)
            descarga_total = min(bateria_disponible, potencia_obj)
            energia_util = descarga_total * self.EFICIENCIA_DESCARGA
            bateria_kwh -= descarga_total
            descargado += descarga_total

            para_casa = min(energia_util, def_cub)
            para_red = energia_util - para_casa

            comprado += (def_cub - para_casa)
            vendido += (para_red + exc_disp)

        # 3. Actualizar Estado (guardar SoC antes para cálculo de degradación)
        soc_antes = self.soc
        self.soc = np.clip(bateria_kwh / self.BATERIA_CAPACIDAD, 0.0, 1.0)

        # 4. Calcular Economia
        # Modelo asimétrico real: compra a PVPC (retail), venta a precio excedentaria (mayorista)
        # RD 244/2019 Art.14: excedentes valorados a Pmh - CDSVh (ind. ESIOS 1739)
        ingresos = vendido * precio_venta
        gastos = comprado * precio_compra

        # Degradación calculada con SoC medio del ciclo (más fiel al stress real)
        energia_movida = cargado + descargado
        soc_medio = (soc_antes + self.soc) / 2
        coste_deg = self.calcular_degradacion_no_lineal(energia_movida, soc_medio)

        beneficio = ingresos - gastos - coste_deg

        # Baseline IDLE: qué pasaría sin batería esta hora (reward shaping)
        # Es stateless: solo depende de los datos de la hora actual
        beneficio_idle = exc_disp * precio_venta - def_cub * precio_compra
        beneficio_marginal = beneficio - beneficio_idle

        return {
            "beneficio":          beneficio,
            "beneficio_marginal": beneficio_marginal,
            "soc":                self.soc,
            "comprado":           comprado,
            "cargado":            cargado,
            "descargado":         descargado,
        }