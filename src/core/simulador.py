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
        self.BATERIA_CAPACIDAD = 100.0 # kWh
        self.POTENCIA_INVERSOR = 50.0  # kW
        self.SOC_INICIAL = 0.5
        
        # Configuración Económica
        self.PRECIO_VENTA_EXCEDENTE = 0.05 
        self.COSTE_DEGRADACION_BASE = 0.005
        
        # Estado interno
        self.soc = self.SOC_INICIAL 
        self.current_step = 0

        # --- DEFINICIÓN DE ACCIONES (EL MAPA) ---
        # Definimos 3 niveles de potencia: Bajo (33%), Medio (66%), Alto (100%)
        self.NIVELES = [0.33, 0.66, 1.0]
        
        # Creamos el mapa de acciones: ID -> (Estrategia, Nivel)
        # 0: IDLE
        # 1-3: CARGAR_SOLAR (Nivel 1, 2, 3)
        # 4-6: CARGAR_MIXTA (Nivel 1, 2, 3)
        # 7-9: DESCARGAR_CASA (Nivel 1, 2, 3)
        # 10-12: DESCARGAR_RED (Nivel 1, 2, 3)
        
        self.action_map = {0: ("IDLE", 0.0)}
        idx = 1
        estrategias = ["CARGAR_SOLAR", "CARGAR_MIXTA", "DESCARGAR_CASA", "DESCARGAR_RED"]
        
        for estrat in estrategias:
            for nivel in self.NIVELES:
                self.action_map[idx] = (estrat, nivel)
                idx += 1
        
        # Total acciones: 1 + (4 estrategias * 3 niveles) = 13

    def get_data_window(self, step, horizon=24):
        end = step + horizon
        if end > self.max_steps:
            padding = end - self.max_steps
            real_data = self.df.iloc[step : self.max_steps][['consumo_total', 'generacion_total', 'precio_kwh']].values
            return np.pad(real_data, ((0, padding), (0, 0)), mode='constant')
        return self.df.iloc[step : end][['consumo_total', 'generacion_total', 'precio_kwh']].values

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
        gen, cons, precio = row['generacion_total'], row['consumo_total'], row['precio_kwh']
        
        balance = gen - cons
        exc_disp = max(0, balance)
        def_cub = abs(min(0, balance))
        
        bateria_kwh = self.soc * self.BATERIA_CAPACIDAD
        espacio_libre = self.BATERIA_CAPACIDAD - bateria_kwh
        
        # 2. DECODIFICAR ACCIÓN
        estrategia, nivel_potencia = self.action_map[action_idx]
        
        # Calcular potencia objetivo en kW (ej: 0.33 * 50 = 16.5 kW)
        potencia_obj = nivel_potencia * self.POTENCIA_INVERSOR
        
        # Flujos
        comprado, vendido, cargado, descargado = 0.0, 0.0, 0.0, 0.0

        # --- LÓGICA GENERALIZADA ---

        if estrategia == "IDLE":
            vendido += exc_disp
            comprado += def_cub

        elif estrategia == "CARGAR_SOLAR":
            # Cargar solo con lo que sobre (limitado por potencia obj)
            carga = min(exc_disp, espacio_libre, potencia_obj)
            bateria_kwh += carga
            cargado += carga
            vendido += (exc_disp - carga)
            comprado += def_cub

        elif estrategia == "CARGAR_MIXTA":
            # Cargar hasta el objetivo SÍ O SÍ (usando red si hace falta)
            carga_real = min(espacio_libre, potencia_obj)
            
            de_sol = min(exc_disp, carga_real)
            de_red = carga_real - de_sol
            
            bateria_kwh += carga_real
            cargado += carga_real
            vendido += (exc_disp - de_sol)
            comprado += (def_cub + de_red)

        elif estrategia == "DESCARGAR_CASA":
            # Descargar solo lo necesario para la casa (limitado por potencia obj)
            descarga = min(def_cub, bateria_kwh, potencia_obj)
            bateria_kwh -= descarga
            descargado += descarga
            comprado += (def_cub - descarga)
            vendido += exc_disp

        elif estrategia == "DESCARGAR_RED":
            # Descargar a tope hacia la red (Cascada: Casa -> Red)
            # Limitado por la potencia objetivo (ej: 33%)
            descarga_total = min(bateria_kwh, potencia_obj)
            bateria_kwh -= descarga_total
            descargado += descarga_total
            
            para_casa = min(descarga_total, def_cub)
            para_red = descarga_total - para_casa
            
            comprado += (def_cub - para_casa)
            vendido += (para_red + exc_disp)

        # 3. Actualizar Estado (guardar SoC antes para cálculo de degradación)
        soc_antes = self.soc
        self.soc = np.clip(bateria_kwh / self.BATERIA_CAPACIDAD, 0.0, 1.0)

        # 4. Calcular Economía
        # Beneficio = cash flow neto: ingresos por venta - gastos por compra - degradación
        ingresos = vendido * self.PRECIO_VENTA_EXCEDENTE
        gastos = comprado * precio

        # Degradación calculada con SoC medio del ciclo (más fiel al stress real)
        energia_movida = cargado + descargado
        soc_medio = (soc_antes + self.soc) / 2
        coste_deg = self.calcular_degradacion_no_lineal(energia_movida, soc_medio)

        beneficio = ingresos - gastos - coste_deg
        
        return {"beneficio": beneficio, "soc": self.soc, "comprado": comprado}