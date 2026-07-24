import numpy as np
import pandas as pd

# Parámetros de batería desde config/system.yaml vía la fuente única src.core.config
# (antes hardcodeados, lo que provocó que el config dijera 80 kWh y el simulador
# entrenara con 100). Fallback a {} por si no se encuentra (lo cubren los .get()).
from src.core.config import bateria as _bateria_cfg
try:
    _BAT = _bateria_cfg()
except Exception:
    _BAT = {}


class ComunidadSimulador:
    """
    Motor físico de la comunidad energética con 9 acciones discretas.

    Args:
        data_path: Ruta al dataset_final.csv.
        mode: 'all' (todo el dataset), 'train' (pool de entrenamiento),
              'eval' (pool de evaluación). Los pools se definen por muestreo
              aleatorio de semanas completas (168h) con semilla fija.
    """

    # Parámetros del split (deben coincidir con config/system.yaml)
    _SEED_SPLIT = 42
    _N_SEMANAS_EVAL = 50
    _HORAS_POR_SEMANA = 168

    # --- DEFINICIÓN DE ACCIONES (9 acciones) — FUENTE ÚNICA ---
    # Diseño v2 (2026-05-17): eliminados los niveles de potencia de CARGAR_SOLAR
    # y DESCARGAR_CASA por degeneración estructural.
    #   0     : IDLE
    #   1     : CARGAR_SOLAR   (nivel único — limita exc_disp, no el inversor)
    #   2,3,4 : CARGAR_MIXTA   33/66/100%  (inversor = cuello de botella)
    #   5     : DESCARGAR_CASA (nivel único — limita def_cub)
    #   6,7,8 : DESCARGAR_RED  33/66/100%
    # La usan ejecutar_accion_fisica (entreno) y accion_a_flujos (decode, también
    # en eval vía DiscreteRLController).
    ACCION_MAP = {
        0: ("IDLE",           0.0),
        1: ("CARGAR_SOLAR",   1.0),
        2: ("CARGAR_MIXTA",   0.33),
        3: ("CARGAR_MIXTA",   0.66),
        4: ("CARGAR_MIXTA",   1.0),
        5: ("DESCARGAR_CASA", 1.0),
        6: ("DESCARGAR_RED",  0.33),
        7: ("DESCARGAR_RED",  0.66),
        8: ("DESCARGAR_RED",  1.0),
    }

    @staticmethod
    def accion_a_flujos(action_idx, exc_disp, def_cub, p_inversor, eff_c, eff_d):
        """Decode discreto (FUENTE ÚNICA): acción 0-8 -> 4 flujos kW (cs, cm, dc, dr).

        Devuelve los flujos *intencionados* (sin recortar por batería); el recorte
        por espacio_libre/bat_disponible lo hace `aplicar_fisica_4flujos`. Así
        entreno (`ejecutar_accion_fisica`) y eval (`DiscreteRLController`)
        decodifican EXACTAMENTE igual.

        cs = carga solar, cm = carga red, dc = descarga a casa, dr = descarga a red
        (dc/dr en kW de lado batería; aplicar_fisica_4flujos aplica EFF_D al entregar).
        """
        estrategia, nivel = ComunidadSimulador.ACCION_MAP[action_idx]
        potencia_obj = nivel * p_inversor
        cs = cm = dc = dr = 0.0

        if estrategia == "CARGAR_SOLAR":
            cs = exc_disp                                   # se recorta a espacio en la física
        elif estrategia == "CARGAR_MIXTA":
            cs = min(exc_disp, potencia_obj)
            cm = potencia_obj - cs
        elif estrategia == "DESCARGAR_CASA":
            dc = def_cub / eff_d                            # cubrir todo el déficit (se recorta a batería)
        elif estrategia == "DESCARGAR_RED":
            energia_util = potencia_obj * eff_d
            para_casa = min(energia_util, def_cub)
            para_red = energia_util - para_casa
            dc = para_casa / eff_d
            dr = para_red / eff_d
        return cs, cm, dc, dr

    def __init__(self, data_path, mode='all'):
        # 1. Cargar datos
        try:
            self.df = pd.read_csv(data_path)
            self.df.columns = self.df.columns.str.strip()
        except FileNotFoundError:
            raise Exception(f"ERROR: No se encuentra {data_path}.")

        self.max_steps = len(self.df)

        # 2. Calcular pools de semanas
        total_semanas = self.max_steps // self._HORAS_POR_SEMANA
        rng_split = np.random.RandomState(self._SEED_SPLIT)
        indices_eval = sorted(rng_split.choice(
            total_semanas, self._N_SEMANAS_EVAL, replace=False))
        set_eval = set(indices_eval)
        indices_train = [i for i in range(total_semanas) if i not in set_eval]

        # Convertir a índices horarios de inicio de semana
        self.semanas_train = [i * self._HORAS_POR_SEMANA for i in indices_train]
        self.semanas_eval = [i * self._HORAS_POR_SEMANA for i in indices_eval]
        self._mode = mode
        
        # 2. Configuración Física (desde config/system.yaml — fuente única)
        self.BATERIA_CAPACIDAD = _BAT.get('capacidad_kwh', 80.0)         # kWh
        self.POTENCIA_INVERSOR = _BAT.get('potencia_inversor_kw', 50.0)  # kW
        self.SOC_INICIAL = _BAT.get('soc_inicial', 0.5)
        self.SOC_MIN = _BAT.get('soc_min', 0.10)               # Límite operativo inferior
        self.SOC_MAX = _BAT.get('soc_max', 0.90)               # Límite operativo superior
        self.EFICIENCIA_CARGA = _BAT.get('eficiencia_carga', 0.95)       # pérdidas AC→DC
        self.EFICIENCIA_DESCARGA = _BAT.get('eficiencia_descarga', 0.95) # pérdidas DC→AC
        self.AUTODESCARGA_POR_HORA = _BAT.get('autodescarga_por_hora', 0.00004)

        # Configuración Económica
        # Modelo asimétrico real (regulación española):
        # - Compra de red: precio PVPC completo (ind. ESIOS 1001) → precio_kwh
        # - Venta excedentes: precio compensación simplificada (ind. ESIOS 1739) → precio_excedente
        # RD 244/2019 Art.14: excedentes se valoran a Pmh - CDSVh (≈ precio mayorista OMIE)
        self.COSTE_DEGRADACION_BASE = _BAT.get('degradacion_base', 0.005)

        # Estado interno
        self.soc = self.SOC_INICIAL
        self.current_step = 0
        # El mapa de acciones es ComunidadSimulador.ACCION_MAP (atributo de clase,
        # fuente única). Antes estaba duplicado aquí como self.action_map.

    def get_data_window(self, step, horizon=24):
        cols = ['consumo_total', 'generacion_total', 'precio_kwh', 'precio_excedente']
        end = step + horizon
        if end > self.max_steps:
            padding = end - self.max_steps
            real_data = self.df.iloc[step : self.max_steps][cols].values
            return np.pad(real_data, ((0, padding), (0, 0)), mode='constant')
        return self.df.iloc[step : end][cols].values

    def get_tiempo(self, step):
        """(hora, dia_sem, mes0) para un step, desde la columna 'fecha'.

        Fuente ÚNICA del tiempo, compartida por el entorno de entreno
        (energy_env) y el constructor de obs de eval/producción (obs_builder),
        para que las features temporales sean idénticas en ambos.
        Si no hay 'fecha', cae a aritmética de step (hora=step%24, mes=0).
        """
        if not hasattr(self, '_fechas'):
            try:
                self._fechas = pd.to_datetime(self.df['fecha'])
            except Exception:
                self._fechas = None
        if self._fechas is not None and step < len(self._fechas):
            ts = self._fechas.iloc[step]
            return ts.hour, ts.dayofweek, ts.month - 1
        return step % 24, (step // 24) % 7, 0

    def calcular_degradacion_no_lineal(self, energia_kwh, soc_actual):
        if energia_kwh == 0: return 0.0
        # Con paso horario (Δt=1h) la energía movida en kWh coincide numéricamente
        # con la potencia media en kW, de ahí que sirva para el ratio con el inversor.
        potencia = energia_kwh

        # Factor Potencia (I^2)
        ratio_potencia = (potencia / self.POTENCIA_INVERSOR)
        factor_stress_potencia = 1.0 + (ratio_potencia ** 2)
        
        # Factor SoC (Extremos)
        desviacion = abs(soc_actual - 0.5)
        factor_stress_soc = 1.0 + 15.0 * (desviacion ** 4)
        
        return self.COSTE_DEGRADACION_BASE * factor_stress_potencia * factor_stress_soc * energia_kwh

    def aplicar_fisica_4flujos(self, cs, cm, dc, dr):
        """Física de la batería para 4 flujos en kW (cs, cm, dc, dr) sobre la hora
        ACTUAL (self.current_step). FUENTE ÚNICA compartida por el entorno continuo
        (Residual SAC) y el MPC (simular_hora_mpc) → garantiza que mueven la batería
        EXACTAMENTE igual. Lee el dato real del paso, actualiza self.soc y devuelve
        el resultado económico. NO avanza el tiempo (lo hace quien llama).

        cs = carga desde solar, cm = carga desde red, dc = descarga a casa,
        dr = descarga a red.
        """
        row = self.df.iloc[self.current_step]
        gen, cons = row['generacion_total'], row['consumo_total']
        precio_compra, precio_venta = row['precio_kwh'], row['precio_excedente']

        balance = gen - cons
        exc_disp = max(0.0, balance)
        def_cub = max(0.0, -balance)

        # Autodescarga
        self.soc *= (1 - self.AUTODESCARGA_POR_HORA)
        bateria_kwh = self.soc * self.BATERIA_CAPACIDAD
        espacio_libre = max(0.0, self.SOC_MAX * self.BATERIA_CAPACIDAD - bateria_kwh)
        bat_disponible = max(0.0, bateria_kwh - self.SOC_MIN * self.BATERIA_CAPACIDAD)

        cs = max(0.0, float(cs)); cm = max(0.0, float(cm))
        dc = max(0.0, float(dc)); dr = max(0.0, float(dr))

        # Netear carga vs descarga (inversor bidireccional; preserva el ratio interno)
        carga_bruta = cs + cm
        descarga_bruta = dc + dr
        net = carga_bruta - descarga_bruta
        if net >= 0:
            ratio_solar = cs / carga_bruta if carga_bruta > 0 else 0.0
            cs = net * ratio_solar
            cm = net * (1 - ratio_solar)
            dc, dr = 0.0, 0.0
        else:
            ratio_casa = dc / descarga_bruta if descarga_bruta > 0 else 0.0
            dc = abs(net) * ratio_casa
            dr = abs(net) * (1 - ratio_casa)
            cs, cm = 0.0, 0.0

        # Recortar por estado real
        cs = min(cs, exc_disp, espacio_libre / self.EFICIENCIA_CARGA)
        cm = min(cm, max(0.0, espacio_libre / self.EFICIENCIA_CARGA - cs))
        carga_total = cs + cm
        dc = min(dc, bat_disponible)
        dr = min(dr, max(0.0, bat_disponible - dc))
        descarga_total = dc + dr

        # Actualizar batería
        soc_antes = self.soc
        bateria_kwh += carga_total * self.EFICIENCIA_CARGA - descarga_total
        self.soc = float(np.clip(bateria_kwh / self.BATERIA_CAPACIDAD, 0.0, 1.0))

        # Economía
        comprado = max(0.0, def_cub - dc * self.EFICIENCIA_DESCARGA) + cm
        vendido = (exc_disp - cs) + dr * self.EFICIENCIA_DESCARGA
        ingresos = vendido * precio_venta
        gastos = comprado * precio_compra

        energia_movida = carga_total + descarga_total
        soc_medio = (soc_antes + self.soc) / 2
        coste_deg = self.calcular_degradacion_no_lineal(energia_movida, soc_medio)

        beneficio = ingresos - gastos - coste_deg
        beneficio_idle = exc_disp * precio_venta - def_cub * precio_compra

        return {
            'soc': self.soc,
            'comprado': comprado,
            'vendido': vendido,
            'cargado': carga_total,
            'descargado': descarga_total,
            'beneficio': beneficio,
            'beneficio_idle': beneficio_idle,
            'beneficio_marginal': beneficio - beneficio_idle,
        }

    def ejecutar_accion_fisica(self, action_idx, step):
        """Ejecuta una acción discreta (0-8) sobre la hora `step`.

        Decodifica la acción a 4 flujos con la FUENTE ÚNICA
        `accion_a_flujos` y delega la física en `aplicar_fisica_4flujos`
        (el mismo núcleo que usan el MPC y los controladores continuo/residual).
        Sobre el `beneficio_marginal` LIMPIO que devuelve la física, re-aplica el
        shaping `coste_oportunidad` (penaliza DESCARGAR_RED→red a SoC bajo), que es
        propio del reward de ENTRENO (la eval usa el beneficio limpio).
        """
        # Asegurar que aplicar_fisica_4flujos lee la misma hora.
        self.current_step = step
        row = self.df.iloc[step]
        gen, cons = row['generacion_total'], row['consumo_total']
        precio_compra = row['precio_kwh']           # PVPC (ind. 1001)
        precio_venta = row['precio_excedente']       # Compensación simplificada (ind. 1739)

        balance = gen - cons
        exc_disp = max(0.0, balance)
        def_cub = max(0.0, -balance)

        estrategia, _nivel = self.ACCION_MAP[action_idx]
        # SoC tras autodescarga (lo que usará aplicar_fisica_4flujos) — para el shaping.
        soc_antes = self.soc * (1 - self.AUTODESCARGA_POR_HORA)

        cs, cm, dc, dr = self.accion_a_flujos(
            action_idx, exc_disp, def_cub,
            self.POTENCIA_INVERSOR, self.EFICIENCIA_CARGA, self.EFICIENCIA_DESCARGA)

        # Física + economía (beneficio_marginal LIMPIO, sin shaping).
        r = self.aplicar_fisica_4flujos(cs, cm, dc, dr)

        # Coste de oportunidad dinámico para DESCARGAR_RED→red a SoC < SOC_LIBRE
        # (shaping del entreno). para_red se reconstruye de la descarga final, igual
        # que en la versión histórica.
        _SOC_LIBRE = 0.70
        coste_oportunidad = 0.0
        if estrategia == "DESCARGAR_RED":
            energia_util = r["descargado"] * self.EFICIENCIA_DESCARGA
            para_casa = min(energia_util, def_cub)
            para_red = energia_util - para_casa
            if para_red > 0 and soc_antes < _SOC_LIBRE:
                factor = (_SOC_LIBRE - soc_antes) / (_SOC_LIBRE - self.SOC_MIN)
                coste_oportunidad = para_red * (precio_compra - precio_venta) * factor * 4.0

        return {
            "beneficio":          r["beneficio"],
            "beneficio_marginal": r["beneficio_marginal"] - coste_oportunidad,
            "soc":                r["soc"],
            "comprado":           r["comprado"],
            "cargado":            r["cargado"],
            "descargado":         r["descargado"],
        }