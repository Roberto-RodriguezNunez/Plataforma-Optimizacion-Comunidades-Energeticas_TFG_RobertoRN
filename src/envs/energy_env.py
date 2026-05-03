import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.core.simulador import ComunidadSimulador

class EnergyEnv(gym.Env):
    """
    Entorno Gymnasium compatible con Stable-Baselines3.
    """
    def __init__(self):
        super(EnergyEnv, self).__init__()
        
        # Instanciar el motor físico
        # Asegúrate de haber ejecutado la Tarea 1 para tener este archivo
        self.simulador = ComunidadSimulador('data/processed/dataset_final.csv')
        
        # --- ACCIONES: 9 Botones (Eco/Turbo/Mixtas) ---
        self.action_space = spaces.Discrete(13)
        
        # --- ESTADO: 101 Variables ---
        # 5 actuales (SoC, Precio_compra, Precio_venta, Excedente, Deficit)
        # + 96 futuras (24h * 4 variables: Consumo, Generacion, Precio_compra, Precio_venta)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(101,), dtype=np.float32
        )
        
        # Configuración del Episodio
        self.EPISODE_LENGTH = 24 * 7  # Episodios de 1 semana
        self.steps_in_episode = 0
        self._pendiente_coste_inicial = 0.0  # Se resta en el primer step

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Elegir inicio aleatorio con margen suficiente
        max_start = self.simulador.max_steps - self.EPISODE_LENGTH - 25
        start_step = np.random.randint(0, max_start)
        
        # Resetear simulador
        self.simulador.current_step = start_step
        self.simulador.soc = self.simulador.SOC_INICIAL
        self.steps_in_episode = 0

        # Calcular coste de la energía inicial (se resta en el primer step)
        soc_util = max(0, self.simulador.SOC_INICIAL - self.simulador.SOC_MIN)
        energia_inicial = soc_util * self.simulador.BATERIA_CAPACIDAD
        datos_hora = self.simulador.get_data_window(start_step, horizon=1)[0]
        precio_compra = datos_hora[2]
        self._pendiente_coste_inicial = energia_inicial * precio_compra * self.simulador.EFICIENCIA_DESCARGA

        return self._get_obs(), {}

    def _get_obs(self):
        """Construye el vector de estado completo."""
        t = self.simulador.current_step

        # 1. Datos Actuales
        datos_hoy = self.simulador.get_data_window(t, horizon=1)[0]
        # [consumo, generacion, precio_compra, precio_venta]
        cons, gen, precio_compra, precio_venta = datos_hoy

        balance = gen - cons
        exc = max(0, balance)
        def_ = abs(min(0, balance))

        # 2. Pronóstico (24 horas futuras)
        window_future = self.simulador.get_data_window(t+1, horizon=24)
        forecast_flat = window_future.flatten()  # 24 * 4 = 96 valores

        # 3. Concatenar todo (5 + 96 = 101)
        obs = np.concatenate(([self.simulador.soc, precio_compra, precio_venta, exc, def_], forecast_flat))

        return obs.astype(np.float32)

    def step(self, action):
        # 1. Ejecutar en el simulador
        resultado = self.simulador.ejecutar_accion_fisica(action, self.simulador.current_step)
        
        # 2. Obtener Recompensa (marginal vs IDLE para reducir varianza)
        reward = resultado["beneficio_marginal"]

        # Restar valor de la energía inicial en el primer paso (simetría con valor terminal)
        if self.steps_in_episode == 0:
            reward -= self._pendiente_coste_inicial
        
        # 3. Avanzar tiempo
        self.simulador.current_step += 1
        self.steps_in_episode += 1
        
        # 4. Comprobar fin
        terminated = (self.steps_in_episode >= self.EPISODE_LENGTH)
        truncated = False

        # 5. Valor terminal: la energía en batería al final no se pierde
        # Valorada al precio medio de compra evitada (PVPC medio del dataset)
        if terminated:
            soc_util = max(0, self.simulador.soc - self.simulador.SOC_MIN)
            energia_restante = soc_util * self.simulador.BATERIA_CAPACIDAD
            # Valorar a precio de compra actual (lo que costaría recargarla)
            datos_hora = self.simulador.get_data_window(self.simulador.current_step - 1, horizon=1)[0]
            precio_compra_actual = datos_hora[2]
            reward += energia_restante * precio_compra_actual * self.simulador.EFICIENCIA_DESCARGA

        # Info extra (útil para gráficas luego)
        info = {
            "soc": resultado["soc"],
            "beneficio": resultado["beneficio"],
            "comprado": resultado["comprado"]
        }

        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        pass