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
        
        # --- ESTADO: 76 Variables ---
        # 4 actuales (SoC, Precio, Excedente, Deficit)
        # + 72 futuras (24h * 3 variables: Precio, Gen, Cons)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(76,), dtype=np.float32
        )
        
        # Configuración del Episodio
        self.EPISODE_LENGTH = 24 * 7  # Episodios de 1 semana
        self.steps_in_episode = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Elegir inicio aleatorio con margen suficiente
        max_start = self.simulador.max_steps - self.EPISODE_LENGTH - 25
        start_step = np.random.randint(0, max_start)
        
        # Resetear simulador
        self.simulador.current_step = start_step
        self.simulador.soc = self.simulador.SOC_INICIAL
        self.steps_in_episode = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Construye el vector de estado completo."""
        t = self.simulador.current_step
        
        # 1. Datos Actuales
        datos_hoy = self.simulador.get_data_window(t, horizon=1)[0]
        # [consumo, generacion, precio]
        cons, gen, precio = datos_hoy
        
        balance = gen - cons
        exc = max(0, balance)
        def_ = abs(min(0, balance))
        
        # 2. Pronóstico (24 horas futuras)
        window_future = self.simulador.get_data_window(t+1, horizon=24)
        forecast_flat = window_future.flatten() # Aplanar matriz a vector
        
        # 3. Concatenar todo (4 + 72 = 76)
        obs = np.concatenate(([self.simulador.soc, precio, exc, def_], forecast_flat))
        
        return obs.astype(np.float32)

    def step(self, action):
        # 1. Ejecutar en el simulador
        resultado = self.simulador.ejecutar_accion_fisica(action, self.simulador.current_step)
        
        # 2. Obtener Recompensa
        reward = resultado["beneficio"]
        
        # 3. Avanzar tiempo
        self.simulador.current_step += 1
        self.steps_in_episode += 1
        
        # 4. Comprobar fin
        terminated = (self.steps_in_episode >= self.EPISODE_LENGTH)
        truncated = False
        
        # Info extra (útil para gráficas luego)
        info = {
            "soc": resultado["soc"],
            "beneficio": resultado["beneficio"],
            "comprado": resultado["comprado"]
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        pass