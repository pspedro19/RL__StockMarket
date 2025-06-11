import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from gymnasium import spaces
import gymnasium as gym

class TradingEnv(gym.Env):
    """Entorno de trading simple para RL"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        super().__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        # Espacio de acciones: [0=Hold, 1=Buy, 2=Sell]
        self.action_space = spaces.Discrete(3)
        
        # Espacio de observaciones: [price, volume, balance, position]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, True, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        # Ejecutar acción
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 0:  # Sell
            self.position = -1
            self.entry_price = current_price
        elif action == 0 and self.position != 0:  # Close position
            if self.position == 1:  # Close long
                reward = (current_price - self.entry_price) / self.entry_price
            else:  # Close short
                reward = (self.entry_price - current_price) / self.entry_price
            self.position = 0
            self.entry_price = 0
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        if self.current_step >= len(self.data):
            return np.array([0, 0, self.balance, self.position], dtype=np.float32)
        
        row = self.data.iloc[self.current_step]
        return np.array([
            row['close'] / 1000,
            row['volume'] / 1000,
            self.balance / self.initial_balance,
            self.position
        ], dtype=np.float32)

def test_agent():
    """Probar agente básico"""
    print("🤖 Probando agente RL básico...")
    
    try:
        data = pd.read_csv('data/raw/EURUSD_7days.csv')
        print(f"📊 Datos cargados: {len(data)} registros")
    except FileNotFoundError:
        print("❌ No se encontraron datos. Ejecuta primero data_generator.py")
        return
    
    env = TradingEnv(data)
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3)
    print("✅ Agente DQN creado")
    
    print("🏋️ Entrenando por 1000 pasos...")
    model.learn(total_timesteps=1000)
    
    obs, _ = env.reset()
    total_reward = 0
    actions_taken = []
    
    for i in range(100):
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(int(action))  # Convertir a int
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    
    print(f"🎯 Recompensa total: {total_reward:.4f}")
    print(f"📈 Acciones únicas: {len(set(actions_taken))} tipos")
    print("✅ Agente funcionando correctamente!")

if __name__ == "__main__":
    test_agent()
