"""
Script simple para entrenar A2C compatible
"""

import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
import gymnasium as gym

# Añadir el directorio src al path
sys.path.append('src')

class TradingEnvA2CSimple(gym.Env):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.current_step = 50
        self.total_steps = len(data)
        
        # Mismo espacio de observación que el sistema principal
        self.observation_space = gym.spaces.Box(
            low=np.array([-4.8000002, -3.4028235e+38, -0.41887903, -3.4028235e+38], dtype=np.float32),
            high=np.array([4.8000002, 3.4028235e+38, 0.41887903, 3.4028235e+38], dtype=np.float32),
            dtype=np.float32
        )
        
        # A2C con acciones discretas (igual que el sistema principal)
        self.action_space = gym.spaces.Discrete(2)
        
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 50
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        return self._get_observation(), {}
        
    def _get_observation(self):
        return np.array([
            self.data.iloc[self.current_step]['norm_price'],
            self.data.iloc[self.current_step]['norm_volume'],
            self.data.iloc[self.current_step]['price_change'],
            self.data.iloc[self.current_step]['volume_change']
        ], dtype=np.float32)
        
    def step(self, action):
        done = False
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        if action == 1:  # Compra
            if self.position <= 0:
                self.position = 1
                self.entry_price = current_price
        elif action == 0:  # Venta
            if self.position >= 0:
                self.position = -1
                self.entry_price = current_price
                
        if self.position != 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            reward = price_change * self.position
            
        reward -= 0.0001
        self.total_reward += reward
        self.current_step += 1
        
        if self.current_step >= self.total_steps - 1:
            done = True
            
        return self._get_observation(), reward, done, False, {}

def main():
    print('🚀 Entrenando A2C Simple Compatible...')
    
    # Cargar datos
    data = pd.read_csv('data/training_data.csv')
    print(f"📊 Datos cargados: {len(data)} registros")
    
    # Crear entorno
    env = TradingEnvA2CSimple(data)
    
    # Crear modelo A2C simple
    model = A2C(
        'MlpPolicy',
        env,
        learning_rate=0.0003,
        n_steps=5,  # Muy pequeño para A2C
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1
    )
    
    print('📈 Entrenando por 25000 pasos...')
    model.learn(total_timesteps=25000)
    
    # Limpiar modelos anteriores
    import shutil
    if os.path.exists('data/models/a2c'):
        shutil.rmtree('data/models/a2c')
    if os.path.exists('data/models/best_a2c'):
        shutil.rmtree('data/models/best_a2c')
    
    # Guardar modelo nuevo
    os.makedirs('data/models/a2c', exist_ok=True)
    model.save('data/models/a2c/model.zip')
    print('✅ Modelo guardado en data/models/a2c/model.zip')
    
    os.makedirs('data/models/best_a2c', exist_ok=True)
    model.save('data/models/best_a2c/model.zip')
    print('✅ Mejor modelo guardado en data/models/best_a2c/model.zip')
    
    # Probar la carga del modelo
    try:
        loaded_model = A2C.load('data/models/a2c/model.zip')
        obs = env.reset()[0]
        action = loaded_model.predict(obs)
        print(f'✅ Modelo cargado y probado exitosamente: {action}')
    except Exception as e:
        print(f'❌ Error probando modelo: {e}')
    
    print('🎉 Entrenamiento A2C Simple completado!')

if __name__ == "__main__":
    main() 