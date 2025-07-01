"""
Script corregido para entrenar A2C compatible con el sistema
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import torch

class TradingEnvA2C(gym.Env):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.current_step = 50
        self.total_steps = len(data)
        
        self.observation_space = gym.spaces.Box(
            low=np.array([-4.8000002, -3.4028235e+38, -0.41887903, -3.4028235e+38], dtype=np.float32),
            high=np.array([4.8000002, 3.4028235e+38, 0.41887903, 3.4028235e+38], dtype=np.float32),
            dtype=np.float32
        )
        
        # A2C necesita acciones continuas
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
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
        
        # Convertir acción continua a discreta
        action_value = float(action[0])
        if action_value > 0.3:
            discrete_action = 1  # Compra
        elif action_value < -0.3:
            discrete_action = 0  # Venta
        else:
            discrete_action = -1  # Hold
            
        if discrete_action == 1:
            if self.position <= 0:
                self.position = 1
                self.entry_price = current_price
        elif discrete_action == 0:
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
    print('🚀 Entrenando modelo A2C corregido...')
    
    # Cargar datos
    data = pd.read_csv('data/training_data.csv')
    print(f"📊 Datos cargados: {len(data)} registros")
    
    # Crear entorno
    env = DummyVecEnv([lambda: TradingEnvA2C(data)])
    
    # Crear modelo A2C con configuración simplificada
    model = A2C(
        'MlpPolicy',
        env,
        learning_rate=0.0001,
        n_steps=1024,  # Reducido para mayor estabilidad
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),  # Red más simple
            activation_fn=torch.nn.ReLU
        ),
        verbose=1
    )
    
    print('📈 Entrenando por 50000 pasos...')
    model.learn(total_timesteps=50000)  # Menos pasos para prueba
    
    # Eliminar modelos anteriores problemáticos
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
        test_env = DummyVecEnv([lambda: TradingEnvA2C(data)])
        loaded_model = A2C.load('data/models/a2c/model.zip', env=test_env)
        obs = test_env.reset()
        action = loaded_model.predict(obs)
        print(f'✅ Modelo cargado y probado exitosamente: {action}')
    except Exception as e:
        print(f'❌ Error probando modelo: {e}')
    
    print('🎉 Entrenamiento A2C completado!')

if __name__ == "__main__":
    main() 