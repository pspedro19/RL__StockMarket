"""
🤖 ENTRENAMIENTO DE MODELOS DE TRADING
Entrena QDN, DeepQDN y A2C con datos históricos normalizados
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import MlpPolicy
import gymnasium as gym

class TradingEnv(gym.Env):
    """Entorno de trading para RL"""
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        self.current_step = 0
        self.total_steps = len(data)
        
        # Espacio de observación: 8 features normalizados
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        # Espacio de acción: 0 (Venta), 1 (Hold), 2 (Compra)
        self.action_space = gym.spaces.Discrete(3)
        
        # Estado del trading
        self.position = 0  # -1 (Short), 0 (Neutral), 1 (Long)
        self.entry_price = 0
        self.total_reward = 0
        
    def reset(self, seed=None, options=None):
        """Reiniciar entorno"""
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        return self._get_observation(), {}
        
    def _get_observation(self):
        """Obtener observación actual"""
        features = [
            self.data.iloc[self.current_step]['returns'],
            self.data.iloc[self.current_step]['volatility'],
            self.data.iloc[self.current_step]['momentum_1'],
            self.data.iloc[self.current_step]['momentum_5'],
            self.data.iloc[self.current_step]['momentum_10'],
            self.data.iloc[self.current_step]['trend'],
            self.data.iloc[self.current_step]['volume_ratio'],
            self.data.iloc[self.current_step]['RSI']
        ]
        return np.array(features, dtype=np.float32)
        
    def step(self, action):
        """Ejecutar acción"""
        done = False
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calcular reward
        reward = 0
        
        # Ejecutar acción
        if action == 2:  # Compra
            if self.position <= 0:
                self.position = 1
                self.entry_price = current_price
        elif action == 0:  # Venta
            if self.position >= 0:
                self.position = -1
                self.entry_price = current_price
                
        # Calcular P&L si hay posición
        if self.position != 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            reward = price_change * self.position
            
        # Penalizar cambios frecuentes
        if action != 1:  # Si no es Hold
            reward -= 0.0001  # Pequeña penalización por trading
            
        self.total_reward += reward
        
        # Avanzar
        self.current_step += 1
        
        # Verificar fin de episodio
        if self.current_step >= self.total_steps - 1:
            done = True
            
        return self._get_observation(), reward, done, False, {}

def train_model(model_name, model_class, env, policy_kwargs=None, learning_rate=0.001):
    """Entrenar un modelo específico"""
    print(f"\n🚀 Entrenando modelo {model_name}...")
    
    # Configurar modelo
    if policy_kwargs is None:
        model = model_class(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            verbose=1
        )
    else:
        model = model_class(
            MlpPolicy,
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            verbose=1
        )
    
    # Entrenar modelo
    total_timesteps = 100000  # Más pasos para mejor aprendizaje
    print(f"📈 Entrenando por {total_timesteps} pasos...")
    model.learn(total_timesteps=total_timesteps)
    
    # Guardar modelo
    os.makedirs(f"data/models/{model_name.lower()}", exist_ok=True)
    model_path = f"data/models/{model_name.lower()}/model.zip"
    model.save(model_path)
    print(f"✅ Modelo guardado en {model_path}")
    
    # Guardar mejor modelo
    os.makedirs(f"data/models/best_{model_name.lower()}", exist_ok=True)
    best_path = f"data/models/best_{model_name.lower()}/model.zip"
    model.save(best_path)
    print(f"✅ Mejor modelo guardado en {best_path}")
    
def main():
    """Función principal"""
    print("🎮 Iniciando entrenamiento de modelos...")
    
    # Cargar datos
    data = pd.read_csv('data/training_data.csv')
    print(f"📊 Datos cargados: {len(data)} registros")
    
    # Crear entorno
    env = DummyVecEnv([lambda: TradingEnv(data)])
    
    # 1. QDN - Red simple
    train_model(
        model_name="QDN",
        model_class=DQN,
        env=env,
        policy_kwargs=dict(net_arch=[64, 32]),
        learning_rate=0.001
    )
    
    # 2. DeepQDN - Red profunda
    train_model(
        model_name="DeepQDN",
        model_class=DQN,
        env=env,
        policy_kwargs=dict(net_arch=[256, 256, 128, 64]),
        learning_rate=0.0005
    )
    
    # 3. A2C - Configuración por defecto
    train_model(
        model_name="A2C",
        model_class=A2C,
        env=env,
        learning_rate=0.0003
    )
    
    print("\n🎉 Entrenamiento completado!")
    
if __name__ == "__main__":
    main()
