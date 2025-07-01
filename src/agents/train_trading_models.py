"""
🤖 ENTRENAMIENTO DE MODELOS DE TRADING
Entrena QDN, DeepQDN y A2C con datos históricos de trading
"""

import os
import numpy as np
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import MlpPolicy

from ml_enhanced_system import MLEnhancedTradingSystem

def train_model(model_name, model_class, policy_kwargs=None, learning_rate=0.001):
    """Entrenar un modelo específico"""
    print(f"\n🚀 Entrenando modelo {model_name}...")
    
    # Crear sistema
    env = MLEnhancedTradingSystem(skip_selection=True)
    env.algorithm_name = model_name
    env.algorithm_class = model_class
    
    # Generar datos de entrenamiento
    env.generate_market_data(n_points=5000)  # Más datos para entrenamiento
    
    # Configurar modelo
    if policy_kwargs is None:
        model = model_class(
            "MlpPolicy",
            DummyVecEnv([lambda: env]),
            learning_rate=learning_rate,
            verbose=1
        )
    else:
        model = model_class(
            MlpPolicy,
            DummyVecEnv([lambda: env]),
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
    
    # 1. QDN - Red simple
    train_model(
        model_name="QDN",
        model_class=DQN,
        policy_kwargs=dict(net_arch=[64, 32]),
        learning_rate=0.001
    )
    
    # 2. DeepQDN - Red profunda
    train_model(
        model_name="DeepQDN",
        model_class=DQN,
        policy_kwargs=dict(net_arch=[256, 256, 128, 64]),
        learning_rate=0.0005
    )
    
    # 3. A2C - Configuración por defecto
    train_model(
        model_name="A2C",
        model_class=A2C,
        learning_rate=0.0003
    )
    
    print("\n🎉 Entrenamiento completado!")
    
if __name__ == "__main__":
    main() 