#!/usr/bin/env python3
"""
Script para entrenar A2C con acciones discretas
Compatible con el sistema de trading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ml_enhanced_system import MLEnhancedTradingSystem, HAS_RL
from stable_baselines3 import A2C
import numpy as np

def train_a2c_discrete():
    """Entrenar A2C con acciones discretas"""
    print("🚀 Entrenando A2C con acciones discretas...")
    
    if not HAS_RL:
        print("❌ Stable-baselines3 no disponible")
        return
    
    # Crear sistema con acciones discretas
    system = MLEnhancedTradingSystem(skip_selection=True)
    system.algorithm_choice = "4"
    system.algorithm_name = "A2C"
    
    # FORZAR acciones discretas para A2C
    import gymnasium as gym
    system.action_space = gym.spaces.Discrete(2)  # 0=Vender, 1=Comprar
    
    # Generar datos
    print("📊 Generando datos de entrenamiento...")
    system.generate_market_data(n_points=2000)
    
    # Configurar modelo A2C con acciones discretas
    print("🤖 Configurando modelo A2C...")
    model = A2C(
        "MlpPolicy",
        system,
        learning_rate=0.0003,
        n_steps=5,
        verbose=1,
        device='cpu'
    )
    
    # Entrenar
    print("📈 Entrenando por 20,000 pasos...")
    model.learn(total_timesteps=20000)
    
    # Guardar
    print("💾 Guardando modelo...")
    os.makedirs("data/models/a2c", exist_ok=True)
    os.makedirs("data/models/best_a2c", exist_ok=True)
    
    model.save("data/models/a2c/model.zip")
    model.save("data/models/best_a2c/model.zip")
    
    # Probar modelo
    print("🧪 Probando modelo...")
    obs = system.reset()[0]
    action = model.predict(obs, deterministic=True)
    print(f"✅ Modelo probado exitosamente: {action}")
    
    print("🎉 A2C con acciones discretas entrenado!")

if __name__ == "__main__":
    train_a2c_discrete() 