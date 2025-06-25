#!/usr/bin/env python3
"""
Script para debuggear las predicciones de los modelos
"""

import sys
import os
sys.path.append('.')

from src.agents.ml_enhanced_system import MLEnhancedTradingSystem, HAS_RL
from stable_baselines3 import DQN, PPO, A2C
import numpy as np

def debug_models():
    """Debuggear predicciones de modelos"""
    print("🔍 DEBUGGING MODELOS...")
    
    if not HAS_RL:
        print("❌ RL no disponible")
        return
    
    # Crear sistema de prueba
    system = MLEnhancedTradingSystem(skip_selection=True)
    system.generate_market_data(n_points=100)
    
    print(f"📊 Datos generados: {len(system.data)} registros")
    print(f"📈 Precio inicial: {system.data.iloc[50]['close']:.2f}")
    print(f"📈 Precio final: {system.data.iloc[-1]['close']:.2f}")
    
    # Preparar features para test
    test_step = 60
    features = system.prepare_ml_features(test_step)
    print(f"\n🧪 Features de prueba (step {test_step}):")
    print(f"  Precio: {system.data.iloc[test_step]['close']:.2f}")
    print(f"  RSI: {system.data.iloc[test_step]['rsi']:.1f}")
    print(f"  Features: {features}")
    
    # Test DQN
    print("\n🔥 TESTING DQN...")
    try:
        dqn_model = DQN.load("data/models/best_qdn/model.zip")
        dqn_pred = dqn_model.predict(features.reshape(1, -1), deterministic=True)
        print(f"  DQN Prediction: {dqn_pred}")
        print(f"  Action: {dqn_pred[0]} ({'COMPRA' if dqn_pred[0] == 1 else 'VENTA/HOLD'})")
    except Exception as e:
        print(f"  ❌ Error DQN: {e}")
    
    # Test DeepDQN
    print("\n🎯 TESTING DeepDQN...")
    try:
        deepdqn_model = DQN.load("data/models/best_deepqdn/model.zip")
        deepdqn_pred = deepdqn_model.predict(features.reshape(1, -1), deterministic=True)
        print(f"  DeepDQN Prediction: {deepdqn_pred}")
        print(f"  Action: {deepdqn_pred[0]} ({'COMPRA' if deepdqn_pred[0] == 1 else 'VENTA/HOLD'})")
    except Exception as e:
        print(f"  ❌ Error DeepDQN: {e}")
    
    # Test PPO
    print("\n⚖️ TESTING PPO...")
    try:
        ppo_model = PPO.load("data/models/best_ppo/best_model.zip")
        ppo_pred = ppo_model.predict(features.reshape(1, -1), deterministic=True)
        print(f"  PPO Prediction: {ppo_pred}")
        print(f"  Action: {ppo_pred[0]} ({'COMPRA' if ppo_pred[0] == 1 else 'VENTA/HOLD'})")
    except Exception as e:
        print(f"  ❌ Error PPO: {e}")
    
    # Test A2C
    print("\n🛡️ TESTING A2C...")
    try:
        a2c_model = A2C.load("data/models/best_a2c/model.zip")
        a2c_pred = a2c_model.predict(features.reshape(1, -1), deterministic=True)
        print(f"  A2C Prediction: {a2c_pred}")
        if hasattr(a2c_pred[0], '__len__'):
            action_val = a2c_pred[0][0]
            print(f"  Action Value: {action_val:.3f} ({'COMPRA' if action_val > 0.3 else 'VENTA' if action_val < -0.3 else 'HOLD'})")
        else:
            print(f"  Action: {a2c_pred[0]} ({'COMPRA' if a2c_pred[0] == 1 else 'VENTA/HOLD'})")
    except Exception as e:
        print(f"  ❌ Error A2C: {e}")
    
    # Test de situación de mercado
    print(f"\n📊 SITUACIÓN DEL MERCADO (step {test_step}):")
    print(f"  RSI: {system.data.iloc[test_step]['rsi']:.1f} ({'Sobrecompra' if system.data.iloc[test_step]['rsi'] > 70 else 'Sobreventa' if system.data.iloc[test_step]['rsi'] < 30 else 'Normal'})")
    print(f"  MACD: {system.data.iloc[test_step]['macd_normalized']:.3f}")
    print(f"  Volatilidad: {system.data.iloc[test_step]['volatility']:.4f}")
    
    # Análisis técnico
    tech_signal = system.generate_technical_signal(test_step)
    print(f"  Señal Técnica: {tech_signal:.3f} ({'COMPRA' if tech_signal > 0.3 else 'VENTA' if tech_signal < -0.3 else 'NEUTRAL'})")
    
    print("\n✅ Debug completado!")

if __name__ == "__main__":
    debug_models() 