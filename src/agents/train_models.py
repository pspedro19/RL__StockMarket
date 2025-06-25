"""
🚀 ENTRENAMIENTO DE MODELOS IA PARA TRADING
Entrena DQN, DeepDQN, PPO y A2C con datos históricos normalizados
"""

import sys
import os
sys.path.append('.')

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import MlpPolicy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar nuestro entorno personalizado
from src.agents.ml_enhanced_system import MLEnhancedTradingSystem

def load_training_data():
    """Cargar y preparar datos históricos"""
    print("📊 Cargando datos de entrenamiento...")
    
    try:
        # Intentar cargar datos reales
        data_path = "data/training_data.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"✅ Datos cargados: {len(df)} registros")
            return df
        else:
            print("⚠️ No se encontraron datos históricos")
            print("💡 Generando datos sintéticos para entrenamiento...")
            return None
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None

def create_training_environment(data=None):
    """Crear entorno de trading para entrenamiento"""
    print("🏗️ Creando entorno de entrenamiento...")
    
    # Crear ambiente base
    env = MLEnhancedTradingSystem(skip_selection=True)
    
    # Para DQN, DeepDQN, PPO: 0 (Venta), 1 (Compra)
    env.action_space = env.action_space  # Ya está configurado correctamente
    
    if data is not None:
        env.data = data
        env.calculate_indicators(env.data)
    else:
        # Generar datos sintéticos con más variabilidad
        env.generate_market_data(n_points=50000)  # Más datos para mejor entrenamiento
    
    env.initialize_tracking_arrays()
    env.current_step = 20  # Empezar después de los indicadores
    
    return env

def train_model(model_class, model_name, env, config):
    """Entrenar un modelo específico"""
    print(f"\n🚀 Entrenando {model_name}...")
    print(f"📈 Configuración: {config}")
    
    try:
        # Crear directorios
        model_dir = f"data/models/{model_name.lower().replace('deep', '')}"
        best_model_dir = f"data/models/best_{model_name.lower().replace('deep', '')}"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)
        
        # Crear modelo
        if model_class == DQN:
            # Configuración específica para DQN
            model = model_class(
                MlpPolicy,
                env,
                policy_kwargs=config['policy_kwargs'],
                learning_rate=config['learning_rate'],
                buffer_size=config.get('buffer_size', 100000),
                batch_size=config.get('batch_size', 32),
                gamma=config.get('gamma', 0.99),
                exploration_fraction=config.get('exploration_fraction', 0.1),
                exploration_final_eps=config.get('exploration_final_eps', 0.02),
                target_update_interval=config.get('target_update_interval', 1000),
                verbose=1
            )
        else:
            # Para PPO y A2C
            model = model_class(
                "MlpPolicy",
                env,
                learning_rate=config['learning_rate'],
                verbose=1
            )
        
        # Entrenar modelo
        print(f"⏱️ Iniciando entrenamiento ({config['timesteps']} steps)...")
        model.learn(total_timesteps=config['timesteps'])
        
        # Guardar modelo
        model_path = f"{model_dir}/model.zip"
        best_path = f"{best_model_dir}/model.zip"
        model.save(model_path)
        model.save(best_path)
        
        print(f"✅ {model_name} entrenado y guardado")
        print(f"📁 Modelo: {model_path}")
        print(f"📁 Mejor: {best_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error entrenando {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def evaluate_model(model, env, episodes=10):
    """Evaluar performance del modelo"""
    print(f"📊 Evaluando modelo ({episodes} episodios)...")
    
    total_rewards = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
        total_rewards.append(episode_reward)
        print(f"  Episodio {episode + 1}: Reward = {episode_reward:.2f}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"📈 Reward promedio: {avg_reward:.2f} ± {std_reward:.2f}")
    return avg_reward

def main():
    """Entrenar todos los modelos"""
    print("="*60)
    print("🤖 ENTRENAMIENTO DE MODELOS IA PARA TRADING")
    print("="*60)
    
    # Cargar datos
    data = load_training_data()
    
    # Crear entorno base
    base_env = create_training_environment(data)
    
    # Configuraciones de entrenamiento para cada modelo
    configs = [
        # 1. DQN - Red simple (acciones discretas)
        {
            "model_class": DQN,
            "model_name": "DQN",
            "config": {
                "policy_kwargs": dict(net_arch=[64, 32]),
                "learning_rate": 0.001,
                "buffer_size": 50000,
                "batch_size": 32,
                "timesteps": 200000
            }
        },
        # 2. DeepDQN - Red profunda (acciones discretas)
        {
            "model_class": DQN,
            "model_name": "DeepDQN",
            "config": {
                "policy_kwargs": dict(net_arch=[256, 256, 128, 64]),
                "learning_rate": 0.0005,
                "buffer_size": 100000,
                "batch_size": 64,
                "timesteps": 300000
            }
        },
        # 3. PPO - Política optimizada (acciones discretas)
        {
            "model_class": PPO,
            "model_name": "PPO",
            "config": {
                "learning_rate": 0.0003,
                "timesteps": 250000
            }
        },
        # 4. A2C - Actor-Critic (acciones continuas)
        {
            "model_class": A2C,
            "model_name": "A2C",
            "config": {
                "learning_rate": 0.0003,
                "timesteps": 200000
            }
        }
    ]
    
    # Entrenar cada modelo
    successful_models = 0
    
    for config in configs:
        print(f"\n{'='*40}")
        print(f"🎯 MODELO: {config['model_name']}")
        print(f"{'='*40}")
        
        # Crear entorno específico para este modelo
        env = create_training_environment(data)
        
        # Entrenar modelo
        success = train_model(
            config["model_class"],
            config["model_name"],
            env,
            config["config"]
        )
        
        if success:
            successful_models += 1
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN DE ENTRENAMIENTO")
    print(f"{'='*60}")
    print(f"✅ Modelos entrenados exitosamente: {successful_models}/{len(configs)}")
    print(f"📁 Modelos guardados en: data/models/")
    print(f"💡 Para probar: python src/agents/comparison_four_models.py")
    
    if successful_models > 0:
        print("\n🎉 ¡Entrenamiento completado!")
    else:
        print("\n❌ No se pudo entrenar ningún modelo")

if __name__ == "__main__":
    main()
