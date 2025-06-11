import os
import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import logging
from datetime import datetime
from src.agents.environment import TradingEnvironment

class RLTrainer:
    """Entrenador básico de modelos RL"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """Cargar datos para entrenamiento"""
        # Buscar archivos en data/processed/
        processed_files = []
        if os.path.exists('data/processed'):
            processed_files = [f for f in os.listdir('data/processed') if f.endswith('.csv')]
        
        if not processed_files:
            # Buscar en data/raw/
            if os.path.exists('data/raw'):
                raw_files = [f for f in os.listdir('data/raw') if f.endswith('.csv')]
                if raw_files:
                    file_path = f'data/raw/{raw_files[0]}'
                    self.logger.info(f"Cargando datos de {file_path}")
                    return pd.read_csv(file_path)
            
            raise FileNotFoundError("No se encontraron datos en data/processed/ ni data/raw/")
        
        # Usar el primer archivo con features
        file_path = f'data/processed/{processed_files[0]}'
        self.logger.info(f"Cargando datos de {file_path}")
        return pd.read_csv(file_path)

    def train_model(self, data: pd.DataFrame, model_name: str = "sac_basic"):
        """Entrenar modelo SAC básico"""
        self.logger.info("Iniciando entrenamiento...")
        
        # Crear ambiente
        env = DummyVecEnv([lambda: Monitor(TradingEnvironment(data))])
        
        # Crear modelo SAC
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            verbose=1,
            device='auto'
        )
        
        # Entrenar
        try:
            model.learn(total_timesteps=10000, progress_bar=True)
            
            # Guardar modelo
            os.makedirs('data/models', exist_ok=True)
            model_path = f'data/models/{model_name}'
            model.save(model_path)
            
            self.logger.info(f"Modelo guardado en {model_path}")
            return model
            
        except KeyboardInterrupt:
            self.logger.info("Entrenamiento interrumpido")
            return model

def main():
    """Función principal"""
    logging.basicConfig(level=logging.INFO)
    
    trainer = RLTrainer()
    
    try:
        # Cargar datos
        data = trainer.load_data()
        print(f"✅ Datos cargados: {len(data)} filas, {len(data.columns)} columnas")
        
        # Entrenar modelo
        model = trainer.train_model(data)
        
        print("✅ Entrenamiento completado!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
