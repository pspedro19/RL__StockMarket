"""
Script para preparar datos de entrenamiento para el modelo de RL
"""

import pandas as pd
import numpy as np
from datetime import datetime

def prepare_data(input_file='data/historical_data.csv', output_file='data/training_data.csv'):
    """Preparar datos para entrenamiento"""
    print("🔄 Cargando datos históricos...")
    
    # Cargar datos
    df = pd.read_csv(input_file)
    
    # Convertir tiempo a datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calcular features
    print("📊 Calculando features...")
    
    # 1. Normalizar precio usando ventana de 20 periodos
    df['norm_price'] = df['close'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std()
    )
    
    # 2. Normalizar volumen
    df['norm_volume'] = df['tick_volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std()
    )
    
    # 3. Calcular cambios porcentuales
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['tick_volume'].pct_change()
    
    # Eliminar filas con NaN
    df = df.dropna()
    
    # Guardar solo las columnas necesarias
    df_final = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 
                   'norm_price', 'norm_volume', 'price_change', 'volume_change']]
    
    # Guardar datos procesados
    df_final.to_csv(output_file, index=False)
    print(f"✅ Datos guardados en {output_file}")
    print(f"📈 Total registros: {len(df_final)}")
    
    # Mostrar estadísticas
    print("\n📊 Estadísticas de los features:")
    stats = df_final[['norm_price', 'norm_volume', 'price_change', 'volume_change']].describe()
    print(stats)
    
if __name__ == "__main__":
    prepare_data() 