import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class SyntheticDataGenerator:
    """Generador de datos sintéticos para pruebas"""
    
    def __init__(self):
        self.symbols = ['EURUSD', 'GBPUSD', 'US500']
        self.base_prices = {
            'EURUSD': 1.0800,
            'GBPUSD': 1.2500, 
            'US500': 4500.0
        }
    
    def generate_ohlcv_data(self, symbol: str, days: int = 30, interval_minutes: int = 1) -> pd.DataFrame:
        """Generar datos OHLCV sintéticos"""
        
        base_price = self.base_prices.get(symbol, 100.0)
        volatility = 0.001  # 0.1% por minuto
        
        # Número total de barras
        total_bars = days * 24 * 60 // interval_minutes
        
        # Generar timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start_time, end_time, periods=total_bars)
        
        # Generar precios usando random walk
        prices = [base_price]
        for i in range(1, total_bars):
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Crear datos OHLCV
        data = []
        for i, timestamp in enumerate(timestamps):
            # Para cada barra, generar OHLC basado en el precio base
            base = prices[i]
            high_factor = 1 + abs(np.random.normal(0, volatility/2))
            low_factor = 1 - abs(np.random.normal(0, volatility/2))
            
            open_price = base if i == 0 else data[-1]['close']
            close_price = base
            high_price = max(open_price, close_price) * high_factor
            low_price = min(open_price, close_price) * low_factor
            volume = np.random.randint(100, 1000)
            
            data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def save_data(self, symbol: str, days: int = 30):
        """Generar y guardar datos"""
        df = self.generate_ohlcv_data(symbol, days)
        
        # Crear directorio si no existe
        os.makedirs('data/raw', exist_ok=True)
        
        # Guardar como CSV
        filename = f'data/raw/{symbol}_{days}days.csv'
        df.to_csv(filename, index=False)
        print(f"✅ Datos guardados: {filename} ({len(df)} registros)")
        
        return filename

def test_data_generation():
    """Probar generación de datos"""
    generator = SyntheticDataGenerator()
    
    for symbol in ['EURUSD', 'GBPUSD', 'US500']:
        filename = generator.save_data(symbol, days=7)
        
        # Verificar datos
        df = pd.read_csv(filename)
        print(f"📊 {symbol}: {len(df)} registros, precio promedio: {df['close'].mean():.4f}")

if __name__ == "__main__":
    test_data_generation()
