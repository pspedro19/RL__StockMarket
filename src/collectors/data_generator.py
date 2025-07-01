"""
Generador de Datos de Mercado
Simula datos realistas cuando MT5 no está disponible
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('data_generator')

class DataGenerator:
    """Generador de datos de mercado simulados"""
    
    def __init__(self, symbol="US500", base_price=4500):
        self.symbol = symbol
        self.base_price = base_price
        
    def generate_realistic_data(self, n_points=1500, start_date=None):
        """Generar datos de mercado realistas"""
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=n_points//1440)  # Aprox días
            
            logger.info(f"🔄 Generando {n_points} puntos de datos para {self.symbol}")
            
            # Crear timestamps
            dates = [start_date + timedelta(minutes=i) for i in range(n_points)]
            
            # Generar precios con tendencia y volatilidad realista
            prices = self._generate_price_series(n_points)
            
            # Crear DataFrame similar a MT5
            data = {
                'time': dates,
                'open': prices,
                'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_points))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_points))),
                'close': prices,
                'tick_volume': np.random.randint(100, 1000, n_points),
                'spread': np.random.randint(1, 5, n_points),
                'real_volume': np.random.randint(1000, 10000, n_points)
            }
            
            df = pd.DataFrame(data)
            df.set_index('time', inplace=True)
            
            # Ajustar high/low para que sean coherentes
            df['high'] = np.maximum(df[['open', 'close']].max(axis=1), df['high'])
            df['low'] = np.minimum(df[['open', 'close']].min(axis=1), df['low'])
            
            logger.info(f"✅ Datos generados: {len(df)} puntos desde {df.index[0]} hasta {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"Error generando datos: {e}")
            return None
    
    def _generate_price_series(self, n_points):
        """Generar serie de precios con tendencias realistas"""
        # Parámetros para simulación realista
        dt = 1/252/1440  # 1 minuto en años
        mu = 0.10  # Retorno anual esperado
        sigma = 0.20  # Volatilidad anual
        
        # Proceso de Wiener (movimiento browniano)
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_points)
        
        # Agregar algunos eventos de mercado (gaps, trends)
        for i in range(0, n_points, 200):
            if np.random.random() < 0.3:  # 30% probabilidad de evento
                returns[i:i+20] += np.random.normal(0, 0.01, 20)  # Volatilidad aumentada
        
        # Agregar tendencias periódicas
        trend = 0.0005 * np.sin(np.arange(n_points) * 2 * np.pi / 100)
        returns += trend
        
        # Calcular precios
        prices = [self.base_price]
        for i in range(1, n_points):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, self.base_price * 0.5))  # Evitar precios negativos
        
        return np.array(prices)
    
    def add_market_events(self, df, n_events=5):
        """Agregar eventos de mercado aleatorios"""
        try:
            indices = np.random.choice(len(df), n_events, replace=False)
            
            for idx in indices:
                # Tipo de evento aleatorio
                event_type = np.random.choice(['gap_up', 'gap_down', 'high_vol'])
                
                if event_type == 'gap_up':
                    multiplier = 1 + np.random.uniform(0.02, 0.05)
                    df.iloc[idx:idx+10] *= multiplier
                elif event_type == 'gap_down':
                    multiplier = 1 - np.random.uniform(0.02, 0.05)
                    df.iloc[idx:idx+10] *= multiplier
                elif event_type == 'high_vol':
                    volatility = np.random.normal(0, 0.02, 20)
                    for i, vol in enumerate(volatility):
                        if idx + i < len(df):
                            df.iloc[idx + i] *= (1 + vol)
            
            logger.info(f"✅ {n_events} eventos de mercado agregados")
            return df
            
        except Exception as e:
            logger.error(f"Error agregando eventos: {e}")
            return df
    
    def get_current_price(self):
        """Obtener precio actual simulado"""
        # Simular precio actual con pequeña variación
        variation = np.random.normal(0, 0.001)
        return self.base_price * (1 + variation) 