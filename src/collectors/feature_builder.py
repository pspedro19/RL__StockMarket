"""
Constructor de Features para Trading
Crea características técnicas para modelos ML
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('feature_builder')

class FeatureBuilder:
    """Constructor de features técnicas para ML"""
    
    def __init__(self):
        self.features = []
    
    def add_technical_indicators(self, df):
        """Agregar indicadores técnicos como features"""
        try:
            # RSI
            df['rsi'] = self.calculate_rsi(df['close'])
            
            # MACD
            df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Moving Averages
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_21'] = df['close'].ewm(span=21).mean()
            
            # Volume indicators
            df['volume_sma'] = df['tick_volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
            
            # Price momentum
            df['momentum'] = df['close'].pct_change(periods=10)
            
            logger.info("✅ Features técnicas agregadas")
            return df
            
        except Exception as e:
            logger.error(f"Error agregando features: {e}")
            return df
    
    def calculate_rsi(self, prices, period=14):
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100  # Normalizado 0-1
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcular MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        
        # Normalizar
        macd_norm = macd / prices
        signal_norm = macd_signal / prices
        
        return macd_norm, signal_norm
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calcular Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower
    
    def prepare_ml_features(self, df, lookback=10):
        """Preparar features para modelos ML"""
        try:
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'bb_position',
                'volume_ratio', 'momentum'
            ]
            
            # Agregar features de precio normalizado
            df['price_norm'] = df['close'] / df['close'].rolling(window=50).mean()
            feature_columns.append('price_norm')
            
            # Crear matriz de features
            features = []
            for i in range(lookback, len(df)):
                row_features = []
                for col in feature_columns:
                    if col in df.columns:
                        row_features.append(df[col].iloc[i])
                    else:
                        row_features.append(0)
                features.append(row_features)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error preparando features ML: {e}")
            return np.array([])
    
    def get_feature_names(self):
        """Obtener nombres de features"""
        return [
            'rsi', 'macd', 'macd_signal', 'bb_position',
            'volume_ratio', 'momentum', 'price_norm'
        ] 