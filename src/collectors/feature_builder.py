import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

class FeatureBuilder:
    """Constructor de features técnicas para trading"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def build_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Construir todas las features para un DataFrame"""
        df = data.copy()
        
        # Verificar que tenemos datos suficientes
        if len(df) < 50:
            self.logger.warning(f"Datos insuficientes para {symbol}: {len(df)} filas")
            return df
        
        try:
            # 1. Features básicas de precio
            df = self._add_price_features(df)
            
            # 2. Indicadores técnicos básicos
            df = self._add_basic_indicators(df)
            
            # 3. Features de volumen (si disponible)
            if 'volume' in df.columns:
                df = self._add_volume_features(df)
            
            # 4. Limpieza
            df = df.fillna(method='ffill').fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            
            self.logger.info(f"Features creadas para {symbol}: {len(df)} filas, {len(df.columns)} columnas")
            
        except Exception as e:
            self.logger.error(f"Error creando features para {symbol}: {e}")
            
        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir features básicas de precio"""
        # Retornos
        df['return_1'] = df['close'].pct_change()
        df['return_5'] = df['close'].pct_change(5)
        df['return_20'] = df['close'].pct_change(20)
        
        # Volatilidad
        df['volatility_20'] = df['return_1'].rolling(20).std()
        
        # Posición en el rango
        df['close_location'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Rangos
        df['high_low_ratio'] = df['high'] / df['low']
        df['hl_pct'] = (df['high'] - df['low']) / df['close']
        
        return df

    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir indicadores técnicos básicos"""
        try:
            # Moving averages
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # RSI simple
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands simple
            bb_middle = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = bb_middle + (bb_std * 2)
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_middle - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
            
        except Exception as e:
            self.logger.error(f"Error en indicadores técnicos: {e}")
            
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir features de volumen"""
        # Volume Moving Average
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
        
        return df
