"""
Conector directo para MetaTrader5
Versión simplificada para conexión rápida
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger('mt5_direct_connector')

class MT5DirectConnector:
    """Conector directo y simple para MT5"""
    
    def __init__(self):
        self.connected = False
        
    def connect(self):
        """Conectar directamente a MT5"""
        try:
            if not mt5.initialize():
                logger.error("Error inicializando MT5")
                return False
            
            self.connected = True
            logger.info("✅ MT5 conectado directamente")
            return True
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    def get_data(self, symbol="US500", timeframe=mt5.TIMEFRAME_M1, count=1000):
        """Obtener datos directamente"""
        if not self.connected:
            return None
            
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo datos: {e}")
            return None
    
    def disconnect(self):
        """Desconectar"""
        if self.connected:
            mt5.shutdown()
            self.connected = False 