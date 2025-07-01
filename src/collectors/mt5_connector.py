"""
Conector especializado para MetaTrader5
Maneja la conexión, obtención de datos y validación de símbolos
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import logging

# Configurar logger
logger = logging.getLogger('mt5_connector')

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
    logger.info("✅ MetaTrader5 disponible")
except ImportError:
    HAS_MT5 = False
    logger.warning("⚠️ MetaTrader5 no disponible")

class MT5Connector:
    """Conector especializado para MetaTrader5"""
    
    def __init__(self, symbol="US500", timeframe=None):
        self.symbol = symbol
        self.timeframe = timeframe if timeframe else (mt5.TIMEFRAME_M1 if HAS_MT5 else None)
        self.connected = False
        self.last_update = datetime.now()
        
        # Símbolos alternativos para diferentes brokers
        self.symbol_alternatives = [
            "US500", "SP500", "SPX500", "US500m", "USTEC", "SPX"
        ]
        
    def connect(self):
        """Conectar a MetaTrader5 con diagnóstico completo"""
        if not HAS_MT5:
            logger.error("❌ MetaTrader5 no está instalado")
            logger.info("💡 Solución: Instala MT5 desde https://www.metaquotes.net/es/metatrader5")
            return False
            
        try:
            logger.info("🔄 Intentando conectar a MetaTrader5...")
            
            # Intentar inicializar
            if not mt5.initialize():
                error_code = mt5.last_error()
                logger.error(f"❌ Error inicializando MT5: {error_code}")
                logger.info("💡 Soluciones:")
                logger.info("   1. Abre MetaTrader5 manualmente")
                logger.info("   2. Configura una cuenta demo")
                logger.info("   3. Asegúrate de que MT5 esté ejecutándose")
                return False
            
            logger.info("✅ MT5 inicializado correctamente")
            
            # Verificar conexión al servidor
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("❌ No hay cuenta configurada en MT5")
                logger.info("💡 Solución: Configura una cuenta demo en MT5")
                return False
            
            logger.info(f"✅ Cuenta conectada: {account_info.login} - {account_info.server}")
            
            # Verificar y configurar símbolo
            if not self._setup_symbol():
                return False
            
            self.connected = True
            logger.info(f"🎉 Conexión MT5 exitosa - Símbolo: {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error conectando MT5: {e}")
            logger.info("💡 Soluciones:")
            logger.info("   1. Reinstala MetaTrader5")
            logger.info("   2. Ejecuta como administrador")
            logger.info("   3. Verifica que no esté bloqueado por antivirus")
            return False
    
    def _setup_symbol(self):
        """Configurar y verificar el símbolo de trading"""
        logger.info(f"🔍 Buscando símbolo {self.symbol}...")
        symbol_info = mt5.symbol_info(self.symbol)
        
        if symbol_info is None:
            logger.warning(f"❌ Símbolo {self.symbol} no encontrado")
            logger.info("💡 Símbolos alternativos para SP500:")
            
            # Buscar símbolos alternativos
            for alt in self.symbol_alternatives:
                if mt5.symbol_info(alt) is not None:
                    logger.info(f"   ✅ Encontrado: {alt}")
                    self.symbol = alt
                    symbol_info = mt5.symbol_info(alt)
                    break
                else:
                    logger.info(f"   ❌ No disponible: {alt}")
            
            if symbol_info is None:
                logger.error("❌ No se encontró ningún símbolo SP500")
                logger.info("💡 Contacta a tu broker para obtener el símbolo correcto")
                return False
        
        # Activar símbolo si no está visible
        if not symbol_info.visible:
            logger.info(f"🔄 Activando símbolo {self.symbol}...")
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"❌ Error activando símbolo {self.symbol}")
                return False
        
        logger.info(f"✅ Símbolo {self.symbol} listo - Spread: {symbol_info.spread}")
        return True
    
    def get_historical_data(self, count=1000):
        """Obtener datos históricos"""
        if not self.connected:
            logger.error("❌ No conectado a MT5")
            return None
            
        try:
            # Obtener datos históricos
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.error("❌ No se pudieron obtener datos históricos")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            logger.info(f"✅ Obtenidos {len(df)} datos históricos desde {df.index[0]} hasta {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos históricos: {e}")
            return None
    
    def get_real_time_data(self, count=1000):
        """Obtener datos en tiempo real"""
        if not self.connected:
            return None
            
        try:
            # Obtener hora actual
            now = datetime.now(timezone.utc)
            
            # Obtener datos más recientes
            rates = mt5.copy_rates_from(self.symbol, self.timeframe, now, count)
            
            if rates is None or len(rates) == 0:
                logger.warning("⚠️ No se pudieron obtener datos en tiempo real")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Actualizar timestamp
            self.last_update = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos en tiempo real: {e}")
            return None
    
    def get_symbol_info(self):
        """Obtener información del símbolo"""
        if not self.connected:
            return None
            
        try:
            return mt5.symbol_info(self.symbol)
        except Exception as e:
            logger.error(f"❌ Error obteniendo info del símbolo: {e}")
            return None
    
    def disconnect(self):
        """Desconectar de MT5"""
        if HAS_MT5 and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("🔌 Desconectado de MT5")
    
    def __del__(self):
        """Cleanup al destruir el objeto"""
        self.disconnect() 