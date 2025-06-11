import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MT5Config:
    login: int
    password: str
    server: str
    timeout: int = 60000
    path: Optional[str] = None

class MT5Connector:
    """Conector principal para MetaTrader 5"""
    
    def __init__(self, config: MT5Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connected = False

    def connect(self) -> bool:
        """Conectar a MT5"""
        try:
            # Inicializar MT5
            if self.config.path:
                if not mt5.initialize(self.config.path):
                    self.logger.error(f"Error al inicializar MT5: {mt5.last_error()}")
                    return False
            else:
                if not mt5.initialize():
                    self.logger.error(f"Error al inicializar MT5: {mt5.last_error()}")
                    return False

            # Login
            authorized = mt5.login(
                self.config.login,
                password=self.config.password,
                server=self.config.server,
                timeout=self.config.timeout
            )
            
            if not authorized:
                self.logger.error(f"Error de login: {mt5.last_error()}")
                mt5.shutdown()
                return False

            # Obtener info de la cuenta
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("No se pudo obtener info de la cuenta")
                return False

            self.logger.info(f"Conectado exitosamente a MT5")
            self.logger.info(f"Cuenta: {account_info.login}")
            self.logger.info(f"Balance: ${account_info.balance:,.2f}")
            self.logger.info(f"Servidor: {account_info.server}")
            
            self.connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error al conectar: {e}")
            return False

    def disconnect(self):
        """Desconectar de MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Desconectado de MT5")

    def get_symbols(self) -> List[str]:
        """Obtener lista de símbolos disponibles"""
        symbols = mt5.symbols_get()
        if symbols is None:
            return []

        # Filtrar símbolos activos y visibles
        active_symbols = [s.name for s in symbols if s.visible]
        
        # Filtrar los que nos interesan
        desired = os.getenv('SYMBOLS', '').split(',')
        available = []
        for symbol in active_symbols:
            for desired_symbol in desired:
                if desired_symbol.strip() in symbol:
                    available.append(symbol)
                    break
        
        self.logger.info(f"Símbolos disponibles: {available}")
        return available

    def get_tick(self, symbol: str) -> Optional[Dict]:
        """Obtener tick actual"""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
            
        return {
            'time': datetime.fromtimestamp(tick.time, tz=timezone.utc),
            'symbol': symbol,
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'flags': tick.flags,
            'spread': tick.ask - tick.bid,
            'mid_price': (tick.bid + tick.ask) / 2
        }

# Función de utilidad para probar conexión
def test_connection():
    """Probar conexión a MT5"""
    config = MT5Config(
        login=int(os.getenv('MT5_LOGIN', 0)),
        password=os.getenv('MT5_PASSWORD', ''),
        server=os.getenv('MT5_SERVER', ''),
        path=os.getenv('MT5_PATH')
    )
    
    connector = MT5Connector(config)
    if connector.connect():
        print("✅ Conexión exitosa!")
        
        # Probar obtener símbolos
        symbols = connector.get_symbols()
        print(f"📊 Símbolos disponibles: {symbols}")
        
        # Probar obtener tick
        if symbols:
            tick = connector.get_tick(symbols[0])
            print(f"📈 Tick actual de {symbols[0]}: {tick}")
            
        connector.disconnect()
    else:
        print("❌ Error al conectar")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_connection()
