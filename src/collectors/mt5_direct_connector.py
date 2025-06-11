import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
import numpy as np
import os
from dotenv import load_dotenv

class MT5DirectConnector:
    """Conector directo a MT5"""
    
    def __init__(self):
        load_dotenv()
        self.login = int(os.getenv('MT5_LOGIN', 0))
        self.password = os.getenv('MT5_PASSWORD', '')
        self.server = os.getenv('MT5_SERVER', '')
        self.connected = False
        
        if not all([self.login, self.password, self.server]):
            raise ValueError("Configura MT5_LOGIN, MT5_PASSWORD, MT5_SERVER en .env")
    
    def connect(self) -> bool:
        """Conectar a MT5"""
        try:
            print("🔌 Inicializando MT5...")
            
            if not mt5.initialize():
                error = mt5.last_error()
                print(f"❌ Error inicializando MT5: {error}")
                return False
            
            print(f"🔐 Conectando a cuenta {self.login}...")
            
            if not mt5.login(self.login, password=self.password, server=self.server):
                error = mt5.last_error()
                print(f"❌ Error conectando: {error}")
                mt5.shutdown()
                return False
            
            account_info = mt5.account_info()
            print("✅ ¡Conectado a MT5!")
            print(f"   Broker: {account_info.company}")
            print(f"   Balance: ${account_info.balance:.2f}")
            print(f"   Servidor: {account_info.server}")
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def get_live_price(self, symbol: str) -> dict:
        """Obtener precio actual"""
        if not self.connected:
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        return {
            'symbol': symbol,
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid,
            'time': datetime.fromtimestamp(tick.time)
        }
    
    def get_historical_data(self, symbol: str, timeframe, bars: int = 1000) -> pd.DataFrame:
        """Obtener datos históricos"""
        if not self.connected:
            return pd.DataFrame()
        
        try:
            print(f"📈 Obteniendo {bars} barras de {symbol}...")
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            
            if rates is None:
                print(f"❌ No se obtuvieron datos para {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df['symbol'] = symbol
            
            print(f"✅ Obtenidos {len(df)} registros")
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return pd.DataFrame()
    
    def disconnect(self):
        """Desconectar"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("🔌 Desconectado de MT5")

def test_mt5_connection():
    """Probar conexión MT5"""
    print("🚀 PRUEBA DE CONEXIÓN MT5")
    print("=" * 30)
    
    try:
        connector = MT5DirectConnector()
        
        if connector.connect():
            # Probar precios
            symbols = ['EURUSD', 'GBPUSD']
            for symbol in symbols:
                price = connector.get_live_price(symbol)
                if price:
                    print(f"💰 {symbol}: Bid={price['bid']:.5f}, Ask={price['ask']:.5f}")
            
            # Datos históricos
            data = connector.get_historical_data('EURUSD', mt5.TIMEFRAME_M1, 100)
            if not data.empty:
                print(f"📊 Datos EURUSD: {len(data)} registros")
                # Guardar datos reales
                data.to_csv('data/raw/EURUSD_MT5_real.csv', index=False)
                print("💾 Datos guardados: data/raw/EURUSD_MT5_real.csv")
            
            connector.disconnect()
        else:
            print("❌ No se pudo conectar")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_mt5_connection()
