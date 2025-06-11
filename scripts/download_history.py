#!/usr/bin/env python3
"""
Script mejorado para descargar datos históricos de MT5
"""
import os
import sys
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from dotenv import load_dotenv

def download_symbol_data(symbol, days=30, timeframe=mt5.TIMEFRAME_M1):
    """Descargar datos de un símbolo específico"""
    
    # Verificar que el símbolo esté disponible
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Símbolo {symbol} no encontrado")
        return None
    
    # Asegurar que el símbolo esté seleccionado
    if not mt5.symbol_select(symbol, True):
        print(f"❌ No se pudo seleccionar {symbol}")
        return None
    
    # Calcular fechas
    date_to = datetime.now()
    date_from = date_to - timedelta(days=days)
    
    print(f"📊 Descargando {symbol} ({days} días)...")
    
    # Obtener datos
    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
    
    if rates is None or len(rates) == 0:
        print(f"❌ No se pudieron obtener datos de {symbol}")
        print(f"   Error: {mt5.last_error()}")
        return None
    
    # Convertir a DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['symbol'] = symbol
    
    # Reordenar columnas
    df = df[['time', 'symbol', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread']]
    
    print(f"✅ {symbol}: {len(df)} registros descargados")
    print(f"   Período: {df['time'].min()} a {df['time'].max()}")
    
    return df

def main():
    """Función principal"""
    print("🚀 DESCARGA DE DATOS HISTÓRICOS")
    print("=" * 50)
    
    # Cargar configuración
    load_dotenv()
    
    # Conectar a MT5
    print("📡 Conectando a MT5...")
    
    if not mt5.initialize():
        print(f"❌ Error inicializando MT5: {mt5.last_error()}")
        return
    
    # Login
    login = int(os.getenv('MT5_LOGIN'))
    password = os.getenv('MT5_PASSWORD')
    server = os.getenv('MT5_SERVER')
    
    authorized = mt5.login(login, password=password, server=server)
    
    if not authorized:
        print(f"❌ Error de login: {mt5.last_error()}")
        mt5.shutdown()
        return
    
    print("✅ Conectado a MT5")
    
    # Obtener lista de símbolos
    symbols_config = os.getenv('SYMBOLS', 'EURUSD,GBPUSD,USDCAD')
    symbols = [s.strip() for s in symbols_config.split(',') if s.strip()]
    
    print(f"📋 Símbolos a descargar: {symbols}")
    
    # Crear directorio de datos
    os.makedirs('data/raw', exist_ok=True)
    
    # Descargar cada símbolo
    all_data = []
    successful_downloads = 0
    
    for symbol in symbols:
        try:
            df = download_symbol_data(symbol, days=30)  # 30 días de datos
            
            if df is not None:
                # Guardar archivo individual
                filename = f'data/raw/{symbol}_30days.csv'
                df.to_csv(filename, index=False)
                print(f"💾 Guardado: {filename}")
                
                all_data.append(df)
                successful_downloads += 1
            
        except Exception as e:
            print(f"❌ Error descargando {symbol}: {e}")
    
    # Combinar todos los datos
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['symbol', 'time'])
        
        # Guardar archivo combinado
        combined_filename = 'data/raw/all_symbols_combined.csv'
        combined_df.to_csv(combined_filename, index=False)
        print(f"💾 Archivo combinado guardado: {combined_filename}")
        
        # Estadísticas finales
        print("\n📊 ESTADÍSTICAS DE DESCARGA:")
        print("=" * 30)
        print(f"Símbolos exitosos: {successful_downloads}/{len(symbols)}")
        print(f"Total registros: {len(combined_df):,}")
        print(f"Rango de fechas: {combined_df['time'].min()} a {combined_df['time'].max()}")
        
        # Mostrar datos por símbolo
        for symbol in combined_df['symbol'].unique():
            symbol_data = combined_df[combined_df['symbol'] == symbol]
            print(f"  {symbol}: {len(symbol_data):,} registros")
    
    # Desconectar
    mt5.shutdown()
    print("\n✅ Descarga completada!")

if __name__ == "__main__":
    main()
