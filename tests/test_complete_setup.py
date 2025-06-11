#!/usr/bin/env python3
"""
Script para probar todo el setup del sistema
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

def test_imports():
    """Probar imports críticos"""
    print("🔍 Probando imports...")
    
    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader5: OK")
    except ImportError:
        print("❌ MetaTrader5: NO INSTALADO")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas/Numpy: OK")
    except ImportError:
        print("❌ Pandas/Numpy: ERROR")
        return False
    
    try:
        from stable_baselines3 import SAC
        print("✅ Stable-Baselines3: OK")
    except ImportError:
        print("❌ Stable-Baselines3: ERROR")
        return False
    
    try:
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"✅ PyTorch: OK ({device})")
    except ImportError:
        print("❌ PyTorch: ERROR")
        return False
    
    return True

def test_environment():
    """Probar variables de entorno"""
    print("\n🔧 Probando configuración...")
    
    load_dotenv()
    
    # Variables críticas
    critical_vars = [
        'MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER',
        'DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD'
    ]
    
    all_ok = True
    for var in critical_vars:
        value = os.getenv(var)
        if value:
            # No mostrar passwords completos
            display_value = value if 'PASSWORD' not in var else f"{value[:3]}***"
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: NO CONFIGURADO")
            all_ok = False
    
    return all_ok

def test_mt5_connection():
    """Probar conexión real a MT5"""
    print("\n📡 Probando conexión MT5...")
    
    try:
        import MetaTrader5 as mt5
        
        # Intentar inicializar
        if not mt5.initialize():
            print(f"❌ Error inicializando MT5: {mt5.last_error()}")
            return False
        
        # Intentar login
        login = int(os.getenv('MT5_LOGIN'))
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')
        
        authorized = mt5.login(login, password=password, server=server)
        
        if not authorized:
            print(f"❌ Error de login MT5: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        # Obtener info de cuenta
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ No se pudo obtener info de cuenta")
            mt5.shutdown()
            return False
        
        print(f"✅ Conectado a MT5!")
        print(f"   Cuenta: {account_info.login}")
        print(f"   Servidor: {account_info.server}")
        print(f"   Balance: ${account_info.balance:,.2f}")
        print(f"   Moneda: {account_info.currency}")
        
        # Probar obtener símbolos
        symbols = mt5.symbols_get()
        if symbols:
            available_symbols = [s.name for s in symbols if s.visible]
            desired_symbols = os.getenv('SYMBOLS', '').split(',')
            found_symbols = [s for s in desired_symbols if s in available_symbols]
            print(f"   Símbolos disponibles: {len(found_symbols)}/{len(desired_symbols)}")
            print(f"   Encontrados: {found_symbols[:5]}...")  # Mostrar solo los primeros 5
        
        # Probar obtener un tick
        test_symbol = 'EURUSD'
        tick = mt5.symbol_info_tick(test_symbol)
        if tick:
            print(f"   Tick {test_symbol}: Bid={tick.bid}, Ask={tick.ask}")
        
        mt5.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Error probando MT5: {e}")
        return False

async def test_database():
    """Probar conexión a base de datos"""
    print("\n🗄️ Probando base de datos...")
    
    try:
        import asyncpg
        
        # Configuración de BD
        config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }
        
        # Intentar conectar
        conn = await asyncpg.connect(**config)
        
        # Probar query simple
        result = await conn.fetchval('SELECT version()')
        print(f"✅ PostgreSQL conectado!")
        print(f"   Versión: {result.split(',')[0]}")
        
        # Verificar si TimescaleDB está instalado
        try:
            result = await conn.fetchval("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
            if result:
                print("✅ TimescaleDB: Instalado")
            else:
                print("⚠️ TimescaleDB: No detectado")
        except:
            print("⚠️ TimescaleDB: No se pudo verificar")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error conectando a BD: {e}")
        print("   💡 Asegúrate de que Docker esté corriendo: docker-compose up -d")
        return False

def test_docker_services():
    """Verificar servicios Docker"""
    print("\n🐳 Verificando servicios Docker...")
    
    import subprocess
    
    try:
        # Verificar docker-compose
        result = subprocess.run(['docker-compose', 'ps'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout
            services = ['trading_db', 'trading_kafka', 'trading_zookeeper', 'trading_redis']
            
            for service in services:
                if service in output and 'Up' in output:
                    print(f"✅ {service}: Corriendo")
                else:
                    print(f"❌ {service}: No está corriendo")
            
            return True
        else:
            print("❌ Error ejecutando docker-compose ps")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout verificando Docker")
        return False
    except FileNotFoundError:
        print("❌ docker-compose no encontrado")
        return False
    except Exception as e:
        print(f"❌ Error verificando Docker: {e}")
        return False

def main():
    """Ejecutar todas las pruebas"""
    print("🚀 DIAGNÓSTICO COMPLETO DEL SISTEMA")
    print("=" * 50)
    
    # Lista de pruebas
    tests = [
        ("Imports de Python", test_imports),
        ("Variables de entorno", test_environment),
        ("Servicios Docker", test_docker_services),
        ("Conexión MT5", test_mt5_connection),
    ]
    
    results = []
    
    # Ejecutar pruebas síncronas
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error en {name}: {e}")
            results.append((name, False))
    
    # Ejecutar prueba de BD (asíncrona)
    try:
        db_result = asyncio.run(test_database())
        results.append(("Base de datos", db_result))
    except Exception as e:
        print(f"❌ Error probando BD: {e}")
        results.append(("Base de datos", False))
    
    # Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ¡TODOS LOS TESTS PASARON!")
        print("\n�� PRÓXIMOS PASOS:")
        print("1. python scripts/download_history.py  # Descargar datos")
        print("2. python src/agents/train.py          # Entrenar modelo")  
        print("3. python scripts/run_backtest.py      # Hacer backtest")
        print("4. python src/trading/executor.py      # Trading en papel")
    else:
        print("⚠️ ALGUNOS TESTS FALLARON")
        print("\n🔧 SOLUCIONES:")
        print("1. Si Docker falla: docker-compose up -d")
        print("2. Si MT5 falla: verificar credenciales en .env")
        print("3. Si imports fallan: pip install -r requirements.txt")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
