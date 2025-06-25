#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 INSTALADOR DE DEPENDENCIAS AVANZADAS
Script para instalar todas las dependencias necesarias para el sistema avanzado de trading
"""

import subprocess
import sys
import os

def install_package(package):
    """Instalar un paquete usando pip"""
    try:
        print(f"📦 Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} instalado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {package}: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INSTALADOR DE DEPENDENCIAS AVANZADAS")
    print("="*60)
    
    # Lista de dependencias adicionales
    advanced_packages = [
        "yfinance",           # Yahoo Finance para datos de SP500
        "ccxt",               # Intercambio de criptomonedas (Binance)
        "python-dotenv",      # Variables de entorno (.env)
        "seaborn",            # Visualizaciones avanzadas
        "plotly",             # Gráficos interactivos
        "dash",               # Dashboard web
        "sqlalchemy",         # Base de datos
        "pandas-ta",          # Indicadores técnicos adicionales
        "ta-lib",             # TA-Lib (requiere instalación especial)
        "scikit-learn",       # Machine learning adicional
        "scipy",              # Computación científica
        "numba",              # Aceleración de cálculos
        "requests",           # HTTP requests
        "websocket-client",   # WebSockets para tiempo real
        "asyncio",            # Programación asíncrona
    ]
    
    print(f"📋 Se instalarán {len(advanced_packages)} paquetes adicionales")
    print()
    
    # Instalar cada paquete
    successful = 0
    failed = 0
    
    for package in advanced_packages:
        if install_package(package):
            successful += 1
        else:
            failed += 1
        print()
    
    # Instalación especial para TA-Lib
    print("🔧 Instalación especial para TA-Lib...")
    print("💡 Si falla, instala manualmente:")
    print("   Windows: pip install TA-Lib")
    print("   Linux: sudo apt-get install libta-lib-dev && pip install TA-Lib")
    print("   macOS: brew install ta-lib && pip install TA-Lib")
    print()
    
    # Resumen
    print("="*60)
    print("📊 RESUMEN DE INSTALACIÓN:")
    print(f"✅ Exitosos: {successful}")
    print(f"❌ Fallidos: {failed}")
    print()
    
    if failed == 0:
        print("🎉 ¡Todas las dependencias se instalaron correctamente!")
        print("🚀 Ahora puedes ejecutar el sistema avanzado de trading")
    else:
        print("⚠️ Algunas dependencias fallaron")
        print("💡 Revisa los errores arriba e instala manualmente si es necesario")
    
    print()
    print("📚 PRÓXIMOS PASOS:")
    print("1. Copia configs/binance.env.example a .env")
    print("2. Completa tus API keys de Binance en .env")
    print("3. Ejecuta: python src/agents/advanced_trading_analytics.py")
    print()

if __name__ == "__main__":
    main() 