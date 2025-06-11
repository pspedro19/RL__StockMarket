#!/usr/bin/env python3
"""Script de diagnóstico del sistema"""

import sys
import os
import subprocess
from dotenv import load_dotenv

def check_system():
    """Diagnóstico completo del sistema"""
    print("🔍 DIAGNÓSTICO DEL SISTEMA DE TRADING")
    print("=" * 50)
    
    # 1. Python
    print(f"\n✓ Python: {sys.version}")
    
    # 2. Paquetes críticos
    packages = ['MetaTrader5', 'pandas', 'numpy', 'stable_baselines3']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg}: Instalado")
        except ImportError:
            print(f"✗ {pkg}: NO instalado")
    
    # 3. Variables de entorno
    load_dotenv()
    env_vars = ['MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER']
    for var in env_vars:
        if os.getenv(var):
            print(f"✓ {var}: Configurado")
        else:
            print(f"✗ {var}: NO configurado")
    
    # 4. Estructura de archivos
    critical_files = [
        '.env',
        'requirements.txt',
        'src/collectors/mt5_connector.py',
        'src/agents/train.py'
    ]
    
    print("\nArchivos críticos:")
    for file in critical_files:
        if os.path.exists(file):
            print(f"✓ {file}: Existe")
        else:
            print(f"✗ {file}: NO existe")
    
    print("\n" + "=" * 50)
    print("Diagnóstico completado")

if __name__ == "__main__":
    check_system()
