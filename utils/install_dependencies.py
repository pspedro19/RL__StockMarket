#!/usr/bin/env python3
"""
🚀 Script de Instalación Automática - Sistema de Trading IA
Instala todas las dependencias necesarias para el funcionamiento del sistema

Uso: python install_dependencies.py
"""

import subprocess
import sys
import os

def install_package(package):
    """Instalar un paquete usando pip"""
    try:
        print(f"📦 Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {package}: {e}")
        return False

def main():
    """Función principal de instalación"""
    print("=" * 60)
    print("🤖 INSTALADOR AUTOMÁTICO - SISTEMA DE TRADING IA")
    print("=" * 60)
    print()
    
    # Dependencias críticas en orden de prioridad
    critical_packages = [
        "numpy>=1.26.4",
        "pandas>=2.2.3", 
        "matplotlib>=3.10.1",
        "gymnasium==1.1.1",
        "stable-baselines3==2.6.0",
        "MetaTrader5==5.0.5120",
        "ta>=0.11.0"
    ]
    
    print("📋 Instalando dependencias críticas...")
    failed_packages = []
    
    for package in critical_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print("\n" + "=" * 60)
    
    if not failed_packages:
        print("🎉 ¡INSTALACIÓN COMPLETADA CON ÉXITO!")
        print("✅ Todas las dependencias críticas están instaladas")
        print("\n💡 Ahora puedes ejecutar el sistema:")
        print("   python src/agents/ml_enhanced_system.py")
        print("   o")
        print("   python main.py")
    else:
        print("⚠️ Instalación completada con algunos errores")
        print("❌ Paquetes que fallaron:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print("\n💡 Intenta instalar manualmente:")
        print("   pip install -r requirements.txt")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 