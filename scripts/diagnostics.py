"""
Diagnósticos del Sistema
Script para verificar la salud y configuración del sistema de trading
"""

import logging
import sys
import os

logger = logging.getLogger('diagnostics')

def check_mt5_connection():
    """Verificar conexión a MetaTrader5"""
    try:
        import MetaTrader5 as mt5
        logger.info("✅ MetaTrader5 disponible")
        return True
    except ImportError:
        logger.warning("⚠️ MetaTrader5 no instalado")
        return False

def check_ml_components():
    """Verificar componentes de Machine Learning"""
    try:
        import stable_baselines3
        logger.info("✅ Stable-baselines3 disponible")
        return True
    except ImportError:
        logger.warning("⚠️ Stable-baselines3 no instalado")
        return False

def check_dependencies():
    """Verificar dependencias principales"""
    dependencies = [
        'numpy', 'pandas', 'matplotlib', 'yaml'
    ]
    
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            logger.info(f"✅ {dep} disponible")
        except ImportError:
            logger.error(f"❌ {dep} faltante")
            missing.append(dep)
    
    return len(missing) == 0

def check_directory_structure():
    """Verificar estructura de directorios"""
    required_dirs = [
        'src', 'configs', 'data', 'tests', 'scripts',
        'data/models', 'data/results', 'data/raw', 'data/processed',
        'logs', 'monitoring'
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if os.path.exists(directory):
            logger.info(f"✅ Directorio {directory} existe")
        else:
            logger.warning(f"⚠️ Directorio {directory} faltante")
            missing_dirs.append(directory)
    
    return len(missing_dirs) == 0

def check_config_files():
    """Verificar archivos de configuración"""
    config_files = [
        'configs/config.yaml',
        'configs/trading_params.yaml', 
        'configs/logging.yaml'
    ]
    
    missing_configs = []
    
    for config in config_files:
        if os.path.exists(config):
            logger.info(f"✅ Configuración {config} existe")
        else:
            logger.warning(f"⚠️ Configuración {config} faltante")
            missing_configs.append(config)
    
    return len(missing_configs) == 0

def run_full_diagnostics():
    """Ejecutar diagnósticos completos"""
    logger.info("🔍 Iniciando diagnósticos del sistema...")
    
    results = {
        'mt5': check_mt5_connection(),
        'ml': check_ml_components(),
        'deps': check_dependencies(),
        'dirs': check_directory_structure(),
        'configs': check_config_files()
    }
    
    all_good = all(results.values())
    
    if all_good:
        logger.info("🎉 Todos los diagnósticos pasaron - Sistema listo")
    else:
        logger.warning("⚠️ Algunos diagnósticos fallaron - Revisar logs")
    
    return results

def main():
    """Función principal"""
    print("🔧 System Diagnostics - Diagnósticos del Sistema")
    print("Verificando configuración y dependencias")
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Ejecutar diagnósticos
    results = run_full_diagnostics()
    
    print("\n📊 Resumen:")
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}: {'OK' if status else 'FALLO'}")

if __name__ == "__main__":
    main() 