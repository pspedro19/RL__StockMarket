#!/usr/bin/env python3
"""
ML ENHANCED TRADING SYSTEM v2.0
Arquitectura modular reorganizada
Combina modelo RL entrenado con análisis técnico tradicional
Integrado con MetaTrader5 para datos en tiempo real
"""

import sys
import os
import logging
import logging.config
import yaml
from pathlib import Path

# Agregar el directorio src al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Configurar logging desde archivo YAML"""
    try:
        with open('configs/logging.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Crear directorio de logs si no existe
        os.makedirs('logs', exist_ok=True)
        
        logging.config.dictConfig(config)
        logger = logging.getLogger('trading_system')
        logger.info("[OK] Logging configurado correctamente")
        return logger
    except Exception as e:
        # Fallback a configuración básica
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger('trading_system')
        logger.warning(f"[WARN] Error configurando logging desde YAML: {e}")
        logger.info("[INFO] Usando configuración básica de logging")
        return logger

def load_config():
    """Cargar configuración desde archivos YAML"""
    try:
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        with open('configs/trading_params.yaml', 'r') as f:
            trading_config = yaml.safe_load(f)
        
        # Combinar configuraciones
        config.update(trading_config)
        return config
    except Exception as e:
        logger.error(f"[ERROR] Error cargando configuración: {e}")
        return {}

def main():
    """Función principal"""
    print("[START] Iniciando ML Enhanced Trading System v2.0")
    print("[INFO] Arquitectura modular reorganizada")
    
    # Configurar logging
    logger = setup_logging()
    
    # Cargar configuración
    config = load_config()
    if not config:
        logger.error("[ERROR] No se pudo cargar la configuración")
        return
    
    logger.info(f"[OK] Configuración cargada: {config.get('system', {}).get('name', 'Sistema')}")
    
    try:
        # Importar el sistema principal
        from agents.ml_enhanced_system import MLEnhancedTradingSystem
        
        # Crear e inicializar el sistema
        logger.info("[INFO] Inicializando sistema de trading...")
        trading_system = MLEnhancedTradingSystem()
        
        # Cargar modelo ML si está disponible
        logger.info("[INFO] Cargando modelo de IA...")
        if not trading_system.load_ml_model():
            logger.info("[INFO] Creando modelo técnico avanzado...")
            trading_system.create_simple_ml_model()
        
        # Generar datos (históricos o simulados)
        logger.info("[INFO] Generando datos de mercado...")
        trading_system.generate_market_data(config.get('data', {}).get('history_size', 1500))
        
        # Intentar conectar a MT5 si está habilitado
        if config.get('mt5', {}).get('enabled', True):
            logger.info("[INFO] Intentando conectar a MetaTrader5...")
            if trading_system.connect_mt5():
                logger.info("[OK] MT5 conectado - datos en tiempo real disponibles")
            else:
                logger.info("[INFO] Usando datos simulados")
        
        # Crear interfaz gráfica
        logger.info("[INFO] Creando interfaz gráfica...")
        trading_system.create_interface()
        
        logger.info("[OK] Sistema iniciado correctamente")
        logger.info("[INFO] Usa los controles para navegar y operar")
        
    except ImportError as e:
        logger.error(f"[ERROR] Error importando módulos: {e}")
        logger.info("[INFO] Verifica que todos los archivos estén en su lugar")
        return
    except Exception as e:
        logger.error(f"[ERROR] Error inesperado: {e}")
        logger.info("[INFO] Revisa los logs para más detalles")
        return

if __name__ == "__main__":
    # Crear directorios necesarios si no existen
    for directory in ['logs', 'data/raw', 'data/processed', 'data/models', 'data/results']:
        os.makedirs(directory, exist_ok=True)
    
    # Ejecutar sistema
    main() 