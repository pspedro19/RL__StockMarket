#!/usr/bin/env python3
"""
🔄 ARCHIVO DE COMPATIBILIDAD HACIA ATRÁS
Este archivo mantiene la funcionalidad original pero ahora usa la nueva estructura modular.
Tu sistema ml_enhanced_trading seguirá funcionando exactamente igual.
"""

import sys
import os

# Agregar src al path para imports modulares
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Importar desde la nueva estructura
from src.agents.ml_enhanced_system import MLEnhancedTradingSystem

def main():
    """Función principal que mantiene la compatibilidad"""
    print("🔄 Ejecutando con nueva arquitectura modular...")
    
    # Crear e inicializar el sistema (funciona exactamente igual que antes)
    trading_system = MLEnhancedTradingSystem()
    
    # Cargar modelo ML
    if not trading_system.load_ml_model():
        trading_system.create_simple_ml_model()
    
    # Generar datos
    trading_system.generate_market_data(1500)
    
    # Intentar conectar MT5
    trading_system.connect_mt5()
    
    # Crear interfaz (idéntica a la original)
    trading_system.create_interface()

if __name__ == "__main__":
    main() 