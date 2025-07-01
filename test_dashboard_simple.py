#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simple del dashboard con TkAgg
"""

import sys
import os
import time

# Agregar path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

def main():
    print("🎨 PROBANDO DASHBOARD con TkAgg...")
    print("✅ Sin PyQt5 - usando TkAgg nativo")
    
    try:
        from src.agents.real_time_trading_system import RealTimeTradingSystem
        
        print("📊 Creando sistema...")
        system = RealTimeTradingSystem('technical')
        
        if not system.mt5_connected:
            print("❌ MT5 no conectado")
            return
        
        print("✅ MT5 conectado")
        print("🎨 Creando dashboard...")
        
        dashboard_ok = system.create_live_dashboard()
        
        if dashboard_ok:
            print("🎯 ¡ÉXITO! Dashboard creado")
            print("📺 ¿Puedes ver la ventana gráfica?")
            
            # Iniciar sistema
            system.start_real_time()
            
            # Mantener activo
            input("Presiona Enter para salir...")
            
        else:
            print("❌ Dashboard no se creó")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 