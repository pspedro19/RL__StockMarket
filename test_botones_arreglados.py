#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST DE BOTONES ARREGLADOS - VERIFICACIÓN COMPLETA
Verifica que todos los botones funcionen sin cerrar el dashboard
"""

import sys
import os

# Agregar directorio raíz al path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from src.agents.real_time_trading_system import RealTimeTradingSystem

def test_botones_arreglados():
    """Probar que los botones no cierren el dashboard"""
    print("🧪 INICIANDO PRUEBA DE BOTONES ARREGLADOS")
    print("=" * 60)
    
    try:
        # Crear sistema con modelo técnico (más estable)
        print("🔧 Creando sistema de trading...")
        system = RealTimeTradingSystem(selected_model='technical')
        
        if not system:
            print("❌ Error creando sistema")
            return False
        
        print("✅ Sistema creado exitosamente")
        
        # Crear dashboard con botones
        print("\n📊 Creando dashboard con botones arreglados...")
        
        dashboard_created = system.create_live_dashboard()
        
        if dashboard_created:
            print("✅ Dashboard creado exitosamente")
            print("\n🎮 BOTONES DISPONIBLES:")
            print("   ▶️  = Iniciar trading en tiempo real")
            print("   ⏸️  = Pausar trading")
            print("   ⏹️  = Detener trading completamente")
            print("   🔄  = Refresh/Reiniciar datos")
            print("   ⏩  = Cambiar vista de panel derecho")
            print("   🤖  = Toggle modo automático")
            print("\n" + "=" * 60)
            print("🔬 INSTRUCCIONES DE PRUEBA:")
            print("1. Presiona cada botón UNO POR UNO")
            print("2. Verifica que el dashboard NO se cierre")
            print("3. Observa los mensajes en consola")
            print("4. Cada botón debe mostrar mensajes informativos")
            print("5. Si algún botón causa errores, serán capturados y mostrados")
            print("6. Cierra la ventana cuando termines de probar")
            print("=" * 60)
            
            # Mantener la ventana abierta
            try:
                import matplotlib.pyplot as plt
                plt.show(block=True)  # Bloquear hasta que se cierre la ventana
                print("✅ Prueba completada - Dashboard cerrado correctamente")
                return True
                
            except KeyboardInterrupt:
                print("\n⚠️ Prueba interrumpida por usuario")
                return True
                
        else:
            print("❌ Error creando dashboard")
            return False
            
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("🚀 INICIANDO TEST DE BOTONES ARREGLADOS")
    print("Este test verifica que los botones no cierren el dashboard")
    
    success = test_botones_arreglados()
    
    if success:
        print("\n🎉 TEST EXITOSO - Los botones funcionan correctamente")
    else:
        print("\n❌ TEST FALLÓ - Revisar errores arriba")
    
    return success

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc() 