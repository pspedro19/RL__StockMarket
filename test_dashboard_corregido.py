#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test del dashboard CORREGIDO - Garantiza que funcione
"""

import sys
import os
import time

# Agregar path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

def main():
    print("🎨 PROBANDO DASHBOARD CORREGIDO")
    print("=" * 60)
    print("✅ Backend configurado automáticamente")
    print("✅ Estructura simplificada (2x3 paneles)")
    print("✅ Configuración robusta para Windows")
    print("✅ Actualización automática cada 2 segundos")
    print("=" * 60)
    
    try:
        from src.agents.real_time_trading_system import RealTimeTradingSystem
        
        # Crear sistema con análisis técnico (más estable)
        print("\n📊 Creando sistema con análisis técnico...")
        system = RealTimeTradingSystem('technical')
        
        if not system.mt5_connected:
            print("❌ Error: MT5 no conectado")
            print("   1. Abre MetaTrader 5")
            print("   2. Inicia sesión en tu cuenta")
            print("   3. Vuelve a ejecutar este script")
            return
        
        print("✅ MT5 conectado exitosamente")
        print(f"📊 Símbolo: {system.symbol}")
        print(f"🤖 Modelo: {system.model_name}")
        
        # Crear dashboard - DEBE funcionar ahora
        print("\n🎨 Creando dashboard CORREGIDO...")
        dashboard_created = system.create_live_dashboard()
        
        if dashboard_created:
            print("✅ ¡DASHBOARD CREADO EXITOSAMENTE!")
            print("📺 Buscando la ventana del dashboard...")
            print("   Título: '🚀 TRADING SYSTEM - TIEMPO REAL MT5'")
            
            # Dar tiempo para que aparezca la ventana
            time.sleep(2)
            
            # Iniciar sistema de trading
            print("\n🚀 Iniciando sistema de trading...")
            system.start_real_time()
            
            print("\n" + "=" * 60)
            print("🎯 SISTEMA FUNCIONANDO CON DASHBOARD GRÁFICO")
            print("=" * 60)
            print("📊 El dashboard se actualiza automáticamente cada 2 segundos")
            print("🔺 Triángulos VERDES = Compras")
            print("🔻 Triángulos ROJOS = Ventas")
            print("📈 Línea VERDE = Precio en tiempo real")
            print("⚡ Datos vienen directamente de MT5")
            print("💾 Trades se guardan automáticamente en CSV")
            print("\n📋 PANELES DISPONIBLES:")
            print("  • Principal: Precio con señales de compra/venta")
            print("  • Señales: Análisis técnico en tiempo real")
            print("  • Estado: Información del sistema y capital")
            print("  • RSI: Índice de fuerza relativa")
            print("  • Volumen: Volumen de trading")
            print("  • Trades: Estadísticas de operaciones")
            print("=" * 60)
            print("\n🎮 COMANDOS:")
            print("  'stop'  - Detener sistema")
            print("  'start' - Reiniciar sistema")
            print("  'quit'  - Salir")
            
            # Loop de control
            while True:
                try:
                    command = input("\n>>> ").strip().lower()
                    
                    if command == 'stop':
                        if system.is_real_time:
                            print("🛑 Deteniendo sistema...")
                            system.stop_real_time()
                        else:
                            print("⚠️ Sistema ya está detenido")
                    
                    elif command == 'start':
                        if not system.is_real_time:
                            print("🚀 Reiniciando sistema...")
                            system.start_real_time()
                        else:
                            print("⚠️ Sistema ya está funcionando")
                    
                    elif command in ['quit', 'exit']:
                        print("👋 Saliendo...")
                        break
                    
                    else:
                        print("❌ Comando no reconocido. Usa: stop, start, quit")
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n🛑 Deteniendo sistema...")
                    break
        
        else:
            print("❌ No se pudo crear el dashboard")
            print("🔧 Verifica la instalación de matplotlib")
            
            # Ofrecer alternativa sin GUI
            usar_consola = input("\n¿Usar sistema sin GUI? (s/n): ").strip().lower()
            if usar_consola == 's':
                print("\n🚀 Iniciando sistema en modo CONSOLA...")
                system.start_real_time()
                
                print("📊 Sistema funcionando en consola")
                print("🔄 Señales y trades aparecerán aquí")
                
                try:
                    # Mantener funcionando
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n🛑 Deteniendo...")
        
        # Limpiar al salir
        try:
            system.stop_real_time()
            print("✅ Sistema detenido correctamente")
        except:
            pass
            
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("🔧 Verifica que todas las dependencias estén instaladas")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 