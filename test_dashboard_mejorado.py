#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 TEST DASHBOARD MEJORADO - CON CORRECCIONES DE TRADING
"""

import sys
import os
import time

# Agregar path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

def main():
    print("🎯 TESTING DASHBOARD MEJORADO CON CORRECCIONES")
    print("=" * 70)
    print("✅ CORRECCIONES APLICADAS:")
    print("   🔧 Frecuencia de trading REDUCIDA (cooldown: 5 minutos)")
    print("   📊 Paneles de información MEJORADOS con estadísticas de trades")
    print("   🎮 Botón ⏩ ahora CONTROLA la vista del panel derecho")
    print("   🎯 Umbrales de señal MÁS ESTRICTOS (0.5 en lugar de 0.3)")
    print("   💰 Mejor tracking de trades completados y P&L")
    print("   🚫 Prevención de múltiples posiciones simultáneas")
    print("=" * 70)
    
    try:
        from src.agents.real_time_trading_system import RealTimeTradingSystem
        
        # Crear sistema con análisis técnico (más estable)
        print("\n📊 Creando sistema con todas las mejoras...")
        system = RealTimeTradingSystem('technical')
        
        if not system.mt5_connected:
            print("❌ Error: MT5 no conectado")
            print("   1. Abre MetaTrader 5")
            print("   2. Inicia sesión en tu cuenta")
            print("   3. Vuelve a ejecutar este script")
            return
        
        print("✅ MT5 conectado exitosamente")
        print(f"📊 Símbolo: {system.symbol}")
        print(f"🤖 Modelo: {system.selected_model_type}")
        
        # Mostrar configuración de trading mejorada
        print("\n🔧 CONFIGURACIÓN DE TRADING MEJORADA:")
        print(f"   ⏱️ Cooldown entre trades: {system.cooldown_period}s ({system.cooldown_period//60}min)")
        print(f"   📈 Máximo trades diarios: {system.max_daily_trades}")
        print(f"   💰 Tamaño máximo posición: {system.max_position_size*100}%")
        print(f"   🎯 Umbral mínimo señal: ±0.5 (más estricto)")
        print(f"   🚫 Control múltiples posiciones: ACTIVADO")
        
        # Crear dashboard mejorado
        print("\n🎨 Creando dashboard MEJORADO...")
        dashboard_created = system.create_live_dashboard()
        
        if dashboard_created:
            print("✅ ¡DASHBOARD MEJORADO CREADO EXITOSAMENTE!")
            print("\n🎮 CONTROLES MEJORADOS:")
            print("   ▶️  PLAY   - Iniciar sistema")
            print("   ⏸️  PAUSE  - Pausar sistema") 
            print("   ⏹️  STOP   - Detener sistema")
            print("   ⏪  BACK   - (reservado)")
            print("   ⏩  FORWARD - 🎯 CAMBIAR VISTA DEL PANEL DERECHO")
            print("   🤖  AUTO   - Toggle automático")
            
            print("\n📊 MODOS DEL PANEL DERECHO (botón ⏩):")
            print("   0️⃣ AUTO   - Cambio automático cada 30s")
            print("   1️⃣ STATS  - Estadísticas completas de trading")
            print("   2️⃣ ACTIVE - Trades activos y P&L en tiempo real")
            print("   3️⃣ PERF   - Performance del sistema y señales")
            
            # Dar tiempo para que aparezca la ventana
            time.sleep(2)
            
            # Iniciar sistema de trading
            print("\n🚀 Iniciando sistema de trading MEJORADO...")
            system.start_real_time()
            
            print("\n" + "=" * 70)
            print("🎯 SISTEMA FUNCIONANDO CON MEJORAS APLICADAS")
            print("=" * 70)
            print("📊 Dashboard actualizado con:")
            print("   • 📈 Gráfico principal con señales menos agresivas")
            print("   • 💰 Panel izquierdo con estado general (siempre visible)")
            print("   • 🎮 Panel derecho CONTROLABLE con botón ⏩")
            print("   • 🔄 Trades menos frecuentes (cooldown 5 minutos)")
            print("   • 📊 Estadísticas completas de trades completados")
            print("   • 🎯 Mejor control de posiciones múltiples")
            
            print("\n🎯 DIFERENCIAS CLAVE:")
            print("   ❌ ANTES: Trades cada 30 segundos (muy agresivo)")
            print("   ✅ AHORA: Trades cada 5 minutos (más conservador)")
            print("   ❌ ANTES: Señales débiles (±0.3)")
            print("   ✅ AHORA: Señales fuertes (±0.5)")
            print("   ❌ ANTES: Panel cambiaba automáticamente")
            print("   ✅ AHORA: Usuario controla con botón ⏩")
            print("   ❌ ANTES: Múltiples posiciones simultáneas")
            print("   ✅ AHORA: Una posición a la vez")
            
            print("\n🎮 CÓMO USAR EL CONTROL DE PANELES:")
            print("   1. Presiona el botón ⏩ en la interfaz gráfica")
            print("   2. El panel derecho cambiará entre 4 modos:")
            print("      • AUTO: Cambio automático")
            print("      • STATS: Estadísticas completas")
            print("      • ACTIVE: Trades activos") 
            print("      • PERF: Performance y señales")
            print("   3. El modo actual se muestra en la parte superior del panel")
            
            print("=" * 70)
            print("\n🎮 COMANDOS DE CONSOLA:")
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
                    
                    elif command == 'panel':
                        # Comando para cambiar panel manualmente desde consola
                        system._switch_panel_mode()
                    
                    elif command == 'stats':
                        # Mostrar estadísticas actuales
                        stats = system.trade_manager.get_trade_statistics()
                        print("\n📊 ESTADÍSTICAS ACTUALES:")
                        print(f"   📈 Trades completados: {stats.get('total_trades', 0)}")
                        print(f"   🟢 Ganadores: {stats.get('winning_trades', 0)}")
                        print(f"   🔴 Perdedores: {stats.get('losing_trades', 0)}")
                        print(f"   📊 Win Rate: {stats.get('win_rate', 0):.1f}%")
                        print(f"   💰 P&L Total: ${stats.get('total_pnl', 0):+.2f}")
                        print(f"   🔓 Trades abiertos: {stats.get('open_trades', 0)}")
                    
                    else:
                        print("❌ Comando no reconocido.")
                        print("   Comandos: stop, start, quit, panel, stats")
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n🛑 Deteniendo sistema...")
                    break
        
        else:
            print("❌ No se pudo crear el dashboard")
            print("🔧 Verifica la instalación de matplotlib")
        
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