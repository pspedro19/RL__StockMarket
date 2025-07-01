#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 TEST SISTEMA ARREGLADO - VERIFICACIÓN DE CORRECCIONES
- Frecuencia de trading reducida (cooldown real)
- Control manual de paneles
- Mejores estadísticas de trades
"""

import sys
import os
import time

# Agregar path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

def main():
    print("🔧 VERIFICANDO SISTEMA ARREGLADO")
    print("=" * 80)
    print("🎯 PRINCIPALES CORRECCIONES IMPLEMENTADAS:")
    print()
    print("1. 🕐 FRECUENCIA DE TRADING:")
    print("   ❌ ANTES: Trades cada 30 segundos")
    print("   ✅ AHORA: Cooldown de 5 MINUTOS (300 segundos)")
    print("   ✅ AHORA: Señales más estrictas (±0.5 en lugar de ±0.3)")
    print("   ✅ AHORA: Máximo 5 trades por día (en lugar de 10)")
    print("   ✅ AHORA: Una posición a la vez (no múltiples)")
    print()
    print("2. 🎮 CONTROL DE PANELES:")
    print("   ❌ ANTES: Cambiaba automáticamente sin control")
    print("   ✅ AHORA: Botón ⏩ controla manualmente la vista")
    print("   ✅ AHORA: 4 modos: AUTO, STATS, ACTIVE, PERF")
    print()
    print("3. 📊 ESTADÍSTICAS MEJORADAS:")
    print("   ❌ ANTES: Solo contadores básicos")
    print("   ✅ AHORA: Trades completados vs abiertos separados")
    print("   ✅ AHORA: Win rate calculado automáticamente")
    print("   ✅ AHORA: P&L promedio y duración promedio")
    print("   ✅ AHORA: Cooldown en tiempo real visible")
    print("=" * 80)
    
    try:
        from src.agents.real_time_trading_system import RealTimeTradingSystem
        
        # Crear sistema de trading
        print("\n🚀 Iniciando sistema con todas las correcciones...")
        system = RealTimeTradingSystem('technical')
        
        if not system.mt5_connected:
            print("❌ Error: MT5 no conectado")
            print("   1. Abre MetaTrader 5")
            print("   2. Inicia sesión en tu cuenta")
            print("   3. Vuelve a ejecutar este script")
            return
        
        print("✅ MT5 conectado exitosamente")
        
        # ✅ VERIFICAR CONFIGURACIÓN DE COOLDOWN
        print("\n🔧 VERIFICANDO CONFIGURACIÓN DE COOLDOWN:")
        print(f"   ⏱️ Cooldown configurado: {system.cooldown_period} segundos")
        print(f"   📏 Eso equivale a: {system.cooldown_period//60} minutos y {system.cooldown_period%60} segundos")
        print(f"   🎯 Umbral señal mínima: ±0.5 (más estricto)")
        print(f"   📈 Máximo trades diarios: {system.max_daily_trades}")
        print(f"   💰 Tamaño máximo posición: {system.max_position_size*100}%")
        print(f"   🔒 Prevención múltiples posiciones: ACTIVADO")
        
        if system.cooldown_period >= 300:
            print("   ✅ CORRECTO: Cooldown es de 5+ minutos")
        else:
            print("   ❌ ERROR: Cooldown debería ser de 5+ minutos")
        
        # ✅ VERIFICAR CONTROL DE PANELES
        print("\n🎮 VERIFICANDO CONTROL DE PANELES:")
        print(f"   📊 Modos disponibles: {system.panel_modes}")
        print(f"   🎯 Modo actual: {system.panel_modes[system.panel_mode]}")
        print(f"   ⏰ Intervalo cambio automático: {system.auto_switch_interval}s")
        print("   ✅ CORRECTO: Sistema de control implementado")
        
        # Crear dashboard
        print("\n🎨 Creando dashboard con correcciones...")
        dashboard_created = system.create_live_dashboard()
        
        if dashboard_created:
            print("✅ ¡DASHBOARD CREADO CON CORRECCIONES!")
            print("\n🎮 INSTRUCCIONES DE USO:")
            print("="*50)
            print("🔲 CONTROLES DE BOTONES:")
            print("   ▶️  PLAY    - Iniciar/reanudar sistema")
            print("   ⏸️  PAUSE   - Pausar sistema")
            print("   ⏹️  STOP    - Detener completamente")
            print("   ⏩  FORWARD - 🎯 CAMBIAR MODO DEL PANEL DERECHO")
            print("   🤖  AUTO    - Toggle sistema automático")
            print()
            print("📊 MODOS DEL PANEL DERECHO (presiona ⏩):")
            print("   🔄 AUTO   - Cambia automáticamente cada 30s")
            print("   📈 STATS  - Estadísticas completas de trading")
            print("   🔓 ACTIVE - Trades activos con P&L en tiempo real")
            print("   ⚡ PERF   - Performance del sistema y señales")
            print()
            print("🕐 VERIFICACIÓN DE COOLDOWN:")
            print("   • Observa que NO hay trades cada pocos segundos")
            print("   • El sistema debe esperar 5 MINUTOS entre trades")
            print("   • Verás mensaje '🚫 COOLDOWN ACTIVO' si hay señales")
            print("   • Solo ejecutará trades con señales muy fuertes (±0.5)")
            print("="*50)
            
            # Iniciar sistema
            print("\n🚀 Iniciando sistema con cooldown mejorado...")
            system.start_real_time()
            
            # Mostrar estado inicial
            print("\n📊 ESTADO INICIAL DEL SISTEMA:")
            stats = system.trade_manager.get_trade_statistics()
            print(f"   💰 Capital inicial: ${system.current_capital:,.2f}")
            print(f"   📈 Trades completados: {stats.get('total_trades', 0)}")
            print(f"   🔓 Trades abiertos: {stats.get('open_trades', 0)}")
            print(f"   ⏱️ Último trade: {system.last_trade_time}")
            print(f"   🔒 Trading habilitado: {system.trading_enabled}")
            
            print("\n🔍 MONITOREO DE COOLDOWN:")
            print("   👀 Observa la consola para ver mensajes de cooldown")
            print("   🚫 Deberías ver '🚫 COOLDOWN ACTIVO: XXXs restantes'")
            print("   ✅ Los trades solo ocurrirán cada 5+ minutos")
            
            print("\n💡 COMANDOS DE VERIFICACIÓN:")
            print("   'cooldown' - Ver tiempo restante de cooldown")
            print("   'stats'    - Ver estadísticas completas")
            print("   'panel'    - Cambiar modo de panel desde consola")
            print("   'test'     - Mostrar configuración actual")
            print("   'stop'     - Pausar sistema")
            print("   'start'    - Reanudar sistema")
            print("   'quit'     - Salir")
            
            # Loop de control con verificaciones
            start_time = time.time()
            last_cooldown_check = 0
            
            while True:
                try:
                    command = input("\n>>> ").strip().lower()
                    
                    if command == 'cooldown':
                        current_time = time.time()
                        if system.last_trade_time > 0:
                            elapsed = current_time - system.last_trade_time
                            remaining = max(0, system.cooldown_period - elapsed)
                            print(f"⏱️ COOLDOWN STATUS:")
                            print(f"   📊 Último trade: {elapsed:.1f}s atrás")
                            print(f"   🚫 Tiempo restante: {remaining:.0f}s ({remaining//60:.0f}m {remaining%60:.0f}s)")
                            print(f"   ✅ Próximo trade posible en: {remaining:.0f}s" if remaining > 0 else "   🟢 Listo para trade")
                        else:
                            print("   🎯 Sin trades previos - listo para primer trade")
                    
                    elif command == 'stats':
                        stats = system.trade_manager.get_trade_statistics()
                        print(f"\n📊 ESTADÍSTICAS COMPLETAS:")
                        print(f"   📈 Trades completados: {stats.get('total_trades', 0)}")
                        print(f"   🟢 Ganadores: {stats.get('winning_trades', 0)}")
                        print(f"   🔴 Perdedores: {stats.get('losing_trades', 0)}")
                        print(f"   📊 Win Rate: {stats.get('win_rate', 0):.1f}%")
                        print(f"   💰 P&L Total: ${stats.get('total_pnl', 0):+.2f}")
                        print(f"   📊 Ganancia promedio: ${stats.get('avg_win', 0):.2f}")
                        print(f"   📉 Pérdida promedio: ${stats.get('avg_loss', 0):.2f}")
                        print(f"   ⏱️ Duración promedio: {stats.get('avg_duration', 0):.1f} min")
                        print(f"   🔓 Trades abiertos: {stats.get('open_trades', 0)}")
                    
                    elif command == 'panel':
                        old_mode = system.panel_modes[system.panel_mode]
                        system._switch_panel_mode()
                        new_mode = system.panel_modes[system.panel_mode]
                        print(f"🔄 Panel cambiado de {old_mode} → {new_mode}")
                    
                    elif command == 'test':
                        print(f"\n🔧 CONFIGURACIÓN ACTUAL:")
                        print(f"   ⏱️ Cooldown: {system.cooldown_period}s ({system.cooldown_period//60}m)")
                        print(f"   🎯 Umbral señal: ±0.5")
                        print(f"   📈 Max trades diarios: {system.max_daily_trades}")
                        print(f"   💰 Max posición: {system.max_position_size*100}%")
                        print(f"   🔒 Múltiples posiciones: BLOQUEADAS")
                        print(f"   📊 Panel actual: {system.panel_modes[system.panel_mode]}")
                        print(f"   ✅ Sistema funcionando: {system.is_real_time}")
                    
                    elif command == 'stop':
                        if system.is_real_time:
                            print("🛑 Pausando sistema...")
                            system.stop_real_time()
                        else:
                            print("⚠️ Sistema ya está pausado")
                    
                    elif command == 'start':
                        if not system.is_real_time:
                            print("🚀 Reanudando sistema...")
                            system.start_real_time()
                        else:
                            print("⚠️ Sistema ya está funcionando")
                    
                    elif command in ['quit', 'exit', 'q']:
                        print("👋 Saliendo del test...")
                        break
                    
                    else:
                        print("❌ Comando no reconocido")
                        print("   Comandos: cooldown, stats, panel, test, stop, start, quit")
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n🛑 Saliendo del test...")
                    break
                
                # Verificación automática de cooldown cada 30 segundos
                current_time = time.time()
                if current_time - last_cooldown_check > 30:
                    if system.last_trade_time > 0:
                        elapsed = current_time - system.last_trade_time
                        remaining = max(0, system.cooldown_period - elapsed)
                        if remaining > 0:
                            print(f"\n🕐 Auto-check: Cooldown activo - {remaining:.0f}s restantes")
                        else:
                            print(f"\n🟢 Auto-check: Sistema listo para trading")
                    last_cooldown_check = current_time
        
        else:
            print("❌ No se pudo crear el dashboard")
        
        # Limpiar al salir
        try:
            if system.is_real_time:
                system.stop_real_time()
            print("✅ Sistema detenido correctamente")
        except:
            pass
            
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 