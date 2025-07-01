#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de trading simplificado SIN GUI
Funciona solo en consola para evitar problemas de matplotlib
"""

import sys
import os
import time
from datetime import datetime

# Agregar path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

def main():
    print("🚀 SISTEMA DE TRADING SIMPLIFICADO (SIN GUI)")
    print("="*60)
    print("✅ Solo consola - sin problemas de matplotlib")
    print("✅ Trades y señales en tiempo real")
    print("✅ CSV se guarda automáticamente")
    print("="*60)
    
    try:
        from src.agents.real_time_trading_system import RealTimeTradingSystem
        
        # Crear sistema sin GUI
        print("📊 Creando sistema...")
        system = RealTimeTradingSystem('technical')
        
        if not system.mt5_connected:
            print("❌ Error: MT5 no conectado")
            print("   Asegúrate de que MT5 esté abierto")
            return
        
        print("✅ MT5 conectado exitosamente")
        print(f"📊 Símbolo: {system.symbol}")
        print(f"⏰ Cooldown: {system.trade_cooldown} segundos")
        
        # NO CREAR DASHBOARD - solo funcionar en consola
        print("🎨 Saltando creación de GUI...")
        
        # Iniciar sistema
        print("🚀 Iniciando sistema en tiempo real...")
        system.start_real_time()
        
        print("\n" + "="*60)
        print("🎯 SISTEMA ACTIVO - SOLO CONSOLA")
        print("="*60)
        print("📈 Observa la consola para ver:")
        print("   • Señales detectadas")
        print("   • Compras y ventas ejecutadas")
        print("   • Precios en tiempo real")
        print("   • Estado del sistema")
        print("\n⏰ Funcionará por 2 minutos...")
        print("🎮 Presiona Ctrl+C para detener")
        print("="*60)
        
        # Monitoreo por 2 minutos con reportes cada 30 segundos
        start_time = time.time()
        last_report = 0
        
        while time.time() - start_time < 120:  # 2 minutos
            current_time = time.time() - start_time
            
            # Reporte cada 30 segundos
            if current_time - last_report >= 30:
                print(f"\n📊 REPORTE - {int(current_time)}s")
                print(f"   Posición: {system.current_position or 'NEUTRAL'}")
                print(f"   Trades abiertos: {len(system.trade_manager.open_trades)}")
                print(f"   Trades totales: {len(system.trade_manager.trades)}")
                print(f"   Capital: ${system.current_capital:,.2f}")
                print(f"   P&L: ${system.total_profit_loss:+.2f}")
                print(f"   Señales BUY: {len(system.buy_signals)}")
                print(f"   Señales SELL: {len(system.sell_signals)}")
                
                # Mostrar trades abiertos
                if system.trade_manager.open_trades:
                    print("   Trades abiertos:")
                    for trade_id, trade in system.trade_manager.open_trades.items():
                        print(f"     {trade_id}: {trade['trade_type']} @ ${trade['entry_price']}")
                
                last_report = current_time
            
            time.sleep(1)
        
        print("\n⏰ Tiempo completado!")
        
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo por Ctrl+C...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if 'system' in locals():
                print("\n🛑 Deteniendo sistema...")
                system.stop_real_time()
                
                # Mostrar resumen final
                print("\n" + "="*60)
                print("📊 RESUMEN FINAL")
                print("="*60)
                print(f"💰 Capital inicial: ${system.initial_capital:,.2f}")
                print(f"💰 Capital final: ${system.current_capital:,.2f}")
                print(f"📈 P&L total: ${system.total_profit_loss:+.2f} ({system.total_profit_loss_pct:+.2f}%)")
                print(f"🔢 Trades ejecutados: {len(system.trade_manager.trades)}")
                print(f"📁 CSV guardado: {system.trade_manager.csv_filename}")
                
                if system.trade_manager.trades:
                    print("\n📋 TRADES EJECUTADOS:")
                    for trade in system.trade_manager.trades:
                        print(f"   {trade['trade_id']}: {trade['trade_type']} ")
                        print(f"      Entry: ${trade['entry_price']} -> Exit: ${trade['exit_price']}")
                        print(f"      Return: {trade['return_pct']}% ({trade['exit_reason']})")
            
            print("\n✅ Sistema detenido correctamente")
        except:
            pass

if __name__ == "__main__":
    main() 