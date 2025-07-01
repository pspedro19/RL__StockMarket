#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test del sistema corregido con señales dinámicas
"""

import sys
import os
import time

# Agregar path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from src.agents.real_time_trading_system import RealTimeTradingSystem

def main():
    print("🚀 Iniciando sistema CORREGIDO con mejoras...")
    print("✅ Señales ocupan todo el gráfico")
    print("✅ Cooldown reducido a 30 segundos")
    print("✅ Lógica de trading mejorada")
    print("✅ Señales más dinámicas y reactivas")
    print("="*60)
    
    # Crear sistema
    system = RealTimeTradingSystem('technical')
    
    if not system.mt5_connected:
        print("❌ Error: MT5 no conectado")
        return
    
    print("✅ MT5 conectado exitosamente")
    print(f"📊 Símbolo: {system.symbol}")
    print(f"⏰ Cooldown: {system.trade_cooldown} segundos")
    
    # Crear dashboard
    print("🎨 Creando dashboard con señales mejoradas...")
    system.create_live_dashboard()
    
    # Iniciar sistema
    print("🚀 Iniciando trading en tiempo real...")
    system.start_real_time()
    
    print("\n" + "="*60)
    print("🎯 SISTEMA ACTIVO CON MEJORAS")
    print("="*60)
    print("📈 Las señales ahora ocupan todo el gráfico")
    print("🔄 El sistema cierra posiciones opuestas automáticamente")
    print("⚡ Señales más sensibles y reactivas")
    print("💰 Ejecutará compras y ventas sin quedarse atascado")
    print("\n🎮 Presiona Ctrl+C para detener o espera 2 minutos...")
    print("="*60)
    
    try:
        # Monitorear por 2 minutos
        for i in range(120):
            time.sleep(1)
            if i % 30 == 0:  # Cada 30 segundos mostrar estado
                print(f"\n⏰ {i//60}:{i%60:02d} - Estado:")
                print(f"  Posición: {system.current_position or 'NEUTRAL'}")
                print(f"  Trades abiertos: {len(system.trade_manager.open_trades)}")
                print(f"  Trades totales: {len(system.trade_manager.trades)}")
                print(f"  Señales BUY: {len(system.buy_signals)}")
                print(f"  Señales SELL: {len(system.sell_signals)}")
                
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo por solicitud del usuario...")
    
    # Cleanup
    system.stop_real_time()
    print("✅ Sistema detenido correctamente")
    
    # Mostrar resumen final
    print("\n" + "="*60)
    print("📊 RESUMEN FINAL")
    print("="*60)
    print(f"💰 Capital final: ${system.current_capital:,.2f}")
    print(f"📈 P&L total: ${system.total_profit_loss:+.2f} ({system.total_profit_loss_pct:+.2f}%)")
    print(f"🔢 Trades ejecutados: {len(system.trade_manager.trades)}")
    print(f"📁 CSV guardado en: {system.trade_manager.csv_filename}")

if __name__ == "__main__":
    main() 