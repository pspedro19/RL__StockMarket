#!/usr/bin/env python3
"""
Script para diagnosticar por qué no se ejecutan trades
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.agents.ml_enhanced_system import MLEnhancedTradingSystem
    
    print("="*60)
    print("🔍 DIAGNÓSTICO DE TRADING")
    print("="*60)
    
    # Crear sistema DQN
    system = MLEnhancedTradingSystem(skip_selection=True)
    system.algorithm_choice = "1"
    system.algorithm_name = "DQN"
    
    print("✅ Sistema DQN creado")
    
    # Cargar datos
    system.generate_market_data(1000)
    
    if system.data is not None and len(system.data) > 0:
        print(f"✅ Datos cargados: {len(system.data)} registros")
        
        # Cargar modelo
        model_loaded = system.load_ml_model()
        if model_loaded:
            print("✅ Modelo DQN cargado")
            
            # Verificar configuración del sistema
            print(f"\n📊 CONFIGURACIÓN DEL SISTEMA:")
            print(f"  💰 Capital inicial: ${system.initial_capital:,.0f}")
            print(f"  💰 Capital actual: ${system.current_capital:,.0f}")
            print(f"  📊 Posición actual: {system.position_size}")
            print(f"  🎯 Peso ML: {system.ml_weight:.1%}")
            print(f"  📈 Max trades/día: {system.max_daily_trades}")
            print(f"  🔄 Trades hoy: {system.daily_trades}")
            print(f"  ⏱️ Separación mínima: {system.min_trade_separation}")
            print(f"  🛑 Stop Loss: {system.stop_loss_pct:.1%}")
            print(f"  🎯 Take Profit: {system.take_profit_pct:.1%}")
            
            # Probar señales en varios puntos
            test_steps = [100, 200, 500, 800, len(system.data)-50, len(system.data)-20, len(system.data)-1]
            
            print(f"\n🎯 ANÁLISIS DE SEÑALES EN DIFERENTES PUNTOS:")
            for step in test_steps:
                if step >= len(system.data):
                    continue
                    
                try:
                    # Obtener datos del punto
                    price = system.data.iloc[step]['price']
                    rsi = system.data.iloc[step]['rsi']
                    
                    # Generar señales
                    ml_signal = system.generate_ml_signal(step)
                    tech_signal = system.generate_technical_signal(step)
                    combined_signal = system.generate_combined_signal(step)
                    
                    # Verificar si ejecutaría trade
                    would_buy = combined_signal > 0.3 and system.position_size == 0
                    would_sell = combined_signal < -0.3 and system.position_size > 0
                    
                    # Verificar filtros de riesgo
                    risk_passed, risk_reason = system.check_risk_filters(step)
                    
                    print(f"\n  📍 Paso {step}:")
                    print(f"    💰 Precio: ${price:.2f}")
                    print(f"    📊 RSI: {rsi:.1f}")
                    print(f"    🤖 ML: {ml_signal:.3f}")
                    print(f"    📈 Técnica: {tech_signal:.3f}")
                    print(f"    ⚖️ Combinada: {combined_signal:.3f}")
                    print(f"    🛡️ Riesgo: {'✅' if risk_passed else '❌'} {risk_reason}")
                    
                    if would_buy and risk_passed:
                        position_size = system.calculate_position_size(price)
                        print(f"    🟢 COMPRARÍA: Tamaño={position_size}")
                    elif would_sell and risk_passed:
                        print(f"    🔴 VENDERÍA")
                    elif would_buy and not risk_passed:
                        print(f"    🟡 COMPRARÍA pero filtro de riesgo bloquea")
                    elif would_sell and not risk_passed:
                        print(f"    🟡 VENDERÍA pero filtro de riesgo bloquea")
                    else:
                        print(f"    ⚪ HOLD - Señal insuficiente")
                        
                except Exception as e:
                    print(f"    ❌ Error: {e}")
            
            # Simular algunos pasos de trading
            print(f"\n🎮 SIMULANDO TRADING:")
            original_step = system.current_step
            system.current_step = 100  # Empezar desde paso 100
            
            for i in range(10):  # Simular 10 pasos
                current_step = system.current_step
                if current_step >= len(system.data) - 1:
                    break
                    
                try:
                    # Generar señal
                    combined_signal = system.generate_combined_signal(current_step)
                    
                    # Intentar ejecutar trade
                    trade_executed = system.execute_trade(current_step, combined_signal)
                    
                    price = system.data.iloc[current_step]['price']
                    rsi = system.data.iloc[current_step]['rsi']
                    
                    print(f"  Paso {current_step}: Precio=${price:.2f}, RSI={rsi:.1f}, Señal={combined_signal:.3f} → {'✅ TRADE' if trade_executed else '⚪ HOLD'}")
                    
                    # Avanzar
                    system.current_step += 5  # Avanzar 5 pasos
                    
                except Exception as e:
                    print(f"  ❌ Error en paso {current_step}: {e}")
                    break
            
            # Restaurar step original
            system.current_step = original_step
            
        else:
            print("❌ No se pudo cargar modelo")
    else:
        print("❌ No se pudieron cargar datos")
        
except Exception as e:
    print(f"❌ Error general: {e}")
    import traceback
    traceback.print_exc() 