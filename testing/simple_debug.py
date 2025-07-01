#!/usr/bin/env python3
"""
Script simple para diagnosticar señales de trading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.agents.ml_enhanced_system import MLEnhancedTradingSystem
    
    print("="*50)
    print("🔍 DIAGNÓSTICO SIMPLE DE SEÑALES")
    print("="*50)
    
    # Crear sistema
    system = MLEnhancedTradingSystem(skip_selection=True)
    
    # Configurar como DQN
    system.algorithm_choice = "1"
    system.algorithm_name = "DQN"
    
    print("✅ Sistema creado")
    
    # Cargar datos usando el método correcto
    system.generate_market_data(1000)  # Generar o cargar datos del mercado
    
    if system.data is not None and len(system.data) > 0:
        print(f"✅ Datos cargados: {len(system.data)} registros")
        
        # Ver datos recientes
        if hasattr(system.data, 'tail'):
            recent = system.data.tail(5)
            print("\n📊 Últimos 5 datos:")
            for i, (timestamp, row) in enumerate(recent.iterrows()):
                close = row.get('close', row.get('price', 'N/A'))
                rsi = row.get('rsi', 'N/A')
                print(f"  {i+1}. {timestamp}: Precio=${close}, RSI={rsi}")
        
        # Verificar condiciones del mercado actual
        last_step = len(system.data) - 1
        last_row = system.data.iloc[last_step]
        last_close = last_row.get('close', last_row.get('price', 0))
        last_rsi = last_row.get('rsi', 50)
        
        print(f"\n🎯 ANÁLISIS ACTUAL:")
        print(f"  💰 Precio actual: ${last_close:.2f}")
        print(f"  📊 RSI actual: {last_rsi:.1f}")
        
        if last_rsi < 30:
            print(f"  📉 RSI en SOBREVENTA → Debería COMPRAR")
        elif last_rsi > 70:
            print(f"  📈 RSI en SOBRECOMPRA → Debería VENDER")
        else:
            print(f"  ⚖️ RSI neutral → Sin señal clara")
        
        # Intentar cargar modelo
        model_loaded = system.load_ml_model()
        if model_loaded:
            print("\n✅ Modelo DQN cargado")
            
            # Probar predicción en último punto
            try:
                signal = system.generate_ml_signal(last_step)
                print(f"🎯 Señal ML: {signal:.3f}")
                
                # Probar señal técnica
                tech_signal = system.generate_technical_signal(last_step)
                print(f"📊 Señal Técnica: {tech_signal:.3f}")
                
                # Señal combinada
                combined_signal = system.generate_combined_signal(last_step)
                print(f"⚖️ Señal Combinada: {combined_signal:.3f}")
                
                if combined_signal > 0.3:
                    print("🟢 SEÑAL FUERTE DE COMPRA")
                elif combined_signal > 0.1:
                    print("🟡 SEÑAL DÉBIL DE COMPRA")
                elif combined_signal < -0.3:
                    print("🔴 SEÑAL FUERTE DE VENTA")
                elif combined_signal < -0.1:
                    print("🟠 SEÑAL DÉBIL DE VENTA")
                else:
                    print("⚪ HOLD - Sin señal clara")
                    
                # Verificar filtros de riesgo
                print(f"\n🛡️ VERIFICANDO FILTROS DE RIESGO:")
                risk_passed = system.check_risk_filters(last_step)
                if risk_passed:
                    print("✅ Todos los filtros de riesgo pasados")
                else:
                    print("❌ Algunos filtros de riesgo NO pasados")
                    
            except Exception as e:
                print(f"❌ Error generando señales: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ No se pudo cargar modelo DQN")
    else:
        print("❌ No se pudieron cargar datos")
        
except Exception as e:
    print(f"❌ Error general: {e}")
    import traceback
    traceback.print_exc() 