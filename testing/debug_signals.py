#!/usr/bin/env python3
"""
Script de diagnóstico para verificar señales de modelos RL
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.ml_enhanced_system import MLEnhancedTradingSystem
import numpy as np

def test_model_signals():
    """Probar señales de cada modelo"""
    print("="*60)
    print("🔍 DIAGNÓSTICO DE SEÑALES DE MODELOS RL")
    print("="*60)
    
    # Configuraciones de cada modelo
    configs = {
        "DQN": {"algorithm_choice": "1", "risk_percentage": 12},
        "DeepDQN": {"algorithm_choice": "2", "risk_percentage": 8},
        "PPO": {"algorithm_choice": "3", "risk_percentage": 10},
        "A2C": {"algorithm_choice": "4", "risk_percentage": 6}
    }
    
    for model_name, config in configs.items():
        print(f"\n🤖 Probando modelo {model_name}...")
        
        try:
            # Crear sistema
            system = MLEnhancedTradingSystem(skip_selection=True)
            
            # Configurar parámetros manualmente
            system.algorithm_choice = config["algorithm_choice"]
            system.algorithm_name = list({"1": "DQN", "2": "DeepDQN", "3": "PPO", "4": "A2C"}.values())[int(config["algorithm_choice"])-1]
            
            # Cargar datos
            success = system.load_real_time_data()
            if not success:
                print(f"❌ {model_name}: No se pudieron cargar datos")
                continue
                
            print(f"✅ {model_name}: Datos cargados - {len(system.data)} registros")
            
            # Cargar modelo
            model_loaded = system.load_ml_model()
            if not model_loaded:
                print(f"❌ {model_name}: No se pudo cargar modelo ML")
                continue
                
            print(f"✅ {model_name}: Modelo ML cargado")
            
            # Probar predicciones en diferentes puntos
            test_points = [100, 500, 1000, 1500, -50, -20, -10]  # Incluyendo puntos recientes
            
            print(f"\n📊 Predicciones de {model_name}:")
            for point in test_points:
                try:
                    if point < 0:
                        # Puntos desde el final
                        step = len(system.data) + point
                    else:
                        step = point
                        
                    if step < 50 or step >= len(system.data):
                        continue
                        
                    # Preparar features
                    features = system.prepare_ml_features(step)
                    features = features.reshape(1, -1)
                    
                    # Hacer predicción
                    prediction = system.ml_model.predict(features, deterministic=True)
                    if isinstance(prediction, tuple):
                        action = prediction[0]
                    else:
                        action = prediction
                        
                    if hasattr(action, '__len__'):
                        action = action[0] if len(action) > 0 else 0
                    
                    # Convertir a señal
                    if action == 0:  # Lógica invertida
                        signal = 1.0  # COMPRA
                        signal_text = "🟢 COMPRA"
                    else:  # action == 1
                        signal = -1.0  # VENTA
                        signal_text = "🔴 VENTA"
                    
                    # Obtener datos técnicos del punto
                    if 'rsi' in system.data.columns:
                        rsi = system.data.iloc[step]['rsi']
                        close = system.data.iloc[step]['close']
                        timestamp = system.data.index[step]
                        
                        print(f"  Paso {step} ({timestamp}): Precio={close:.2f}, RSI={rsi:.1f} → Action={action} → {signal_text}")
                    else:
                        print(f"  Paso {step}: Action={action} → {signal_text}")
                        
                except Exception as e:
                    print(f"  ❌ Error en paso {step}: {e}")
            
            # Probar la función generate_ml_signal directamente
            print(f"\n🎯 Probando generate_ml_signal de {model_name}:")
            recent_steps = list(range(max(0, len(system.data)-10), len(system.data)))
            
            for step in recent_steps[-5:]:  # Solo últimos 5
                try:
                    signal = system.generate_ml_signal(step)
                    if 'rsi' in system.data.columns:
                        rsi = system.data.iloc[step]['rsi']
                        close = system.data.iloc[step]['close']
                        timestamp = system.data.index[step]
                        
                        if signal > 0.3:
                            signal_text = "🟢 COMPRA"
                        elif signal < -0.3:
                            signal_text = "🔴 VENTA"
                        else:
                            signal_text = "⚪ HOLD"
                            
                        print(f"  Paso {step} ({timestamp}): Precio={close:.2f}, RSI={rsi:.1f} → Signal={signal:.2f} → {signal_text}")
                    else:
                        print(f"  Paso {step}: Signal={signal:.2f}")
                        
                except Exception as e:
                    print(f"  ❌ Error en generate_ml_signal paso {step}: {e}")
                    
        except Exception as e:
            print(f"❌ Error general con {model_name}: {e}")
            import traceback
            traceback.print_exc()

def check_current_market_conditions():
    """Verificar condiciones actuales del mercado"""
    print(f"\n📈 CONDICIONES ACTUALES DEL MERCADO")
    print("="*40)
    
    try:
        system = MLEnhancedTradingSystem(skip_selection=True)
        success = system.load_real_time_data()
        
        if not success:
            print("❌ No se pudieron cargar datos del mercado")
            return
            
        # Datos más recientes
        latest_data = system.data.tail(10)
        print(f"📅 Últimos datos ({len(latest_data)} registros):")
        
        for i, (timestamp, row) in enumerate(latest_data.iterrows()):
            close = row['close']
            rsi = row.get('rsi', 'N/A')
            print(f"  {i+1}. {timestamp}: Precio=${close:.2f}, RSI={rsi}")
            
        # Análisis del último punto
        last_row = system.data.iloc[-1]
        last_close = last_row['close']
        last_rsi = last_row.get('rsi', 0)
        
        print(f"\n🎯 ANÁLISIS ACTUAL:")
        print(f"  💰 Precio actual: ${last_close:.2f}")
        print(f"  📊 RSI actual: {last_rsi:.1f}")
        
        if last_rsi < 30:
            print(f"  📉 RSI en SOBREVENTA → Debería COMPRAR")
        elif last_rsi > 70:
            print(f"  📈 RSI en SOBRECOMPRA → Debería VENDER")
        else:
            print(f"  ⚖️ RSI neutral → Sin señal clara")
            
    except Exception as e:
        print(f"❌ Error verificando mercado: {e}")

if __name__ == "__main__":
    check_current_market_conditions()
    test_model_signals() 