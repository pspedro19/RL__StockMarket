#!/usr/bin/env python3

"""
Test específico para verificar carga de modelos ML de PPO y A2C
"""

import sys
import os

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.agents.comparison_four_models import FourModelsComparison
    import numpy as np
    import traceback
    
    def test_models_loading():
        """
        Test que verifica que TODOS los modelos cargan correctamente sus ML models
        """
        
        print("\n" + "="*70)
        print("🧪 TEST: Verificación de carga de modelos ML")
        print("="*70)
        
        # Crear comparación
        comparison = FourModelsComparison()
        
        # Inicializar sistemas sin crear interfaz
        comparison.initialize_systems()
        
        print("\n🔍 ESTADO DE MODELOS ML:")
        print("-" * 50)
        
        models = ['DQN', 'DeepDQN', 'PPO', 'A2C']
        systems = [comparison.dqn_system, comparison.deepdqn_system, 
                  comparison.ppo_system, comparison.a2c_system]
        
        ml_loaded = {}
        
        for name, system in zip(models, systems):
            has_ml = system.ml_model is not None
            ml_loaded[name] = has_ml
            
            status = "✅ CARGADO" if has_ml else "❌ NO CARGADO"
            action_space = system.action_space
            ml_weight = system.ml_weight
            
            print(f"{name:8s}: {status} | Action Space: {action_space} | ML Weight: {ml_weight}")
            
            if has_ml:
                # Test de predicción
                try:
                    state = system.get_state()
                    action = system.ml_model.predict(state, deterministic=True)[0]
                    print(f"          Test Prediction: {action}")
                except Exception as e:
                    print(f"          ⚠️ Error en predicción: {e}")
        
        print("\n🎯 OBJETIVO:")
        print("-" * 50)
        print("Todos los modelos deberían cargar exitosamente:")
        print("- DQN: ✅ (Discrete(2))")
        print("- DeepDQN: ✅ (Discrete(2))")
        print("- PPO: ✅ (Discrete(2))")
        print("- A2C: ✅ (Discrete(2) o Box continuo)")
        
        print("\n📊 RESULTADOS:")
        print("-" * 50)
        
        success_count = sum(ml_loaded.values())
        total_count = len(ml_loaded)
        
        print(f"Modelos cargados exitosamente: {success_count}/{total_count}")
        
        if success_count == total_count:
            print("🎉 ¡ÉXITO! Todos los modelos ML están funcionando")
            return True
        else:
            print("⚠️ Algunos modelos no cargaron - usando análisis técnico como fallback")
            
            # Verificar que al menos funcionan con análisis técnico
            print("\n🔍 VERIFICANDO FALLBACK A ANÁLISIS TÉCNICO:")
            print("-" * 50)
            
            for name, system in zip(models, systems):
                if not ml_loaded[name]:
                    try:
                        # Test de señal técnica
                        signal = system.generate_combined_signal(55)  # Step con suficiente historia
                        print(f"{name:8s}: Señal técnica = {signal:+6.3f} ✅")
                    except Exception as e:
                        print(f"{name:8s}: Error en señal técnica = {e} ❌")
            
            return False
        
    if __name__ == "__main__":
        success = test_models_loading()
        if success:
            print("\n✅ CONCLUSIÓN: Sistema listo para trading con modelos ML")
        else:
            print("\n⚠️ CONCLUSIÓN: Sistema funciona pero algunos modelos usan análisis técnico")
        
except Exception as e:
    print(f"\n❌ Error en test: {e}")
    traceback.print_exc() 