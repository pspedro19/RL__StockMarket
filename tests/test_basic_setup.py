#!/usr/bin/env python3

def test_basic_imports():
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ Pandas, NumPy, Matplotlib: OK")
        
        import torch
        print("✅ PyTorch: OK")
        
        import stable_baselines3
        print("✅ Stable Baselines3: OK")
        
        import gym
        print("✅ Gym: OK")
        
        print("\n🎉 ¡Setup básico completado!")
        return True
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_basic_imports()
