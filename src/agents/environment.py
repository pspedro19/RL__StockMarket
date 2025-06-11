import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import logging

class TradingEnvironment(gym.Env):
    """Ambiente básico de trading para RL"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        
        # Espacios de acción y observación
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observación: features del mercado + estado de cuenta
        n_features = min(20, len(data.columns) - 5)  # Máximo 20 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                          shape=(n_features + 5,), dtype=np.float32)
        
        # Estado
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        self.done = False
        self.equity_curve = [initial_balance]
        self.logger = logging.getLogger(__name__)

    def reset(self) -> np.ndarray:
        """Reset ambiente"""
        self.current_step = np.random.randint(100, max(200, len(self.data) - 1000))
        self.balance = self.initial_balance
        self.position = 0
        self.done = False
        self.equity_curve = [self.initial_balance]
        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Ejecutar acción"""
        # Procesar acción (simplificado)
        action_value = float(np.clip(action[0], -1, 1))
        
        # Obtener precio actual
        if self.current_step >= len(self.data):
            self.done = True
            return self._get_observation(), 0, True, {}
            
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calcular reward simple
        if len(self.equity_curve) > 1:
            price_change = (current_price - self.data.iloc[self.current_step-1]['close']) / self.data.iloc[self.current_step-1]['close']
            reward = action_value * price_change * 100  # Simplificado
        else:
            reward = 0
        
        # Actualizar estado
        self.current_step += 1
        current_equity = self.balance  # Simplificado
        self.equity_curve.append(current_equity)
        
        # Check si terminó
        self.done = self.current_step >= len(self.data) - 1
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'equity': current_equity,
            'current_price': current_price
        }
        
        return self._get_observation(), reward, self.done, info

    def _get_observation(self) -> np.ndarray:
        """Obtener observación actual"""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape)
        
        # Features del mercado (primeras 20 columnas numéricas)
        row = self.data.iloc[self.current_step]
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        market_features = []
        
        for col in numeric_cols:
            if col not in ['time'] and len(market_features) < 20:
                value = row[col] if not pd.isna(row[col]) else 0
                market_features.append(float(value))
        
        # Rellenar si faltan features
        while len(market_features) < 20:
            market_features.append(0.0)
        
        # Tomar solo las primeras 20
        market_features = market_features[:20]
        
        # Estado de cuenta (5 features)
        account_features = [
            self.balance / self.initial_balance,  # Balance normalizado
            self.position,                        # Posición
            0.0,                                 # PnL
            0.0,                                 # Drawdown  
            float(self.current_step) / len(self.data)  # Progreso
        ]
        
        observation = np.array(market_features + account_features, dtype=np.float32)
        return observation

    def render(self, mode='human'):
        """Renderizar estado"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, Position: {self.position}")
