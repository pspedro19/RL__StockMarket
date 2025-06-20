"""
Entorno de Trading para RL
Define el environment de Gymnasium para entrenamiento de agentes
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import logging

logger = logging.getLogger('trading_environment')

class TradingEnvironment(gym.Env):
    """Entorno de trading compatible con Gymnasium"""
    
    def __init__(self, data, initial_balance=100000, lookback_window=10):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.current_step = 0
        
        # Espacios de acción y observación
        self.action_space = spaces.Discrete(3)  # 0: Venta, 1: Compra, 2: Hold
        
        # Espacio de observación: [price_norm, rsi, macd, bb_pos, volume_ratio, momentum, position, balance_ratio]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        # Estado del entorno
        self.balance = initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.entry_price = 0
        self.total_trades = 0
        self.profitable_trades = 0
        
        # Para tracking
        self.trade_history = []
        self.balance_history = []
        
    def reset(self, seed=None):
        """Resetear el entorno"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_trades = 0
        self.profitable_trades = 0
        self.trade_history = []
        self.balance_history = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Ejecutar un paso en el entorno"""
        prev_balance = self.balance
        reward = 0
        
        # Obtener precio actual
        current_price = self.data['close'].iloc[self.current_step]
        
        # Ejecutar acción
        if action == 0:  # Venta
            if self.position > 0:  # Cerrar posición larga
                profit = (current_price - self.entry_price) * self.position
                self.balance += profit
                reward = profit / self.initial_balance * 100  # Recompensa normalizada
                self._record_trade(profit)
                self.position = 0
                self.entry_price = 0
            elif self.position == 0:  # Abrir posición corta
                self.position = -1
                self.entry_price = current_price
                
        elif action == 1:  # Compra
            if self.position < 0:  # Cerrar posición corta
                profit = (self.entry_price - current_price) * abs(self.position)
                self.balance += profit
                reward = profit / self.initial_balance * 100
                self._record_trade(profit)
                self.position = 0
                self.entry_price = 0
            elif self.position == 0:  # Abrir posición larga
                self.position = 1
                self.entry_price = current_price
        
        # Hold (action == 2) no hace nada
        
        # Calcular recompensa por holding si hay posición abierta
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = 0
            if self.position > 0:  # Long
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:  # Short
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            
            reward += unrealized_pnl * 0.1  # Pequeña recompensa por unrealized PnL
        
        # Penalización por exceso de trading
        if action != 2:  # Si no es hold
            reward -= 0.001  # Pequeña penalización por transacción
        
        # Avanzar paso
        self.current_step += 1
        
        # Verificar si el episodio terminó
        done = self.current_step >= len(self.data) - 1
        
        # Guardar historial de balance
        self.balance_history.append(self.balance)
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        """Obtener observación actual del entorno"""
        try:
            # Obtener datos actuales
            current_data = self.data.iloc[self.current_step]
            
            # Features básicas (normalizadas)
            price_norm = current_data['close'] / self.data['close'].rolling(50).mean().iloc[self.current_step]
            rsi = current_data.get('rsi', 0.5)
            macd = current_data.get('macd', 0)
            bb_pos = current_data.get('bb_position', 0.5)
            volume_ratio = current_data.get('volume_ratio', 1)
            momentum = current_data.get('momentum', 0)
            
            # Estado del agente
            position_norm = self.position  # -1, 0, 1
            balance_ratio = self.balance / self.initial_balance
            
            observation = np.array([
                price_norm, rsi, macd, bb_pos, volume_ratio, 
                momentum, position_norm, balance_ratio
            ], dtype=np.float32)
            
            # Reemplazar NaN por valores por defecto
            observation = np.nan_to_num(observation, nan=0.0)
            
            return observation
            
        except Exception as e:
            logger.error(f"Error obteniendo observación: {e}")
            return np.zeros(8, dtype=np.float32)
    
    def _record_trade(self, profit):
        """Registrar operación"""
        self.total_trades += 1
        if profit > 0:
            self.profitable_trades += 1
        
        self.trade_history.append({
            'step': self.current_step,
            'profit': profit,
            'balance': self.balance
        })
    
    def get_portfolio_value(self):
        """Obtener valor actual del portfolio"""
        current_price = self.data['close'].iloc[self.current_step]
        unrealized_pnl = 0
        
        if self.position != 0 and self.entry_price > 0:
            if self.position > 0:  # Long
                unrealized_pnl = (current_price - self.entry_price) * self.position
            else:  # Short
                unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
        
        return self.balance + unrealized_pnl
    
    def get_stats(self):
        """Obtener estadísticas del trading"""
        win_rate = (self.profitable_trades / max(self.total_trades, 1)) * 100
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        return {
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_balance': self.balance
        } 