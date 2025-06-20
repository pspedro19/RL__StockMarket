"""
Agente Simple de Trading
Implementación básica de un agente de RL para trading
"""

import numpy as np
import logging

logger = logging.getLogger('simple_agent')

class SimpleAgent:
    """Agente de trading simple basado en reglas"""
    
    def __init__(self, action_space=3):
        """
        Inicializar agente simple
        action_space: 0=Venta, 1=Compra, 2=Hold
        """
        self.action_space = action_space
        self.last_action = 2  # Hold por defecto
        self.position = 0
        
    def predict(self, observation):
        """
        Predecir acción basada en observación
        observation: array con features [price, rsi, macd, bb_pos, volume_ratio, momentum]
        """
        try:
            if len(observation) < 6:
                return np.array([[2]])  # Hold si no hay suficientes features
            
            # Extraer features
            price_norm = observation[0] if len(observation) > 0 else 0.5
            rsi = observation[1] if len(observation) > 1 else 0.5
            macd = observation[2] if len(observation) > 2 else 0
            bb_pos = observation[3] if len(observation) > 3 else 0.5
            volume_ratio = observation[4] if len(observation) > 4 else 1
            momentum = observation[5] if len(observation) > 5 else 0
            
            # Lógica de decisión simple
            signals = []
            
            # Señal RSI
            if rsi < 0.3:  # Sobreventa
                signals.append(1)  # Compra
            elif rsi > 0.7:  # Sobrecompra
                signals.append(0)  # Venta
            
            # Señal MACD
            if macd > 0.01:
                signals.append(1)  # Compra
            elif macd < -0.01:
                signals.append(0)  # Venta
            
            # Señal Bollinger Bands
            if bb_pos < 0.2:  # Cerca del límite inferior
                signals.append(1)  # Compra
            elif bb_pos > 0.8:  # Cerca del límite superior
                signals.append(0)  # Venta
            
            # Señal de momentum
            if momentum > 0.02:
                signals.append(1)  # Compra
            elif momentum < -0.02:
                signals.append(0)  # Venta
            
            # Decidir por mayoría
            if len(signals) >= 2:
                if signals.count(1) > signals.count(0):
                    action = 1  # Compra
                elif signals.count(0) > signals.count(1):
                    action = 0  # Venta
                else:
                    action = 2  # Hold
            else:
                action = 2  # Hold por defecto
            
            self.last_action = action
            return np.array([[action]])
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return np.array([[2]])  # Hold en caso de error
    
    def get_action_name(self, action):
        """Obtener nombre de la acción"""
        actions = {0: "VENTA", 1: "COMPRA", 2: "HOLD"}
        return actions.get(action, "UNKNOWN")
    
    def update_position(self, action):
        """Actualizar posición basada en acción"""
        if action == 0:  # Venta
            self.position = -1
        elif action == 1:  # Compra
            self.position = 1
        else:  # Hold
            pass  # Mantener posición actual
    
    def reset(self):
        """Resetear agente"""
        self.last_action = 2
        self.position = 0
        logger.info("🔄 Agente simple reseteado") 