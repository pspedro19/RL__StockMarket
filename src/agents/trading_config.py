"""
Configuración robusta para el sistema de trading
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TradingConfig:
    """Configuración principal del sistema de trading"""
    
    # Conexión y datos
    max_retries: int = 3
    retry_delay: int = 5
    health_check_interval: int = 30
    data_timeout: int = 300  # 5 minutos
    
    # Trading
    initial_capital: float = 10000.0
    max_position_size: float = 0.1  # 10% del capital
    stop_loss_pct: float = 0.02     # 2%
    take_profit_pct: float = 0.04   # 4%
    max_daily_trades: int = 10
    cooldown_period: int = 30       # segundos
    
    # Gestión de riesgo
    max_consecutive_losses: int = 3
    min_capital_pct: float = 0.5    # No perder más del 50%
    max_drawdown_pct: float = 0.2   # 20% máximo drawdown
    
    # Señales
    min_signal_strength: float = 0.3
    technical_weight: float = 0.4
    ml_weight: float = 0.6
    
    # Sistema
    update_interval: float = 1.0
    dashboard_update_interval: float = 5.0
    max_consecutive_errors: int = 5
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/trading_system.log"
    
    # MT5
    symbol: str = "US500"
    timeframe: str = "1s"
    
    @classmethod
    def from_file(cls, config_file: str = "configs/trading_params.yaml"):
        """Cargar configuración desde archivo"""
        try:
            import yaml
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                return cls(**config_data.get('trading', {}))
        except Exception as e:
            print(f"⚠️ Error cargando config: {e}, usando valores por defecto")
        
        return cls()
    
    def validate(self):
        """Validar configuración"""
        errors = []
        
        if self.initial_capital <= 0:
            errors.append("Capital inicial debe ser positivo")
        
        if self.max_position_size <= 0 or self.max_position_size > 1:
            errors.append("Tamaño máximo de posición debe estar entre 0 y 1")
        
        if self.stop_loss_pct <= 0 or self.stop_loss_pct > 0.5:
            errors.append("Stop loss debe estar entre 0 y 0.5")
        
        if self.min_signal_strength <= 0 or self.min_signal_strength > 1:
            errors.append("Fuerza mínima de señal debe estar entre 0 y 1")
        
        if errors:
            raise ValueError(f"Errores de configuración: {'; '.join(errors)}")
        
        return True

class RiskManager:
    """Gestor de riesgo robusto"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.max_drawdown = 0.0
        self.peak_capital = config.initial_capital
        
    def can_trade(self, current_capital: float, last_trade_time: float, current_time: float) -> tuple[bool, str]:
        """Verificar si se puede ejecutar un trade"""
        
        # Verificar límite diario
        if self.daily_trades >= self.config.max_daily_trades:
            return False, "Límite diario de trades alcanzado"
        
        # Verificar cooldown
        if current_time - last_trade_time < self.config.cooldown_period:
            return False, "En periodo de cooldown"
        
        # Verificar capital mínimo
        if current_capital < self.config.initial_capital * self.config.min_capital_pct:
            return False, "Capital por debajo del mínimo"
        
        # Verificar pérdidas consecutivas
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False, "Demasiadas pérdidas consecutivas"
        
        # Verificar drawdown
        current_drawdown = (self.peak_capital - current_capital) / self.peak_capital
        if current_drawdown > self.config.max_drawdown_pct:
            return False, "Drawdown máximo alcanzado"
        
        return True, "OK"
    
    def update_after_trade(self, pnl: float, current_capital: float):
        """Actualizar métricas después de un trade"""
        self.daily_trades += 1
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Actualizar peak capital
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        # Actualizar drawdown máximo
        current_drawdown = (self.peak_capital - current_capital) / self.peak_capital
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def reset_daily(self):
        """Reset contadores diarios"""
        self.daily_trades = 0

class SystemMonitor:
    """Monitor de salud del sistema"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.last_data_time = None
        self.connection_stable = False
        self.consecutive_errors = 0
        self.system_healthy = True
        
    def check_data_freshness(self, last_data_timestamp) -> bool:
        """Verificar que los datos son recientes"""
        try:
            from datetime import datetime, timedelta
            
            if last_data_timestamp is None:
                return False
            
            # Convertir a datetime si es necesario
            if isinstance(last_data_timestamp, str):
                last_data_timestamp = datetime.fromisoformat(last_data_timestamp)
            
            # Verificar que no son muy viejos
            time_diff = datetime.now() - last_data_timestamp
            return time_diff.total_seconds() < self.config.data_timeout
            
        except Exception:
            return False
    
    def report_error(self):
        """Reportar error del sistema"""
        self.consecutive_errors += 1
        
        if self.consecutive_errors >= self.config.max_consecutive_errors:
            self.system_healthy = False
            return False
        
        return True
    
    def report_success(self):
        """Reportar éxito del sistema"""
        self.consecutive_errors = 0
        self.system_healthy = True
    
    def is_healthy(self) -> bool:
        """Verificar si el sistema está saludable"""
        return (self.system_healthy and 
                self.connection_stable and 
                self.consecutive_errors < self.config.max_consecutive_errors)

# Configuración global por defecto
DEFAULT_CONFIG = TradingConfig()

# Validar configuración al importar
try:
    DEFAULT_CONFIG.validate()
except ValueError as e:
    print(f"❌ Error en configuración por defecto: {e}")
    # Usar configuración mínima válida
    DEFAULT_CONFIG = TradingConfig(
        initial_capital=1000.0,
        max_position_size=0.05,
        max_daily_trades=5
    ) 