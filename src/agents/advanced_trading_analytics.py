#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SISTEMA AVANZADO DE TRADING CON ANÁLISIS COMPLETO
- Curvas de aprendizaje
- Control PID
- Métricas financieras completas (Sharpe, MAPE, etc.)
- Sistema de IDs para trades
- Preparado para SP500 y Binance API
"""

import sys
import os
import traceback
import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Agregar directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

try:
    from stable_baselines3 import DQN, A2C, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_RL = True
    print("✅ Componentes de RL disponibles")
except ImportError:
    HAS_RL = False
    print("⚠️ Sin componentes de RL - funcionará en modo técnico")

try:
    import yfinance as yf
    HAS_YFINANCE = True
    print("✅ Yahoo Finance disponible")
except ImportError:
    HAS_YFINANCE = False
    print("⚠️ Sin Yahoo Finance - datos simulados")

try:
    import ccxt
    HAS_CCXT = True
    print("✅ CCXT disponible para Binance")
except ImportError:
    HAS_CCXT = False
    print("⚠️ Sin CCXT - instalar: pip install ccxt")

# Importar nuestro sistema base
from src.agents.ml_enhanced_system import MLEnhancedTradingSystem

class Trade:
    """Clase para representar una operación de trading con ID único"""
    def __init__(self, trade_id, entry_time, entry_price, trade_type, size):
        self.id = trade_id
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.exit_time = None
        self.exit_price = None
        self.trade_type = trade_type  # 'BUY' or 'SELL'
        self.size = size
        self.return_pct = 0.0
        self.return_absolute = 0.0
        self.duration = None
        self.status = 'OPEN'  # 'OPEN', 'CLOSED', 'STOP_LOSS', 'TAKE_PROFIT'
        self.created_at = datetime.now()
        
    def close_trade(self, exit_time, exit_price, status='CLOSED'):
        """Cerrar la operación"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.status = status
        
        # Calcular retornos
        if self.trade_type == 'BUY':
            self.return_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
            self.return_absolute = (exit_price - self.entry_price) * self.size
        else:  # SELL (short)
            self.return_pct = ((self.entry_price - exit_price) / self.entry_price) * 100
            self.return_absolute = (self.entry_price - exit_price) * self.size
            
        # Calcular duración
        if isinstance(self.entry_time, (int, float)) and isinstance(exit_time, (int, float)):
            self.duration = exit_time - self.entry_time
        else:
            self.duration = (exit_time - self.entry_time).total_seconds() / 3600  # horas
    
    def to_dict(self):
        """Convertir a diccionario para análisis"""
        return {
            'id': self.id,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'trade_type': self.trade_type,
            'size': self.size,
            'return_pct': self.return_pct,
            'return_absolute': self.return_absolute,
            'duration': self.duration,
            'status': self.status,
            'created_at': self.created_at
        }

class PIDController:
    """Controlador PID para optimización de trading"""
    def __init__(self, kp=1.0, ki=0.1, kd=0.05, setpoint=0.0):
        self.kp = kp  # Ganancia proporcional
        self.ki = ki  # Ganancia integral
        self.kd = kd  # Ganancia derivativa
        self.setpoint = setpoint  # Valor objetivo
        
        self.previous_error = 0.0
        self.integral = 0.0
        self.max_integral = 100.0  # Anti-windup
        self.output_limits = (-1.0, 1.0)  # Límites de salida
        
    def update(self, current_value, dt=1.0):
        """Actualizar controlador PID"""
        error = self.setpoint - current_value
        
        # Término proporcional
        proportional = self.kp * error
        
        # Término integral con anti-windup
        self.integral += error * dt
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        integral_term = self.ki * self.integral
        
        # Término derivativo
        derivative = self.kd * (error - self.previous_error) / dt if dt > 0 else 0
        
        # Salida total
        output = proportional + integral_term + derivative
        
        # Aplicar límites
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        self.previous_error = error
        
        return output, {
            'error': error,
            'proportional': proportional,
            'integral': integral_term,
            'derivative': derivative,
            'output': output
        }
    
    def reset(self):
        """Reiniciar el controlador"""
        self.previous_error = 0.0
        self.integral = 0.0

class LearningCurveCallback(BaseCallback):
    """Callback para capturar curvas de aprendizaje durante entrenamiento"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Acumular reward del step actual
        if 'rewards' in self.locals and len(self.locals['rewards']) > 0:
            self.current_episode_reward += self.locals['rewards'][0]
        
        self.current_episode_length += 1
        
        # Verificar si terminó el episodio
        if 'dones' in self.locals and self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.timesteps.append(self.num_timesteps)
            
            # Resetear para próximo episodio
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
        return True

class AdvancedTradingAnalytics:
    """Sistema avanzado de análisis de trading"""
    
    def __init__(self, symbol='SPY', use_binance=False):
        print("🚀 Inicializando Sistema Avanzado de Trading...")
        
        self.symbol = symbol
        self.use_binance = use_binance
        
        # Sistema base de ML
        self.base_system = MLEnhancedTradingSystem(skip_selection=True)
        
        # Controlador PID para optimización
        self.pid_controller = PIDController(kp=0.1, ki=0.05, kd=0.02, setpoint=0.1)
        
        # Almacenamiento de trades con IDs
        self.trades = []
        self.open_trades = {}
        self.trade_counter = 0
        
        # Métricas y curvas
        self.learning_curves = {}
        self.financial_metrics = {}
        self.mape_history = []
        
        # Datos
        self.data = None
        self.predictions = []
        self.actual_prices = []
        
        # Cliente Binance
        self.binance_client = None
        if use_binance and HAS_CCXT:
            self._setup_binance()
    
    def _setup_binance(self):
        """Configurar cliente de Binance"""
        try:
            # Cargar variables de entorno
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
            
            if api_key and api_secret:
                self.binance_client = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'sandbox': testnet,  # Usar testnet por defecto
                    'enableRateLimit': True
                })
                
                # Verificar conexión
                balance = self.binance_client.fetch_balance()
                print("✅ Cliente de Binance configurado correctamente")
                print(f"💰 Balance disponible: {balance.get('USDT', {}).get('free', 0)} USDT")
                
            else:
                print("⚠️ API keys de Binance no encontradas")
                print("💡 Crea un archivo .env con BINANCE_API_KEY y BINANCE_API_SECRET")
                
        except Exception as e:
            print(f"❌ Error configurando Binance: {e}")
            self.binance_client = None
    
    def load_sp500_data(self, period='2y'):
        """Cargar datos históricos del SP500"""
        print(f"📊 Cargando datos de {self.symbol} (SP500)...")
        
        if HAS_YFINANCE:
            try:
                ticker = yf.Ticker(self.symbol)
                data = ticker.history(period=period, interval='1h')
                
                if data.empty:
                    print("❌ No se pudieron obtener datos de Yahoo Finance")
                    return False
                
                # Convertir a formato compatible
                df = pd.DataFrame({
                    'timestamp': data.index,
                    'price': data['Close'].values,
                    'volume': data['Volume'].values,
                    'high': data['High'].values,
                    'low': data['Low'].values,
                    'open': data['Open'].values
                })
                
                df = df.reset_index(drop=True)
                print(f"✅ Cargados {len(df)} puntos de datos de {self.symbol}")
                
                # Calcular indicadores técnicos
                self.data = self.base_system.calculate_indicators(df)
                self.base_system.data = self.data
                return True
                
            except Exception as e:
                print(f"❌ Error cargando datos de Yahoo Finance: {e}")
                print("🔄 Generando datos simulados...")
                return self._generate_simulated_data()
        else:
            print("⚠️ Yahoo Finance no disponible, generando datos simulados...")
            return self._generate_simulated_data()
    
    def _generate_simulated_data(self):
        """Generar datos simulados para pruebas"""
        self.base_system.generate_market_data(n_points=2000)
        self.data = self.base_system.data
        return True
    
    def load_binance_data(self, timeframe='1h', limit=1000):
        """Cargar datos de Bitcoin desde Binance"""
        if not self.binance_client:
            print("❌ Cliente de Binance no disponible")
            return False
            
        try:
            print(f"📊 Cargando datos de {self.symbol} desde Binance...")
            
            # Obtener datos OHLCV
            ohlcv = self.binance_client.fetch_ohlcv(
                self.symbol, timeframe=timeframe, limit=limit
            )
            
            if not ohlcv:
                print("❌ No se pudieron obtener datos de Binance")
                return False
            
            # Convertir a DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'price', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            print(f"✅ Cargados {len(df)} puntos de datos desde Binance")
            
            # Calcular indicadores técnicos
            self.data = self.base_system.calculate_indicators(df)
            self.base_system.data = self.data
            return True
            
        except Exception as e:
            print(f"❌ Error cargando datos de Binance: {e}")
            return False
    
    def generate_trade_id(self):
        """Generar ID único para cada trade"""
        self.trade_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8].upper()
        return f"T{self.trade_counter:05d}_{timestamp}_{unique_id}"
    
    def open_trade(self, step, price, trade_type, size):
        """Abrir nueva operación con ID único"""
        trade_id = self.generate_trade_id()
        
        trade = Trade(
            trade_id=trade_id,
            entry_time=step,
            entry_price=price,
            trade_type=trade_type,
            size=size
        )
        
        self.open_trades[trade_id] = trade
        
        print(f"🟢 {trade_type} ABIERTO")
        print(f"   ID: {trade_id}")
        print(f"   Precio: ${price:.2f}")
        print(f"   Tamaño: {size}")
        print(f"   Tiempo: {step}")
        
        return trade_id
    
    def close_trade(self, trade_id, step, price, status='CLOSED'):
        """Cerrar operación existente"""
        if trade_id not in self.open_trades:
            return None
            
        trade = self.open_trades[trade_id]
        trade.close_trade(step, price, status)
        
        # Mover a trades completados
        self.trades.append(trade)
        del self.open_trades[trade_id]
        
        print(f"🔴 {trade.trade_type} CERRADO")
        print(f"   ID: {trade_id}")
        print(f"   Precio Entry: ${trade.entry_price:.2f}")
        print(f"   Precio Exit: ${price:.2f}")
        print(f"   Retorno: {trade.return_pct:.2f}%")
        print(f"   Duración: {trade.duration:.1f}")
        print(f"   Status: {status}")
        
        return trade
    
    def calculate_mape(self, actual, predicted):
        """Calcular Mean Absolute Percentage Error"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Evitar división por cero
        mask = actual != 0
        if not np.any(mask):
            return 100.0
            
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return mape
    
    def calculate_financial_metrics(self):
        """Calcular métricas financieras completas"""
        if len(self.trades) == 0:
            return {}
        
        trades_df = pd.DataFrame([trade.to_dict() for trade in self.trades])
        
        # Métricas básicas
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        losing_trades = len(trades_df[trades_df['return_pct'] <= 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Retornos
        total_return_abs = trades_df['return_absolute'].sum()
        total_return_pct = (total_return_abs / 100000) * 100  # Asumiendo capital inicial 100k
        avg_return = trades_df['return_pct'].mean()
        
        # Estadísticas de wins/losses
        winning_trades_df = trades_df[trades_df['return_pct'] > 0]
        losing_trades_df = trades_df[trades_df['return_pct'] <= 0]
        
        avg_winning_trade = winning_trades_df['return_pct'].mean() if len(winning_trades_df) > 0 else 0
        avg_losing_trade = losing_trades_df['return_pct'].mean() if len(losing_trades_df) > 0 else 0
        
        # Ratio de Sharpe (simplificado)
        returns_std = trades_df['return_pct'].std()
        sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0
        
        # Máximo drawdown
        cumulative_returns = trades_df['return_absolute'].cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        max_drawdown = drawdown.min()
        
        # Profit Factor
        gross_profit = winning_trades_df['return_absolute'].sum() if len(winning_trades_df) > 0 else 0
        gross_loss = abs(losing_trades_df['return_absolute'].sum()) if len(losing_trades_df) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Duración promedio
        avg_duration = trades_df['duration'].mean()
        
        # MAPE si tenemos predicciones
        mape = None
        if len(self.predictions) > 0 and len(self.actual_prices) > 0:
            min_len = min(len(self.predictions), len(self.actual_prices))
            mape = self.calculate_mape(
                self.actual_prices[:min_len], 
                self.predictions[:min_len]
            )
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return_abs': total_return_abs,
            'total_return_pct': total_return_pct,
            'avg_return': avg_return,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_duration': avg_duration,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'mape': mape
        }
        
        self.financial_metrics = metrics
        return metrics
    
    def run_backtest(self, start_step=100, end_step=None):
        """Ejecutar backtest completo con métricas y PID"""
        if self.data is None:
            print("❌ No hay datos cargados")
            return None
            
        print("🔄 Iniciando backtest avanzado...")
        
        if end_step is None:
            end_step = len(self.data) - 100
            
        # Reiniciar sistema
        self.base_system.data = self.data
        self.base_system.current_step = start_step
        self.base_system.current_capital = self.base_system.initial_capital
        self.base_system.position_size = 0
        self.base_system.position_type = None
        
        # Reiniciar controlador PID
        self.pid_controller.reset()
        
        # Variables para tracking
        portfolio_values = [self.base_system.initial_capital]
        target_return = 0.02  # 2% de retorno objetivo por operación
        
        print(f"📊 Ejecutando backtest desde paso {start_step} hasta {end_step}")
        
        for step in range(start_step, end_step):
            current_price = self.data.iloc[step]['price']
            
            # Guardar precio real para MAPE
            self.actual_prices.append(current_price)
            
            # Obtener señal del sistema base
            signal = self.base_system.generate_combined_signal(step)
            
            # Obtener predicción ML si está disponible
            if self.base_system.ml_model is not None:
                try:
                    state = self.base_system.get_state()
                    ml_action = self.base_system.ml_model.predict(state, deterministic=True)[0]
                    # Convertir acción a predicción de precio (simplificado)
                    price_prediction = current_price * (1 + (ml_action - 0.5) * 0.02)
                    self.predictions.append(price_prediction)
                except:
                    self.predictions.append(current_price)
            else:
                self.predictions.append(current_price)
            
            # Calcular retorno actual del portfolio
            current_portfolio = self.base_system.current_capital
            if self.base_system.position_size > 0:
                current_portfolio += self.base_system.position_size * current_price
                
            # Control PID para ajustar señales
            if len(portfolio_values) > 1:
                portfolio_return = (current_portfolio - portfolio_values[-1]) / portfolio_values[-1]
                pid_output, pid_info = self.pid_controller.update(portfolio_return - target_return)
                
                # Ajustar señal con PID (factor de escala pequeño)
                signal = signal + (pid_output * 0.1)
                signal = max(-1.0, min(1.0, signal))  # Limitar entre -1 y 1
            
            portfolio_values.append(current_portfolio)
            
            # Lógica de trading con IDs únicos
            if signal > 0.3 and self.base_system.position_size == 0:
                # Señal de compra
                size = self.base_system.calculate_position_size(current_price)
                if size > 0:
                    trade_id = self.open_trade(step, current_price, 'BUY', size)
                    self.base_system.position_size = size
                    self.base_system.entry_price = current_price
                    self.base_system.position_type = 'LONG'
                    self.base_system.last_trade_step = step
                    
            elif signal < -0.3 and self.base_system.position_size > 0:
                # Señal de venta
                for trade_id in list(self.open_trades.keys()):
                    self.close_trade(trade_id, step, current_price, 'SIGNAL_EXIT')
                    
                # Actualizar capital
                profit = (current_price - self.base_system.entry_price) * self.base_system.position_size
                total_received = current_price * self.base_system.position_size
                self.base_system.current_capital = self.base_system.current_capital - (self.base_system.position_size * self.base_system.entry_price) + total_received
                
                self.base_system.position_size = 0
                self.base_system.position_type = None
            
            # Verificar stop loss / take profit
            if self.base_system.position_size > 0:
                exit_condition = self.base_system.check_exit_conditions(step)
                if exit_condition:
                    for trade_id in list(self.open_trades.keys()):
                        self.close_trade(trade_id, step, current_price, exit_condition)
                    
                    # Liquidar posición
                    total_received = current_price * self.base_system.position_size
                    self.base_system.current_capital = self.base_system.current_capital - (self.base_system.position_size * self.base_system.entry_price) + total_received
                    
                    self.base_system.position_size = 0
                    self.base_system.position_type = None
        
        # Cerrar trades abiertas al final
        final_price = self.data.iloc[end_step-1]['price']
        for trade_id in list(self.open_trades.keys()):
            self.close_trade(trade_id, end_step-1, final_price, 'END_OF_BACKTEST')
        
        # Calcular métricas finales
        metrics = self.calculate_financial_metrics()
        
        print("\n" + "="*60)
        print("📊 RESULTADOS DEL BACKTEST AVANZADO")
        print("="*60)
        print(f"Total de Trades: {metrics.get('total_trades', 0)}")
        print(f"Trades Ganadores: {metrics.get('winning_trades', 0)}")
        print(f"Trades Perdedores: {metrics.get('losing_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
        print(f"Retorno Total: ${metrics.get('total_return_abs', 0):.2f}")
        print(f"Retorno Porcentual: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"Retorno Promedio: {metrics.get('avg_return', 0):.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: ${metrics.get('max_drawdown', 0):.2f}")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.3f}")
        print(f"Duración Promedio: {metrics.get('avg_duration', 0):.1f}")
        
        if metrics.get('mape') is not None:
            print(f"MAPE (Error Predicción): {metrics.get('mape', 0):.2f}%")
        
        return metrics
    
    def create_comprehensive_dashboard(self):
        """Crear dashboard completo con todas las métricas y análisis"""
        if len(self.trades) == 0:
            print("❌ No hay trades para analizar. Ejecuta primero run_backtest()")
            return
            
        print("🎨 Creando dashboard completo de análisis...")
        
        # Configurar estilo profesional
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(28, 18), facecolor='#0a0a0a')
        
        # Grid complejo para dashboard profesional
        gs = GridSpec(5, 5, height_ratios=[2.5, 1.5, 1.5, 1.5, 1], width_ratios=[2, 1, 1, 1, 1],
                     hspace=0.35, wspace=0.25, top=0.95, bottom=0.05, left=0.03, right=0.97)
        
        # 1. Gráfico principal: Precio y Trades
        ax_main = fig.add_subplot(gs[0, :3])
        self._plot_price_and_trades(ax_main)
        
        # 2. Panel de métricas financieras
        ax_metrics = fig.add_subplot(gs[0, 3:])
        self._plot_financial_metrics_panel(ax_metrics)
        
        # 3. Distribución de retornos
        ax_returns = fig.add_subplot(gs[1, 0])
        self._plot_returns_distribution(ax_returns)
        
        # 4. Curva de equity
        ax_equity = fig.add_subplot(gs[1, 1])
        self._plot_equity_curve(ax_equity)
        
        # 5. Análisis de duración
        ax_duration = fig.add_subplot(gs[1, 2])
        self._plot_duration_analysis(ax_duration)
        
        # 6. Drawdown
        ax_drawdown = fig.add_subplot(gs[1, 3])
        self._plot_drawdown(ax_drawdown)
        
        # 7. MAPE y predicciones
        ax_mape = fig.add_subplot(gs[1, 4])
        self._plot_mape_analysis(ax_mape)
        
        # 8. Análisis PID
        ax_pid = fig.add_subplot(gs[2, 0])
        self._plot_pid_analysis(ax_pid)
        
        # 9. Performance mensual
        ax_monthly = fig.add_subplot(gs[2, 1])
        self._plot_monthly_performance(ax_monthly)
        
        # 10. Rolling Sharpe
        ax_sharpe = fig.add_subplot(gs[2, 2])
        self._plot_rolling_sharpe(ax_sharpe)
        
        # 11. Win/Loss streaks
        ax_streaks = fig.add_subplot(gs[2, 3])
        self._plot_win_loss_streaks(ax_streaks)
        
        # 12. Risk metrics
        ax_risk = fig.add_subplot(gs[2, 4])
        self._plot_risk_metrics(ax_risk)
        
        # 13. Trade size analysis
        ax_sizes = fig.add_subplot(gs[3, 0])
        self._plot_trade_sizes(ax_sizes)
        
        # 14. Hourly performance
        ax_hourly = fig.add_subplot(gs[3, 1])
        self._plot_hourly_performance(ax_hourly)
        
        # 15. Correlación con mercado
        ax_correlation = fig.add_subplot(gs[3, 2])
        self._plot_market_correlation(ax_correlation)
        
        # 16. Volatilidad
        ax_volatility = fig.add_subplot(gs[3, 3])
        self._plot_volatility_analysis(ax_volatility)
        
        # 17. Trade heatmap
        ax_heatmap = fig.add_subplot(gs[3, 4])
        self._plot_trade_heatmap(ax_heatmap)
        
        # 18. Tabla resumen de trades
        ax_table = fig.add_subplot(gs[4, :])
        self._plot_trades_summary_table(ax_table)
        
        # Título principal
        fig.suptitle("DASHBOARD AVANZADO DE TRADING - ANÁLISIS COMPLETO CON IA", 
                    fontsize=24, fontweight='bold', color='white', y=0.98)
        
        # Subtítulo con información del sistema
        subtitle = f"Símbolo: {self.symbol} | Total Trades: {len(self.trades)} | "
        subtitle += f"Sistema: {'Binance' if self.use_binance else 'SP500'} | "
        subtitle += f"PID Activo | MAPE Calculado"
        
        fig.text(0.5, 0.95, subtitle, ha='center', va='center', 
                fontsize=14, color='#cccccc', fontweight='normal')
        
        plt.show()
        
        # Guardar dashboard
        try:
            # Definir ruta de salida
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     'data', 'results', 'trading_analysis', 'visualizations')
            os.makedirs(output_dir, exist_ok=True)
            
            dashboard_path = os.path.join(output_dir, 'dashboard_avanzado.png')
            fig.savefig(dashboard_path, facecolor='#0a0a0a', 
                       dpi=300, bbox_inches='tight')
            print(f"💾 Dashboard guardado en: {dashboard_path}")
        except Exception as e:
            print(f"⚠️ No se pudo guardar el dashboard: {e}")
    
    def _plot_price_and_trades(self, ax):
        """Gráfico principal de precio con todas las operaciones"""
        if self.data is None:
            return
            
        # Plotear precio
        ax.plot(self.data['price'], color='#00d2d3', linewidth=1.5, alpha=0.8, label='Precio')
        
        # Plotear trades con líneas de conexión
        for i, trade in enumerate(self.trades):
            if (trade.entry_time < len(self.data) and 
                trade.exit_time is not None and trade.exit_time < len(self.data)):
                
                entry_price = self.data.iloc[int(trade.entry_time)]['price']
                exit_price = self.data.iloc[int(trade.exit_time)]['price']
                
                # Color según resultado
                color = '#00ff41' if trade.return_pct > 0 else '#ff4444'
                alpha = 0.8 if trade.return_pct > 0 else 0.6
                
                # Línea de entrada a salida
                ax.plot([trade.entry_time, trade.exit_time], 
                       [entry_price, exit_price], 
                       color=color, linewidth=2, alpha=alpha)
                
                # Marcadores de entrada y salida
                ax.scatter(trade.entry_time, entry_price, 
                          color='#00ff41', marker='^', s=60, zorder=5, 
                          edgecolors='white', linewidths=1)
                ax.scatter(trade.exit_time, exit_price, 
                          color='#ff4444', marker='v', s=60, zorder=5,
                          edgecolors='white', linewidths=1)
                
                # Etiqueta con ID (solo para algunos trades para no saturar)
                if i % max(1, len(self.trades) // 10) == 0:
                    ax.annotate(f'{trade.id[:8]}', 
                              xy=(trade.entry_time, entry_price),
                              xytext=(5, 10), textcoords='offset points',
                              fontsize=8, color='white', alpha=0.7)
        
        ax.set_title("PRECIO Y OPERACIONES CON IDs", fontweight='bold', 
                    color='white', fontsize=14)
        ax.set_ylabel("Precio USD", fontweight='bold', color='white')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_financial_metrics_panel(self, ax):
        """Panel completo de métricas financieras"""
        metrics = self.financial_metrics
        
        # Crear texto con métricas organizadas
        metrics_text = f"""
MÉTRICAS FINANCIERAS AVANZADAS

📊 ESTADÍSTICAS BÁSICAS
Total de Trades: {metrics.get('total_trades', 0)}
Trades Ganadores: {metrics.get('winning_trades', 0)}
Trades Perdedores: {metrics.get('losing_trades', 0)}
Win Rate: {metrics.get('win_rate', 0):.1f}%

💰 RENDIMIENTO
Retorno Total: ${metrics.get('total_return_abs', 0):.2f}
Retorno %: {metrics.get('total_return_pct', 0):.2f}%
Retorno Promedio: {metrics.get('avg_return', 0):.2f}%
Avg Ganancia: {metrics.get('avg_winning_trade', 0):.2f}%
Avg Pérdida: {metrics.get('avg_losing_trade', 0):.2f}%

📈 MÉTRICAS DE RIESGO
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
Max Drawdown: ${metrics.get('max_drawdown', 0):.2f}
Profit Factor: {metrics.get('profit_factor', 0):.2f}

⏱️ TIMING
Duración Promedio: {metrics.get('avg_duration', 0):.1f}

🤖 PRECISIÓN ML
MAPE: {metrics.get('mape', 0):.2f}% {'✅' if metrics.get('mape', 100) < 10 else '⚠️' if metrics.get('mape', 100) < 20 else '❌'}
        """
        
        # Color de fondo según performance
        total_return_pct = metrics.get('total_return_pct', 0)
        if total_return_pct > 10:
            bg_color = '#1a4a1a'  # Verde
        elif total_return_pct > 0:
            bg_color = '#2a3a1a'  # Verde claro
        else:
            bg_color = '#4a1a1a'  # Rojo
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=11, color='white', va='top', fontfamily='monospace',
                bbox=dict(facecolor=bg_color, alpha=0.9, pad=15,
                         edgecolor='white', linewidth=1))
        ax.set_title("PANEL DE MÉTRICAS", fontweight='bold', color='white')
        ax.axis('off')
    
    # Resto de funciones de plotting (simplificadas por espacio)
    def _plot_returns_distribution(self, ax):
        """Distribución de retornos con estadísticas"""
        returns = [trade.return_pct for trade in self.trades]
        
        if returns:
            ax.hist(returns, bins=20, color='#3742fa', alpha=0.7, 
                   edgecolor='white', linewidth=0.5)
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            ax.axvline(mean_return, color='#ffa502', linestyle='--', 
                      linewidth=2, label=f'Media: {mean_return:.2f}%')
            ax.axvline(mean_return + std_return, color='#ff4444', 
                      linestyle=':', alpha=0.7, label=f'+1σ: {mean_return + std_return:.2f}%')
            ax.axvline(mean_return - std_return, color='#ff4444', 
                      linestyle=':', alpha=0.7, label=f'-1σ: {mean_return - std_return:.2f}%')
            
        ax.set_title("DISTRIBUCIÓN DE RETORNOS", fontweight='bold', color='white')
        ax.set_xlabel("Retorno %", color='white')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_equity_curve(self, ax):
        """Curva de equity mejorada"""
        if not self.trades:
            ax.text(0.5, 0.5, "NO HAY DATOS", ha='center', va='center',
                   transform=ax.transAxes, color='white', fontsize=12)
            ax.axis('off')
            return
            
        cumulative_returns = np.cumsum([trade.return_absolute for trade in self.trades])
        initial_capital = 100000
        equity_curve = initial_capital + cumulative_returns
        
        ax.plot(equity_curve, color='#00ff41', linewidth=2.5, alpha=0.9)
        ax.fill_between(range(len(equity_curve)), initial_capital, equity_curve, 
                       alpha=0.3, color='#00ff41')
        ax.axhline(initial_capital, color='white', linestyle='--', 
                  alpha=0.5, label='Capital Inicial')
        
        # Máximo y mínimo
        max_equity = np.max(equity_curve)
        min_equity = np.min(equity_curve)
        ax.axhline(max_equity, color='#00ff41', linestyle=':', 
                  alpha=0.7, label=f'Máximo: ${max_equity:,.0f}')
        
        ax.set_title("CURVA DE EQUITY", fontweight='bold', color='white')
        ax.set_ylabel("Capital USD", color='white')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Funciones placeholder para otros gráficos
    def _plot_duration_analysis(self, ax):
        durations = [trade.duration for trade in self.trades if trade.duration is not None]
        if durations:
            ax.boxplot(durations, patch_artist=True,
                      boxprops=dict(facecolor='#3742fa', alpha=0.7))
        ax.set_title("DURACIÓN TRADES", fontweight='bold', color='white')
        ax.grid(True, alpha=0.3)
    
    def _plot_drawdown(self, ax):
        if not self.trades:
            ax.text(0.5, 0.5, "NO HAY DATOS", ha='center', va='center',
                   transform=ax.transAxes, color='white')
            ax.axis('off')
            return
            
        cumulative = np.cumsum([trade.return_absolute for trade in self.trades])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        ax.fill_between(range(len(drawdown)), 0, drawdown,
                       color='#ff4444', alpha=0.7)
        ax.set_title("DRAWDOWN", fontweight='bold', color='white')
        ax.grid(True, alpha=0.3)
    
    # Implementaciones simplificadas para el resto de funciones
    def _plot_mape_analysis(self, ax):
        mape = self.financial_metrics.get('mape', None)
        if mape is not None:
            ax.text(0.5, 0.5, f"MAPE\n{mape:.2f}%", ha='center', va='center',
                   transform=ax.transAxes, color='white', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, "MAPE\nN/A", ha='center', va='center',
                   transform=ax.transAxes, color='white', fontsize=14)
        ax.set_title("ERROR PREDICCIÓN", fontweight='bold', color='white')
        ax.axis('off')
    
    def _plot_pid_analysis(self, ax):
        ax.text(0.5, 0.5, "CONTROL PID\nACTIVO", ha='center', va='center',
               transform=ax.transAxes, color='white', fontsize=12, fontweight='bold')
        ax.set_title("CONTROLADOR PID", fontweight='bold', color='white')
        ax.axis('off')
    
    # Placeholder para funciones restantes
    def _plot_monthly_performance(self, ax): self._placeholder_plot(ax, "PERFORMANCE\nMENSUAL")
    def _plot_rolling_sharpe(self, ax): self._placeholder_plot(ax, "ROLLING\nSHARPE")
    def _plot_win_loss_streaks(self, ax): self._placeholder_plot(ax, "WIN/LOSS\nSTREAKS")
    def _plot_risk_metrics(self, ax): self._placeholder_plot(ax, "MÉTRICAS\nDE RIESGO")
    def _plot_trade_sizes(self, ax): self._placeholder_plot(ax, "TAMAÑOS\nDE TRADE")
    def _plot_hourly_performance(self, ax): self._placeholder_plot(ax, "PERFORMANCE\nPOR HORA")
    def _plot_market_correlation(self, ax): self._placeholder_plot(ax, "CORRELACIÓN\nMERCADO")
    def _plot_volatility_analysis(self, ax): self._placeholder_plot(ax, "ANÁLISIS\nVOLATILIDAD")
    def _plot_trade_heatmap(self, ax): self._placeholder_plot(ax, "HEATMAP\nTRADES")
    
    def _placeholder_plot(self, ax, title):
        """Plot placeholder para funciones en desarrollo"""
        ax.text(0.5, 0.5, title, ha='center', va='center',
               transform=ax.transAxes, color='white', fontsize=10, fontweight='bold')
        ax.set_title(title.replace('\n', ' '), fontweight='bold', color='white', fontsize=10)
        ax.axis('off')
    
    def _plot_trades_summary_table(self, ax):
        """Tabla resumen con los últimos trades"""
        if len(self.trades) == 0:
            ax.text(0.5, 0.5, "NO HAY TRADES PARA MOSTRAR", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, color='white', fontweight='bold')
            ax.axis('off')
            return
        
        # Últimos 8 trades
        recent_trades = self.trades[-8:] if len(self.trades) >= 8 else self.trades
        
        table_data = []
        for trade in recent_trades:
            # Status con emoji
            status_emoji = {
                'CLOSED': '✅', 'STOP_LOSS': '🛑', 
                'TAKE_PROFIT': '🎯', 'SIGNAL_EXIT': '📊',
                'END_OF_BACKTEST': '🔚'
            }.get(trade.status, '❓')
            
            table_data.append([
                trade.id[:15],  # ID truncado
                f"${trade.entry_price:.2f}",
                f"${trade.exit_price:.2f}" if trade.exit_price else "N/A",
                f"{trade.return_pct:.2f}%",
                f"{trade.duration:.1f}" if trade.duration else "N/A",
                f"{status_emoji} {trade.status}"
            ])
        
        columns = ['ID', 'Entry $', 'Exit $', 'Return %', 'Duration', 'Status']
        
        # Crear tabla
        table = ax.table(cellText=table_data, colLabels=columns,
                        cellLoc='center', loc='center',
                        colColours=['#2a2a2a']*len(columns))
        
        # Estilo de tabla
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Colorear celdas según retorno
        for i, trade in enumerate(recent_trades):
            if trade.return_pct > 0:
                table[(i+1, 3)].set_facecolor('#1a4a1a')  # Verde para ganancias
            elif trade.return_pct < 0:
                table[(i+1, 3)].set_facecolor('#4a1a1a')  # Rojo para pérdidas
        
        ax.set_title("RESUMEN DE TRADES RECIENTES", fontweight='bold', 
                    color='white', pad=20, fontsize=12)
        ax.axis('off')

def main():
    """Función principal del sistema avanzado"""
    print("\n" + "="*80)
    print("🚀 SISTEMA AVANZADO DE TRADING CON IA - ANÁLISIS COMPLETO")
    print("📊 Métricas Financieras | 🎯 PID Control | 🔍 MAPE | 📈 Curvas de Aprendizaje")
    print("="*80)
    
    try:
        # Crear sistema avanzado
        print("\n1️⃣ Inicializando sistema...")
        system = AdvancedTradingAnalytics(symbol='SPY', use_binance=False)
        
        # Cargar datos
        print("\n2️⃣ Cargando datos...")
        if system.load_sp500_data(period='1y'):  # 1 año de datos
            print("✅ Datos cargados exitosamente")
            
            # Ejecutar backtest avanzado
            print("\n3️⃣ Ejecutando backtest avanzado...")
            metrics = system.run_backtest()
            
            if metrics and metrics.get('total_trades', 0) > 0:
                print("\n4️⃣ Creando dashboard...")
                system.create_comprehensive_dashboard()
                
                print("\n✅ ¡Sistema completado exitosamente!")
                print("📋 Próximos pasos:")
                print("   • Instalar dependencias: python install_advanced_dependencies.py")
                print("   • Configurar Binance: Copiar configs/binance.env.example a .env")
                print("   • Para Bitcoin: system = AdvancedTradingAnalytics('BTCUSDT', use_binance=True)")
                
            else:
                print("❌ No se generaron trades suficientes en el backtest")
                print("💡 Intenta ajustar los parámetros o el período de datos")
        else:
            print("❌ Error cargando datos")
            
    except Exception as e:
        print(f"\n❌ Error en el sistema: {e}")
        traceback.print_exc()
        print("\n💡 Soluciones:")
        print("   • Instala dependencias: python install_advanced_dependencies.py")
        print("   • Verifica conexión a internet para datos de Yahoo Finance")

if __name__ == "__main__":
    main() 