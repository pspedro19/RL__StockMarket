#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SISTEMA DE TRADING EN TIEMPO REAL - MODO CONSOLA
- Solo terminal, sin gráficos
- Datos reales de MT5
- Seguimiento financiero completo
- CSV export automático
"""

import sys
import os
import uuid
import csv
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Agregar directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# MT5
try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
    print("✅ MetaTrader5 disponible")
except ImportError:
    HAS_MT5 = False
    print("❌ MetaTrader5 NO disponible - REQUERIDO")
    sys.exit(1)

# RL Components
try:
    from stable_baselines3 import DQN, A2C, PPO
    HAS_RL = True
    print("✅ Stable-baselines3 disponible")
except ImportError:
    HAS_RL = False
    print("⚠️ Sin componentes de RL")

class RealTimeTradeManager:
    """Gestor de trades en tiempo real con IDs únicos y export CSV"""
    
    def __init__(self, csv_filename=None):
        self.trades = []
        self.open_trades = {}
        self.trade_counter = 0
        
        # CSV filename con timestamp
        if csv_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_filename = f"trades_realtime_{timestamp}.csv"
        else:
            self.csv_filename = csv_filename
            
        # Crear headers del CSV
        self.csv_headers = [
            'trade_id', 'status', 'symbol', 'trade_type', 'size',
            'entry_time', 'entry_price', 'exit_time', 'exit_price',
            'duration_minutes', 'return_pct', 'return_absolute',
            'stop_loss_price', 'take_profit_price', 'exit_reason',
            'ml_signal', 'technical_signal', 'combined_signal',
            'rsi', 'macd', 'volume', 'portfolio_value'
        ]
        
        # Inicializar CSV
        self.init_csv()
        
    def init_csv(self):
        """Inicializar archivo CSV con headers"""
        try:
            output_dir = os.path.join(root_dir, 'data', 'results', 'trading_analysis', 'csv_exports')
            os.makedirs(output_dir, exist_ok=True)
            
            self.csv_path = os.path.join(output_dir, self.csv_filename)
            
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.csv_headers)
                writer.writeheader()
                
            print(f"📁 CSV inicializado: {self.csv_path}")
            
        except Exception as e:
            print(f"❌ Error inicializando CSV: {e}")
            
    def generate_trade_id(self):
        """Generar ID único para trade"""
        self.trade_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:6].upper()
        return f"T{self.trade_counter:04d}_{timestamp}_{unique_id}"
    
    def open_trade(self, symbol, trade_type, size, entry_price, entry_time, 
                   ml_signal=0, technical_signal=0, combined_signal=0,
                   rsi=50, macd=0, volume=0, portfolio_value=0):
        """Abrir nuevo trade y registrar en CSV"""
        
        trade_id = self.generate_trade_id()
        
        trade_data = {
            'trade_id': trade_id,
            'status': 'OPEN',
            'symbol': symbol,
            'trade_type': trade_type,
            'size': size,
            'entry_time': entry_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(entry_time, datetime) else str(entry_time),
            'entry_price': round(entry_price, 4),
            'exit_time': '',
            'exit_price': '',
            'duration_minutes': '',
            'return_pct': '',
            'return_absolute': '',
            'stop_loss_price': round(entry_price * 0.99, 4) if trade_type == 'BUY' else round(entry_price * 1.01, 4),
            'take_profit_price': round(entry_price * 1.02, 4) if trade_type == 'BUY' else round(entry_price * 0.98, 4),
            'exit_reason': '',
            'ml_signal': round(ml_signal, 4),
            'technical_signal': round(technical_signal, 4),
            'combined_signal': round(combined_signal, 4),
            'rsi': round(rsi, 2),
            'macd': round(macd, 6),
            'volume': int(volume),
            'portfolio_value': round(portfolio_value, 2)
        }
        
        # Guardar en memoria
        self.open_trades[trade_id] = trade_data
        
        # Escribir al CSV inmediatamente
        self.write_trade_to_csv(trade_data)
        
        return trade_id
    
    def close_trade(self, trade_id, exit_price, exit_time, exit_reason='MANUAL'):
        """Cerrar trade y actualizar CSV"""
        
        if trade_id not in self.open_trades:
            return None
            
        trade_data = self.open_trades[trade_id].copy()
        
        # Calcular duración
        entry_time = datetime.strptime(trade_data['entry_time'], '%Y-%m-%d %H:%M:%S')
        if isinstance(exit_time, datetime):
            duration = (exit_time - entry_time).total_seconds() / 60  # minutos
            exit_time_str = exit_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            duration = 0
            exit_time_str = str(exit_time)
        
        # Calcular retornos
        entry_price = trade_data['entry_price']
        if trade_data['trade_type'] == 'BUY':
            return_pct = ((exit_price - entry_price) / entry_price) * 100
        else:  # SELL
            return_pct = ((entry_price - exit_price) / entry_price) * 100
            
        return_absolute = return_pct * trade_data['size'] * entry_price / 100
        
        # Actualizar datos
        trade_data.update({
            'status': 'CLOSED',
            'exit_time': exit_time_str,
            'exit_price': round(exit_price, 4),
            'duration_minutes': round(duration, 1),
            'return_pct': round(return_pct, 2),
            'return_absolute': round(return_absolute, 2),
            'exit_reason': exit_reason
        })
        
        # Mover a trades cerrados
        self.trades.append(trade_data)
        del self.open_trades[trade_id]
        
        # Actualizar CSV
        self.update_trade_in_csv(trade_data)
        
        return trade_data
    
    def write_trade_to_csv(self, trade_data):
        """Escribir trade al CSV"""
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.csv_headers)
                writer.writerow(trade_data)
        except Exception as e:
            print(f"❌ Error escribiendo al CSV: {e}")
    
    def update_trade_in_csv(self, updated_trade):
        """Actualizar trade en CSV (reescribir archivo)"""
        try:
            # Leer todos los trades
            all_trades = []
            
            # Leer existentes
            with open(self.csv_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['trade_id'] == updated_trade['trade_id']:
                        all_trades.append(updated_trade)
                    else:
                        all_trades.append(row)
            
            # Reescribir archivo
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.csv_headers)
                writer.writeheader()
                writer.writerows(all_trades)
                
        except Exception as e:
            print(f"❌ Error actualizando CSV: {e}")

class ConsoleTradingSystem:
    """Sistema de trading en tiempo real - SOLO CONSOLA"""
    
    def __init__(self, selected_model=None):
        """Inicializar sistema"""
        self.symbol = "US500"
        self.timeframe = mt5.TIMEFRAME_M1
        self.update_interval = 3  # segundos
        self.is_real_time = False
        self.mt5_connected = False
        
        # Seleccionar modelo
        if selected_model:
            self.selected_model = selected_model
            self.model_name = self._model_name_from_selected_model()
        else:
            self.selected_model = None
            self.model_name = ""
        
        # Sistema base simplificado
        self.data = pd.DataFrame(columns=['timestamp', 'price', 'volume', 'rsi', 'macd'])
        
        # Modelos ML
        self.models = {'dqn': None, 'deepdqn': None, 'a2c': None, 'ppo': None}
        self._load_models()
        
        # Trade manager
        self.trade_manager = RealTimeTradeManager()
        
        # Variables financieras
        self.initial_capital = 100000.0
        self.current_capital = 100000.0
        self.total_profit_loss = 0.0
        self.total_profit_loss_pct = 0.0
        self.last_trade_pnl = 0.0
        self.trade_size_usd = 1000.0
        
        # Control de posiciones
        self.current_position = None
        self.last_operation_type = None
        self.last_trade_time = None
        self.trade_cooldown = 30  # segundos
        
        # Conectar MT5
        if not self.connect_mt5():
            print("❌ Error: MT5 es REQUERIDO")
            sys.exit(1)
        
        # Datos iniciales
        if not self._download_initial_data():
            print("❌ Error obteniendo datos")
            sys.exit(1)
    
    def _model_name_from_selected_model(self):
        """Obtener nombre del modelo"""
        names = {
            'dqn': 'DQN (Deep Q-Network)',
            'deepdqn': 'DeepDQN (Deep DQN)', 
            'ppo': 'PPO (Proximal Policy)',
            'a2c': 'A2C (Advantage Actor-Critic)',
            'all': 'TODOS los modelos',
            'technical': 'Análisis Técnico'
        }
        return names.get(self.selected_model, 'Desconocido')
    
    def _load_models(self):
        """Cargar modelos ML"""
        if self.selected_model == 'technical':
            return
            
        model_config = {
            'dqn': {'class': DQN, 'paths': ["data/models/qdn/model.zip", "data/models/best_qdn/model.zip"]},
            'deepdqn': {'class': DQN, 'paths': ["data/models/deepqdn/model.zip", "data/models/best_deepqdn/model.zip"]},
            'ppo': {'class': PPO, 'paths': ["data/models/ppo/model.zip", "data/models/best_ppo/best_model.zip"]},
            'a2c': {'class': A2C, 'paths': ["data/models/a2c/model.zip", "data/models/best_a2c/model.zip"]}
        }
        
        models_to_load = []
        if self.selected_model == 'all':
            models_to_load = ['dqn', 'deepdqn', 'ppo', 'a2c']
        elif self.selected_model in model_config:
            models_to_load = [self.selected_model]
        
        for model_key in models_to_load:
            config = model_config[model_key]
            for path in config['paths']:
                try:
                    if os.path.exists(path):
                        model = config['class'].load(path, device='cpu')
                        self.models[model_key] = model
                        print(f"✅ {model_key.upper()} cargado")
                        break
                except Exception as e:
                    continue
    
    def connect_mt5(self):
        """Conectar a MT5"""
        try:
            if not mt5.initialize():
                return False
                
            account_info = mt5.account_info()
            if account_info is None:
                return False
                
            print(f"✅ MT5 conectado - Cuenta: {account_info.login}")
            
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                alternative_symbols = ["US30", "USTEC", "SPX500"]
                for alt_symbol in alternative_symbols:
                    symbol_info = mt5.symbol_info(alt_symbol)
                    if symbol_info is not None:
                        self.symbol = alt_symbol
                        break
                else:
                    return False
            
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    return False
            
            print(f"✅ Símbolo: {self.symbol}")
            self.mt5_connected = True
            return True
            
        except Exception as e:
            return False
    
    def _download_initial_data(self):
        """Descargar datos iniciales"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 100)
            if rates is None or len(rates) == 0:
                return False
            
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df['price'] = df['close']
            df['volume'] = df['tick_volume']
            
            # Calcular indicadores
            df = self._calculate_indicators(df)
            self.data = df[['timestamp', 'price', 'volume', 'rsi', 'macd']].copy()
            
            print(f"✅ {len(rates)} datos cargados")
            return True
            
        except Exception as e:
            return False
    
    def _calculate_indicators(self, df):
        """Calcular indicadores técnicos"""
        try:
            # RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.inf)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['price'].ewm(span=12).mean()
            ema_26 = df['price'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            
            df['rsi'] = df['rsi'].fillna(50)
            df['macd'] = df['macd'].fillna(0)
            
            return df
        except:
            df['rsi'] = 50
            df['macd'] = 0
            return df
    
    def get_latest_data(self):
        """Obtener datos en tiempo real"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 1)
            if rates is None or len(rates) == 0:
                return None
            
            rate = rates[0]
            return {
                'timestamp': datetime.now(),
                'price': float(rate['close']),
                'volume': float(rate['tick_volume'])
            }
        except:
            return None
    
    def calculate_indicators(self, data_point):
        """Calcular indicadores para punto actual"""
        if len(self.data) == 0:
            return {'rsi': 50.0, 'macd': 0.0}
        
        recent_prices = self.data['price'].tail(20).tolist()
        recent_prices.append(data_point['price'])
        
        # RSI simplificado
        deltas = np.diff(recent_prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # MACD simplificado
        if len(recent_prices) >= 26:
            ema_12 = pd.Series(recent_prices).ewm(span=12).mean().iloc[-1]
            ema_26 = pd.Series(recent_prices).ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
        else:
            macd = 0.0
        
        return {'rsi': rsi, 'macd': macd}
    
    def analyze_signals(self, data_point, indicators):
        """Analizar señales"""
        try:
            technical_signal = self._calculate_technical_signal(indicators)
            ml_signal = 0.0
            
            if self.selected_model == "technical":
                selected_signal = technical_signal
                model_name = "Técnico"
            elif self.selected_model in self.models and self.models[self.selected_model]:
                ml_signal = self._get_model_prediction(self.selected_model, data_point, indicators)
                selected_signal = ml_signal
                model_name = self.selected_model.upper()
            else:
                selected_signal = technical_signal
                model_name = "Técnico"
            
            return {
                'ml_signal': ml_signal,
                'technical_signal': technical_signal,
                'selected_signal': selected_signal,
                'model_name': model_name
            }
        except:
            return {'ml_signal': 0.0, 'technical_signal': 0.0, 'selected_signal': 0.0, 'model_name': 'Error'}
    
    def _calculate_technical_signal(self, indicators):
        """Calcular señal técnica"""
        try:
            rsi = indicators['rsi']
            macd = indicators['macd']
            signal = 0.0
            
            if rsi < 30:
                signal += 0.5
            elif rsi < 40:
                signal += 0.2
            elif rsi > 70:
                signal -= 0.5
            elif rsi > 60:
                signal -= 0.2
            
            if macd > 0:
                signal += 0.3
            else:
                signal -= 0.3
            
            return max(-1.0, min(1.0, signal))
        except:
            return 0.0
    
    def _get_model_prediction(self, model_type, data_point, indicators):
        """Obtener predicción del modelo"""
        try:
            model = self.models[model_type]
            if model is None:
                return 0.0
            
            state = np.array([
                (data_point['price'] - 6000) / 6000,  # Normalizado
                data_point['volume'] / 1000,  # Normalizado
                (indicators['rsi'] - 50) / 50,
                indicators['macd'] / data_point['price'] if data_point['price'] > 0 else 0
            ], dtype=np.float32)
            
            action = model.predict(state.reshape(1, -1), deterministic=True)[0]
            signal = 0.8 if action == 1 else -0.8
            
            rsi_factor = (indicators['rsi'] - 50) / 100
            signal += rsi_factor * 0.2
            
            return max(-1.0, min(1.0, signal))
        except:
            return 0.0
    
    def calculate_trade_pnl(self, entry_price, exit_price, trade_type, size_usd):
        """Calcular P&L"""
        try:
            units = size_usd / entry_price
            
            if trade_type == 'BUY':
                price_diff = exit_price - entry_price
                pnl_absolute = units * price_diff
                pnl_percentage = ((exit_price - entry_price) / entry_price) * 100
            else:
                price_diff = entry_price - exit_price
                pnl_absolute = units * price_diff
                pnl_percentage = ((entry_price - exit_price) / entry_price) * 100
            
            return pnl_absolute, pnl_percentage
        except:
            return 0.0, 0.0
    
    def update_capital(self, pnl_absolute):
        """Actualizar capital"""
        self.current_capital += pnl_absolute
        self.total_profit_loss += pnl_absolute
        self.total_profit_loss_pct = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        self.last_trade_pnl = pnl_absolute
    
    def execute_trading_logic(self, data_point, indicators, signals):
        """Ejecutar lógica de trading"""
        selected_signal = signals['selected_signal']
        model_name = signals['model_name']
        price = data_point['price']
        timestamp = data_point['timestamp']
        
        # Verificar cooldown
        current_time = datetime.now()
        if (self.last_trade_time and 
            (current_time - self.last_trade_time).total_seconds() < self.trade_cooldown):
            return
        
        # Señal de compra
        if selected_signal > 0.5:
            can_buy = (len(self.trade_manager.open_trades) == 0 and 
                      (self.current_position != 'LONG' or self.current_position is None))
            
            if can_buy:
                trade_id = self.trade_manager.open_trade(
                    symbol=self.symbol,
                    trade_type='BUY',
                    size=self.trade_size_usd / price,
                    entry_price=price,
                    entry_time=timestamp,
                    ml_signal=signals.get('ml_signal', selected_signal),
                    technical_signal=signals['technical_signal'],
                    combined_signal=selected_signal,
                    rsi=indicators['rsi'],
                    macd=indicators['macd'],
                    volume=data_point['volume'],
                    portfolio_value=self.current_capital
                )
                
                print(f"\n{'='*60}")
                print(f"[BUY] COMPRA EJECUTADA")
                print(f"{'='*60}")
                print(f"Hora: {timestamp.strftime('%H:%M:%S')}")
                print(f"Modelo: {model_name}")
                print(f"Precio: ${price:.2f}")
                print(f"Señal: {selected_signal:.2f}")
                print(f"RSI: {indicators['rsi']:.1f}")
                print(f"Trade ID: {trade_id}")
                print(f"{'='*60}")
                
                self.last_trade_time = current_time
                self.current_position = 'LONG'
                self.last_operation_type = 'BUY'
        
        # Señal de venta
        elif selected_signal < -0.5:
            can_sell = (len(self.trade_manager.open_trades) == 0 and 
                       (self.current_position != 'SHORT' or self.current_position is None))
            
            if can_sell:
                trade_id = self.trade_manager.open_trade(
                    symbol=self.symbol,
                    trade_type='SELL',
                    size=self.trade_size_usd / price,
                    entry_price=price,
                    entry_time=timestamp,
                    ml_signal=signals.get('ml_signal', selected_signal),
                    technical_signal=signals['technical_signal'],
                    combined_signal=selected_signal,
                    rsi=indicators['rsi'],
                    macd=indicators['macd'],
                    volume=data_point['volume'],
                    portfolio_value=self.current_capital
                )
                
                print(f"\n{'='*60}")
                print(f"[SELL] VENTA EJECUTADA")
                print(f"{'='*60}")
                print(f"Hora: {timestamp.strftime('%H:%M:%S')}")
                print(f"Modelo: {model_name}")
                print(f"Precio: ${price:.2f}")
                print(f"Señal: {selected_signal:.2f}")
                print(f"RSI: {indicators['rsi']:.1f}")
                print(f"Trade ID: {trade_id}")
                print(f"{'='*60}")
                
                self.last_trade_time = current_time
                self.current_position = 'SHORT'
                self.last_operation_type = 'SELL'
        
        # Cerrar trades con ganancia/pérdida
        trades_to_close = []
        for trade_id, trade_data in self.trade_manager.open_trades.items():
            entry_price = trade_data['entry_price']
            trade_type = trade_data['trade_type']
            
            if trade_type == 'BUY':
                return_pct = ((price - entry_price) / entry_price) * 100
            else:
                return_pct = ((entry_price - price) / entry_price) * 100
            
            if return_pct > 1.0 or return_pct < -0.5:
                trades_to_close.append(trade_id)
        
        # Ejecutar cierres
        for trade_id in trades_to_close:
            trade_data = self.trade_manager.open_trades.get(trade_id)
            if trade_data:
                trade_type = trade_data['trade_type']
                entry_price = trade_data['entry_price']
                
                pnl_absolute, pnl_percentage = self.calculate_trade_pnl(
                    entry_price, price, trade_type, self.trade_size_usd
                )
                
                self.update_capital(pnl_absolute)
                
            self.trade_manager.close_trade(trade_id, price, timestamp)
            
            if trade_data:
                self.current_position = None
                profit_symbol = "[PROFIT]" if pnl_absolute > 0 else "[LOSS]"
                
                print(f"\n{'='*60}")
                print(f"[CLOSE] TRADE CERRADO")
                print(f"{'='*60}")
                print(f"Hora: {timestamp.strftime('%H:%M:%S')}")
                print(f"Trade ID: {trade_id}")
                print(f"Tipo: {trade_type}")
                print(f"Precio Entrada: ${entry_price:.2f}")
                print(f"Precio Salida: ${price:.2f}")
                print(f"{profit_symbol} P&L: ${pnl_absolute:.2f} ({pnl_percentage:+.2f}%)")
                print(f"[CAPITAL] Capital actual: ${self.current_capital:.2f}")
                print(f"{'='*60}")
    
    def start_real_time(self):
        """Iniciar sistema en tiempo real"""
        if self.is_real_time:
            return
        
        print("🚀 Iniciando sistema en tiempo real...")
        print(f"🤖 Modelo: {self.model_name}")
        
        self.is_real_time = True
        self.is_running = True
        
        self.real_time_thread = threading.Thread(target=self._real_time_loop, daemon=True)
        self.real_time_thread.start()
        
        print("✅ Sistema iniciado")
    
    def stop_real_time(self):
        """Detener sistema"""
        if not self.is_real_time:
            return
            
        print("🛑 Deteniendo sistema...")
        self.is_real_time = False
        self.is_running = False
        
        if self.real_time_thread and self.real_time_thread.is_alive():
            self.real_time_thread.join(timeout=5)
        
        # Cerrar trades abiertos
        current_time = datetime.now()
        for trade_id in list(self.trade_manager.open_trades.keys()):
            last_price = getattr(self, 'last_price', 6000)
            self.trade_manager.close_trade(trade_id, last_price, current_time, 'SYSTEM_STOP')
        
        mt5.shutdown()
        print("✅ Sistema detenido")
    
    def _real_time_loop(self):
        """Loop principal en tiempo real"""
        print("🔄 Loop iniciado...")
        
        while self.is_running:
            try:
                data_point = self.get_latest_data()
                if data_point is None:
                    print("⚠️ Sin datos, reintentando...")
                    time.sleep(5)
                    continue
                
                indicators = self.calculate_indicators(data_point)
                signals = self.analyze_signals(data_point, indicators)
                self.execute_trading_logic(data_point, indicators, signals)
                
                # Agregar datos
                new_row = pd.DataFrame([{
                    'timestamp': data_point['timestamp'],
                    'price': data_point['price'],
                    'volume': data_point['volume'],
                    'rsi': indicators['rsi'],
                    'macd': indicators['macd']
                }])
                
                self.data = pd.concat([self.data, new_row], ignore_index=True)
                
                if len(self.data) > 200:
                    self.data = self.data.tail(200).reset_index(drop=True)
                
                # Status cada 30 segundos
                current_time = datetime.now()
                if not hasattr(self, '_last_status') or (current_time - self._last_status).total_seconds() >= 30:
                    self._print_status(data_point, indicators, signals)
                    self._last_status = current_time
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"❌ Error en loop: {e}")
                time.sleep(5)
    
    def _print_status(self, data_point, indicators, signals):
        """Imprimir status del sistema"""
        timestamp = data_point['timestamp'].strftime('%H:%M:%S')
        price = data_point['price']
        signal = signals['selected_signal']
        model = signals['model_name']
        
        # Símbolo de estado financiero
        if self.total_profit_loss > 0:
            pnl_symbol = "[+]"
        elif self.total_profit_loss < 0:
            pnl_symbol = "[-]"
        else:
            pnl_symbol = "[=]"
        
        print(f"\n{'='*80}")
        print(f"SISTEMA DE TRADING EN TIEMPO REAL - {timestamp}")
        print(f"CAPITAL: ${self.current_capital:,.2f} | {pnl_symbol} P&L: ${self.total_profit_loss:+,.2f} ({self.total_profit_loss_pct:+.2f}%)")
        print(f"{'='*80}")
        print(f"Precio: ${price:.2f} | RSI: {indicators['rsi']:.1f} | MACD: {indicators['macd']:.3f}")
        print(f"Modelo: {model} | Señal: {signal:.2f}")
        print(f"Posición: {self.current_position or 'NEUTRAL'} | Trades abiertos: {len(self.trade_manager.open_trades)}")
        print(f"Trades totales: {len(self.trade_manager.trades)}")
        print(f"{'='*80}")

def main():
    """Función principal"""
    try:
        print("🚀 SISTEMA DE TRADING EN TIEMPO REAL - MODO CONSOLA")
        print("✅ MetaTrader5 disponible" if HAS_MT5 else "❌ MetaTrader5 NO disponible")
        
        if not HAS_MT5:
            print("❌ ERROR: MetaTrader5 es REQUERIDO")
            sys.exit(1)
        
        # Menú de modelos
        print("\n🤖 Selecciona el modelo:")
        print("1. DQN (Deep Q-Network)")
        print("2. DeepDQN (Deep DQN)")
        print("3. PPO (Proximal Policy)")
        print("4. A2C (Advantage Actor-Critic)")
        print("5. TODOS los modelos")
        print("6. Solo Análisis Técnico")
        
        while True:
            try:
                choice = input("\nSelecciona (1-6): ").strip()
                if choice in ['1', '2', '3', '4', '5', '6']:
                    break
                else:
                    print("❌ Opción inválida")
            except KeyboardInterrupt:
                print("\n👋 Saliendo...")
                return
        
        # Mapear elección a modelo
        model_map = {
            '1': 'dqn',
            '2': 'deepdqn', 
            '3': 'ppo',
            '4': 'a2c',
            '5': 'all',
            '6': 'technical'
        }
        
        selected_model = model_map[choice]
        print(f"\n🎯 Modelo seleccionado: {selected_model}")
        
        # Crear sistema
        system = ConsoleTradingSystem(selected_model=selected_model)
        
        # Iniciar automáticamente
        system.start_real_time()
        
        print("\n" + "="*60)
        print("🤖 SISTEMA FUNCIONANDO EN MODO CONSOLA")
        print("="*60)
        print("📊 El sistema opera automáticamente")
        print("📈 Operaciones aparecerán en tiempo real")
        print("📁 Trades se guardan automáticamente en CSV")
        print("\n🎯 COMANDOS:")
        print("  'stop'     - Detener tiempo real")
        print("  'start'    - Reiniciar tiempo real")
        print("  'status'   - Ver estado actual")
        print("  'quit'     - Salir")
        print("="*60)
        
        # Loop de comandos
        while True:
            try:
                command = input("\n>>> ").strip().lower()
                
                if command == 'start':
                    system.start_real_time()
                elif command == 'stop':
                    system.stop_real_time()
                elif command == 'status':
                    print(f"\n📊 ESTADO:")
                    print(f"  Modelo: {system.model_name}")
                    print(f"  Sistema: {'ACTIVO' if system.is_real_time else 'DETENIDO'}")
                    print(f"  MT5: {'CONECTADO' if system.mt5_connected else 'DESCONECTADO'}")
                    print(f"  Capital: ${system.current_capital:.2f}")
                    print(f"  P&L: ${system.total_profit_loss:+.2f} ({system.total_profit_loss_pct:+.2f}%)")
                    print(f"  Posición: {system.current_position or 'NEUTRAL'}")
                    print(f"  Trades: {len(system.trade_manager.trades)} total, {len(system.trade_manager.open_trades)} abiertos")
                elif command == 'quit' or command == 'exit':
                    break
                else:
                    print("❌ Comando no reconocido: start, stop, status, quit")
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n🛑 Deteniendo...")
                break
        
        # Limpiar
        try:
            system.stop_real_time()
            print("✅ Sistema finalizado")
        except:
            pass
        
    except KeyboardInterrupt:
        print("\n👋 Saliendo...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 