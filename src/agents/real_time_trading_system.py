#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SISTEMA DE TRADING EN TIEMPO REAL COMPLETO - SOLO MT5
- Escalas de tiempo reales (no steps)
- Datos reales SOLO de MT5 
- CSV export automático con IDs únicos
- Dashboard live con fechas
- Continuidad perfecta
"""

import matplotlib
matplotlib.use('TkAgg')  # Usar TkAgg como backend

import sys
import os
import uuid
import csv
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from matplotlib.gridspec import GridSpec
import threading
import time
import warnings
import queue
import collections
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

# Importar sistema base
try:
    from src.agents.ml_enhanced_system import MLEnhancedTradingSystem
    print("✅ Sistema base importado")
except ImportError:
    print("⚠️ Sistema base no encontrado - usando sistema simplificado")

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
        
        print(f"🟢 TRADE ABIERTO")
        print(f"   ID: {trade_id}")
        print(f"   Tipo: {trade_type}")
        print(f"   Precio: ${entry_price:.4f}")
        print(f"   Tamaño: {size}")
        print(f"   Tiempo: {entry_time}")
        
        return trade_id
    
    def close_trade(self, trade_id, exit_price, exit_time, exit_reason='MANUAL'):
        """Cerrar trade y actualizar CSV"""
        
        if trade_id not in self.open_trades:
            print(f"❌ Trade {trade_id} no encontrado")
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
        
        print(f"🔴 TRADE CERRADO")
        print(f"   ID: {trade_id}")
        print(f"   Precio Exit: ${exit_price:.4f}")
        print(f"   Retorno: {return_pct:.2f}%")
        print(f"   Duración: {duration:.1f} min")
        print(f"   Razón: {exit_reason}")
        
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

class SimpleBaseSystem:
    """Sistema base simplificado sin menú"""
    def __init__(self):
        self.data = pd.DataFrame(columns=['timestamp', 'price', 'volume', 'rsi', 'macd'])
        self.ml_model = None
        self.current_capital = 100000

class RealTimeTradingSystem:
    """Sistema de trading en tiempo real mejorado"""
    
    def __init__(self, selected_model=None):
        """Inicializar sistema COMPLETO"""
        self.symbol = "US500"  # S&P 500
        self.timeframe = mt5.TIMEFRAME_M1  # 1 minuto
        self.update_interval = 2  # segundos
        self.is_real_time = False
        self.mt5_connected = False
        
        # ✅ NUEVO: Cola thread-safe para datos
        self.data_queue = queue.Queue(maxsize=1000)
        self.update_lock = threading.Lock()
        
        # ✅ NUEVO: Buffer circular para datos eficiente
        self.max_data_points = 500
        self.data_buffer = collections.deque(maxlen=self.max_data_points)
        
        # ✅ NUEVO: Líneas de matplotlib reutilizables (no recrear cada vez)
        self.plot_lines = {}
        self.last_update_time = time.time()
        self.min_update_interval = 1.0  # Mínimo 1 segundo entre actualizaciones de gráfico
        
        # Seleccionar modelo ANTES de cargar
        if selected_model:
            self.selected_model = selected_model
            self.selected_model_type = selected_model  
            self.model_name = self._model_name_from_selected_model()
        else:
            self.selected_model = None
            self.selected_model_type = None
            self.model_name = ""
        
        # Usar sistema base simplificado SIN menú
        self.base_system = SimpleBaseSystem()
        
        self.models = {
            'dqn': None,
            'deepdqn': None,
            'a2c': None,
            'ppo': None
        }
        
        # Tracking de señales para triángulos - COMO comparison_four_models.py
        self.buy_signals = []  # Lista de índices donde ocurrieron compras
        self.sell_signals = []  # Lista de índices donde ocurrieron ventas
        self.signal_index = 0  # Índice actual para el tracking
        
        # NUEVO: Tracking con timestamps para fijar triángulos
        self.buy_timestamps = []  # Lista de timestamps de compras
        self.sell_timestamps = []  # Lista de timestamps de ventas
        self.buy_prices = []  # Lista de precios de compras
        self.sell_prices = []  # Lista de precios de ventas
        
        # VENTANA DESLIZANTE - Para evitar recortar el gráfico
        self.window_size = 100  # Mantener últimos 100 puntos visibles
        self.display_window_size = 200  # Ventana más grande para visualización
        self.all_data = []  # Todos los datos históricos
        
        # Cargar todos los modelos disponibles
        self._load_all_models()
        
        self.trade_manager = RealTimeTradeManager()
        
        # Configuración de visualización
        plt.style.use('dark_background')
        self.fig = None
        self.axes = {}
        
        # Estado inicial
        self.prev_price = None
        self.last_trade_time = None
        self.trade_cooldown = 60  # segundos entre trades
        self.dashboard_mode = 'visual'  # 'visual' o 'console'
        
        # ✅ CONTROL DE POSICIONES - Evitar operaciones ilógicas
        self.current_position = None  # 'LONG', 'SHORT', o None
        self.last_operation_type = None  # 'BUY', 'SELL', o None
        self.position_history = []  # Historial de posiciones
        
        # ✅ CONTROL FINANCIERO EN TIEMPO REAL
        self.initial_capital = 100000.0  # Capital inicial
        self.current_capital = 100000.0  # Capital actual
        self.total_profit_loss = 0.0     # Total P&L
        self.total_profit_loss_pct = 0.0 # Total P&L en porcentaje
        self.last_trade_pnl = 0.0        # P&L del último trade
        self.last_trade_pnl_pct = 0.0    # P&L del último trade en %
        self.trade_size_usd = 1000.0     # Tamaño de cada trade en USD
        
        # ✅ CONECTAR MT5 Y OBTENER DATOS INICIALES
        if not self.connect_mt5():
            print("❌ Error: MT5 es REQUERIDO para este sistema")
            sys.exit(1)
        
        # Descargar datos iniciales para ventana (más datos para continuidad)
        if not self._download_initial_mt5_data():
            print("❌ Error obteniendo datos iniciales de MT5")
            sys.exit(1)
    
    def connect_mt5(self):
        """Conectar a MT5 y configurar símbolo con reconexión robusta"""
        try:
            # ✅ CERRAR CONEXIÓN ANTERIOR SI EXISTE
            try:
                mt5.shutdown()
            except:
                pass
            
            # ✅ INICIALIZAR MT5
            if not mt5.initialize():
                print("❌ Error inicializando MT5 - Verifica que MT5 esté abierto")
                self.mt5_connected = False
                return False
                
            account_info = mt5.account_info()
            if account_info is None:
                print("❌ Error obteniendo información de cuenta MT5")
                self.mt5_connected = False
                return False
                
            print(f"✅ MT5 conectado - Cuenta: {account_info.login}")
            
            # ✅ CONFIGURAR SÍMBOLO
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                print(f"❌ {self.symbol} no encontrado en MT5, probando alternativas...")
                # Intentar con símbolos alternativos
                alternative_symbols = ["US30", "USTEC", "SPX500", "US500Cash", "USTECH", "SP500"]
                for alt_symbol in alternative_symbols:
                    symbol_info = mt5.symbol_info(alt_symbol)
                    if symbol_info is not None:
                        print(f"✅ Usando símbolo alternativo: {alt_symbol}")
                        self.symbol = alt_symbol
                        break
                else:
                    print("❌ No se encontraron símbolos disponibles")
                    print("   Símbolos intentados:", ["US500"] + alternative_symbols)
                    self.mt5_connected = False
                    return False
            
            # ✅ HABILITAR SÍMBOLO SI ES NECESARIO
            if not symbol_info.visible:
                print(f"⚠️ {self.symbol} no visible, intentando habilitar...")
                if not mt5.symbol_select(self.symbol, True):
                    print(f"❌ Error habilitando {self.symbol}")
                    self.mt5_connected = False
                    return False
            
            # ✅ VERIFICAR QUE PODEMOS OBTENER DATOS
            test_rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 1)
            if test_rates is None or len(test_rates) == 0:
                print(f"❌ No se pueden obtener datos de {self.symbol}")
                self.mt5_connected = False
                return False
            
            print(f"✅ Símbolo configurado y funcionando: {self.symbol}")
            print(f"📊 Último precio: ${test_rates[0]['close']:.2f}")
            
            self.mt5_connected = True
            return True
            
        except Exception as e:
            print(f"❌ Error conectando MT5: {e}")
            self.mt5_connected = False
            return False
    
    def _download_initial_mt5_data(self):
        """Descargar datos iniciales de MT5 para la ventana - MÁS DATOS"""
        print("📊 Descargando datos iniciales de MT5...")
        
        try:
            # Obtener últimos 200 puntos para tener más datos iniciales
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 200)
            
            if rates is None or len(rates) == 0:
                print(f"❌ Error obteniendo datos de {self.symbol}")
                return False
            
            print(f"✅ Descargados {len(rates)} datos de MT5")
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            
            # Usar close como precio principal
            df['price'] = df['close']
            df['volume'] = df['tick_volume']
            
            # Calcular indicadores técnicos
            df = self._calculate_technical_indicators(df)
            
            # ✅ GUARDAR MÁS DATOS para mejor continuidad
            self.base_system.data = df[['timestamp', 'price', 'volume', 'rsi', 'macd']].copy()
            
            print(f"📈 Rango de precios: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
            print(f"📈 Último precio: ${df['price'].iloc[-1]:.2f}")
            print(f"📅 Desde: {df['timestamp'].iloc[0].strftime('%H:%M:%S')}")
            print(f"📅 Hasta: {df['timestamp'].iloc[-1].strftime('%H:%M:%S')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error descargando datos MT5: {e}")
            return False

    def _model_name_from_selected_model(self):
        """Obtener nombre del modelo seleccionado"""
        names = {
            'dqn': 'DQN (Deep Q-Network) - AGRESIVO',
            'deepdqn': 'DeepDQN (Deep DQN) - PRECISO',
            'ppo': 'PPO (Proximal Policy Optimization) - BALANCEADO',
            'a2c': 'A2C (Advantage Actor-Critic) - CONSERVADOR',
            'all': 'TODOS los 4 modelos (DQN+DeepDQN+PPO+A2C)',
            'technical': 'Análisis Técnico'
        }
        return names.get(self.selected_model, 'Desconocido')

    def _load_all_models(self):
        """Cargar todos los modelos disponibles - EXACTAMENTE COMO comparison_four_models.py"""
        print("\n🤖 Cargando modelos ML con configuración idéntica a comparison_four_models.py...")
        
        # Si solo queremos análisis técnico, no cargar modelos
        if self.selected_model == 'technical':
            print("📊 Usando solo análisis técnico")
            return
        
        # Configuración EXACTA de comparison_four_models.py
        model_config = {
            'dqn': {
                'class': DQN,
                'paths': [
                    "data/models/qdn/model.zip",
                    "data/models/best_qdn/model.zip"
                ],
                'action_space': 'discrete',
                'obs_space': 4,
                'name': 'DQN'
            },
            'deepdqn': {
                'class': DQN,
                'paths': [
                    "data/models/deepqdn/model.zip",
                    "data/models/best_deepqdn/model.zip"
                ],
                'action_space': 'discrete',
                'obs_space': 4,
                'name': 'DeepDQN'
            },
            'ppo': {
                'class': PPO,
                'paths': [
                    "data/models/ppo/model.zip",
                    "data/models/best_ppo/best_model.zip"
                ],
                'action_space': 'discrete',
                'obs_space': 4,
                'name': 'PPO'
            },
            'a2c': {
                'class': A2C,
                'paths': [
                    "data/models/a2c/model.zip",
                    "data/models/best_a2c/model.zip"
                ],
                'action_space': 'discrete',
                'obs_space': 4,
                'name': 'A2C'
            }
        }
        
        # Determinar qué modelos cargar
        models_to_load = []
        if self.selected_model == 'all':
            models_to_load = ['dqn', 'deepdqn', 'ppo', 'a2c']
        elif self.selected_model in model_config:
            models_to_load = [self.selected_model]
        elif self.selected_model == 'dqn':  # Mapear 'dqn' a 'dqn'
            models_to_load = ['dqn']
        elif self.selected_model == 'sac':  # Si piden SAC, usar DeepDQN en su lugar
            print("⚠️ SAC no disponible, usando DeepDQN en su lugar")
            models_to_load = ['deepdqn']
        
        # Cargar cada modelo
        for model_key in models_to_load:
            config = model_config[model_key]
            model_loaded = False
            
            print(f"\n🤖 Cargando modelo {config['name']}...")
            
            # Intentar cargar cada path disponible
            for path in config['paths']:
                try:
                    print(f"🔄 Intentando: {path}")
                    
                    if not os.path.exists(path):
                        print(f"⚠️ Archivo no encontrado: {path}")
                        continue
                    
                    # Cargar modelo directamente (sin environment)
                    model = config['class'].load(path, device='cpu')
                    
                    self.models[model_key] = model
                    model_loaded = True
                    
                    print(f"✅ {config['name']} cargado exitosamente desde {path}")
                    break
                    
                except Exception as e:
                    print(f"❌ Error cargando {path}: {e}")
                    continue
            
            if not model_loaded:
                print(f"❌ No se pudo cargar {config['name']}")
                self.models[model_key] = None
        
        # Mostrar resumen
        loaded_models = [k for k, v in self.models.items() if v is not None]
        print(f"\n✅ Modelos cargados: {loaded_models}")
        
        if not loaded_models:
            print("⚠️ Ningún modelo cargado, usando solo análisis técnico")

    def get_latest_data(self):
        """✅ CORREGIDO: Obtener datos REALES EN TIEMPO REAL de MT5 con reconexión automática"""
        
        # ✅ VERIFICAR Y RECONECTAR MT5 SI ES NECESARIO
        if not self.mt5_connected:
            print("🔄 MT5 desconectado, reconectando...")
            if not self.connect_mt5():
                print("❌ Error reconectando MT5")
                return None
        
        try:
            # ✅ OBTENER DATO MÁS RECIENTE DE MT5
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 1)
            
            if rates is None or len(rates) == 0:
                print(f"⚠️ Sin datos de {self.symbol}, intentando reconectar...")
                # Intentar reconectar una vez más
                if self.connect_mt5():
                    rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 1)
                
                if rates is None or len(rates) == 0:
                    print(f"❌ No se pudieron obtener datos de {self.symbol}")
                    return None
            
            rate = rates[0]
            current_time = datetime.now()  # Usar tiempo actual
            
            data_point = {
                'timestamp': current_time,
                'price': float(rate['close']),
                'open': float(rate['open']),
                'high': float(rate['high']),
                'low': float(rate['low']),
                'volume': float(rate['tick_volume'])
            }
            
            # Log del nuevo dato con información de ventana
            data_count = len(self.base_system.data) if self.base_system.data is not None else 0
            print(f"📡 MT5: {current_time.strftime('%H:%M:%S')} | ${data_point['price']:.2f} | Datos: {data_count}")
            
            return data_point
            
        except Exception as e:
            print(f"❌ Error obteniendo datos MT5: {e}")
            # Marcar como desconectado para intentar reconectar en la siguiente iteración
            self.mt5_connected = False
            return None

    def calculate_indicators(self, data_point):
        """Calcular indicadores para el punto actual"""
        # Si no tenemos datos base, usar valores por defecto
        if self.base_system.data is None or len(self.base_system.data) == 0:
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'sma_20': data_point['price'],
                'volume_sma': data_point['volume']
            }
        
        # Calcular indicadores basados en historia
        recent_prices = self.base_system.data['price'].tail(20).tolist()
        recent_prices.append(data_point['price'])
        
        recent_volumes = self.base_system.data['volume'].tail(20).tolist()
        recent_volumes.append(data_point['volume'])
        
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
        
        # SMA
        sma_20 = np.mean(recent_prices[-20:]) if len(recent_prices) >= 20 else data_point['price']
        volume_sma = np.mean(recent_volumes[-20:]) if len(recent_volumes) >= 20 else data_point['volume']
        
        return {
            'rsi': rsi,
            'macd': macd,
            'sma_20': sma_20,
            'volume_sma': volume_sma
        }
    
    def analyze_signals(self, data_point, indicators):
        """Analizar señales de ML y técnicas"""
        try:
            # Inicializar valores por defecto
            ml_signal = 0.0
            technical_signal = self._calculate_technical_signal(indicators)
            
            # Determinar modelo y señal según el tipo seleccionado
            if self.selected_model_type == "technical":
                # Solo análisis técnico
                selected_signal = technical_signal
                model_name = "Análisis Técnico"
                
            elif self.selected_model_type == "all":
                # Usar todos los modelos con pesos iguales
                weighted_signals = []
                total_weight = 0
                
                # Pesos para cada modelo (igual que comparison_four_models.py)
                weights = {
                    'dqn': 0.3,
                    'deepdqn': 0.2,
                    'ppo': 0.25,
                    'a2c': 0.25
                }
                
                # Evaluar cada modelo
                for model_type, model in self.models.items():
                    if model is not None:
                        try:
                            model_signal = self._get_model_prediction(model_type, data_point, indicators)
                            weight = weights.get(model_type, 0.25)
                            weighted_signals.append(model_signal * weight)
                            total_weight += weight
                        except Exception as e:
                            print(f"⚠️ Error en modelo {model_type}: {e}")
                
                # Combinar señales ML
                if total_weight > 0:
                    ml_signal = sum(weighted_signals) / total_weight
                
                # Combinar ML y técnico
                ml_weight = 0.8
                tech_weight = 0.2
                selected_signal = (ml_signal * ml_weight) + (technical_signal * tech_weight)
                model_name = "Combinado (ML + Técnico)"
                
            else:
                # Usar modelo individual seleccionado
                if (self.selected_model_type in self.models and 
                    self.models[self.selected_model_type] is not None):
                    ml_signal = self._get_model_prediction(self.selected_model_type, data_point, indicators)
                    selected_signal = ml_signal
                    model_name = self.selected_model_type.upper()
                    
                    # Mostrar predicción del modelo individual (solo si es significativa)
                    if abs(ml_signal) > 0.3:
                        prediction_type = "FUERTE COMPRA" if ml_signal > 0.5 else "FUERTE VENTA" if ml_signal < -0.5 else "NEUTRAL"
                        signal_color = "🟢" if ml_signal > 0.5 else "🔴" if ml_signal < -0.5 else "🟡"
                        print(f"{signal_color} {model_name} predice: {prediction_type} (Señal: {ml_signal:.3f})")
                else:
                    # Fallback si el modelo no está disponible
                    selected_signal = technical_signal
                    model_name = "Análisis Técnico (Fallback)"
            
            return {
                'ml_signal': ml_signal,
                'technical_signal': technical_signal,
                'selected_signal': selected_signal,
                'model_name': model_name
            }
            
        except Exception as e:
            print(f"❌ Error en analyze_signals: {e}")
            return {
                'ml_signal': 0.0,
                'technical_signal': 0.0,
                'selected_signal': 0.0,
                'model_name': "Error"
            }
    
    def _calculate_technical_signal(self, indicators):
        """Calcular señal basada en análisis técnico"""
        try:
            rsi = indicators['rsi']
            macd = indicators['macd']
            
            signal = 0.0
            
            # RSI: Sobrecompra/sobreventa
            if rsi < 30:
                signal += 0.5  # Señal de compra fuerte
            elif rsi < 40:
                signal += 0.2  # Señal de compra débil
            elif rsi > 70:
                signal -= 0.5  # Señal de venta fuerte
            elif rsi > 60:
                signal -= 0.2  # Señal de venta débil
            
            # MACD: Momentum
            if macd > 0:
                signal += 0.3  # Tendencia alcista
            else:
                signal -= 0.3  # Tendencia bajista
            
            # Normalizar entre -1 y 1
            signal = max(-1.0, min(1.0, signal))
            
            return signal
            
        except Exception as e:
            print(f"⚠️ Error en análisis técnico: {e}")
            return 0.0
    
    def _get_model_prediction(self, model_type, data_point, indicators):
        """Obtener predicción de un modelo específico - COMPATIBLE CON comparison_four_models.py"""
        try:
            model = self.models[model_type]
            
            if model is None:
                return 0.0
            
            # TODOS los modelos usan estado de 4 dimensiones y espacios discretos
            state = np.array([
                (data_point['price'] - indicators['sma_20']) / indicators['sma_20'] if indicators['sma_20'] > 0 else 0,
                data_point['volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1,
                (indicators['rsi'] - 50) / 50,
                indicators['macd'] / data_point['price'] if data_point['price'] > 0 else 0
            ], dtype=np.float32)
            
            # Obtener acción discreta y convertir a señal continua
            action = model.predict(state.reshape(1, -1), deterministic=True)[0]
            
            # Convertir acción discreta (0 o 1) a señal continua
            if action == 1:
                # Acción de compra - señal positiva fuerte
                signal = 0.8
            else:
                # Acción de venta - señal negativa fuerte
                signal = -0.8
            
            # Añadir algo de variabilidad basado en RSI para hacer las señales más realistas
            rsi_factor = (indicators['rsi'] - 50) / 100  # Entre -0.5 y 0.5
            signal += rsi_factor * 0.2  # Ajuste fino
            
            # Normalizar señal entre -1 y 1
            signal = max(-1.0, min(1.0, signal))
            
            return signal
            
        except Exception as e:
            print(f"⚠️ Error obteniendo predicción de {model_type}: {e}")
            return 0.0
    
    def execute_trading_logic(self, data_point, indicators, signals):
        """Ejecutar lógica de trading con timestamps precisos"""
        selected_signal = signals['selected_signal']
        model_name = signals['model_name']
        
        price = data_point['price']
        timestamp = data_point['timestamp']
        timestamp_str = timestamp.strftime('%H:%M:%S')
        
        # Verificar cooldown entre trades
        current_time = datetime.now()
        if (self.last_trade_time and 
            (current_time - self.last_trade_time).total_seconds() < self.trade_cooldown):
            return
        
        # ✅ DECISIÓN DE TRADING CON LÓGICA DE POSICIONES CORRECTA
        if selected_signal > 0.5:  # Señal fuerte de compra
            # Verificar condiciones para COMPRA:
            # 1. No hay trades abiertos
            # 2. No estamos en posición LONG (evitar compras consecutivas)
            # 3. O estamos cerrando una posición SHORT
            can_buy = (len(self.trade_manager.open_trades) == 0 and 
                      (self.current_position != 'LONG' or self.current_position is None))
            
            if can_buy:
                trade_id = self.trade_manager.open_trade(
                    symbol=self.symbol,
                    trade_type='BUY',
                    size=self.trade_size_usd / price,  # Tamaño basado en USD
                    entry_price=price,
                    entry_time=timestamp,
                    ml_signal=signals.get('ml_signal', selected_signal),
                    technical_signal=signals['technical_signal'],
                    combined_signal=selected_signal,
                    rsi=indicators['rsi'],
                    macd=indicators['macd'],
                    volume=data_point['volume'],
                    portfolio_value=self.current_capital  # Usar capital actual
                )
                
                # Mensaje destacado de COMPRA
                print("\n" + "="*60)
                print("🟢🟢🟢 OPERACIÓN DE COMPRA EJECUTADA 🟢🟢🟢")
                print("="*60)
                print(f"⏰ HORA: {timestamp_str}")
                print(f"🤖 MODELO: {model_name}")
                print(f"💰 PRECIO: ${price:.2f}")
                print(f"📊 SEÑAL: {selected_signal:.2f}")
                print(f"📈 RSI: {indicators['rsi']:.1f}")
                print(f"🆔 TRADE ID: {trade_id}")
                print("="*60 + "\n")
                
                self.last_trade_time = current_time
                
                # ✅ REGISTRAR COMPRA CON TIMESTAMP EXACTO
                actual_timestamp = datetime.now()  # Usar tiempo exacto de la operación
                self.buy_signals.append(self.signal_index)
                self.buy_timestamps.append(actual_timestamp)
                self.buy_prices.append(price)
                
                # ✅ ACTUALIZAR ESTADO DE POSICIÓN
                self.current_position = 'LONG'
                self.last_operation_type = 'BUY'
                self.position_history.append({
                    'timestamp': actual_timestamp,
                    'operation': 'BUY',
                    'price': price,
                    'position': 'LONG'
                })
                
                print(f"🎯 COMPRA registrada: {actual_timestamp.strftime('%H:%M:%S')} @ ${price:.2f}")
                print(f"📊 Nueva posición: {self.current_position}")
            else:
                print(f"⚠️ COMPRA bloqueada - Posición actual: {self.current_position}, Último trade: {self.last_operation_type}")
                
        elif selected_signal < -0.5:  # Señal fuerte de venta
            # Verificar condiciones para VENTA:
            # 1. No hay trades abiertos
            # 2. No estamos en posición SHORT (evitar ventas consecutivas)
            # 3. O estamos cerrando una posición LONG
            can_sell = (len(self.trade_manager.open_trades) == 0 and 
                       (self.current_position != 'SHORT' or self.current_position is None))
            
            if can_sell:
                trade_id = self.trade_manager.open_trade(
                    symbol=self.symbol,
                    trade_type='SELL',
                    size=self.trade_size_usd / price,  # Tamaño basado en USD
                    entry_price=price,
                    entry_time=timestamp,
                    ml_signal=signals.get('ml_signal', selected_signal),
                    technical_signal=signals['technical_signal'],
                    combined_signal=selected_signal,
                    rsi=indicators['rsi'],
                    macd=indicators['macd'],
                    volume=data_point['volume'],
                    portfolio_value=self.current_capital  # Usar capital actual
                )
                
                # Mensaje destacado de VENTA
                print("\n" + "="*60)
                print("🔴🔴🔴 OPERACIÓN DE VENTA EJECUTADA 🔴🔴🔴")
                print("="*60)
                print(f"⏰ HORA: {timestamp_str}")
                print(f"🤖 MODELO: {model_name}")
                print(f"💰 PRECIO: ${price:.2f}")
                print(f"📊 SEÑAL: {selected_signal:.2f}")
                print(f"📈 RSI: {indicators['rsi']:.1f}")
                print(f"🆔 TRADE ID: {trade_id}")
                print("="*60 + "\n")
                
                self.last_trade_time = current_time
                
                # ✅ REGISTRAR VENTA CON TIMESTAMP EXACTO
                actual_timestamp = datetime.now()  # Usar tiempo exacto de la operación
                self.sell_signals.append(self.signal_index)
                self.sell_timestamps.append(actual_timestamp)
                self.sell_prices.append(price)
                
                # ✅ ACTUALIZAR ESTADO DE POSICIÓN
                self.current_position = 'SHORT'
                self.last_operation_type = 'SELL'
                self.position_history.append({
                    'timestamp': actual_timestamp,
                    'operation': 'SELL',
                    'price': price,
                    'position': 'SHORT'
                })
                
                print(f"🎯 VENTA registrada: {actual_timestamp.strftime('%H:%M:%S')} @ ${price:.2f}")
                print(f"📊 Nueva posición: {self.current_position}")
            else:
                print(f"⚠️ VENTA bloqueada - Posición actual: {self.current_position}, Último trade: {self.last_operation_type}")
        
        # Incrementar índice para tracking de señales
        self.signal_index += 1
        
        # Cerrar trades abiertos con ganancia/pérdida
        trades_to_close = []
        for trade_id, trade_data in self.trade_manager.open_trades.items():
            entry_price = trade_data['entry_price']
            trade_type = trade_data['trade_type']
            
            # Calcular retorno actual
            if trade_type == 'BUY':
                return_pct = ((price - entry_price) / entry_price) * 100
            else:  # SELL
                return_pct = ((entry_price - price) / entry_price) * 100
            
            # Cerrar si ganancia > 1% o pérdida > 0.5%
            if return_pct > 1.0 or return_pct < -0.5:
                trades_to_close.append(trade_id)
                action = "GANANCIA" if return_pct > 0 else "PÉRDIDA"
                
                # Mensaje destacado de CIERRE
                print("\n" + "="*60)
                if return_pct > 0:
                    print("💚💚💚 TRADE CERRADO CON GANANCIA 💚💚💚")
                else:
                    print("💔💔💔 TRADE CERRADO CON PÉRDIDA 💔💔💔")
                print("="*60)
                print(f"⏰ HORA: {timestamp_str}")
                print(f"🆔 TRADE ID: {trade_id}")
                print(f"📊 TIPO: {trade_type}")
                print(f"💰 PRECIO ENTRADA: ${entry_price:.2f}")
                print(f"💰 PRECIO SALIDA: ${price:.2f}")
                print(f"📈 RETORNO: {return_pct:+.2f}%")
                print(f"🎯 RAZÓN: {action}")
                print("="*60 + "\n")
        
        # Ejecutar cierres
        for trade_id in trades_to_close:
            # Obtener información del trade antes de cerrarlo
            trade_data = self.trade_manager.open_trades.get(trade_id)
            if trade_data:
                trade_type = trade_data['trade_type']
                entry_price = trade_data['entry_price']
                
                # ✅ CALCULAR P&L DEL TRADE
                pnl_absolute, pnl_percentage = self.calculate_trade_pnl(
                    entry_price, price, trade_type, self.trade_size_usd
                )
                
                # ✅ ACTUALIZAR CAPITAL
                self.update_capital(pnl_absolute)
                
            # Cerrar el trade
            self.trade_manager.close_trade(trade_id, price, timestamp)
            
            # ✅ RESETEAR POSICIÓN AL CERRAR TRADE
            if trade_data:
                self.current_position = None
                self.position_history.append({
                    'timestamp': datetime.now(),
                    'operation': f'CLOSE_{trade_type}',
                    'price': price,
                    'position': None,
                    'pnl': pnl_absolute,
                    'pnl_pct': pnl_percentage
                })
                
                # Mensaje mejorado con P&L (sin emojis problemáticos)
                profit_symbol = "[PROFIT]" if pnl_absolute > 0 else "[LOSS]"
                print(f"[CLOSE] Posicion cerrada: {trade_type} -> NEUTRAL")
                print(f"{profit_symbol} P&L: ${pnl_absolute:.2f} ({pnl_percentage:+.2f}%)")
                print(f"[CAPITAL] Capital actual: ${self.current_capital:.2f}")
        
        # Log periódico más compacto
        if hasattr(self, '_last_log_time'):
            if (current_time - self._last_log_time).total_seconds() >= 10:  # Log cada 10 segundos
                price_change = ((price - self.prev_price) / self.prev_price * 100) if self.prev_price else 0
                print(f"📊 [{timestamp_str}] ${data_point['price']:.2f} ({price_change:+.4f}%) | "
                      f"RSI: {indicators['rsi']:.1f} | "
                      f"Señal {model_name}: {selected_signal:.2f} | "
                      f"Trades: {len(self.trade_manager.open_trades)}")
                self._last_log_time = current_time
        else:
            self._last_log_time = current_time

    def _calculate_technical_indicators(self, df):
        """Calcular indicadores técnicos para los datos descargados"""
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
            
            # Rellenar NaN con valores por defecto
            df['rsi'] = df['rsi'].fillna(50)
            df['macd'] = df['macd'].fillna(0)
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error calculando indicadores: {e}")
            # Valores por defecto si falla
            df['rsi'] = 50
            df['macd'] = 0
            return df
        
    def start_real_time(self):
        """Iniciar trading en tiempo real SIN menú adicional"""
        if self.is_real_time:
            print("⚠️ Sistema ya está en tiempo real")
            return
        
        # ✅ INTENTAR RECONECTAR MT5 SI NO ESTÁ CONECTADO
        if not self.mt5_connected:
            print("🔄 MT5 desconectado, intentando reconectar...")
            if not self.connect_mt5():
                print("❌ Error: No se pudo reconectar MT5")
                print("   Verifica que MT5 esté abierto y funcionando")
                return
            else:
                print("✅ MT5 reconectado exitosamente")
            
        print("🚀 Iniciando sistema en tiempo real...")
        print(f"🤖 Usando modelo {self.model_name}")
        
        self.is_real_time = True
        self.is_running = True
        
        # Iniciar thread de tiempo real
        self.real_time_thread = threading.Thread(target=self._real_time_loop, daemon=True)
        self.real_time_thread.start()
        
        print("✅ Sistema en tiempo real iniciado")
        print(f"📊 Analizando {self.symbol} cada {self.update_interval} segundos")
        print(f"📁 Trades guardándose en: {self.trade_manager.csv_path}")
    
    def stop_real_time(self):
        """Detener trading en tiempo real"""
        if not self.is_real_time:
            return
            
        print("🛑 Deteniendo sistema en tiempo real...")
        self.is_real_time = False
        self.is_running = False
        
        if self.real_time_thread and self.real_time_thread.is_alive():
            self.real_time_thread.join(timeout=5)
        
        # Cerrar trades abiertos
        current_time = datetime.now()
        for trade_id in list(self.trade_manager.open_trades.keys()):
            # Usar último precio conocido
            last_price = getattr(self, 'last_price', 6000)
            self.trade_manager.close_trade(trade_id, last_price, current_time, 'SYSTEM_STOP')
        
        mt5.shutdown()
            
        print("✅ Sistema detenido")
    
    def _real_time_loop(self):
        """Loop principal de tiempo real MEJORADO con cola thread-safe"""
        print("🔄 Iniciando loop de tiempo real MEJORADO...")
        
        connection_check_interval = 30  # Verificar conexión cada 30 segundos
        last_connection_check = datetime.now()
        
        while self.is_running:
            try:
                # ✅ VERIFICAR CONEXIÓN PERIÓDICAMENTE
                current_time = datetime.now()
                if (current_time - last_connection_check).total_seconds() >= connection_check_interval:
                    if self.mt5_connected:
                        # Verificar que MT5 sigue funcionando
                        try:
                            account_info = mt5.account_info()
                            if account_info is None:
                                print("⚠️ MT5 perdió conexión, marcando como desconectado")
                                self.mt5_connected = False
                        except:
                            print("⚠️ Error verificando estado MT5")
                            self.mt5_connected = False
                    
                    last_connection_check = current_time
                
                # ✅ OBTENER DATOS REALES DE MT5
                data_point = self.get_latest_data()
                
                if data_point is None:
                    print("⚠️ Sin datos, reintentando en 5 segundos...")
                    time.sleep(5)
                    continue
                
                # Calcular indicadores
                indicators = self.calculate_indicators(data_point)
                
                # Analizar señales
                signals = self.analyze_signals(data_point, indicators)
                
                # Ejecutar trading
                self.execute_trading_logic(data_point, indicators, signals)
                
                # ✅ AGREGAR DATOS DE FORMA THREAD-SAFE
                with self.update_lock:
                    # ✅ AGREGAR A BUFFER CIRCULAR (más eficiente)
                    self.data_buffer.append({
                        'timestamp': data_point['timestamp'],
                        'price': data_point['price'],
                        'volume': data_point['volume'],
                        'rsi': indicators['rsi'],
                        'macd': indicators['macd']
                    })
                    
                    # ✅ ACTUALIZAR DATAFRAME PRINCIPAL
                    if self.base_system.data is None:
                        self.base_system.data = pd.DataFrame(columns=['timestamp', 'price', 'volume', 'rsi', 'macd'])
                    
                    # Crear nueva fila
                    new_row = pd.DataFrame([{
                        'timestamp': data_point['timestamp'],
                        'price': data_point['price'],
                        'volume': data_point['volume'],
                        'rsi': indicators['rsi'],
                        'macd': indicators['macd']
                    }])
                    
                    # Verificar que no sea duplicado (mismo timestamp)
                    if len(self.base_system.data) > 0:
                        last_timestamp = self.base_system.data['timestamp'].iloc[-1]
                        if data_point['timestamp'] <= last_timestamp:
                            # Si es el mismo tiempo o anterior, esperar al siguiente ciclo
                            time.sleep(self.update_interval)
                            continue
                    
                    # Concatenar datos
                    self.base_system.data = pd.concat([self.base_system.data, new_row], ignore_index=True)
                    
                    # Mantener ventana eficiente (500 puntos)
                    if len(self.base_system.data) > 500:
                        self.base_system.data = self.base_system.data.tail(500).reset_index(drop=True)
                
                # Guardar precio anterior
                self.prev_price = data_point['price']
                
                # Debug: Mostrar información de la ventana de datos
                print(f"📊 Buffer: {len(self.data_buffer)} | "
                      f"DataFrame: {len(self.base_system.data)} | "
                      f"Precio: ${data_point['price']:.2f} | "
                      f"RSI: {indicators['rsi']:.1f}")
                
                # Esperar siguiente ciclo
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"❌ Error en loop tiempo real MEJORADO: {e}")
                print("🔄 Reintentando en 5 segundos...")
                time.sleep(5)

    def create_live_dashboard(self):
        """Crear dashboard MEJORADO con actualización eficiente"""
        print("🎨 Creando dashboard OPTIMIZADO con datos REALES de MT5...")
        
        # Verificar que tenemos datos
        if self.base_system.data is None or len(self.base_system.data) == 0:
            print("❌ No hay datos para mostrar")
            return
        
        print("✅ Datos MT5 cargados exitosamente")
        
        # Configurar matplotlib para tiempo real
        plt.ion()
        
        # Crear figura
        self.fig = plt.figure(figsize=(20, 12), facecolor='#0a0a0a')
        
        # Título dinámico con capital
        self._update_title()
        self.fig.suptitle(self._get_dynamic_title(), 
                         fontsize=16, color='white', y=0.95)
        
        # Grid layout
        gs = GridSpec(3, 4, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1, 1],
                     hspace=0.3, wspace=0.2, top=0.9, bottom=0.1)
        
        # Crear axes
        self.axes['main'] = self.fig.add_subplot(gs[0, :2])
        self.axes['status'] = self.fig.add_subplot(gs[0, 2:])
        self.axes['rsi'] = self.fig.add_subplot(gs[1, 0])
        self.axes['portfolio'] = self.fig.add_subplot(gs[1, 1])
        self.axes['signals'] = self.fig.add_subplot(gs[1, 2])
        self.axes['trades'] = self.fig.add_subplot(gs[1, 3])
        self.axes['volume'] = self.fig.add_subplot(gs[2, 0])
        self.axes['macd'] = self.fig.add_subplot(gs[2, 1])
        self.axes['performance'] = self.fig.add_subplot(gs[2, 2])
        self.axes['controls'] = self.fig.add_subplot(gs[2, 3])
        
        # Configurar estilo
        for ax in self.axes.values():
            ax.set_facecolor('#111111')
            ax.grid(True, color='#333333', linestyle='--', alpha=0.5)
            ax.tick_params(colors='#aaaaaa', labelsize=8)
        
        # ✅ INICIALIZAR LÍNEAS REUTILIZABLES
        self._initialize_plot_lines()
        
        # Crear botones de control
        self._create_control_buttons()
        
        print("✅ Dashboard OPTIMIZADO creado - iniciando animación eficiente...")
        
        # ✅ NUEVA ANIMACIÓN OPTIMIZADA - NO redibuja todo
        try:
            self.animation = FuncAnimation(self.fig, self._update_dashboard_optimized, 
                                         interval=2000, blit=False, repeat=True)  # 2 segundos
            plt.tight_layout()
            
            # FORZAR VENTANA VISIBLE EN WINDOWS
            plt.show(block=False)  # No bloquear el programa
            
            # Mantener ventana activa
            self.fig.canvas.manager.window.wm_attributes('-topmost', 1)  # Mantener al frente
            self.fig.canvas.manager.window.wm_attributes('-topmost', 0)  # Permitir minimizar
            
            print("🪟 Ventana gráfica OPTIMIZADA visible")
            print("📈 Actualización eficiente cada 2 segundos")
            
        except Exception as e:
            print(f"⚠️ Error con animación optimizada: {e}")
            # Fallback a dashboard estático
            self._draw_initial_data()
            plt.tight_layout()
            try:
                plt.show(block=False)
                print("📊 Dashboard estático mostrado")
            except Exception as e2:
                print(f"❌ Error mostrando dashboard: {e2}")
                print("🖥️ Ejecutándose en modo consola solamente")

    def _initialize_plot_lines(self):
        """✅ NUEVO: Inicializar líneas reutilizables para actualización eficiente CON DATOS"""
        print("🎨 Inicializando líneas de gráfico reutilizables CON DATOS...")
        
        if self.base_system.data is None or len(self.base_system.data) == 0:
            print("❌ No hay datos para inicializar líneas")
            return
        
        # Datos iniciales
        data = self.base_system.data.tail(50)
        
        # ✅ GRÁFICO PRINCIPAL - Inicializar CON DATOS REALES
        ax_main = self.axes['main']
        ax_main.clear()
        ax_main.set_facecolor('#111111')
        ax_main.grid(True, color='#333333', linestyle='--', alpha=0.5)
        ax_main.set_title("PRECIO EN TIEMPO REAL", color='white', fontweight='bold')
        ax_main.set_ylabel("Precio USD", color='white')
        
        # ✅ DIBUJAR DATOS INICIALES INMEDIATAMENTE
        timestamps = data['timestamp'].values
        prices = data['price'].values
        
        self.plot_lines['price_line'], = ax_main.plot(timestamps, prices, color='#00ff41', linewidth=3, label='Precio')
        ax_main.legend()
        
        # Formatear fechas
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_main.tick_params(axis='x', rotation=45)
        
        # ✅ RSI - Inicializar CON DATOS REALES
        ax_rsi = self.axes['rsi']
        ax_rsi.clear()
        ax_rsi.set_facecolor('#111111')
        ax_rsi.grid(True, color='#333333', linestyle='--', alpha=0.5)
        ax_rsi.set_title("RSI", color='white', fontweight='bold')
        ax_rsi.set_ylim(0, 100)
        
        if 'rsi' in data.columns:
            rsi_values = data['rsi'].values
            self.plot_lines['rsi_line'], = ax_rsi.plot(timestamps, rsi_values, color='#9c88ff', linewidth=2)
        else:
            self.plot_lines['rsi_line'], = ax_rsi.plot([], [], color='#9c88ff', linewidth=2)
            
        ax_rsi.axhline(70, color='#ff4444', linestyle='--', alpha=0.7, label='Sobrecompra')
        ax_rsi.axhline(30, color='#44ff44', linestyle='--', alpha=0.7, label='Sobreventa')
        ax_rsi.legend()
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # ✅ VOLUMEN - Inicializar CON DATOS REALES
        ax_volume = self.axes['volume']
        ax_volume.clear()
        ax_volume.set_facecolor('#111111')
        ax_volume.grid(True, color='#333333', linestyle='--', alpha=0.5)
        ax_volume.set_title("VOLUMEN", color='white', fontweight='bold')
        
        if 'volume' in data.columns:
            volumes = data['volume'].values
            ax_volume.bar(timestamps, volumes, color='#ffa502', alpha=0.7, width=0.0001)
            ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # ✅ INICIALIZAR OTROS PANELES
        self._update_other_panels()
        
        print("✅ Líneas de gráfico inicializadas CON DATOS REALES")
        
        # ✅ FORZAR PRIMERA ACTUALIZACIÓN
        if hasattr(self, 'fig') and self.fig:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _draw_initial_data(self):
        """Dibujar datos iniciales como fallback (método de compatibilidad)"""
        try:
            print("🎨 Dibujando datos iniciales (fallback)...")
            
            if self.base_system.data is None or len(self.base_system.data) == 0:
                print("❌ No hay datos para dibujar inicialmente")
                return
            
            # Usar últimos 50 puntos para vista inicial
            data = self.base_system.data.tail(50)
            
            # Gráfico principal
            ax_main = self.axes['main']
            ax_main.clear()
            ax_main.set_facecolor('#111111')
            ax_main.grid(True, color='#333333', linestyle='--', alpha=0.5)
            
            # Dibujar línea de precios
            ax_main.plot(data['timestamp'], data['price'], color='#00ff41', linewidth=3, label='Precio', marker='o', markersize=3)
            ax_main.set_title("PRECIO EN TIEMPO REAL", color='white', fontweight='bold')
            ax_main.set_ylabel("Precio USD", color='white')
            ax_main.legend()
            
            # Formatear fechas
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_main.tick_params(axis='x', rotation=45)
            
            # RSI
            if 'rsi' in data.columns:
                ax_rsi = self.axes['rsi']
                ax_rsi.clear()
                ax_rsi.set_facecolor('#111111')
                ax_rsi.grid(True, color='#333333', linestyle='--', alpha=0.5)
                ax_rsi.plot(data['timestamp'], data['rsi'], color='#9c88ff', linewidth=2)
                ax_rsi.axhline(70, color='#ff4444', linestyle='--', alpha=0.7)
                ax_rsi.axhline(30, color='#44ff44', linestyle='--', alpha=0.7)
                ax_rsi.set_ylim(0, 100)
                ax_rsi.set_title("RSI", color='white', fontweight='bold')
            
            # Volumen
            if 'volume' in data.columns:
                ax_volume = self.axes['volume']
                ax_volume.clear()
                ax_volume.set_facecolor('#111111')
                ax_volume.grid(True, color='#333333', linestyle='--', alpha=0.5)
                ax_volume.bar(data['timestamp'], data['volume'], color='#ffa502', alpha=0.7)
                ax_volume.set_title("VOLUMEN", color='white', fontweight='bold')
            
            # Status panel
            self._update_status_panel()
            
            print(f"✅ Datos iniciales dibujados: {len(data)} puntos")
            
            # Forzar actualización de canvas
            if hasattr(self, 'fig') and self.fig:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
        except Exception as e:
            print(f"❌ Error dibujando datos iniciales: {e}")
            import traceback
            print(f"   Detalles: {traceback.format_exc()}")

    def _update_dashboard_optimized(self, frame):
        """✅ NUEVA: Actualización SIMPLIFICADA y ROBUSTA del dashboard"""
        try:
            current_time = time.time()
            
            # ✅ CONTROL DE FRECUENCIA - No actualizar demasiado frecuente
            if current_time - self.last_update_time < self.min_update_interval:
                return
            
            print(f"🔄 Actualizando dashboard... Frame: {frame}")
            
            # ✅ VERIFICAR DATOS THREAD-SAFE
            with self.update_lock:
                if not hasattr(self, 'base_system') or self.base_system.data is None:
                    print("⚠️ No hay datos disponibles")
                    return
                
                if len(self.base_system.data) == 0:
                    print("⚠️ DataFrame vacío")
                    return
                
                # Copiar datos de forma segura
                current_data = self.base_system.data.copy()
                current_count = len(current_data)
                last_price = current_data['price'].iloc[-1] if current_count > 0 else 0
                last_time = current_data['timestamp'].iloc[-1] if current_count > 0 else datetime.now()
            
            print(f"📊 Datos: {current_count} puntos | Último precio: ${last_price:.2f} | Hora: {last_time.strftime('%H:%M:%S')}")
            
            # ✅ USAR VENTANA DESLIZANTE
            display_data = current_data.tail(50)  # Últimos 50 puntos para mejor performance
            
            if len(display_data) < 2:
                print("⚠️ Insuficientes datos para gráfico")
                return
            
            # ✅ SIMPLIFICAR: Redibujar gráfico principal completo pero solo cuando sea necesario
            try:
                timestamps = display_data['timestamp'].values
                prices = display_data['price'].values
                
                # Actualizar línea de precio
                if 'price_line' in self.plot_lines:
                    self.plot_lines['price_line'].set_data(timestamps, prices)
                    
                    # Actualizar límites del eje
                    ax_main = self.axes['main']
                    ax_main.relim()
                    ax_main.autoscale_view()
                    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    
                    print(f"✅ Precio actualizado: {len(timestamps)} puntos")
                else:
                    print("⚠️ price_line no encontrada en plot_lines")
                
            except Exception as e:
                print(f"⚠️ Error actualizando precio: {e}")
            
            # ✅ ACTUALIZAR RSI
            try:
                if 'rsi' in display_data.columns and 'rsi_line' in self.plot_lines:
                    rsi_values = display_data['rsi'].values
                    self.plot_lines['rsi_line'].set_data(timestamps, rsi_values)
                    
                    ax_rsi = self.axes['rsi']
                    ax_rsi.relim()
                    ax_rsi.autoscale_view()
                    ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    
                    print(f"✅ RSI actualizado: último valor {rsi_values[-1]:.1f}")
                    
            except Exception as e:
                print(f"⚠️ Error actualizando RSI: {e}")
            
            # ✅ ACTUALIZAR VOLUMEN (más simple)
            try:
                if 'volume' in display_data.columns:
                    ax_volume = self.axes['volume']
                    ax_volume.clear()
                    ax_volume.set_facecolor('#111111')
                    ax_volume.grid(True, color='#333333', linestyle='--', alpha=0.5)
                    ax_volume.set_title("VOLUMEN", color='white', fontweight='bold')
                    
                    volumes = display_data['volume'].values
                    ax_volume.bar(timestamps, volumes, color='#ffa502', alpha=0.7, width=0.0001)
                    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    
                    print(f"✅ Volumen actualizado")
                    
            except Exception as e:
                print(f"⚠️ Error actualizando volumen: {e}")
            
            # ✅ TRIÁNGULOS - Simplificado
            try:
                ax_main = self.axes['main']
                
                # Dibujar compras
                if hasattr(self, 'buy_timestamps') and self.buy_timestamps:
                    ax_main.scatter(self.buy_timestamps, self.buy_prices, 
                                  color='#00ff41', marker='^', s=400, zorder=20, 
                                  edgecolors='white', linewidths=4, alpha=1.0, label='COMPRAS')
                    print(f"🟢 {len(self.buy_timestamps)} compras dibujadas")
                
                # Dibujar ventas
                if hasattr(self, 'sell_timestamps') and self.sell_timestamps:
                    ax_main.scatter(self.sell_timestamps, self.sell_prices, 
                                  color='#ff4444', marker='v', s=400, zorder=20, 
                                  edgecolors='white', linewidths=4, alpha=1.0, label='VENTAS')
                    print(f"🔴 {len(self.sell_timestamps)} ventas dibujadas")
                        
            except Exception as e:
                print(f"⚠️ Error dibujando triángulos: {e}")
            
            # ✅ ACTUALIZAR PANELES
            try:
                self._update_status_panel()
                self._update_other_panels()
                print("✅ Paneles actualizados")
            except Exception as e:
                print(f"⚠️ Error actualizando paneles: {e}")
            
            # ✅ ACTUALIZAR TÍTULO
            try:
                self.fig.suptitle(self._get_dynamic_title(), 
                                fontsize=16, color='white', y=0.95)
            except Exception as e:
                print(f"⚠️ Error actualizando título: {e}")
            
            # ✅ REFRESH CANVAS
            try:
                self.fig.canvas.draw_idle()
                print("✅ Canvas actualizado")
            except Exception as e:
                print(f"⚠️ Error actualizando canvas: {e}")
            
            self.last_update_time = current_time
            print("✅ Dashboard completamente actualizado")
            
        except Exception as e:
            print(f"❌ Error CRÍTICO en dashboard: {e}")
            import traceback
            print(f"   Traceback completo: {traceback.format_exc()}")

    def _get_dynamic_title(self):
        """Generar título dinámico con información financiera"""
        # Símbolo de estado financiero (sin emojis problemáticos)
        if self.total_profit_loss > 0:
            status_symbol = "[+]"
            pnl_color = "GANANCIA"
        elif self.total_profit_loss < 0:
            status_symbol = "[-]"
            pnl_color = "PERDIDA"
        else:
            status_symbol = "[=]"
            pnl_color = "NEUTRAL"
        
        # Crear título completo (sin emojis problemáticos)
        title = (f"SISTEMA DE TRADING EN TIEMPO REAL - MT5 | "
                f"CAPITAL: ${self.current_capital:,.2f} | "
                f"{status_symbol} {pnl_color}: ${self.total_profit_loss:+,.2f} "
                f"({self.total_profit_loss_pct:+.2f}%)")
        
        return title
    
    def _update_title(self):
        """Actualizar título con información financiera actual"""
        if hasattr(self, 'fig') and self.fig:
            try:
                self.fig.suptitle(self._get_dynamic_title(), 
                                fontsize=16, color='white', y=0.95)
            except:
                pass
        
        # ✅ INICIALIZAR DASHBOARD CORRECTAMENTE
        try:
            self._initialize_dashboard_safely()
        except Exception as e:
            print(f"⚠️ Error inicializando dashboard: {e}")
            print("📊 Sistema funcionará en modo consola")
            self.dashboard_mode = 'console'
            return
        
        # Iniciar animación con manejo de errores
        try:
            self.animation = FuncAnimation(self.fig, self._update_dashboard, 
                                         interval=2000, blit=False, repeat=True)  # 2 segundos
            plt.show()
        except Exception as e:
            print(f"⚠️ Error con animación: {e}")
            print("📊 Dashboard estático disponible")
            plt.show()
    
    def _initialize_dashboard_safely(self):
        """Inicializar dashboard de forma segura"""
        if not hasattr(self, 'axes') or not self.axes:
            print("⚠️ Axes no disponibles, saltando inicialización")
            return
            
        # Evitar re-inicialización
        if hasattr(self, '_dashboard_safely_initialized'):
            return
            
        # Inicializar solo con datos básicos
        for key, ax in self.axes.items():
            if key not in ['controls', 'status']:
                ax.clear()
                ax.set_facecolor('#111111')
                ax.grid(True, color='#333333', linestyle='--', alpha=0.5)
                ax.set_title(key.upper(), color='white', fontweight='bold')
        
        self._dashboard_safely_initialized = True
        print("✅ Dashboard inicializado de forma segura")

    def _create_control_buttons(self):
        """Crear botones de control FUNCIONALES"""
        controls_ax = self.axes['controls']
        controls_ax.clear()
        controls_ax.axis('off')
        
        # Crear botón START/STOP funcional
        button_text = "STOP RT" if self.is_real_time else "START RT"
        button_color = '#cc4444' if self.is_real_time else '#44cc44'
        
        # ✅ CREAR BOTÓN REAL (no solo texto)
        if hasattr(self, 'start_button_ax'):
            self.start_button_ax.remove()
        
        self.start_button_ax = self.fig.add_axes([0.75, 0.45, 0.15, 0.08])  # [x, y, width, height]
        self.start_button = Button(self.start_button_ax, button_text, 
                                color=button_color, hovercolor='#aaaaaa')
        
        # ✅ CONECTAR EVENTO DEL BOTÓN
        self.start_button.on_clicked(self._on_button_click)
        
        # Status text
        status_text = "ACTIVO" if self.is_real_time else "DETENIDO"
        status_color = '#44cc44' if self.is_real_time else '#cc4444'
        
        controls_ax.text(0.5, 0.7, f"Status: {status_text}", ha='center', va='center',
                        fontsize=12, color=status_color, fontweight='bold')
        
        # Información
        info_text = f"Intervalo: {self.update_interval}s\nSímbolo: {self.symbol}"
        controls_ax.text(0.5, 0.3, info_text, ha='center', va='center',
                        fontsize=10, color='white')
        
        controls_ax.set_title("CONTROLES", color='white', fontweight='bold')

    def _on_button_click(self, event):
        """Manejador para el botón START/STOP"""
        print(f"🔘 Botón clickeado! Estado actual: {'ACTIVO' if self.is_real_time else 'DETENIDO'}")
        
        if self.is_real_time:
            print("🛑 Deteniendo sistema...")
            self.stop_real_time()
        else:
            print("🚀 Iniciando sistema...")
            self.start_real_time()
        
        # Forzar redibujado del botón
        self._create_control_buttons()
        self.fig.canvas.draw()
    
    def _draw_initial_data(self):
        """Dibujar datos iniciales para verificar que la gráfica funciona"""
        try:
            print("🎨 Dibujando datos iniciales...")
            
            if self.base_system.data is None or len(self.base_system.data) == 0:
                print("❌ No hay datos para dibujar inicialmente")
                return
            
            # Usar últimos 50 puntos para vista inicial
            data = self.base_system.data.tail(50)
            
            # Gráfico principal
            ax_main = self.axes['main']
            ax_main.clear()
            ax_main.set_facecolor('#111111')
            ax_main.grid(True, color='#333333', linestyle='--', alpha=0.5)
            
            # Dibujar línea de precios
            ax_main.plot(data['timestamp'], data['price'], color='#00ff41', linewidth=3, label='Precio', marker='o', markersize=3)
            ax_main.set_title("PRECIO EN TIEMPO REAL", color='white', fontweight='bold')
            ax_main.set_ylabel("Precio USD", color='white')
            ax_main.legend()
            
            # Formatear fechas
            import matplotlib.dates as mdates
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_main.tick_params(axis='x', rotation=45)
            
            # RSI
            if 'rsi' in data.columns:
                ax_rsi = self.axes['rsi']
                ax_rsi.clear()
                ax_rsi.set_facecolor('#111111')
                ax_rsi.grid(True, color='#333333', linestyle='--', alpha=0.5)
                ax_rsi.plot(data['timestamp'], data['rsi'], color='#9c88ff', linewidth=2)
                ax_rsi.axhline(70, color='#ff4444', linestyle='--', alpha=0.7)
                ax_rsi.axhline(30, color='#44ff44', linestyle='--', alpha=0.7)
                ax_rsi.set_ylim(0, 100)
                ax_rsi.set_title("RSI", color='white', fontweight='bold')
            
            # Volumen
            if 'volume' in data.columns:
                ax_volume = self.axes['volume']
                ax_volume.clear()
                ax_volume.set_facecolor('#111111')
                ax_volume.grid(True, color='#333333', linestyle='--', alpha=0.5)
                ax_volume.bar(data['timestamp'], data['volume'], color='#ffa502', alpha=0.7)
                ax_volume.set_title("VOLUMEN", color='white', fontweight='bold')
            
            # Status panel
            self._update_status_panel()
            
            print(f"✅ Datos iniciales dibujados: {len(data)} puntos")
            
            # Forzar actualización de canvas
            if hasattr(self, 'fig') and self.fig:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
        except Exception as e:
            print(f"❌ Error dibujando datos iniciales: {e}")
            import traceback
            print(f"   Detalles: {traceback.format_exc()}")
    
    def _update_status_panel(self):
        """Actualizar panel de estado de forma simple"""
        ax_status = self.axes['status']
        ax_status.clear()
        ax_status.axis('off')
        
        current_time = datetime.now().strftime('%H:%M:%S')
        last_price = self.base_system.data['price'].iloc[-1] if len(self.base_system.data) > 0 else 0
        total_trades = len(self.trade_manager.trades)
        open_trades = len(self.trade_manager.open_trades)
        
        # Obtener información de posición
        position_status = self.current_position if self.current_position else "NEUTRAL"
        position_color = "🟢" if self.current_position == "LONG" else "🔴" if self.current_position == "SHORT" else "⚪"
        
        # Información financiera (sin emojis problemáticos)
        pnl_symbol = "[+]" if self.total_profit_loss > 0 else "[-]" if self.total_profit_loss < 0 else "[=]"
        last_trade_symbol = "[+]" if self.last_trade_pnl > 0 else "[-]" if self.last_trade_pnl < 0 else "[=]"
        position_color = "[L]" if self.current_position == "LONG" else "[S]" if self.current_position == "SHORT" else "[N]"
        
        status_text = f"""TRADING SYSTEM
        
Tiempo: {current_time}
Precio: ${last_price:.2f}
Simbolo: {self.symbol}

CAPITAL: ${self.current_capital:,.2f}
{pnl_symbol} P&L Total: ${self.total_profit_loss:+,.2f}
ROI: {self.total_profit_loss_pct:+.2f}%
{last_trade_symbol} Ultimo Trade: ${self.last_trade_pnl:+.2f}

Modelo: {self.model_name}
Estado: {'[ON] ACTIVO' if self.is_real_time else '[OFF] PARADO'}
MT5: {'[OK]' if self.mt5_connected else '[ERROR]'}

Posicion: {position_color} {position_status}
Trades: {total_trades} total, {open_trades} abiertos
"""
        
        ax_status.text(0.05, 0.95, status_text, transform=ax_status.transAxes,
                      fontsize=10, color='white', va='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='#1a1a1a', alpha=0.9))
    
    def _update_signals_panel(self, timestamps, recent_data, start_time, end_time):
        """Actualizar panel de señales de forma simple"""
        ax_signals = self.axes['signals']
        ax_signals.clear()
        ax_signals.set_facecolor('#111111')
        ax_signals.grid(True, color='#333333', linestyle='--', alpha=0.5)
        
        # Obtener señal actual
        if len(recent_data) > 0:
            last_data = recent_data.iloc[-1].to_dict()
            indicators = self.calculate_indicators(last_data)
            signals = self.analyze_signals(last_data, indicators)
            
            selected_signal = signals.get('selected_signal', 0)
            signal_name = signals.get('model_name', 'N/A')
            
            # Línea horizontal con la señal actual
            signal_values = [selected_signal] * len(timestamps)
            ax_signals.plot(timestamps, signal_values, 
                          color='#45b7d1', linewidth=3, alpha=0.9,
                          label=f"{signal_name}: {selected_signal:.2f}")
            
            ax_signals.set_title("SEÑAL ACTUAL", color='white', fontweight='bold')
            ax_signals.legend(loc='upper right', fontsize=9)
            ax_signals.set_ylim(-1.2, 1.2)
            ax_signals.axhline(0.5, color='#44ff44', linestyle='--', alpha=0.7)
            ax_signals.axhline(-0.5, color='#ff4444', linestyle='--', alpha=0.7)
            ax_signals.set_xlim(start_time, end_time)
    
    def _update_other_panels(self):
        """Actualizar otros paneles informativos - CORREGIDO"""
        try:
            # ✅ PORTFOLIO - Mostrar evolución del capital
            ax_portfolio = self.axes['portfolio']
            ax_portfolio.clear()
            ax_portfolio.set_facecolor('#111111')
            ax_portfolio.grid(True, color='#333333', linestyle='--', alpha=0.5)
            ax_portfolio.set_title("PORTFOLIO", color='white', fontweight='bold')
            
            # Crear datos de portfolio simple
            if hasattr(self, 'base_system') and self.base_system.data is not None and len(self.base_system.data) > 0:
                # Usar últimos 50 puntos para mostrar evolución del capital
                recent_data = self.base_system.data.tail(50)
                timestamps = recent_data['timestamp'].values
                
                # Simular evolución del capital (simplificado)
                capital_evolution = [self.current_capital] * len(timestamps)
                ax_portfolio.plot(timestamps, capital_evolution, color='#00ff41', linewidth=2, label=f'${self.current_capital:,.0f}')
                ax_portfolio.legend()
                ax_portfolio.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            else:
                ax_portfolio.text(0.5, 0.5, f"CAPITAL\n${self.current_capital:,.0f}", 
                                ha='center', va='center', transform=ax_portfolio.transAxes,
                                fontsize=12, color='#00ff41', fontweight='bold')
            
            # ✅ TRADES INFO - Panel de información de trades
            ax_trades = self.axes['trades']
            ax_trades.clear()
            ax_trades.axis('off')
            total_trades = len(self.trade_manager.trades)
            open_trades = len(self.trade_manager.open_trades)
            
            trades_text = f"""TRADES

Total: {total_trades}
Abiertas: {open_trades}
Compras: {len(self.buy_signals)}
Ventas: {len(self.sell_signals)}

Posición:
{self.current_position or 'NEUTRAL'}"""
            
            ax_trades.text(0.5, 0.5, trades_text, 
                          ha='center', va='center', transform=ax_trades.transAxes,
                          fontsize=10, color='white', fontweight='bold')
            
            # ✅ SIGNALS - Panel de señales actual
            ax_signals = self.axes['signals']
            ax_signals.clear()
            ax_signals.set_facecolor('#111111')
            ax_signals.grid(True, color='#333333', linestyle='--', alpha=0.5)
            ax_signals.set_title("SEÑALES", color='white', fontweight='bold')
            
            # Mostrar señal actual como línea horizontal
            signal_levels = [0.5, -0.5, 0]  # Compra, Venta, Neutral
            signal_colors = ['#44ff44', '#ff4444', '#ffff44']
            signal_labels = ['Compra (>0.5)', 'Venta (<-0.5)', 'Neutral (0)']
            
            for level, color, label in zip(signal_levels, signal_colors, signal_labels):
                ax_signals.axhline(level, color=color, linestyle='--', alpha=0.7, label=label)
            
            ax_signals.set_ylim(-1.2, 1.2)
            ax_signals.legend(fontsize=8)
            
            # ✅ PERFORMANCE - Panel de rendimiento
            ax_performance = self.axes['performance']
            ax_performance.clear()
            ax_performance.axis('off')
            
            # Calcular estadísticas de performance
            if len(self.trade_manager.trades) > 0:
                winning_trades = len([t for t in self.trade_manager.trades if float(t.get('return_pct', 0)) > 0])
                losing_trades = len(self.trade_manager.trades) - winning_trades
                win_rate = (winning_trades / len(self.trade_manager.trades)) * 100
            else:
                winning_trades = 0
                losing_trades = 0
                win_rate = 0
            
            # Símbolo de performance
            if self.total_profit_loss > 100:
                perf_symbol = "[PROFIT]"
                perf_color = '#44ff44'
            elif self.total_profit_loss > 0:
                perf_symbol = "[UP]"
                perf_color = '#44ff44'
            elif self.total_profit_loss < 0:
                perf_symbol = "[DOWN]"
                perf_color = '#ff4444'
            else:
                perf_symbol = "[NEUTRAL]"
                perf_color = '#ffff44'
            
            performance_text = f"""PERFORMANCE {perf_symbol}

Capital: ${self.current_capital:,.0f}
P&L: ${self.total_profit_loss:+,.0f}
ROI: {self.total_profit_loss_pct:+.1f}%

Win Rate: {win_rate:.1f}%
Ganadas: {winning_trades}
Perdidas: {losing_trades}

Modelo: {self.selected_model.upper() if self.selected_model else 'N/A'}"""
            
            ax_performance.text(0.5, 0.5, performance_text, 
                               ha='center', va='center', transform=ax_performance.transAxes,
                               fontsize=9, color=perf_color, fontweight='bold')
            
        except Exception as e:
            print(f"⚠️ Error actualizando paneles: {e}")

    def toggle_real_time(self):
        """Alternar entre tiempo real y detenido"""
        if self.is_real_time:
            self.stop_real_time()
        else:
            self.start_real_time()

    def _model_is_loaded(self):
        """Verificar si el modelo seleccionado está cargado"""
        if self.selected_model == 'technical':
            return True
        elif self.selected_model == 'all':
            return any(model is not None for model in self.models.values())
        else:
            return self.models.get(self.selected_model) is not None
    
    def calculate_trade_pnl(self, entry_price, exit_price, trade_type, size_usd):
        """Calcular P&L de un trade"""
        try:
            # Calcular número de unidades basado en el tamaño en USD
            units = size_usd / entry_price
            
            if trade_type == 'BUY':
                # LONG: ganancia cuando precio sube
                price_diff = exit_price - entry_price
                pnl_absolute = units * price_diff
                pnl_percentage = ((exit_price - entry_price) / entry_price) * 100
            else:  # SELL (SHORT)
                # SHORT: ganancia cuando precio baja
                price_diff = entry_price - exit_price
                pnl_absolute = units * price_diff
                pnl_percentage = ((entry_price - exit_price) / entry_price) * 100
            
            return pnl_absolute, pnl_percentage
            
        except Exception as e:
            print(f"❌ Error calculando P&L: {e}")
            return 0.0, 0.0
    
    def update_capital(self, pnl_absolute):
        """Actualizar capital y estadísticas"""
        self.current_capital += pnl_absolute
        self.total_profit_loss += pnl_absolute
        self.total_profit_loss_pct = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        self.last_trade_pnl = pnl_absolute
        self.last_trade_pnl_pct = (pnl_absolute / self.trade_size_usd) * 100

def main():
    """Función principal del sistema"""
    try:
        print("✅ MetaTrader5 disponible" if HAS_MT5 else "❌ MetaTrader5 NO disponible - REQUERIDO")
        print("✅ Stable-baselines3 disponible" if HAS_RL else "⚠️ Stable-baselines3 NO disponible")
        
        if not HAS_MT5:
            print("❌ ERROR: MetaTrader5 es REQUERIDO para este sistema")
            print("   Instala MT5 e intenta de nuevo")
            sys.exit(1)
        
        # Texto decorativo
        print("⚡ Datos reales MT5 | 📊 CSV automático | 🎯 Dashboard live")
        print("=" * 80)
        
        # Nuevo menú para seleccionar modelo individual - EXACTO DE comparison_four_models.py
        print("\n🤖 Selecciona el modelo de IA a utilizar:")
        print("1. DQN (Deep Q-Network) - AGRESIVO")
        print("2. DeepDQN (Deep DQN) - PRECISO")
        print("3. PPO (Proximal Policy Optimization) - BALANCEADO")
        print("4. A2C (Advantage Actor-Critic) - CONSERVADOR")
        print("5. Comparación de TODOS los 4 modelos")
        print("6. Solo Análisis Técnico")
        
        while True:
            try:
                choice = input("\nIngresa el número de tu elección (1-6): ").strip()
                
                if choice in ['1', '2', '3', '4', '5', '6']:
                    break
                else:
                    print("❌ Opción inválida. Por favor, selecciona un número del 1 al 6.")
            except KeyboardInterrupt:
                print("\n👋 Saliendo...")
                return
            except:
                print("❌ Entrada inválida. Intenta de nuevo.")
        
        # Crear sistema con modelo seleccionado
        selected_model = None
        model_name = ""
        
        if choice == '1':
            selected_model = 'dqn'
            model_name = "DQN (Deep Q-Network) - AGRESIVO"
        elif choice == '2':
            selected_model = 'deepdqn'
            model_name = "DeepDQN (Deep DQN) - PRECISO"
        elif choice == '3':
            selected_model = 'ppo'
            model_name = "PPO (Proximal Policy Optimization) - BALANCEADO"
        elif choice == '4':
            selected_model = 'a2c'
            model_name = "A2C (Advantage Actor-Critic) - CONSERVADOR"
        elif choice == '5':
            selected_model = 'all'
            model_name = "TODOS los 4 modelos (DQN+DeepDQN+PPO+A2C)"
        elif choice == '6':
            selected_model = 'technical'
            model_name = "Análisis Técnico"
        
        print(f"\n🎯 Modelo seleccionado: {model_name}")
        print("🚀 Iniciando sistema...")
        
        # Crear sistema con configuración específica
        system = RealTimeTradingSystem(selected_model=selected_model)
        
        # ✅ INICIAR AUTOMÁTICAMENTE - SIN ESPERAR COMANDOS MANUALES
        print("🚀 Iniciando sistema automáticamente...")
        
        # Iniciar dashboard
        system.create_live_dashboard()
        system.start_real_time()
        
        # Mantener ventana abierta y sistema funcionando
        print("\n" + "="*60)
        print("🤖 SISTEMA FUNCIONANDO AUTOMÁTICAMENTE CON MT5")
        print("="*60)
        print("📊 El sistema está operando en tiempo real de forma autónoma")
        print("📈 Compras y ventas aparecerán automáticamente en el gráfico")
        print("🔺 Triángulo VERDE = COMPRA | 🔻 Triángulo ROJO = VENTA")
        print("📁 Trades guardándose automáticamente en CSV")
        print("\n🎯 COMANDOS DISPONIBLES:")
        print("  'stop'     - Detener tiempo real")
        print("  'start'    - Reiniciar tiempo real")
        print("  'status'   - Ver estado actual")
        print("  'reconnect'- Reconectar MT5 manualmente")
        print("  'quit'     - Salir del programa")
        print("="*60)
        
        # Loop de comandos opcionales (el sistema ya está funcionando)
        while True:
            try:
                command = input("\n>>> ").strip().lower()
                
                if command == 'start':
                    print("🚀 Reiniciando sistema en tiempo real...")
                    system.start_real_time()
                    
                elif command == 'stop':
                    print("🛑 Deteniendo sistema...")
                    system.stop_real_time()
                    
                elif command == 'status':
                    print(f"\n📊 ESTADO ACTUAL:")
                    print(f"  Modelo: {system.model_name}")
                    print(f"  Sistema: {'🟢 ACTIVO' if system.is_real_time else '🔴 DETENIDO'}")
                    print(f"  MT5: {'🟢 CONECTADO' if system.mt5_connected else '🔴 DESCONECTADO'}")
                    if system.mt5_connected:
                        try:
                            account_info = mt5.account_info()
                            if account_info:
                                print(f"  Cuenta MT5: {account_info.login}")
                            else:
                                print(f"  ⚠️ MT5 puede estar desconectado")
                        except:
                            print(f"  ⚠️ Error verificando MT5")
                    print(f"  Símbolo: {system.symbol}")
                    print(f"  Trades abiertos: {len(system.trade_manager.open_trades)}")
                    print(f"  Trades totales: {len(system.trade_manager.trades)}")
                    print(f"  Señales: BUY={len(system.buy_signals)}, SELL={len(system.sell_signals)}")
                    print(f"  CSV: {system.trade_manager.csv_filename}")
                
                elif command == 'reconnect':
                    print("🔄 Intentando reconectar MT5...")
                    if system.connect_mt5():
                        print("✅ MT5 reconectado exitosamente")
                    else:
                        print("❌ Error reconectando MT5")
                        print("   Verifica que MT5 esté abierto y funcionando")
                    
                elif command == 'quit' or command == 'exit':
                    print("👋 Saliendo del sistema...")
                    break
                    
                elif command == 'help':
                    print("📱 Comandos: start, stop, status, reconnect, quit")
                    
                else:
                    print(f"❌ Comando '{command}' no reconocido. Usa 'help' para ver comandos.")
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n🛑 Deteniendo sistema por Ctrl+C...")
                break
                
        # Limpiar al salir
        try:
            system.stop_real_time()
            print("✅ Sistema finalizado correctamente")
        except:
            pass
        
    except KeyboardInterrupt:
        print("\n👋 Saliendo del sistema...")
    except Exception as e:
        print(f"❌ Error en el sistema: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()