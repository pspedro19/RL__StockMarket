"""
🤖 SISTEMA DE TRADING CON IA + TÉCNICO
Combina modelo DQN entrenado con análisis técnico tradicional
Integrado con MetaTrader5 para datos en tiempo real
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Intentar importar MT5
try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
    print("✅ MetaTrader5 disponible")
except ImportError:
    HAS_MT5 = False
    print("⚠️ MetaTrader5 no disponible - usando datos simulados")

# Intentar importar componentes de RL (si están disponibles)
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_RL = True
    print("✅ Componentes de RL disponibles")
except ImportError:
    HAS_RL = False
    print("⚠️ Sin componentes de RL - usando solo técnico")

class MLEnhancedTradingSystem:
    """Sistema de trading que combina IA + análisis técnico con datos en tiempo real"""
    
    def __init__(self):
        # Control de reproducción
        self.is_playing = False
        self.current_step = 50
        self.speed = 1.0
        self.data = None
        
        # MT5 y tiempo real
        self.use_real_time = False
        self.mt5_connected = False
        self.symbol = "US500"  # SP500 en FPMarkets
        self.timeframe = mt5.TIMEFRAME_M1 if HAS_MT5 else None
        self.real_time_thread = None
        self.last_update = datetime.now()
        
        # Fechas para gráficos - USAR FECHA ACTUAL
        self.start_date = datetime.now() - timedelta(hours=24)  # Últimas 24 horas
        self.dates = []
        
        # Portfolio mejorado
        self.initial_capital = 100000
        self.current_capital = self.initial_capital
        self.position_size = 0
        self.entry_price = 0
        self.position_type = None
        
        # Gestión de riesgo RELAJADA para más trading
        self.max_position_risk = 0.03      # 3% (era 2%)
        self.stop_loss_pct = 0.02          # 2% (era 1.5%)
        self.take_profit_pct = 0.04        # 4% (era 3%)
        
        # Límites RELAJADOS
        self.max_daily_trades = 5          # 5 (era 2)
        self.daily_trades = 0
        self.min_trade_separation = 3      # 3 (era 8)
        self.last_trade_step = -10
        
        # Modelo de IA
        self.ml_model = None
        self.use_ml_signals = True
        self.ml_weight = 0.6  # 60% IA, 40% técnico
        
        # Tracking
        self.portfolio_values = []
        self.buy_signals = []
        self.sell_signals = []
        self.stop_losses = []
        self.take_profits = []
        self.trades_history = []
        self.actions = []
        self.ml_predictions = []
        self.technical_signals = []
        
        # Métricas
        self.max_drawdown = 0
        self.peak_value = self.initial_capital
        self.consecutive_losses = 0
        self.total_trades = 0
        self.profitable_trades = 0
        
        # Visual
        self.window_size = 150
        
    def load_ml_model(self):
        """Cargar modelo entrenado si existe"""
        try:
            if HAS_RL:
                # Intentar cargar modelos entrenados
                model_paths = [
                    "dqn_trading_model.zip",
                    "best_dqn_model.zip", 
                    "trading_dqn.zip",
                    "sac_trading_model.zip"
                ]
                
                for model_path in model_paths:
                    try:
                        self.ml_model = DQN.load(model_path)
                        print(f"✅ Modelo cargado: {model_path}")
                        return True
                    except:
                        continue
                
                print("⚠️ No se encontraron modelos entrenados")
                return False
            else:
                print("⚠️ Stable-baselines3 no disponible")
                return False
                
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def create_simple_ml_model(self):
        """Crear un modelo simple basado en reglas si no hay uno entrenado"""
        print("🔄 Creando modelo técnico avanzado...")
        
        class SimpleTechnicalModel:
            def predict(self, obs):
                """Predicción basada en múltiples indicadores"""
                if len(obs.shape) == 1:
                    obs = obs.reshape(1, -1)
                
                predictions = []
                for row in obs:
                    # Asumir que tenemos: [price, rsi, macd, bb_pos, volume_ratio, momentum, ...]
                    if len(row) >= 6:
                        price_idx = 0
                        rsi_idx = 1 if len(row) > 1 else 0
                        macd_idx = 2 if len(row) > 2 else 0
                        bb_idx = 3 if len(row) > 3 else 0
                        vol_idx = 4 if len(row) > 4 else 0
                        mom_idx = 5 if len(row) > 5 else 0
                        
                        # Señales múltiples
                        signals = []
                        
                        # RSI
                        if len(row) > rsi_idx:
                            rsi_val = row[rsi_idx] * 100  # Normalizar
                            if rsi_val < 30:
                                signals.append(1)  # Compra
                            elif rsi_val > 70:
                                signals.append(0)  # Venta
                        
                        # MACD (asumir normalizado)
                        if len(row) > macd_idx:
                            macd_val = row[macd_idx]
                            if macd_val > 0.1:
                                signals.append(1)
                            elif macd_val < -0.1:
                                signals.append(0)
                        
                        # Bollinger Bands
                        if len(row) > bb_idx:
                            bb_val = row[bb_idx]
                            if bb_val < 0.2:
                                signals.append(1)
                            elif bb_val > 0.8:
                                signals.append(0)
                        
                        # Momentum
                        if len(row) > mom_idx:
                            mom_val = row[mom_idx]
                            if mom_val > 0.02:
                                signals.append(1)
                            elif mom_val < -0.02:
                                signals.append(0)
                        
                        # Decisión por mayoría
                        if len(signals) >= 2:
                            if signals.count(1) > signals.count(0):
                                predictions.append([1])  # Compra
                            elif signals.count(0) > signals.count(1):
                                predictions.append([0])  # Venta  
                            else:
                                predictions.append([2])  # Hold
                        else:
                            predictions.append([2])  # Hold por defecto
                    else:
                        predictions.append([2])  # Hold
                
                return np.array(predictions)
        
        self.ml_model = SimpleTechnicalModel()
        print("✅ Modelo técnico avanzado creado")
        return True
    
    def connect_mt5(self):
        """Conectar a MetaTrader5 con diagnóstico mejorado"""
        if not HAS_MT5:
            print("❌ MetaTrader5 no está instalado")
            print("💡 Solución: Instala MT5 desde https://www.metaquotes.net/es/metatrader5")
            return False
            
        try:
            print("🔄 Intentando conectar a MetaTrader5...")
            
            # Intentar inicializar
            if not mt5.initialize():
                error_code = mt5.last_error()
                print(f"❌ Error inicializando MT5: {error_code}")
                print("💡 Soluciones:")
                print("   1. Abre MetaTrader5 manualmente")
                print("   2. Configura una cuenta demo")
                print("   3. Asegúrate de que MT5 esté ejecutándose")
                return False
            
            print("✅ MT5 inicializado correctamente")
            
            # Verificar conexión al servidor
            account_info = mt5.account_info()
            if account_info is None:
                print("❌ No hay cuenta configurada en MT5")
                print("💡 Solución: Configura una cuenta demo en MT5")
                return False
            
            print(f"✅ Cuenta conectada: {account_info.login} - {account_info.server}")
            
            # Verificar símbolo US500
            print(f"🔍 Buscando símbolo {self.symbol}...")
            symbol_info = mt5.symbol_info(self.symbol)
            
            if symbol_info is None:
                print(f"❌ Símbolo {self.symbol} no encontrado")
                print("💡 Símbolos alternativos para SP500:")
                
                # Buscar símbolos alternativos
                alternatives = ["SP500", "SPX500", "US500m", "USTEC", "SPX"]
                for alt in alternatives:
                    if mt5.symbol_info(alt) is not None:
                        print(f"   ✅ Encontrado: {alt}")
                        self.symbol = alt
                        symbol_info = mt5.symbol_info(alt)
                        break
                    else:
                        print(f"   ❌ No disponible: {alt}")
                
                if symbol_info is None:
                    print("❌ No se encontró ningún símbolo SP500")
                    print("💡 Contacta a tu broker para obtener el símbolo correcto")
                    return False
            
            # Activar símbolo si no está visible
            if not symbol_info.visible:
                print(f"🔄 Activando símbolo {self.symbol}...")
                if not mt5.symbol_select(self.symbol, True):
                    print(f"❌ Error activando símbolo {self.symbol}")
                    return False
            
            print(f"✅ Símbolo {self.symbol} listo - Spread: {symbol_info.spread}")
            
            self.mt5_connected = True
            print(f"🎉 Conexión MT5 exitosa - Símbolo: {self.symbol}")
            return True
            
        except Exception as e:
            print(f"❌ Error conectando MT5: {e}")
            print("💡 Soluciones:")
            print("   1. Reinstala MetaTrader5")
            print("   2. Ejecuta como administrador")
            print("   3. Verifica que no esté bloqueado por antivirus")
            return False
    
    def get_real_time_data(self, count=1000):
        """Obtener datos ACTUALES en tiempo real de MT5"""
        if not self.mt5_connected:
            return None
            
        try:
            # Para tiempo real, obtener datos desde ahora hacia atrás
            from datetime import datetime, timezone
            import pytz
            
            # Obtener hora actual
            now = datetime.now(timezone.utc)
            
            # Obtener datos más recientes (últimos minutos)
            rates = mt5.copy_rates_from(self.symbol, self.timeframe, now, count)
            
            if rates is None or len(rates) == 0:
                print("❌ No se pudieron obtener datos actuales de MT5")
                print("🔄 Intentando con copy_rates_from_pos...")
                # Fallback a método anterior
                rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                print("❌ No hay datos disponibles en MT5")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Verificar si los datos son actuales (últimas 24 horas)
            latest_time = df['time'].max()
            time_diff = now.replace(tzinfo=None) - latest_time.replace(tzinfo=None)
            
            print(f"📊 Último dato: {latest_time}")
            print(f"🕐 Ahora: {now.replace(tzinfo=None)}")
            print(f"⏰ Diferencia: {time_diff}")
            
            if time_diff.total_seconds() > 86400:  # Más de 24 horas
                print("⚠️ DATOS ANTIGUOS DETECTADOS - Usando datos simulados en su lugar")
                return None
            
            df.rename(columns={
                'open': 'price',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            }, inplace=True)
            
            # Usar precio de cierre como precio principal
            df['price'] = df['close']
            
            print(f"✅ Datos actuales obtenidos: {len(df)} registros de MT5")
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos de MT5: {e}")
            return None
    
    def start_real_time_updates(self):
        """Iniciar actualizaciones en tiempo real"""
        if not self.mt5_connected:
            print("❌ MT5 no está conectado")
            return
        
        self.use_real_time = True
        
        def update_loop():
            while self.use_real_time and self.mt5_connected:
                try:
                    # Obtener nuevo dato más frecuentemente
                    new_data = self.get_real_time_data(1)
                    if new_data is not None and len(new_data) > 0:
                        # Agregar a datos existentes
                        latest_time = new_data['time'].iloc[-1]
                        if latest_time > self.last_update:
                            self.append_new_data(new_data.iloc[-1])
                            self.last_update = latest_time
                    
                    time.sleep(15)  # Verificar cada 15 segundos para capturar nuevos minutos
                    
                except Exception as e:
                    print(f"❌ Error en actualización tiempo real: {e}")
                    time.sleep(10)
        
        self.real_time_thread = threading.Thread(target=update_loop, daemon=True)
        self.real_time_thread.start()
        print("✅ Actualizaciones en tiempo real iniciadas")
    
    def append_new_data(self, new_row):
        """Agregar nueva fila de datos, analizar y TOMAR DECISIONES DE TRADING"""
        if self.data is None:
            return
            
        try:
            # Crear nuevo registro con los datos correctos
            new_record = {
                'price': float(new_row['close']),  # Usar close como precio
                'volume': float(new_row['tick_volume']),
                'time': pd.to_datetime(new_row['time'], unit='s'),
                'high': float(new_row['high']),
                'low': float(new_row['low']),
                'open': float(new_row['open'])
            }
            
            # Convertir a DataFrame
            new_df = pd.DataFrame([new_record])
            
            # Calcular indicadores para el nuevo dato
            temp_data = pd.concat([self.data.tail(50), new_df], ignore_index=True)
            temp_data = self.calculate_indicators(temp_data)
            
            # Tomar solo la última fila (con indicadores calculados)
            new_row_processed = temp_data.iloc[-1:].copy()
            
            # Agregar al dataset principal
            self.data = pd.concat([self.data, new_row_processed], ignore_index=True)
            
            # Avanzar el step actual al último dato
            self.current_step = len(self.data) - 1
            
            # 🤖 ANÁLISIS Y DECISIÓN DE TRADING EN TIEMPO REAL
            self.analyze_and_trade_real_time()
            
            # Actualizar arrays de tracking
            self.portfolio_values.append(float(self.current_capital))
            
            # FORZAR ACTUALIZACIÓN DE GRÁFICOS
            self.force_plot_update()
            
            print(f"📈 Análisis en vivo: {new_record['time']} - Precio: ${new_record['price']:.2f}")
            
        except Exception as e:
            print(f"❌ Error procesando nuevo dato: {e}")
            return
    
    def analyze_and_trade_real_time(self):
        """🤖 MOTOR DE DECISIONES DE TRADING EN TIEMPO REAL"""
        
        # Verificar que tengamos suficientes datos
        if len(self.data) < 50:
            print("⏳ Esperando más datos para análisis...")
            return
        
        current_price = self.data['price'].iloc[self.current_step]
        
        print(f"\n{'='*60}")
        print(f"🤖 ANÁLISIS DE TRADING EN TIEMPO REAL")
        print(f"{'='*60}")
        print(f"💰 Precio actual: ${current_price:.2f}")
        print(f"💼 Capital: ${self.current_capital:,.0f}")
        print(f"📊 Posición: {self.position_size}")
        
        # 1. VERIFICAR CONDICIONES DE SALIDA (Stop Loss / Take Profit)
        if self.position_size != 0:
            exit_result = self.check_exit_conditions(self.current_step)
            if exit_result:
                print(f"🚪 Salida ejecutada: {exit_result}")
                return
        
        # 2. GENERAR SEÑALES DE ANÁLISIS
        ml_signal = self.generate_ml_signal(self.current_step)
        tech_signal = self.generate_technical_signal(self.current_step)
        combined_signal = self.generate_combined_signal(self.current_step)
        
        # Actualizar arrays
        self.technical_signals.append(tech_signal)
        
        # Convertir predicción ML
        if ml_signal > 0.5:
            ml_pred = 1  # Buy
        elif ml_signal < -0.5:
            ml_pred = 0  # Sell
        else:
            ml_pred = 2  # Hold
        
        self.ml_predictions.append(ml_pred)
        
        print(f"🔍 Señal Técnica: {tech_signal:.3f}")
        print(f"🤖 Señal IA: {ml_signal:.3f}")
        print(f"⚖️ Señal Combinada: {combined_signal:.3f}")
        
        # 3. INTERPRETAR SEÑALES
        if combined_signal > 0.25:
            signal_type = "🟢 COMPRA FUERTE"
            action = "BUY"
        elif combined_signal > 0.1:
            signal_type = "🟡 COMPRA DÉBIL"
            action = "WEAK_BUY"
        elif combined_signal < -0.25:
            signal_type = "🔴 VENTA FUERTE"
            action = "SELL"
        elif combined_signal < -0.1:
            signal_type = "🟠 VENTA DÉBIL"
            action = "WEAK_SELL"
        else:
            signal_type = "⚪ MANTENER"
            action = "HOLD"
        
        print(f"🎯 Decisión: {signal_type}")
        
        # 4. EJECUTAR TRADING
        trade_executed = False
        
        if action == "BUY" and self.position_size == 0:
            # Solo comprar si no tenemos posición
            trade_executed = self.execute_trade(self.current_step, combined_signal)
            if trade_executed:
                print(f"✅ COMPRA EJECUTADA - Precio: ${current_price:.2f}")
                self.buy_signals.append(self.current_step)
        
        elif action == "SELL" and self.position_size > 0:
            # Solo vender si tenemos posición larga
            trade_executed = self.execute_trade(self.current_step, combined_signal)
            if trade_executed:
                print(f"✅ VENTA EJECUTADA - Precio: ${current_price:.2f}")
                self.sell_signals.append(self.current_step)
        
        elif action == "HOLD":
            print(f"⏸️ MANTENER POSICIÓN - Sin cambios")
        
        # 5. MOSTRAR ESTADO ACTUAL
        if self.position_size > 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position_size
            print(f"📈 P&L No Realizado: ${unrealized_pnl:+.2f}")
        
        print(f"🕐 Siguiente análisis en ~1 minuto...")
        print(f"{'='*60}\n")
    
    def force_plot_update(self):
        """Forzar actualización inmediata de los gráficos"""
        try:
            if hasattr(self, 'fig') and self.fig is not None:
                # Verificar que los datos sean válidos antes de actualizar
                if self.data is not None and len(self.data) > 0:
                    # Limpiar cualquier valor NaN o infinito que pueda causar problemas
                    numeric_columns = self.data.select_dtypes(include=[np.number]).columns
                    self.data[numeric_columns] = self.data[numeric_columns].fillna(0)
                    
                    # Llamar directamente al método de actualización
                    self.update_plots(None)
                    # Redibujar la figura
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
        except Exception as e:
            print(f"⚠️ Error actualizando gráficos: {e}")
            # Intentar reinicializar arrays si hay problemas
            self.fix_data_types()
    
    def fix_data_types(self):
        """Arreglar problemas de tipos de datos"""
        try:
            if self.data is not None:
                # Asegurar que arrays tengan la misma longitud que los datos
                data_len = len(self.data)
                
                # Ajustar arrays si tienen diferente longitud
                if len(self.portfolio_values) != data_len:
                    if len(self.portfolio_values) < data_len:
                        # Extender con el último valor
                        last_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
                        self.portfolio_values.extend([last_value] * (data_len - len(self.portfolio_values)))
                    else:
                        # Truncar
                        self.portfolio_values = self.portfolio_values[:data_len]
                
                if len(self.technical_signals) != data_len:
                    if len(self.technical_signals) < data_len:
                        self.technical_signals.extend([0.0] * (data_len - len(self.technical_signals)))
                    else:
                        self.technical_signals = self.technical_signals[:data_len]
                
                if len(self.ml_predictions) != data_len:
                    if len(self.ml_predictions) < data_len:
                        self.ml_predictions.extend([2] * (data_len - len(self.ml_predictions)))
                    else:
                        self.ml_predictions = self.ml_predictions[:data_len]
                
                print("✅ Tipos de datos corregidos")
        except Exception as e:
            print(f"❌ Error corrigiendo tipos de datos: {e}")
    
    def stop_real_time_updates(self):
        """Detener actualizaciones en tiempo real"""
        self.use_real_time = False
        if self.real_time_thread and self.real_time_thread.is_alive():
            self.real_time_thread.join(timeout=2)
        print("🛑 Actualizaciones en tiempo real detenidas")
    
    def generate_market_data(self, n_points=1500):
        """Generar datos de mercado realistas o usar datos reales de MT5"""
        
        # Intentar conectar MT5 y usar datos reales
        if self.connect_mt5():
            print("📊 Verificando disponibilidad de datos actuales en MT5...")
            real_data = self.get_real_time_data(n_points)
            if real_data is not None:
                # Verificar que los datos sean realmente actuales
                latest_time = real_data['time'].max()
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                time_diff = now.replace(tzinfo=None) - latest_time.replace(tzinfo=None)
                
                if time_diff.total_seconds() < 86400:  # Menos de 24 horas
                    self.data = real_data.copy()
                    self.data = self.calculate_indicators(self.data)
                    self.initialize_tracking_arrays()
                    print(f"✅ Usando {len(self.data)} datos ACTUALES de MT5")
                    print(f"📅 Último dato: {latest_time}")
                    return
                else:
                    print(f"⚠️ Datos de MT5 son antiguos ({time_diff})")
                    print("🔄 Generando datos simulados actuales...")
        
        # Si no hay MT5 o datos antiguos, generar datos simulados ACTUALES
        print("📊 Generando datos simulados con fechas ACTUALES...")
        
        np.random.seed(42)
        base_price = 6000  # Precio más actual del SP500
        
        # Generar fechas realistas (datos por minutos)
        current_time = self.start_date
        self.dates = [current_time]
        
        # Generar precios con diferentes fases
        prices = [base_price]
        volumes = []
        
        for i in range(1, n_points):
            # Generar fecha (saltar fines de semana)
            current_time += timedelta(minutes=1)
            if current_time.weekday() >= 5:  # Sábado (5) o Domingo (6)
                current_time += timedelta(days=2)  # Saltar al lunes
            self.dates.append(current_time)
            
            prev_price = prices[-1]
            
            # Tendencia cíclica
            cycle = np.sin(i / 100) * 0.001
            trend = 0.0002
            
            # Volatilidad variable
            vol_cycle = 0.008 + 0.004 * np.sin(i / 200)
            noise = np.random.normal(0, vol_cycle)
            
            # Mean reversion
            mean_rev = -0.03 * (prev_price - base_price) / base_price
            
            price_change = trend + cycle + noise + mean_rev
            new_price = prev_price * (1 + price_change)
            
            # Límites
            new_price = max(new_price, base_price * 0.85)
            new_price = min(new_price, base_price * 1.30)
            
            prices.append(new_price)
            
            # Volumen correlacionado con volatilidad
            volume = 1000000 * (1 + abs(price_change) * 5) * np.random.uniform(0.8, 1.2)
            volumes.append(max(volume, 600000))
        
        # Asegurar misma longitud
        min_length = min(len(prices), len(volumes))
        prices = prices[:min_length]
        volumes = volumes[:min_length]
        
        # Crear DataFrame con fechas
        df = pd.DataFrame({
            'price': prices,
            'volume': volumes,
            'time': self.dates
        })
        
        # Calcular indicadores
        df = self.calculate_indicators(df)
        
        self.data = df
        print(f"✅ {len(df)} registros simulados generados: ${min(prices):.2f} - ${max(prices):.2f}")
        
        # Inicializar arrays de seguimiento
        self.initialize_tracking_arrays()
    
    def initialize_tracking_arrays(self):
        """Inicializar arrays de seguimiento"""
        n = len(self.data)
        self.portfolio_values = [self.initial_capital] * n
        self.buy_signals = []
        self.sell_signals = []
        self.stop_losses = []
        self.take_profits = []
        self.trades_history = []
        self.actions = [0.0] * n
        self.ml_predictions = [2] * n  # 2 = HOLD
        self.technical_signals = [0.0] * n
    
    def calculate_indicators(self, df):
        """Calcular todos los indicadores técnicos"""
        
        # Básicos
        df['sma_10'] = df['price'].rolling(10).mean()
        df['sma_20'] = df['price'].rolling(20).mean()
        df['sma_50'] = df['price'].rolling(50).mean()
        df['ema_12'] = df['price'].ewm(span=12).mean()
        df['ema_26'] = df['price'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_normalized'] = df['macd'] / df['price']  # Normalizar para ML
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_normalized'] = df['rsi'] / 100  # Normalizar para ML
        
        # Bandas de Bollinger
        df['bb_middle'] = df['price'].rolling(20).mean()
        bb_std = df['price'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Otros
        df['returns'] = df['price'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum'] = df['price'].pct_change(5)
        
        # Volumen
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def prepare_ml_features(self, step):
        """Preparar features para el modelo ML"""
        if step < 50:
            return None
        
        row = self.data.iloc[step]
        
        # Features normalizados para ML
        features = [
            row['price'] / 5000,  # Normalizar precio
            row['rsi_normalized'],
            row['macd_normalized'], 
            row['bb_position'],
            row['volume_ratio'] / 2,  # Normalizar volumen
            row['momentum'],
            row['volatility'] * 10,  # Escalar volatilidad
            (row['sma_10'] - row['sma_20']) / row['price'],  # Tendencia
        ]
        
        # Limpiar NaN
        features = [f if not pd.isna(f) else 0.0 for f in features]
        
        return np.array(features)
    
    def generate_ml_signal(self, step):
        """Generar señal usando modelo ML"""
        if not self.ml_model:
            return 0.0
        
        features = self.prepare_ml_features(step)
        if features is None:
            return 0.0
        
        try:
            prediction = self.ml_model.predict(features.reshape(1, -1))[0][0]
            
            # Convertir predicción a señal
            if prediction == 1:  # BUY
                return 0.7
            elif prediction == 0:  # SELL
                return -0.7
            else:  # HOLD
                return 0.0
                
        except Exception as e:
            print(f"Error en predicción ML: {e}")
            return 0.0
    
    def generate_technical_signal(self, step):
        """Generar señal técnica tradicional MEJORADA"""
        if step < 50:
            return 0.0
        
        row = self.data.iloc[step]
        signals = []
        
        # 1. RSI - más sensible
        if row['rsi'] < 35:  # Era 25
            signals.append(0.4)
        elif row['rsi'] > 65:  # Era 75
            signals.append(-0.4)
        
        # 2. MACD - más sensible
        if row['macd_histogram'] > 0 and row['macd'] > row['macd_signal']:
            signals.append(0.3)
        elif row['macd_histogram'] < 0 and row['macd'] < row['macd_signal']:
            signals.append(-0.3)
        
        # 3. Bollinger Bands - más sensible
        if row['bb_position'] < 0.25:  # Era 0.05
            signals.append(0.3)
        elif row['bb_position'] > 0.75:  # Era 0.95
            signals.append(-0.3)
        
        # 4. Tendencia
        if row['price'] > row['sma_20']:
            signals.append(0.2)
        elif row['price'] < row['sma_20']:
            signals.append(-0.2)
        
        # 5. Momentum
        if row['momentum'] > 0.01:  # Era 0.03
            signals.append(0.2)
        elif row['momentum'] < -0.01:
            signals.append(-0.2)
        
        # 6. Volumen - menos restrictivo
        if row['volume_ratio'] > 1.1:  # Era 1.2
            if any(s > 0 for s in signals):
                signals.append(0.1)
            elif any(s < 0 for s in signals):
                signals.append(-0.1)
        
        # Solo necesitamos 1 confirmación (era 2)
        if len(signals) >= 1:
            return np.clip(sum(signals), -1.0, 1.0)
        else:
            return 0.0
    
    def generate_combined_signal(self, step):
        """Combinar señales ML + técnicas"""
        ml_signal = self.generate_ml_signal(step) if self.use_ml_signals else 0.0
        tech_signal = self.generate_technical_signal(step)
        
        # Guardar para visualización
        self.ml_predictions[step] = 1 if ml_signal > 0.3 else (0 if ml_signal < -0.3 else 2)
        self.technical_signals[step] = tech_signal
        
        # Combinar señales
        if self.use_ml_signals and self.ml_model:
            combined = (ml_signal * self.ml_weight) + (tech_signal * (1 - self.ml_weight))
        else:
            combined = tech_signal
        
        return combined
    
    def execute_trade(self, step, signal):
        """Ejecutar trade con validaciones relajadas"""
        current_price = self.data.iloc[step]['price']
        
        # Verificaciones básicas (RELAJADAS)
        if step - self.last_trade_step < self.min_trade_separation:
            return False
        
        if self.daily_trades >= self.max_daily_trades:
            return False
        
        if self.consecutive_losses >= 4:  # Era 3
            return False
        
        # COMPRA - umbral más bajo
        if signal > 0.25 and self.position_size == 0:  # Era 0.3
            position_size = self.calculate_position_size(current_price)
            
            if position_size > 0:
                self.position_size = position_size
                self.position_type = 'LONG'
                self.entry_price = current_price
                self.last_trade_step = step
                self.daily_trades += 1
                
                self.buy_signals.append(step)
                self.total_trades += 1
                
                trade_info = {
                    'step': step,
                    'type': 'BUY',
                    'price': current_price,
                    'size': position_size,
                    'signal': signal,
                    'ml_pred': self.ml_predictions[step],
                    'tech_signal': self.technical_signals[step]
                }
                self.trades_history.append(trade_info)
                
                print(f"🟢 COMPRA: ${current_price:.2f} | Tamaño: {position_size} | Señal: {signal:.3f} | ML: {self.ml_predictions[step]} | Tech: {self.technical_signals[step]:.2f}")
                return True
        
        # VENTA - umbral más bajo
        elif signal < -0.25 and self.position_type == 'LONG':  # Era -0.3
            profit = (current_price - self.entry_price) * self.position_size
            self.current_capital += profit
            
            self.sell_signals.append(step)
            
            if profit > 0:
                self.profitable_trades += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
            
            print(f"🔴 VENTA: ${current_price:.2f} | P&L: ${profit:.2f} | Señal: {signal:.3f}")
            
            self.position_size = 0
            self.position_type = None
            self.last_trade_step = step
            
            return True
        
        return False
    
    def calculate_position_size(self, entry_price):
        """Calcular tamaño de posición"""
        if entry_price <= 0:
            return 0
        
        stop_loss_price = entry_price * (1 - self.stop_loss_pct)
        risk_per_share = entry_price - stop_loss_price
        
        max_risk_capital = self.current_capital * self.max_position_risk
        shares = int(max_risk_capital / risk_per_share)
        
        max_shares_by_capital = int(self.current_capital * 0.9 / entry_price)
        shares = min(shares, max_shares_by_capital)
        
        return max(shares, 0)
    
    def check_exit_conditions(self, step):
        """Verificar stop loss y take profit"""
        if self.position_size == 0:
            return None
        
        current_price = self.data.iloc[step]['price']
        
        if self.position_type == 'LONG':
            stop_price = self.entry_price * (1 - self.stop_loss_pct)
            profit_price = self.entry_price * (1 + self.take_profit_pct)
            
            if current_price <= stop_price:
                return 'STOP_LOSS'
            elif current_price >= profit_price:
                return 'TAKE_PROFIT'
        
        return None
    
    def step_forward(self):
        """Avanzar un paso"""
        if self.current_step >= len(self.data) - 1:
            return
        
        self.current_step += 1
        current_price = self.data.iloc[self.current_step]['price']
        
        # Reset daily trades cada 50 steps (simular nuevo día)
        if self.current_step % 50 == 0:
            self.daily_trades = 0
        
        # Verificar exits
        exit_condition = self.check_exit_conditions(self.current_step)
        
        if exit_condition == 'STOP_LOSS':
            profit = (current_price - self.entry_price) * self.position_size
            self.current_capital += profit
            self.stop_losses.append(self.current_step)
            self.consecutive_losses += 1
            
            print(f"🛑 STOP LOSS: ${current_price:.2f} | Pérdida: ${profit:.2f}")
            self.position_size = 0
            self.position_type = None
            
        elif exit_condition == 'TAKE_PROFIT':
            profit = (current_price - self.entry_price) * self.position_size
            self.current_capital += profit
            self.take_profits.append(self.current_step)
            self.profitable_trades += 1
            self.consecutive_losses = 0
            
            print(f"🎯 TAKE PROFIT: ${current_price:.2f} | Ganancia: ${profit:.2f}")
            self.position_size = 0
            self.position_type = None
        
        else:
            # Generar señal combinada
            signal = self.generate_combined_signal(self.current_step)
            self.actions[self.current_step] = signal
            
            # Ejecutar trade si es necesario
            if abs(signal) > 0.25:  # Umbral más bajo
                self.execute_trade(self.current_step, signal)
        
        # Actualizar portfolio
        portfolio_value = self.current_capital
        if self.position_size > 0:
            portfolio_value += self.position_size * current_price
        
        self.portfolio_values[self.current_step] = portfolio_value
        
        # Actualizar drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def create_interface(self):
        """Crear interfaz interactiva mejorada"""
        print("🎮 Iniciando interfaz IA + Técnico...")
        
        # Intentar cargar modelo ML
        if not self.load_ml_model():
            self.create_simple_ml_model()
        
        # Generar datos
        self.generate_market_data()
        
        # Configurar figura con tamaño optimizado
        self.fig = plt.figure(figsize=(26, 18))  # Más grande para aprovechar todo el espacio
        self.fig.suptitle('🤖 Sistema de Trading con IA + Análisis Técnico - Dashboard Optimizado', 
                         fontsize=18, fontweight='bold', y=0.97)  # Título más grande
        
        # Grid optimizado: más espacio vertical para evitar superposición
        gs = self.fig.add_gridspec(4, 6, height_ratios=[3.5, 1.8, 1.8, 0.25], 
                                   hspace=0.5, wspace=0.25)  # Más espacio vertical
        
        # Layout optimizado para gráficos más grandes
        self.ax_price = self.fig.add_subplot(gs[0, :4])      # Gráfico principal - MÁS ALTO
        self.ax_info1 = self.fig.add_subplot(gs[0, 4])       # Panel info superior
        self.ax_info2 = self.fig.add_subplot(gs[0, 5])       # Panel info inferior
        
        # Segunda fila - gráficos más grandes
        self.ax_rsi = self.fig.add_subplot(gs[1, :2])        # RSI más grande
        self.ax_portfolio = self.fig.add_subplot(gs[1, 2:4]) # Portfolio más grande
        self.ax_signals = self.fig.add_subplot(gs[1, 4:])    # Señales más grande
        
        # Tercera fila - gráficos más grandes
        self.ax_ml = self.fig.add_subplot(gs[2, :2])         # ML más grande
        self.ax_macd = self.fig.add_subplot(gs[2, 2:4])      # MACD más grande
        self.ax_volume = self.fig.add_subplot(gs[2, 4:])     # Volumen más grande
        
        # Títulos con más padding para evitar superposición
        self.ax_price.set_title('📈 Precio + Señales IA/Técnico', fontsize=14, pad=25, fontweight='bold')
        self.ax_rsi.set_title('📊 RSI (Relative Strength Index)', fontsize=12, pad=20, fontweight='bold')
        self.ax_portfolio.set_title('💼 Portfolio vs Buy & Hold', fontsize=12, pad=20, fontweight='bold')
        self.ax_signals.set_title('🎯 Señales Técnicas Combinadas', fontsize=12, pad=20, fontweight='bold')
        self.ax_ml.set_title('🤖 Predicciones Machine Learning', fontsize=12, pad=20, fontweight='bold')
        self.ax_macd.set_title('📊 MACD (Moving Average Convergence Divergence)', fontsize=12, pad=20, fontweight='bold')
        self.ax_volume.set_title('📊 Volumen de Transacciones', fontsize=12, pad=20, fontweight='bold')
        
        # Configurar paneles de información
        self.ax_info1.axis('off')
        self.ax_info2.axis('off')
        
        # Ajustar tamaños de fuente más grandes para mejor legibilidad
        for ax in [self.ax_price, self.ax_rsi, self.ax_portfolio, self.ax_signals, 
                   self.ax_ml, self.ax_macd, self.ax_volume]:
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=9)
            # Mejorar la apariencia de las leyendas
            ax.grid(True, alpha=0.3, linewidth=0.8)
        
        # Controles
        self.create_controls()
        
        # Animación con intervalo más rápido para tiempo real
        self.animation = FuncAnimation(self.fig, self.update_plots, 
                                     interval=50, blit=False, cache_frame_data=False)
        
        # Ajustar márgenes para evitar superposición de títulos
        plt.subplots_adjust(top=0.94, bottom=0.06, left=0.02, right=0.99, 
                           hspace=0.55, wspace=0.2)
        plt.show()
    
    def create_controls(self):
        """Crear controles con botón de tiempo real"""
        # Botones principales
        ax_play = plt.axes([0.05, 0.02, 0.05, 0.04])
        ax_pause = plt.axes([0.11, 0.02, 0.05, 0.04])
        ax_stop = plt.axes([0.17, 0.02, 0.05, 0.04])
        ax_back = plt.axes([0.23, 0.02, 0.05, 0.04])
        ax_forward = plt.axes([0.29, 0.02, 0.05, 0.04])
        ax_realtime = plt.axes([0.35, 0.02, 0.08, 0.04])
        
        self.btn_play = Button(ax_play, '▶️')
        self.btn_pause = Button(ax_pause, '⏸️')
        self.btn_stop = Button(ax_stop, '⏹️')
        self.btn_back = Button(ax_back, '⏪')
        self.btn_forward = Button(ax_forward, '⏩')
        self.btn_realtime = Button(ax_realtime, '🤖 AUTO')
        
        # Sliders
        ax_speed = plt.axes([0.48, 0.02, 0.12, 0.04])
        ax_ml_weight = plt.axes([0.65, 0.02, 0.12, 0.04])
        
        self.slider_speed = Slider(ax_speed, 'Velocidad', 0.25, 4.0, valinit=1.0)
        self.slider_ml_weight = Slider(ax_ml_weight, 'Peso IA', 0.0, 1.0, valinit=0.6)
        
        # Eventos
        self.btn_play.on_clicked(lambda x: setattr(self, 'is_playing', True))
        self.btn_pause.on_clicked(lambda x: setattr(self, 'is_playing', False))
        self.btn_stop.on_clicked(lambda x: self.reset())
        self.btn_back.on_clicked(lambda x: self.step_back())
        self.btn_forward.on_clicked(lambda x: self.step_manual_forward())
        self.btn_realtime.on_clicked(lambda x: self.toggle_real_time())
        self.slider_speed.on_changed(lambda x: setattr(self, 'speed', x))
        self.slider_ml_weight.on_changed(lambda x: setattr(self, 'ml_weight', x))
    
    def toggle_real_time(self):
        """Activar/Desactivar TRADER AUTOMÁTICO"""
        print("\n" + "="*60)
        
        if not self.use_real_time:
            print("🤖 ACTIVANDO TRADER AUTOMÁTICO...")
            print("="*60)
            print("⚠️  ATENCIÓN: El sistema tomará decisiones de compra/venta automáticamente")
            print("📊 Basado en análisis técnico + Machine Learning")
            print("💰 Capital inicial: ${:,.0f}".format(self.current_capital))
            
            if self.connect_mt5():
                self.start_real_time_updates()
                print("\n🎉 ¡TRADER AUTOMÁTICO ACTIVADO!")
                print(f"📡 Analizando {self.symbol} cada minuto")
                print("🤖 El sistema decidirá automáticamente comprar/vender")
                print("📈 Verás las decisiones en tiempo real en la consola")
            else:
                print("❌ No se pudo conectar a MT5")
                print("🎮 Usa el modo simulación presionando ▶️")
        else:
            print("🛑 DESACTIVANDO TRADER AUTOMÁTICO...")
            self.stop_real_time_updates()
            print("✅ Trader automático desactivado")
            print("🎮 Puedes usar modo simulación con ▶️")
        
        print("="*60)
    
    def update_plots(self, frame):
        """Actualizar visualización"""
        # En modo tiempo real, siempre actualizar, sino solo si está playing
        if not self.is_playing and not self.use_real_time:
            return
        
        # Avanzar
        for _ in range(int(self.speed)):
            if self.current_step < len(self.data) - 1:
                self.step_forward()
        
        # Limpiar
        for ax in [self.ax_price, self.ax_rsi, self.ax_portfolio, 
                   self.ax_signals, self.ax_ml, self.ax_macd, self.ax_volume, self.ax_info1, self.ax_info2]:
            ax.clear()
        
        # Ventana de datos y fechas
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = min(len(self.data), self.current_step + 1)
        window_steps = range(start_idx, end_idx)
        
        # Obtener fechas para el eje X
        if 'time' in self.data.columns:
            try:
                window_dates = self.data['time'].iloc[start_idx:end_idx]
                # Verificar que las fechas sean válidas
                if len(window_dates) > 0 and pd.notna(window_dates).all():
                    use_dates = True
                else:
                    window_dates = window_steps
                    use_dates = False
            except:
                window_dates = window_steps
                use_dates = False
        else:
            window_dates = window_steps
            use_dates = False
        
        # 1. Precio
        prices = self.data['price'].iloc[start_idx:end_idx]
        sma20 = self.data['sma_20'].iloc[start_idx:end_idx]
        bb_upper = self.data['bb_upper'].iloc[start_idx:end_idx]
        bb_lower = self.data['bb_lower'].iloc[start_idx:end_idx]
        
        self.ax_price.plot(window_dates, prices, 'b-', linewidth=1.5, label='Precio')
        self.ax_price.plot(window_dates, sma20, 'orange', alpha=0.7, label='SMA 20')
        self.ax_price.fill_between(window_dates, bb_upper, bb_lower, alpha=0.1, color='gray')
        
        # Señales
        window_buys = [s for s in self.buy_signals if start_idx <= s <= end_idx]
        window_sells = [s for s in self.sell_signals if start_idx <= s <= end_idx]
        window_stops = [s for s in self.stop_losses if start_idx <= s <= end_idx]
        window_profits = [s for s in self.take_profits if start_idx <= s <= end_idx]
        
        if window_buys:
            buy_prices = [self.data['price'].iloc[s] for s in window_buys]
            if use_dates:
                buy_dates = [self.data['time'].iloc[s] for s in window_buys]
                self.ax_price.scatter(buy_dates, buy_prices, c='green', marker='^', 
                                    s=120, label='Compras', edgecolors='darkgreen', linewidth=2)
            else:
                self.ax_price.scatter(window_buys, buy_prices, c='green', marker='^', 
                                    s=120, label='Compras', edgecolors='darkgreen', linewidth=2)
        
        if window_sells:
            sell_prices = [self.data['price'].iloc[s] for s in window_sells]
            if use_dates:
                sell_dates = [self.data['time'].iloc[s] for s in window_sells]
                self.ax_price.scatter(sell_dates, sell_prices, c='red', marker='v', 
                                    s=120, label='Ventas', edgecolors='darkred', linewidth=2)
            else:
                self.ax_price.scatter(window_sells, sell_prices, c='red', marker='v', 
                                    s=120, label='Ventas', edgecolors='darkred', linewidth=2)
        
        if window_stops:
            stop_prices = [self.data['price'].iloc[s] for s in window_stops]
            if use_dates:
                stop_dates = [self.data['time'].iloc[s] for s in window_stops]
                self.ax_price.scatter(stop_dates, stop_prices, c='red', marker='x', s=150)
            else:
                self.ax_price.scatter(window_stops, stop_prices, c='red', marker='x', s=150)
        
        if window_profits:
            profit_prices = [self.data['price'].iloc[s] for s in window_profits]
            if use_dates:
                profit_dates = [self.data['time'].iloc[s] for s in window_profits]
                self.ax_price.scatter(profit_dates, profit_prices, c='gold', marker='*', s=150)
            else:
                self.ax_price.scatter(window_profits, profit_prices, c='gold', marker='*', s=150)
        
        current_price = self.data['price'].iloc[self.current_step]
        if use_dates:
            current_date = self.data['time'].iloc[self.current_step]
            self.ax_price.axvline(current_date, color='red', linestyle='--', alpha=0.7)
            self.ax_price.scatter([current_date], [current_price], c='red', s=100)
        else:
            self.ax_price.axvline(self.current_step, color='red', linestyle='--', alpha=0.7)
            self.ax_price.scatter([self.current_step], [current_price], c='red', s=100)
        
        self.ax_price.set_title('📈 Precio + Señales IA/Técnico', fontsize=14, pad=25, fontweight='bold')
        self.ax_price.legend(fontsize=10, loc='upper left')
        self.ax_price.grid(True, alpha=0.3, linewidth=0.8)
        
        # Formatear fechas en el eje X si están disponibles
        if use_dates:
            self.ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d/%m'))
            self.ax_price.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.setp(self.ax_price.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. RSI
        rsi_data = self.data['rsi'].iloc[start_idx:end_idx]
        self.ax_rsi.plot(window_dates, rsi_data, 'purple')
        self.ax_rsi.axhline(70, color='red', linestyle='--', alpha=0.7)
        self.ax_rsi.axhline(30, color='green', linestyle='--', alpha=0.7)
        self.ax_rsi.axhline(65, color='red', linestyle=':', alpha=0.5, label='Nuevos umbrales')
        self.ax_rsi.axhline(35, color='green', linestyle=':', alpha=0.5)
        self.ax_rsi.fill_between(window_dates, 65, 100, alpha=0.1, color='red')
        self.ax_rsi.fill_between(window_dates, 0, 35, alpha=0.1, color='green')
        if use_dates:
            self.ax_rsi.axvline(current_date, color='red', linestyle='--', alpha=0.5)
        else:
            self.ax_rsi.axvline(self.current_step, color='red', linestyle='--', alpha=0.5)
        self.ax_rsi.set_title('📊 RSI (Relative Strength Index)', fontsize=12, pad=20, fontweight='bold')
        self.ax_rsi.set_ylim(0, 100)
        self.ax_rsi.grid(True, alpha=0.3, linewidth=0.8)
        
        # 3. Portfolio vs Buy&Hold
        portfolio_data = self.portfolio_values[start_idx:end_idx]
        self.ax_portfolio.plot(window_dates, portfolio_data, 'green', linewidth=2, label='IA+Técnico')
        
        if len(prices) > 0:
            initial_price = self.data['price'].iloc[0]
            buy_hold_values = [(self.data['price'].iloc[i] / initial_price) * self.initial_capital 
                              for i in window_steps]
            self.ax_portfolio.plot(window_dates, buy_hold_values, 'gray', linewidth=2, 
                                  linestyle='--', label='Buy & Hold')
        
        self.ax_portfolio.axhline(self.initial_capital, color='black', alpha=0.5)
        if use_dates:
            self.ax_portfolio.axvline(current_date, color='red', linestyle='--', alpha=0.5)
        else:
            self.ax_portfolio.axvline(self.current_step, color='red', linestyle='--', alpha=0.5)
        self.ax_portfolio.set_title('💼 Portfolio vs Buy & Hold', fontsize=12, pad=20, fontweight='bold')
        self.ax_portfolio.legend(fontsize=10)
        self.ax_portfolio.grid(True, alpha=0.3, linewidth=0.8)
        
        # 4. Señales técnicas
        tech_signals = self.technical_signals[start_idx:end_idx]
        self.ax_signals.plot(window_dates, tech_signals, 'blue', linewidth=1.5)
        self.ax_signals.axhline(0.25, color='green', linestyle='--', alpha=0.7, label='Umbrales')
        self.ax_signals.axhline(-0.25, color='red', linestyle='--', alpha=0.7)
        self.ax_signals.axhline(0, color='gray', alpha=0.5)
        self.ax_signals.fill_between(window_dates, 0.25, 1, alpha=0.1, color='green')
        self.ax_signals.fill_between(window_dates, -0.25, -1, alpha=0.1, color='red')
        if use_dates:
            self.ax_signals.axvline(current_date, color='red', linestyle='--', alpha=0.5)
        else:
            self.ax_signals.axvline(self.current_step, color='red', linestyle='--', alpha=0.5)
        self.ax_signals.set_title('🎯 Señales Técnicas Combinadas', fontsize=12, pad=20, fontweight='bold')
        self.ax_signals.set_ylim(-1, 1)
        self.ax_signals.legend(fontsize=10)
        self.ax_signals.grid(True, alpha=0.3, linewidth=0.8)
        
        # 5. Predicciones ML
        ml_preds = self.ml_predictions[start_idx:end_idx]
        colors = ['red' if p == 0 else 'green' if p == 1 else 'gray' for p in ml_preds]
        self.ax_ml.scatter(window_dates, ml_preds, c=colors, alpha=0.6, s=30)
        self.ax_ml.axhline(1, color='green', linestyle='--', alpha=0.7, label='Buy')
        self.ax_ml.axhline(0, color='red', linestyle='--', alpha=0.7, label='Sell')
        self.ax_ml.axhline(2, color='gray', linestyle='--', alpha=0.7, label='Hold')
        if use_dates:
            self.ax_ml.axvline(current_date, color='red', linestyle='--', alpha=0.5)
        else:
            self.ax_ml.axvline(self.current_step, color='red', linestyle='--', alpha=0.5)
        self.ax_ml.set_title('🤖 Predicciones Machine Learning', fontsize=12, pad=20, fontweight='bold')
        self.ax_ml.set_ylim(-0.5, 2.5)
        self.ax_ml.legend(fontsize=10)
        self.ax_ml.grid(True, alpha=0.3, linewidth=0.8)
        
        # 6. MACD
        macd_data = self.data['macd'].iloc[start_idx:end_idx]
        macd_signal_data = self.data['macd_signal'].iloc[start_idx:end_idx]
        macd_hist = self.data['macd_histogram'].iloc[start_idx:end_idx]
        
        self.ax_macd.plot(window_dates, macd_data, 'blue', label='MACD')
        self.ax_macd.plot(window_dates, macd_signal_data, 'red', label='Signal')
        if use_dates:
            # Para barras con fechas, usar un ancho apropiado
            self.ax_macd.bar(window_dates, macd_hist, alpha=0.3, width=timedelta(minutes=30))
            self.ax_macd.axvline(current_date, color='red', linestyle='--', alpha=0.5)
        else:
            self.ax_macd.bar(window_dates, macd_hist, alpha=0.3, width=0.8)
            self.ax_macd.axvline(self.current_step, color='red', linestyle='--', alpha=0.5)
        self.ax_macd.axhline(0, color='gray', alpha=0.5)
        self.ax_macd.set_title('📊 MACD (Moving Average Convergence Divergence)', fontsize=12, pad=20, fontweight='bold')
        self.ax_macd.legend(fontsize=10)
        self.ax_macd.grid(True, alpha=0.3, linewidth=0.8)
        
        # 7. Volumen
        volume_data = self.data['volume'].iloc[start_idx:end_idx]
        volume_sma = self.data['volume_sma'].iloc[start_idx:end_idx]
        
        if use_dates:
            self.ax_volume.bar(window_dates, volume_data, alpha=0.6, color='lightblue', width=timedelta(minutes=30))
            self.ax_volume.axvline(current_date, color='red', linestyle='--', alpha=0.5)
        else:
            self.ax_volume.bar(window_dates, volume_data, alpha=0.6, color='lightblue', width=0.8)
            self.ax_volume.axvline(self.current_step, color='red', linestyle='--', alpha=0.5)
        self.ax_volume.plot(window_dates, volume_sma, 'red', label='Volume SMA')
        self.ax_volume.set_title('📊 Volumen de Transacciones', fontsize=12, pad=20, fontweight='bold')
        self.ax_volume.legend(fontsize=10)
        self.ax_volume.grid(True, alpha=0.3, linewidth=0.8)
        
        # 8. Paneles de información divididos
        self.ax_info1.axis('off')
        self.ax_info2.axis('off')
        
        current_value = self.portfolio_values[self.current_step]
        total_return = (current_value / self.initial_capital - 1) * 100
        win_rate = (self.profitable_trades / max(self.total_trades, 1)) * 100
        
        position_info = "SIN POSICIÓN"
        if self.position_size > 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position_size
            position_info = f"LONG {self.position_size}\nEntry: ${self.entry_price:.2f}\nP&L: ${unrealized_pnl:.2f}"
        
        # Panel 1: Estado actual y posición
        current_time_str = "Simulación"
        if 'time' in self.data.columns:
            current_time_str = self.data['time'].iloc[self.current_step].strftime('%H:%M %d/%m/%Y')
        
        # Verificar si los datos son realmente actuales
        is_current_data = False
        if self.mt5_connected and 'time' in self.data.columns:
            latest_time = self.data['time'].max()
            from datetime import datetime, timezone
            now = datetime.now()
            time_diff = now - latest_time.replace(tzinfo=None)
            is_current_data = time_diff.total_seconds() < 3600  # Menos de 1 hora
        
        if self.use_real_time and is_current_data:
            rt_status = "🟢 TIEMPO REAL"
            data_source = "📡 MT5 Actual"
        elif self.use_real_time and self.mt5_connected:
            rt_status = "🟡 MT5 HISTÓRICO"
            data_source = "📊 MT5 Antiguo"
        elif self.use_real_time:
            rt_status = "🔴 SIN MT5"
            data_source = "📊 Simulado"
        else:
            rt_status = "⚪ SIMULACIÓN"
            data_source = "📊 Simulado"
        
        info_text1 = f"""🤖 SISTEMA IA + TÉCNICO

⏰ Tiempo: {current_time_str}
🕐 Step: {self.current_step}/{len(self.data)-1}
💰 Precio: ${current_price:.2f}
📈 Portfolio: ${current_value:,.0f}
📊 Return: {total_return:+.2f}%
📉 Drawdown: {self.max_drawdown:.1%}

🎯 POSICIÓN:
{position_info}

📡 Tiempo Real: {rt_status}
📊 Datos: {data_source}
🎮 Velocidad: {self.speed:.1f}x
🤖 Modelo: {'IA' if HAS_RL else 'Técnico'}"""
        
        # Panel 2: Configuración y estadísticas
        info_text2 = f"""📊 CONFIGURACIÓN:

• Peso IA: {self.ml_weight:.0%}
• Peso Técnico: {1-self.ml_weight:.0%}
• Max trades/día: {self.max_daily_trades}
• Trades hoy: {self.daily_trades}
• Separación mín: {self.min_trade_separation}

🔄 ESTADÍSTICAS:
Trades: {self.total_trades}
Compras: {len(self.buy_signals)}
Ventas: {len(self.sell_signals)}
Stops: {len(self.stop_losses)}
Profits: {len(self.take_profits)}
Win rate: {win_rate:.1f}%"""
        
        # Mostrar ambos paneles con fuente más grande
        self.ax_info1.text(0.05, 0.95, info_text1, transform=self.ax_info1.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.9))
        
        self.ax_info2.text(0.05, 0.95, info_text2, transform=self.ax_info2.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.9))
    
    def reset(self):
        """Reiniciar simulación"""
        self.is_playing = False
        self.current_step = 50
        self.current_capital = self.initial_capital
        self.position_size = 0
        self.position_type = None
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.profitable_trades = 0
        
        # Limpiar tracking
        self.buy_signals.clear()
        self.sell_signals.clear()
        self.stop_losses.clear()
        self.take_profits.clear()
        self.trades_history.clear()
        
        print("⏹️ Sistema reiniciado")
    
    def step_back(self):
        """Retroceder"""
        if self.current_step > 60:
            self.current_step -= 20
            print(f"⏪ Retrocediendo a step {self.current_step}")
    
    def step_manual_forward(self):
        """Avanzar manualmente"""
        for _ in range(10):
            if self.current_step < len(self.data) - 1:
                self.step_forward()
        print(f"⏩ Avanzando a step {self.current_step}")

def main():
    """
    🚀 SISTEMA DE TRADING IA + TÉCNICO CON TIEMPO REAL
    
    Características:
    - ✅ Datos reales de MetaTrader5 (US500/SP500)
    - ✅ Fechas reales en lugar de steps
    - ✅ Actualizaciones en tiempo real (botón 📡 RT)
    - ✅ Dashboard optimizado y sin superposiciones
    - ✅ Análisis técnico + Machine Learning
    - ✅ Gestión de riesgo profesional
    """
    print("=" * 80)
    print("🤖 INICIANDO SISTEMA DE TRADING IA + TÉCNICO CON TIEMPO REAL")
    print("=" * 80)
    print()
    print("📋 CARACTERÍSTICAS:")
    print("  • 📡 Datos en tiempo real de MetaTrader5")
    print("  • 📅 Fechas reales en gráficos")
    print("  • 🤖 Machine Learning + Análisis Técnico")
    print("  • 📊 Dashboard optimizado")
    print("  • 🎮 Controles interactivos")
    print()
    print("🎯 CONTROLES:")
    print("  • ▶️ Play - Iniciar simulación")
    print("  • ⏸️ Pause - Pausar simulación")
    print("  • 📡 RT - Activar/Desactivar tiempo real")
    print("  • Sliders - Ajustar velocidad y peso IA")
    print()
    
    try:
        system = MLEnhancedTradingSystem()
        system.create_interface()
        
    except KeyboardInterrupt:
        print("\n🛑 Sistema detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Asegúrate de tener MetaTrader5 instalado y configurado")
    finally:
        # Limpiar recursos
        try:
            if HAS_MT5:
                mt5.shutdown()
        except:
            pass
        print("✅ Sistema finalizado correctamente")

if __name__ == "__main__":
    main() 