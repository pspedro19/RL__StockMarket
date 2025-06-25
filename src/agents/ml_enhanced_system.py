"""
🤖 SISTEMA DE TRADING CON IA + TÉCNICO
Combina modelo DQN/DeepDQN/A2C entrenado con análisis técnico tradicional
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
import gymnasium as gym
from gymnasium import spaces
import os
warnings.filterwarnings('ignore')

# Intentar importar MT5
try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
    print("✅ MetaTrader5 disponible")
except ImportError:
    HAS_MT5 = False
    print("⚠️ MetaTrader5 no disponible - usando datos simulados")

# Intentar importar componentes de RL
try:
    from stable_baselines3 import DQN, A2C, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.dqn.policies import MlpPolicy
    HAS_RL = True
    print("✅ Componentes de RL disponibles")
except ImportError:
    HAS_RL = False
    print("⚠️ Sin componentes de RL - usando solo técnico")

# Algoritmos disponibles
AVAILABLE_ALGORITHMS = {
    "1": ("DQN", DQN),
    "2": ("DeepDQN", DQN),  # DeepDQN es una variante de DQN con una red más profunda
    "3": ("A2C", A2C),
    "4": ("Técnico", None)
}

def select_algorithm():
    """Permite al usuario seleccionar el algoritmo de IA a utilizar"""
    print("\n🤖 Selecciona el algoritmo de IA a utilizar:")
    print("1. DQN (Deep Q-Network)")
    print("2. DeepDQN (Deep Q-Network con red neuronal profunda)")
    print("3. A2C (Advantage Actor-Critic)")
    print("4. Solo Análisis Técnico")
    
    while True:
        choice = input("\nIngresa el número de tu elección (1-4): ").strip()
        if choice in AVAILABLE_ALGORITHMS:
            algorithm_name, algorithm_class = AVAILABLE_ALGORITHMS[choice]
            return choice, algorithm_name, algorithm_class
        print("❌ Opción inválida. Por favor, selecciona un número del 1 al 4.")

class MLEnhancedTradingSystem(gym.Env):
    """Sistema de trading que combina IA + análisis técnico con datos en tiempo real"""
    
    def __init__(self, skip_selection=False):
        """Inicializar sistema de trading"""
        super().__init__()
        
        # Configuración de RL - SELECCIONAR ALGORITMO PRIMERO
        if not skip_selection:
            self.algorithm_choice, self.algorithm_name, self.algorithm_class = select_algorithm()
        else:
            self.algorithm_choice, self.algorithm_name, self.algorithm_class = "2", "DeepDQN", DQN
        
        # Configuración inicial
        self.symbol = "US500"  # SP500
        self.timeframe = mt5.TIMEFRAME_M1  # 1 minuto
        self.start_date = datetime.now() - timedelta(days=1)
        self.data = None
        self.current_step = 0
        self.mt5_connected = False
        self.use_real_time = False
        self.last_update = datetime.now()
        self.real_time_thread = None
        
        # Parámetros de trading
        self.initial_capital = 100000  # Capital inicial
        self.current_capital = self.initial_capital  # Capital actual
        self.balance = self.initial_capital  # Balance actual
        self.position_size = 0  # Tamaño de posición actual
        self.position_type = None  # Tipo de posición actual
        self.entry_price = 0  # Precio de entrada
        self.max_position_risk = 0.02  # Máximo riesgo por posición (2%)
        self.stop_loss_pct = 0.01  # Stop loss (1%)
        self.take_profit_pct = 0.02  # Take profit (2%)
        self.min_trade_separation = 5  # Mínimo de velas entre trades
        self.max_daily_trades = 5  # Máximo de trades por día
        self.consecutive_losses = 2  # Máximo de pérdidas consecutivas
        self.ml_weight = 0.5  # Peso del modelo ML vs técnico
        self.last_trade_step = 0  # Último paso donde se ejecutó un trade
        self.daily_trades = 0  # Número de trades del día actual
        
        # Estadísticas de trading
        self.total_trades = 0
        self.profitable_trades = 0
        self.failed_trades = 0
        self.total_return = 0.0
        self.peak_value = self.initial_capital
        self.max_drawdown = 0.0
        
        # Arrays de seguimiento
        self.portfolio_values = []
        self.buy_signals = []
        self.sell_signals = []
        self.stop_losses = []
        self.take_profits = []
        self.trades_history = []
        self.actions = []
        self.ml_predictions = []
        self.technical_signals = []
        
        # Modelo ML
        self.ml_model = None
        self.model_paths = {
            "1": [  # DQN
                "data/models/qdn/model.zip",
                "data/models/best_qdn/model.zip"
            ],
            "2": [  # DeepDQN
                "data/models/deepqdn/model.zip",
                "data/models/best_deepqdn/model.zip"
            ],
            "3": [  # PPO
                "data/models/ppo/model.zip",
                "data/models/best_ppo/best_model.zip"
            ],
            "4": [  # A2C
                "data/models/a2c/model.zip",
                "data/models/best_a2c/model.zip"
            ]
        }
        
        # Definir espacios de observación y acción para RL
        self.state_dim = 4  # [price, sma_diff, rsi, macd]
        self.observation_space = spaces.Box(
            low=np.array([-4.8000002, -3.4028235e+38, -0.41887903, -3.4028235e+38], dtype=np.float32),
            high=np.array([4.8000002, 3.4028235e+38, 0.41887903, 3.4028235e+38], dtype=np.float32),
            dtype=np.float32
        )
        
        # Configurar espacio de acciones según el algoritmo
        # TODOS los modelos entrenados usan Discrete(2)
        self.action_space = spaces.Discrete(2)
        
        # Control de reproducción
        self.is_playing = False
        self.speed = 1.0
        
        # Tracking
        self.initialize_tracking_arrays()
        
        # Métricas
        self.max_drawdown = 0
        self.peak_value = self.initial_capital
        self.consecutive_losses = 0
        self.total_trades = 0
        self.profitable_trades = 0
        
        # Visual
        self.window_size = 150
    
    def step(self, action):
        """Implementar step requerido por gym.Env"""
        # TODOS los modelos usan acciones discretas ahora
        trading_action = action
        
        # Avanzar un paso
        self.current_step += 1
        
        # Obtener estado actual
        state = self.get_state()
        
        # Calcular recompensa basada en el cambio de portfolio
        old_value = self.portfolio_values[self.current_step - 1]
        new_value = self.portfolio_values[self.current_step]
        reward = (new_value - old_value) / old_value
        
        # Verificar si el episodio terminó
        done = self.current_step >= len(self.data) - 1
        
        return state, reward, done, False, {}
        
    def reset(self, **kwargs):
        """Implementar reset requerido por gym.Env"""
        # Ignorar todos los kwargs (seed, options, etc.)
        self.current_step = 50
        self.current_capital = self.initial_capital
        self.position_size = 0
        self.position_type = None
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_step = 0
        
        # Reiniciar tracking
        self.initialize_tracking_arrays()
        
        return self.get_state(), {}
        
    def get_state(self):
        """Obtener estado actual para RL"""
        if self.data is None or self.current_step < 50:
            return np.zeros(self.state_dim, dtype=np.float32)
            
        # Obtener features principales
        try:
            current_price = self.data.iloc[self.current_step]['close']
            price_change = (current_price - self.data.iloc[self.current_step - 1]['close']) / self.data.iloc[self.current_step - 1]['close']
            volume = self.data.iloc[self.current_step]['volume']
            rsi = self.data.iloc[self.current_step]['RSI'] if 'RSI' in self.data.columns else 50.0
            
            # Normalizar features para coincidir con el espacio de observación del modelo existente
            price_change = np.clip(price_change * 100, -4.8000002, 4.8000002)  # Cambio % limitado
            volume = volume * 1e-6  # Escalar volumen
            rsi = (rsi - 50) / 120  # Normalizar RSI a [-0.41, 0.41]
            position = 1.0 if self.position_size > 0 else -1.0  # Indicador de posición [-1,1]
            
            features = np.array([
                price_change,  # Cambio porcentual del precio [-4.8, 4.8]
                volume,       # Volumen escalado
                rsi,         # RSI normalizado [-0.41, 0.41]
                position     # Indicador de posición [-1,1]
            ], dtype=np.float32)
            
            return features
            
        except Exception as e:
            print(f"⚠️ Error preparando features: {e}")
            return np.zeros(self.state_dim, dtype=np.float32)
    
    def load_ml_model(self):
        """Cargar modelo pre-entrenado"""
        if not HAS_RL:
            print("⚠️ Sin componentes de RL - usando solo técnico")
            return
            
        print("\n🤖 Cargando modelo real para", self.algorithm_name)
        
        # Intentar cargar modelo
        for model_path in self.model_paths[self.algorithm_choice]:
            try:
                print("🔄 Intentando cargar:", model_path)
                
                # Cargar modelo según el tipo
                if self.algorithm_name == "A2C":
                    self.ml_model = A2C.load(model_path, env=self)
                elif self.algorithm_name == "PPO":
                    self.ml_model = PPO.load(model_path, env=self)
                else:  # DQN y DeepDQN
                    self.ml_model = DQN.load(model_path, env=self)
                
                # Probar predicción
                test_state = self.get_state()
                test_prediction = self.ml_model.predict(test_state)
                print("✅ Modelo cargado exitosamente")
                print("🧠 Tipo:", self.algorithm_name)
                print("🎯 Test prediction:", test_prediction)
                return
                
            except Exception as e:
                print(f"⚠️ Error cargando {model_path}: {str(e)}")
                continue
                
        print("\n❌ No se pudo cargar ningún modelo real")
        print("💡 Intentando entrenar un nuevo modelo...")
        
        # Intentar entrenar nuevo modelo
        try:
            self.train_new_model()
        except Exception as e:
            print("❌ Error entrenando modelo:", str(e))
            print("❌ No se pudo entrenar un nuevo modelo")
    
    def create_simple_ml_model(self):
        """DEPRECATED: Solo usamos modelos reales entrenados"""
        print("❌ Esta función está deshabilitada - solo usamos modelos reales")
        print("💡 Entrena un modelo real: python src/agents/train_models.py")
        return False
    
    def train_new_model(self):
        """Entrenar un nuevo modelo cuando no existe uno válido"""
        print("\n🚀 Iniciando entrenamiento automático de modelo...")
        
        if not HAS_RL:
            print("❌ Stable-baselines3 no disponible")
            print("💡 Instala: pip install stable-baselines3")
            return None

        try:
            # Generar datos de entrenamiento si no existen
            if self.data is None:
                print("📊 Generando datos de entrenamiento...")
                self.generate_market_data(n_points=5000)  # Más datos para entrenamiento
            
            # Asegurar que tenemos RSI calculado
            if 'RSI' not in self.data.columns:
                self.data = self.calculate_indicators(self.data)
            
            # Configurar modelo según el tipo de algoritmo
            if self.algorithm_choice in ["1", "2"]:  # DQN/DeepDQN
                policy_kwargs = dict(net_arch=[64, 32]) if self.algorithm_choice == "1" else dict(net_arch=[256, 256, 128, 64])
                learning_rate = 0.001 if self.algorithm_choice == "1" else 0.0005
                
                model = DQN(
                    MlpPolicy,
                    self,  # Usar self directamente como env
                    policy_kwargs=policy_kwargs,
                    learning_rate=learning_rate,
                    verbose=1
                )
            elif self.algorithm_choice == "3":  # A2C
                # Para A2C, crear un wrapper compatible
                class A2CEnvWrapper(gym.Env):
                    def __init__(self, base_env):
                        super().__init__()
                        self.base_env = base_env
                        self.observation_space = base_env.observation_space
                        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                        
                    def reset(self, **kwargs):
                        return self.base_env.reset(**kwargs)
                        
                    def step(self, action):
                        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
                        discrete_action = 1 if action_value > 0.3 else 0 if action_value < -0.3 else 1
                        return self.base_env.step(discrete_action)
                
                a2c_env = A2CEnvWrapper(self)
                model = A2C(
                    "MlpPolicy",
                    a2c_env,
                    learning_rate=0.0003,
                    verbose=1
                )
            elif self.algorithm_choice == "4":  # SAC
                try:
                    from stable_baselines3 import SAC
                    model = SAC(
                        "MlpPolicy",
                        self,  # Usar self directamente como env
                        learning_rate=0.0003,
                        verbose=1
                    )
                except ImportError:
                    print("⚠️ SAC no disponible, usando DQN...")
                    model = DQN(
                        MlpPolicy,
                        self,  # Usar self directamente como env
                        learning_rate=0.001,
                        verbose=1
                    )
            
            # Entrenar modelo
            total_timesteps = 100000
            print(f"\n📈 Entrenando modelo por {total_timesteps} pasos...")
            model.learn(total_timesteps=total_timesteps)
            
            # Crear directorios si no existen
            model_dir = f"data/models/{self.algorithm_name.lower()}"
            best_model_dir = f"data/models/best_{self.algorithm_name.lower()}"
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(best_model_dir, exist_ok=True)
            
            # Guardar modelo
            model_path = f"{model_dir}/model.zip"
            best_path = f"{best_model_dir}/model.zip"
            model.save(model_path)
            model.save(best_path)
            
            print(f"✅ Modelo guardado en: {model_path}")
            print(f"✅ Mejor modelo guardado en: {best_path}")
            
            return model
            
        except Exception as e:
            print(f"❌ Error entrenando modelo: {e}")
            return None
    
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
            
            # Agregar columna Action
            df['Action'] = 2  # 2 = Hold (no acción)
            
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
                    # Agregar columna Action
                    self.data['Action'] = 2  # 2 = Hold (no acción)
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
        actions = [2]  # 2 = Hold (no acción)
        
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
            
            # Acción por defecto
            actions.append(2)  # 2 = Hold (no acción)
        
        # Asegurar misma longitud
        min_length = min(len(prices), len(volumes))
        prices = prices[:min_length]
        volumes = volumes[:min_length]
        actions = actions[:min_length]
        
        # Crear DataFrame con fechas
        df = pd.DataFrame({
            'price': prices,
            'volume': volumes,
            'time': self.dates,
            'Action': actions  # Agregar columna Action
        })
        
        # Calcular indicadores
        df = self.calculate_indicators(df)
        
        self.data = df
        print(f"✅ {len(df)} registros simulados generados: ${min(prices):.2f} - ${max(prices):.2f}")
        
        # Inicializar arrays de seguimiento
        self.initialize_tracking_arrays()
    
    def initialize_tracking_arrays(self):
        """Inicializar arrays de seguimiento"""
        if self.data is None:
            n = 2000  # Tamaño por defecto
        else:
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
        
        # Agregar columna Action al DataFrame si no existe
        if self.data is not None and 'Action' not in self.data.columns:
            self.data['Action'] = 2  # 2 = Hold (no acción)
    
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
        """Preparar features para ML"""
        if step < 20:  # Necesitamos suficientes datos históricos
            return np.zeros(4)
            
        # 1. Precio normalizado (usando ventana de 20 periodos)
        prices = self.data.iloc[step-20:step+1]['price'].values
        price_mean = np.mean(prices)
        price_std = np.std(prices)
        norm_price = (prices[-1] - price_mean) / (price_std + 1e-8)
        norm_price = np.clip(norm_price, -4.8, 4.8)
        
        # 2. Volumen normalizado
        volumes = self.data.iloc[step-20:step+1]['volume'].values
        volume_mean = np.mean(volumes)
        volume_std = np.std(volumes)
        norm_volume = (volumes[-1] - volume_mean) / (volume_std + 1e-8)
        
        # 3. RSI normalizado
        rsi = self.data.iloc[step]['rsi'] / 100.0  # Ya está entre 0 y 1
        norm_rsi = (rsi - 0.5) * 0.83775806  # Escalar a [-0.41887903, 0.41887903]
        
        # 4. MACD normalizado
        macd = self.data.iloc[step]['macd_normalized']
        
        # Combinar features en el orden correcto
        features = np.array([
            norm_price,      # [-4.8, 4.8]
            norm_volume,     # [-3.4e38, 3.4e38]
            norm_rsi,        # [-0.41887903, 0.41887903]
            macd            # [-3.4e38, 3.4e38]
        ], dtype=np.float32)
        
        return features
        
    def generate_ml_signal(self, step):
        """Generar señal usando el modelo de ML"""
        try:
            if self.ml_model is None:
                return 0
            
            # Obtener predicción del modelo
            observation = self.prepare_ml_features(step)
            observation = observation.reshape(1, -1)  # Reshape para el modelo
            
            # Predecir acción
            action = self.ml_model.predict(observation)[0]
            
            # Convertir acción a señal de trading
            # TODOS los modelos usan acciones discretas ahora
            if action == 1:  # SELL
                return -1.0
            else:  # BUY (action == 0)
                return 1.0
            
        except Exception as e:
            print(f"⚠️ Error en ML: {e}")
            return 0
            
    def generate_technical_signal(self, step):
        """Generar señal técnica más agresiva"""
        if step < 2:
            return 0.0
            
        # Precios
        current_price = self.data.iloc[step]['price']
        prev_price = self.data.iloc[step-1]['price']
        
        # RSI (acceder desde el DataFrame)
        rsi = self.data.iloc[step]['rsi']
        rsi_signal = 0.0
        if rsi < 30:  # Sobreventa
            rsi_signal = 1.0
        elif rsi > 70:  # Sobrecompra
            rsi_signal = -1.0
            
        # MACD (acceder desde el DataFrame)
        macd = self.data.iloc[step]['macd_normalized']
        macd_signal = 0.0
        if macd > 0.1:  # Señal fuerte de compra
            macd_signal = 1.0
        elif macd < -0.1:  # Señal fuerte de venta
            macd_signal = -1.0
            
        # Momentum
        momentum = (current_price - prev_price) / prev_price
        momentum_signal = 0.0
        if momentum > 0.001:  # Umbral más bajo
            momentum_signal = 1.0
        elif momentum < -0.001:  # Umbral más bajo
            momentum_signal = -1.0
            
        # Bollinger Bands (acceder desde el DataFrame)
        bb_position = self.data.iloc[step]['bb_position']
        bb_signal = 0.0
        if bb_position < 0.2:  # Cerca de banda inferior
            bb_signal = 1.0
        elif bb_position > 0.8:  # Cerca de banda superior
            bb_signal = -1.0
            
        # Volumen (acceder desde el DataFrame)
        volume_ratio = self.data.iloc[step]['volume_ratio']
        volume_signal = 0.0
        if volume_ratio > 1.2:  # Alto volumen
            volume_signal = 1.0 if current_price > prev_price else -1.0
            
        # Combinar señales con pesos
        weights = {
            'rsi': 0.3,      # Más peso a RSI
            'macd': 0.2,     # Peso moderado a MACD
            'momentum': 0.2,  # Peso moderado a Momentum
            'bb': 0.2,       # Peso moderado a Bollinger
            'volume': 0.1    # Menos peso a Volumen
        }
        
        combined_signal = (
            weights['rsi'] * rsi_signal +
            weights['macd'] * macd_signal +
            weights['momentum'] * momentum_signal +
            weights['bb'] * bb_signal +
            weights['volume'] * volume_signal
        )
        
        return combined_signal
    
    def generate_combined_signal(self, step):
        """Generar señal combinada de ML y técnico"""
        # Obtener señales individuales
        ml_signal = self.generate_ml_signal(step)
        tech_signal = self.generate_technical_signal(step)
        
        # Guardar señales para tracking
        self.ml_predictions.append(ml_signal)
        self.technical_signals.append(tech_signal)
        
        # Si el modelo ML no está disponible, usar solo técnico
        if self.ml_model is None:
            return tech_signal
            
        # Combinar señales con pesos
        combined = (ml_signal * self.ml_weight + 
                   tech_signal * (1 - self.ml_weight))
        
        # Umbral más bajo para trades más frecuentes
        if combined > 0.15:  # Era 0.25
            return 1.0  # Compra
        elif combined < -0.15:  # Era -0.25
            return -1.0  # Venta
        else:
            return 0.0  # Hold
    
    def check_risk_filters(self, step):
        """Verificar filtros de riesgo antes de operar
        Retorna: (bool, str) - (passed, reason)"""
        
        # 1. Verificar RSI extremo
        rsi = self.data.iloc[step]['rsi']
        if rsi > 85 or rsi < 15:
            return False, f"RSI extremo: {rsi:.1f}"
            
        # 2. Verificar volatilidad
        volatility = self.data.iloc[step]['volatility']
        avg_volatility = self.data['volatility'].rolling(20).mean().iloc[step]
        if volatility > avg_volatility * 2:
            return False, f"Volatilidad alta: {volatility:.4f} vs {avg_volatility:.4f}"
            
        # 3. Verificar spread (si está disponible)
        if 'spread' in self.data.columns:
            spread = self.data.iloc[step]['spread']
            avg_spread = self.data['spread'].rolling(20).mean().iloc[step]
            if spread > avg_spread * 1.5:
                return False, f"Spread alto: {spread:.1f} vs {avg_spread:.1f}"
                
        # 4. Verificar volumen
        volume = self.data.iloc[step]['volume']
        avg_volume = self.data['volume'].rolling(20).mean().iloc[step]
        if volume < avg_volume * 0.5:
            return False, f"Volumen bajo: {volume:.0f} vs {avg_volume:.0f}"
            
        # 5. Verificar tendencia
        sma_fast = self.data['price'].rolling(10).mean().iloc[step]
        sma_slow = self.data['price'].rolling(30).mean().iloc[step]
        trend_strength = abs(sma_fast/sma_slow - 1)
        if trend_strength < 0.0001:
            return False, f"Mercado sin tendencia: {trend_strength:.6f}"
            
        # 6. Verificar momentum
        returns = self.data['returns'].iloc[step]
        avg_returns = self.data['returns'].rolling(5).mean().iloc[step]
        if abs(returns) > abs(avg_returns) * 3:
            return False, f"Movimiento brusco: {returns:.4f} vs {avg_returns:.4f}"
            
        return True, "OK"

    def execute_trade(self, step, signal):
        """Ejecutar trade con validaciones más permisivas"""
        current_price = self.data.iloc[step]['price']
        
        # Verificaciones básicas (MÁS PERMISIVAS)
        if step - self.last_trade_step < 2:  # Era 3
            return False
        
        if self.daily_trades >= 8:  # Era 5
            return False
        
        if self.consecutive_losses >= 5:  # Era 4
            return False
        
        # COMPRA - umbral más bajo
        if signal > 0.15 and self.position_size == 0:  # Era 0.25
            position_size = self.calculate_position_size(current_price)
            
            if position_size > 0:
                self.position_size = position_size
                self.position_type = 'LONG'
                self.entry_price = current_price
                self.last_trade_step = step
                self.daily_trades += 1
                
                self.buy_signals.append(step)
                self.total_trades += 1
                
                # Actualizar Action en el DataFrame
                self.data.iloc[step, self.data.columns.get_loc('Action')] = 1  # 1 = Compra
                
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
        elif signal < -0.15 and self.position_type == 'LONG':  # Era -0.25
            # ✅ ARREGLO: Liquidar posición correctamente
            # Total recibido por la venta
            total_received = current_price * self.position_size
            # Devolver el dinero al capital disponible
            self.current_capital = self.current_capital - (self.position_size * self.entry_price) + total_received
            profit = total_received - (self.position_size * self.entry_price)
            
            self.sell_signals.append(step)
            
            # Actualizar Action en el DataFrame
            self.data.iloc[step, self.data.columns.get_loc('Action')] = 0  # 0 = Venta
            
            # Actualizar estadísticas de trading
            if profit > 0:
                self.profitable_trades += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
            
            trade_info = {
                'step': step,
                'type': 'SELL',
                'price': current_price,
                'size': self.position_size,
                'profit': profit,
                'signal': signal,
                'ml_pred': self.ml_predictions[step],
                'tech_signal': self.technical_signals[step]
            }
            self.trades_history.append(trade_info)
            
            print(f"🔴 VENTA: ${current_price:.2f} | P&L: ${profit:.2f} | Señal: {signal:.3f}")
            
            self.position_size = 0
            self.position_type = None
            self.last_trade_step = step
            
            return True
        
        return False

    def calculate_position_size(self, entry_price):
        """Calcular tamaño de posición con gestión de riesgo mejorada"""
        if entry_price <= 0:
            return 0
            
        # 1. Calcular riesgo por posición
        stop_loss_price = entry_price * (1 - self.stop_loss_pct)
        risk_per_share = entry_price - stop_loss_price
        
        # 2. Ajustar riesgo según volatilidad
        volatility = self.data['volatility'].iloc[-1]
        avg_volatility = self.data['volatility'].rolling(20).mean().iloc[-1]
        volatility_factor = min(avg_volatility / volatility, 1.0) if volatility > 0 else 0.5
        
        # 3. Ajustar riesgo según drawdown
        max_drawdown = abs(min(0, self.current_capital - self.initial_capital) / self.initial_capital)
        drawdown_factor = 1.0 - (max_drawdown * 2)  # Reducir posición si hay drawdown
        
        # 4. Calcular shares considerando todos los factores
        max_risk_capital = self.current_capital * self.max_position_risk * volatility_factor * drawdown_factor
        shares = int(max_risk_capital / risk_per_share)
        
        # 5. Limitar por capital disponible
        max_shares_by_capital = int(self.current_capital * 0.95 / entry_price)  # 95% del capital
        shares = min(shares, max_shares_by_capital)
        
        # 6. Establecer límites absolutos
        min_shares = 5
        max_shares = 50
        shares = max(min(shares, max_shares), min_shares)
        
        return shares
    
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
            # ✅ ARREGLO: Liquidar posición correctamente en stop loss
            total_received = current_price * self.position_size
            self.current_capital = self.current_capital - (self.position_size * self.entry_price) + total_received
            profit = total_received - (self.position_size * self.entry_price)
            self.stop_losses.append(self.current_step)
            self.consecutive_losses += 1
            
            # Actualizar Action en el DataFrame
            self.data.iloc[self.current_step, self.data.columns.get_loc('Action')] = 0  # 0 = Venta
            
            print(f"🛑 STOP LOSS: ${current_price:.2f} | Pérdida: ${profit:.2f}")
            self.position_size = 0
            self.position_type = None
            
        elif exit_condition == 'TAKE_PROFIT':
            # ✅ ARREGLO: Liquidar posición correctamente en take profit
            total_received = current_price * self.position_size
            self.current_capital = self.current_capital - (self.position_size * self.entry_price) + total_received
            profit = total_received - (self.position_size * self.entry_price)
            self.take_profits.append(self.current_step)
            self.profitable_trades += 1
            self.consecutive_losses = 0
            
            # Actualizar Action en el DataFrame
            self.data.iloc[self.current_step, self.data.columns.get_loc('Action')] = 0  # 0 = Venta
            
            print(f"🎯 TAKE PROFIT: ${current_price:.2f} | Ganancia: ${profit:.2f}")
            self.position_size = 0
            self.position_type = None
        
        else:
            # Generar señal combinada
            signal = self.generate_combined_signal(self.current_step)
            self.actions[self.current_step] = signal
            
            # Ejecutar trade si es necesario
            if signal != 0:  # Si hay señal clara (compra o venta)
                if signal > 0 and self.position_size == 0:
                    # Actualizar Action en el DataFrame
                    self.data.iloc[self.current_step, self.data.columns.get_loc('Action')] = 1  # 1 = Compra
                elif signal < 0 and self.position_type == 'LONG':
                    # Actualizar Action en el DataFrame
                    self.data.iloc[self.current_step, self.data.columns.get_loc('Action')] = 0  # 0 = Venta
                self.execute_trade(self.current_step, signal)
        
        # ✅ ARREGLO: Calcular portfolio value correctamente para evitar duplicación
        if self.position_size > 0:
            # Capital disponible (dinero que no está invertido) + valor actual de la posición
            invested_amount = self.position_size * self.entry_price
            available_cash = self.current_capital - invested_amount
            position_value = self.position_size * current_price
            portfolio_value = available_cash + position_value
        else:
            portfolio_value = self.current_capital
        
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
🤖 Modelo: {self.algorithm_name}"""
        
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
        
        # Cargar modelo ML real - OBLIGATORIO
        print("🤖 Cargando modelo de IA real...")
        if not system.load_ml_model():
            print("\n❌ ERROR CRÍTICO: No se pudo cargar ningún modelo real")
            print("💡 SOLUCIONES:")
            print("   1. Entrena nuevos modelos: python src/agents/train_models.py")
            print("   2. Verifica que existan los archivos en data/models/")
            print("   3. Instala dependencias: pip install stable-baselines3")
            return
        
        # Generar datos (históricos o simulados)
        print("📊 Generando datos de mercado...")
        system.generate_market_data(1500)
        
        # Intentar conectar a MT5 si está habilitado
        print("🔌 Intentando conectar a MetaTrader5...")
        if system.connect_mt5():
            print("✅ MT5 conectado - datos en tiempo real disponibles")
        else:
            print("📈 Usando datos simulados")
        
        # Crear interfaz gráfica
        print("🖥️ Creando interfaz gráfica...")
        system.create_interface()
        
    except KeyboardInterrupt:
        print("\n🛑 Sistema detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Verifica instalación de dependencias y modelos entrenados")
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