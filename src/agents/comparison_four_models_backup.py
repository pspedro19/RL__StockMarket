"""
🤖 COMPARACIÓN DE 4 MODELOS DE TRADING
DQN | DeepQDN | PPO | A2C
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, Slider
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar componentes de RL
try:
    from stable_baselines3 import DQN, PPO, A2C
    HAS_RL = True
    print("✅ Componentes de RL disponibles")
except ImportError:
    HAS_RL = False
    print("❌ Sin componentes de RL")

# Importar nuestro sistema base
from ml_enhanced_system import MLEnhancedTradingSystem

class FourModelComparison:
    """Sistema de comparación de 4 modelos"""
    
    def __init__(self):
        """Inicializar sistema de comparación"""
        print("🚀 Iniciando comparación de modelos...")
        
        # Crear sistemas
        self.dqn_system = MLEnhancedTradingSystem(skip_selection=True)
        self.deepqdn_system = MLEnhancedTradingSystem(skip_selection=True)
        self.ppo_system = MLEnhancedTradingSystem(skip_selection=True)
        self.a2c_system = MLEnhancedTradingSystem(skip_selection=True)
        
        # Configurar sistemas
        self._configure_systems()
        
        # Estado actual
        self.current_step = 0
        self.is_playing = False
        self.speed = 1.0
        self.real_time = False
        
        # Crear interfaz
        self.create_interface()
        
    def _configure_systems(self):
        """Configurar los cuatro sistemas con parámetros específicos"""
        # 1. DQN - Agresivo (Scalping)
        self.dqn_system.algorithm_choice = "1"
        self.dqn_system.algorithm_name = "DQN"
        self.dqn_system.algorithm_class = DQN
        self.dqn_system.model_paths = [
            "data/models/qdn/model.zip",
            "data/models/best_qdn/model.zip"
        ]
        self.dqn_system.policy_kwargs = dict(net_arch=[64, 32])  # Red simple
        self.dqn_system.learning_rate = 0.001
        self.dqn_system.max_position_risk = 0.12     # 12% riesgo por posición
        self.dqn_system.stop_loss_pct = 0.015       # 1.5% stop loss
        self.dqn_system.take_profit_pct = 0.035     # 3.5% take profit
        self.dqn_system.min_trade_separation = 1     # Trades muy frecuentes
        self.dqn_system.max_daily_trades = 15       # Muchos trades por día
        self.dqn_system.ml_weight = 0.6             # Más peso en técnico
        self.dqn_system.consecutive_losses = 5       # Más tolerante a pérdidas
        
        # 2. DeepQDN - Preciso (Swing Trading)
        self.deepqdn_system.algorithm_choice = "2"
        self.deepqdn_system.algorithm_name = "DeepQDN"
        self.deepqdn_system.algorithm_class = DQN
        self.deepqdn_system.model_paths = [
            "data/models/deepqdn/model.zip",
            "data/models/best_deepqdn/model.zip"
        ]
        self.deepqdn_system.policy_kwargs = dict(net_arch=[256, 256, 128, 64])  # Red profunda
        self.deepqdn_system.learning_rate = 0.0005  # Learning rate más bajo
        self.deepqdn_system.max_position_risk = 0.08  # 8% riesgo por posición
        self.deepqdn_system.stop_loss_pct = 0.025    # 2.5% stop loss
        self.deepqdn_system.take_profit_pct = 0.055  # 5.5% take profit
        self.deepqdn_system.min_trade_separation = 3  # Trades más espaciados
        self.deepqdn_system.max_daily_trades = 10
        self.deepqdn_system.ml_weight = 0.6          # Más peso en técnico
        self.deepqdn_system.consecutive_losses = 3    # Más conservador con pérdidas
        
        # 3. PPO - Balanceado (Swing Trading)
        self.ppo_system.algorithm_choice = "3"
        self.ppo_system.algorithm_name = "PPO"
        self.ppo_system.algorithm_class = PPO
        self.ppo_system.model_paths = [
            "data/models/ppo/model.zip",
            "data/models/best_ppo/best_model.zip"
        ]
        self.ppo_system.max_position_risk = 0.10     # 10% riesgo por posición
        self.ppo_system.stop_loss_pct = 0.02        # 2% stop loss
        self.ppo_system.take_profit_pct = 0.045     # 4.5% take profit
        self.ppo_system.min_trade_separation = 2     # Trades moderados
        self.ppo_system.max_daily_trades = 12       # Trades moderados por día
        self.ppo_system.ml_weight = 0.6             # Más peso en técnico
        self.ppo_system.consecutive_losses = 4       # Moderado con pérdidas
        
        # 4. A2C - Conservador (Posicional)
        self.a2c_system.algorithm_choice = "4"
        self.a2c_system.algorithm_name = "A2C"
        self.a2c_system.algorithm_class = A2C
        self.a2c_system.model_paths = [
            "data/models/a2c/model.zip",
            "data/models/best_a2c/model.zip"
        ]
        self.a2c_system.max_position_risk = 0.06     # 6% riesgo por posición
        self.a2c_system.stop_loss_pct = 0.03        # 3% stop loss
        self.a2c_system.take_profit_pct = 0.065     # 6.5% take profit
        self.a2c_system.min_trade_separation = 5     # Trades espaciados
        self.a2c_system.max_daily_trades = 5        # Pocos trades por día
        self.a2c_system.ml_weight = 0.6             # Más peso en técnico
        self.a2c_system.consecutive_losses = 2       # Muy conservador con pérdidas
        
        print("✅ Sistemas configurados con estrategias diferenciadas:")
        print("📊 DQN: Scalping (trades frecuentes)")
        print("📈 DeepQDN: Trading Preciso (swing)")
        print("📉 PPO: Trading Balanceado (swing)")
        print("💼 A2C: Trading Conservador (posicional)")
        
    def create_interface(self):
        """Crear interfaz comparativa"""
        print("🎮 Iniciando interfaz comparativa...")
        
        # Generar datos
        print("📊 Generando datos de mercado...")
        self.dqn_system.generate_market_data()
        self.deepqdn_system.data = self.dqn_system.data.copy()
        self.ppo_system.data = self.dqn_system.data.copy()
        self.a2c_system.data = self.dqn_system.data.copy()
        
        # Inicializar sistemas
        print("🔄 Inicializando sistemas...")
        
        print("\n1️⃣ Inicializando DQN...")
        self.dqn_system.load_ml_model()
        
        print("\n2️⃣ Inicializando DeepQDN...")
        self.deepqdn_system.load_ml_model()
        
        print("\n3️⃣ Inicializando PPO...")
        self.ppo_system.load_ml_model()
        
        print("\n4️⃣ Inicializando A2C...")
        self.a2c_system.load_ml_model()
        
        # Configurar figura
        self.fig = plt.figure(figsize=(26, 18))
        self.fig.suptitle("🤖 Comparación de Modelos de Trading - DQN | DeepQDN | PPO | A2C", 
                         fontsize=18, fontweight="bold", y=0.97)
        
        # Grid para cuatro sistemas
        gs = self.fig.add_gridspec(4, 3, height_ratios=[3, 3, 3, 3], 
                                   hspace=0.4, wspace=0.3)
        
        # DQN
        self.ax_dqn = self.fig.add_subplot(gs[0, :])
        self.ax_dqn.set_title("📊 DQN - Trading de Alta Frecuencia", 
                             fontsize=14, pad=20, fontweight="bold")
        
        # DeepQDN
        self.ax_deepqdn = self.fig.add_subplot(gs[1, :])
        self.ax_deepqdn.set_title("📈 DeepQDN - Trading Preciso", 
                                 fontsize=14, pad=20, fontweight="bold")
        
        # PPO
        self.ax_ppo = self.fig.add_subplot(gs[2, :])
        self.ax_ppo.set_title("📉 PPO - Trading Balanceado", 
                             fontsize=14, pad=20, fontweight="bold")
        
        # A2C
        self.ax_a2c = self.fig.add_subplot(gs[3, :])
        self.ax_a2c.set_title("💼 A2C - Trading Conservador", 
                             fontsize=14, pad=20, fontweight="bold")
        
        # Configurar ejes
        for ax in [self.ax_dqn, self.ax_deepqdn, self.ax_ppo, self.ax_a2c]:
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.grid(True, alpha=0.3, linewidth=0.8)
        
        # Controles
        self.create_controls()
        
        # Animación
        self.animation = FuncAnimation(self.fig, self.update_plots, 
                                     interval=50, blit=False, cache_frame_data=False)
        
        plt.show()
        
    def create_controls(self):
        """Crear controles"""
        # Botones principales
        ax_play = plt.axes([0.05, 0.02, 0.05, 0.04])
        ax_pause = plt.axes([0.11, 0.02, 0.05, 0.04])
        ax_stop = plt.axes([0.17, 0.02, 0.05, 0.04])
        ax_back = plt.axes([0.23, 0.02, 0.05, 0.04])
        ax_forward = plt.axes([0.29, 0.02, 0.05, 0.04])
        ax_realtime = plt.axes([0.35, 0.02, 0.08, 0.04])
        
        self.btn_play = Button(ax_play, "▶️")
        self.btn_pause = Button(ax_pause, "⏸️")
        self.btn_stop = Button(ax_stop, "⏹️")
        self.btn_back = Button(ax_back, "⏪")
        self.btn_forward = Button(ax_forward, "⏩")
        self.btn_realtime = Button(ax_realtime, "🤖 AUTO")
        
        # Slider de velocidad
        ax_speed = plt.axes([0.48, 0.02, 0.12, 0.04])
        self.slider_speed = Slider(ax_speed, "Velocidad", 0.25, 4.0, valinit=1.0)
        
        # Eventos
        self.btn_play.on_clicked(lambda x: setattr(self, "is_playing", True))
        self.btn_pause.on_clicked(lambda x: setattr(self, "is_playing", False))
        self.btn_stop.on_clicked(lambda x: self.reset())
        self.btn_back.on_clicked(lambda x: self.step_back())
        self.btn_forward.on_clicked(lambda x: self.step_forward())
        self.btn_realtime.on_clicked(lambda x: self.toggle_real_time())
        self.slider_speed.on_changed(lambda x: setattr(self, "speed", x))
        
    def update_plots(self, frame):
        """Actualizar gráficos"""
        if not self.is_playing:
            return
            
        # Avanzar
        self.step_forward()
        
        # Limpiar ejes
        self.ax_dqn.clear()
        self.ax_deepqdn.clear()
        self.ax_ppo.clear()
        self.ax_a2c.clear()
        
        # Actualizar DQN
        self._plot_system(self.ax_dqn, self.dqn_system, "DQN")
        
        # Actualizar DeepQDN
        self._plot_system(self.ax_deepqdn, self.deepqdn_system, "DeepQDN")
        
        # Actualizar PPO
        self._plot_system(self.ax_ppo, self.ppo_system, "PPO")
        
        # Actualizar A2C
        self._plot_system(self.ax_a2c, self.a2c_system, "A2C")
        
        # Configurar ejes
        for ax in [self.ax_dqn, self.ax_deepqdn, self.ax_ppo, self.ax_a2c]:
            ax.grid(True, alpha=0.3)
            ax.legend()
            
    def _plot_system(self, ax, system, name):
        """Graficar un sistema específico"""
        # Datos visibles
        start = max(0, system.current_step - 100)
        end = system.current_step + 1
        
        # Precios
        prices = system.data.iloc[start:end]["price"]
        times = range(len(prices))
        ax.plot(times, prices, "b-", label="Precio", alpha=0.7)
        
        # Media móvil y bandas
        if 'sma_20' in system.data.columns:
            sma20 = system.data['sma_20'].iloc[start:end]
            ax.plot(times, sma20, 'orange', alpha=0.7, label='SMA 20')
            
        if 'bb_upper' in system.data.columns and 'bb_lower' in system.data.columns:
            bb_upper = system.data['bb_upper'].iloc[start:end]
            bb_lower = system.data['bb_lower'].iloc[start:end]
            ax.fill_between(times, bb_upper, bb_lower, alpha=0.1, color='gray')
        
        # Señales
        window_buys = [s - start for s in system.buy_signals if start <= s <= end]
        window_sells = [s - start for s in system.sell_signals if start <= s <= end]
        window_stops = [s - start for s in system.stop_losses if start <= s <= end]
        window_profits = [s - start for s in system.take_profits if start <= s <= end]
        
        if window_buys:
            buy_prices = [system.data['price'].iloc[s + start] for s in window_buys]
            ax.scatter(window_buys, buy_prices, c='green', marker='^', 
                      s=120, label='Compras', edgecolors='darkgreen', linewidth=2)
        
        if window_sells:
            sell_prices = [system.data['price'].iloc[s + start] for s in window_sells]
            ax.scatter(window_sells, sell_prices, c='red', marker='v', 
                      s=120, label='Ventas', edgecolors='darkred', linewidth=2)
        
        if window_stops:
            stop_prices = [system.data['price'].iloc[s + start] for s in window_stops]
            ax.scatter(window_stops, stop_prices, c='red', marker='x', s=150)
        
        if window_profits:
            profit_prices = [system.data['price'].iloc[s + start] for s in window_profits]
            ax.scatter(window_profits, profit_prices, c='gold', marker='*', s=150)
        
        # Información
        info = f"{name}\n"
        info += f"Capital: ${system.current_capital:,.2f}\n"
        info += f"Trades: {system.total_trades}\n"
        info += f"Win Rate: {(system.profitable_trades/max(system.total_trades, 1))*100:.1f}%\n"
        info += f"Drawdown: {system.max_drawdown*100:.1f}%"
        
        ax.text(0.02, 0.98, info, transform=ax.transAxes, 
                verticalalignment="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8))
        
    def step_forward(self):
        """Avanzar un paso en todos los sistemas"""
        # Avanzar DQN
        self.dqn_system.step_forward()
        
        # Avanzar DeepQDN
        self.deepqdn_system.step_forward()
        
        # Avanzar PPO
        self.ppo_system.step_forward()
        
        # Avanzar A2C
        self.a2c_system.step_forward()
        
        # Actualizar paso actual
        self.current_step = self.dqn_system.current_step
        
    def step_back(self):
        """Retroceder un paso"""
        self.is_playing = False
        
        # Retroceder DQN
        self.dqn_system.step_back()
        
        # Retroceder DeepQDN
        self.deepqdn_system.step_back()
        
        # Retroceder PPO
        self.ppo_system.step_back()
        
        # Retroceder A2C
        self.a2c_system.step_back()
        
        # Actualizar paso actual
        self.current_step = self.dqn_system.current_step
        
    def reset(self):
        """Reiniciar sistemas"""
        self.is_playing = False
        
        # Reiniciar DQN
        self.dqn_system.reset()
        
        # Reiniciar DeepQDN
        self.deepqdn_system.reset()
        
        # Reiniciar PPO
        self.ppo_system.reset()
        
        # Reiniciar A2C
        self.a2c_system.reset()
        
        # Actualizar paso actual
        self.current_step = 0
        
    def toggle_real_time(self):
        """Alternar modo tiempo real"""
        self.real_time = not self.real_time
        
        if self.real_time:
            # Activar tiempo real en todos los sistemas
            self.dqn_system.start_real_time_updates()
            self.deepqdn_system.start_real_time_updates()
            self.ppo_system.start_real_time_updates()
            self.a2c_system.start_real_time_updates()
        else:
            # Desactivar tiempo real
            self.dqn_system.stop_real_time_updates()
            self.deepqdn_system.stop_real_time_updates()
            self.ppo_system.stop_real_time_updates()
            self.a2c_system.stop_real_time_updates()

def main():
    """Función principal"""
    # Verificar componentes
    try:
        import stable_baselines3
        HAS_RL = True
        print("✅ Componentes de RL disponibles")
    except ImportError:
        HAS_RL = False
        print("❌ stable-baselines3 no disponible")
        return
        
    try:
        import MetaTrader5
        HAS_MT5 = True
        print("✅ MetaTrader5 disponible")
    except ImportError:
        HAS_MT5 = False
        print("❌ MetaTrader5 no disponible")
        return
        
    # Crear sistema de comparación
    system = FourModelComparison()

if __name__ == "__main__":
    main() 