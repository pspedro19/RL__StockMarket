"""
🤖 COMPARACIÓN DE TRES MODELOS DE TRADING
QDN vs DeepQDN vs A2C
"""

from ml_enhanced_system import MLEnhancedTradingSystem
from stable_baselines3 import DQN, A2C
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

class ComparisonSystem:
    """Sistema de comparación de modelos"""
    def __init__(self):
        print("🚀 Iniciando comparación de modelos...")
        
        # Crear sistemas
        self.dqn_system = MLEnhancedTradingSystem(skip_selection=True)
        self.deepdqn_system = MLEnhancedTradingSystem(skip_selection=True)
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
        """Configurar los tres sistemas con parámetros específicos"""
        # 1. QDN - Agresivo y frecuente (Scalping)
        self.dqn_system.algorithm_choice = "1"
        self.dqn_system.algorithm_name = "QDN"
        self.dqn_system.algorithm_class = DQN
        self.dqn_system.model_paths = [
            "data/models/qdn/model.zip",
            "data/models/best_qdn/model.zip"
        ]
        self.dqn_system.policy_kwargs = dict(net_arch=[64, 32])  # Red simple
        self.dqn_system.learning_rate = 0.001
        self.dqn_system.max_position_risk = 0.10     # 10% riesgo por posición
        self.dqn_system.stop_loss_pct = 0.02        # 2% stop loss
        self.dqn_system.take_profit_pct = 0.04      # 4% take profit
        self.dqn_system.min_trade_separation = 1     # Trades muy frecuentes
        self.dqn_system.max_daily_trades = 15       # Muchos trades por día
        self.dqn_system.ml_weight = 0.6             # Más peso en técnico
        self.dqn_system.consecutive_losses = 5       # Más tolerante a pérdidas
        
        # 2. DeepQDN - Swing Trading (Medio plazo)
        self.deepdqn_system.algorithm_choice = "2"
        self.deepdqn_system.algorithm_name = "DeepQDN"
        self.deepdqn_system.algorithm_class = DQN
        self.deepdqn_system.model_paths = [
            "data/models/deepqdn/model.zip",
            "data/models/best_deepqdn/model.zip"
        ]
        self.deepdqn_system.policy_kwargs = dict(net_arch=[256, 256, 128, 64])  # Red profunda
        self.deepdqn_system.learning_rate = 0.0005  # Learning rate más bajo
        self.deepdqn_system.max_position_risk = 0.08  # 8% riesgo por posición
        self.deepdqn_system.stop_loss_pct = 0.03     # 3% stop loss
        self.deepdqn_system.take_profit_pct = 0.06   # 6% take profit
        self.deepdqn_system.min_trade_separation = 3  # Trades más espaciados
        self.deepdqn_system.max_daily_trades = 10     # Trades moderados por día
        self.deepdqn_system.ml_weight = 0.6          # Más peso en técnico
        self.deepdqn_system.consecutive_losses = 3    # Más conservador con pérdidas
        
        # 3. A2C - Posicional (Largo plazo)
        self.a2c_system.algorithm_choice = "3"
        self.a2c_system.algorithm_name = "A2C"
        self.a2c_system.algorithm_class = A2C
        self.a2c_system.model_paths = [
            "data/models/a2c/model.zip",
            "data/models/best_a2c/model.zip"
        ]
        self.a2c_system.max_position_risk = 0.06     # 6% riesgo por posición
        self.a2c_system.stop_loss_pct = 0.04        # 4% stop loss
        self.a2c_system.take_profit_pct = 0.08      # 8% take profit
        self.a2c_system.min_trade_separation = 5    # Trades espaciados
        self.a2c_system.max_daily_trades = 5        # Pocos trades por día
        self.a2c_system.ml_weight = 0.6             # Más peso en técnico
        self.a2c_system.consecutive_losses = 2       # Muy conservador con pérdidas
        
        print("✅ Sistemas configurados con estrategias diferenciadas:")
        print("📊 QDN: Scalping (trades frecuentes)")
        print("📈 DeepQDN: Swing Trading (medio plazo)")
        print("📉 A2C: Trading Posicional (largo plazo)")
        
    def create_interface(self):
        """Crear interfaz comparativa"""
        print("🎮 Iniciando interfaz comparativa...")
        
        # Inicializar sistemas
        print("🔄 Inicializando sistemas...")
        
        print("\n1️⃣ Inicializando QDN...")
        self.dqn_system.load_ml_model()
        
        print("\n2️⃣ Inicializando DeepQDN...")
        self.deepdqn_system.load_ml_model()
        
        print("\n3️⃣ Inicializando A2C...")
        self.a2c_system.load_ml_model()
        
        # Configurar figura
        self.fig = plt.figure(figsize=(26, 18))
        self.fig.suptitle("🤖 Comparación de Modelos de Trading - QDN vs DeepQDN vs A2C", 
                         fontsize=18, fontweight="bold", y=0.97)
        
        # Grid para tres sistemas
        gs = self.fig.add_gridspec(3, 3, height_ratios=[3, 3, 3], 
                                   hspace=0.4, wspace=0.3)
        
        # QDN
        self.ax_qdn = self.fig.add_subplot(gs[0, :])
        self.ax_qdn.set_title("📊 QDN - Trading de Alta Frecuencia", 
                             fontsize=14, pad=20, fontweight="bold")
        
        # DeepQDN
        self.ax_deepqdn = self.fig.add_subplot(gs[1, :])
        self.ax_deepqdn.set_title("📈 DeepQDN - Swing Trading", 
                                 fontsize=14, pad=20, fontweight="bold")
        
        # A2C
        self.ax_a2c = self.fig.add_subplot(gs[2, :])
        self.ax_a2c.set_title("📉 A2C - Trading Posicional", 
                             fontsize=14, pad=20, fontweight="bold")
        
        # Configurar ejes
        for ax in [self.ax_qdn, self.ax_deepqdn, self.ax_a2c]:
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
        self.ax_qdn.clear()
        self.ax_deepqdn.clear()
        self.ax_a2c.clear()
        
        # Actualizar QDN
        self._plot_system(self.ax_qdn, self.dqn_system, "QDN")
        
        # Actualizar DeepQDN
        self._plot_system(self.ax_deepqdn, self.deepdqn_system, "DeepQDN")
        
        # Actualizar A2C
        self._plot_system(self.ax_a2c, self.a2c_system, "A2C")
        
        # Configurar ejes
        for ax in [self.ax_qdn, self.ax_deepqdn, self.ax_a2c]:
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
        
        # Señales
        for i in range(start, end):
            if i in system.buy_signals:
                ax.plot(i - start, prices.iloc[i-start], "^g", markersize=10, label="Compra")
            elif i in system.sell_signals:
                ax.plot(i - start, prices.iloc[i-start], "vr", markersize=10, label="Venta")
            elif i in system.stop_losses:
                ax.plot(i - start, prices.iloc[i-start], "xr", markersize=10, label="Stop Loss")
            elif i in system.take_profits:
                ax.plot(i - start, prices.iloc[i-start], "og", markersize=10, label="Take Profit")
                
        # Información
        info = f"{name}\n"
        info += f"Capital: ${system.current_capital:,.2f}\n"
        info += f"Trades: {system.total_trades}\n"
        info += f"Exitosos: {system.profitable_trades}\n"
        info += f"Drawdown: {system.max_drawdown*100:.1f}%"
        
        ax.text(0.02, 0.98, info, transform=ax.transAxes, 
                verticalalignment="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8))
        
    def step_forward(self):
        """Avanzar un paso en todos los sistemas"""
        # Avanzar QDN
        self.dqn_system.step_forward()
        
        # Avanzar DeepQDN
        self.deepdqn_system.step_forward()
        
        # Avanzar A2C
        self.a2c_system.step_forward()
        
        # Actualizar paso actual
        self.current_step = self.dqn_system.current_step
        
    def step_back(self):
        """Retroceder un paso"""
        self.is_playing = False
        
        # Retroceder QDN
        self.dqn_system.step_back()
        
        # Retroceder DeepQDN
        self.deepdqn_system.step_back()
        
        # Retroceder A2C
        self.a2c_system.step_back()
        
        # Actualizar paso actual
        self.current_step = self.dqn_system.current_step
        
    def reset(self):
        """Reiniciar sistemas"""
        self.is_playing = False
        
        # Reiniciar QDN
        self.dqn_system.reset()
        
        # Reiniciar DeepQDN
        self.deepdqn_system.reset()
        
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
            self.deepdqn_system.start_real_time_updates()
            self.a2c_system.start_real_time_updates()
        else:
            # Desactivar tiempo real
            self.dqn_system.stop_real_time_updates()
            self.deepdqn_system.stop_real_time_updates()
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
    system = ComparisonSystem()

if __name__ == "__main__":
    main()