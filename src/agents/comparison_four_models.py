#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 COMPARACIÓN 4 MODELOS IA REALES: DQN vs DeepDQN vs PPO vs A2C
"""

import sys
import os
import traceback

# Agregar directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# Importar dependencias
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('TkAgg')  # Backend compatible con Windows
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button, Slider
    from matplotlib.gridspec import GridSpec
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    import threading
    import time
    import warnings
    warnings.filterwarnings('ignore')
    
    # Configurar estilo de matplotlib
    plt.style.use('dark_background')
    
    # Importar componentes de RL
    from stable_baselines3 import DQN, A2C, PPO
    HAS_RL = True
    print("✅ Componentes de RL disponibles")
    
    # Importar nuestro sistema base
    from src.agents.ml_enhanced_system import MLEnhancedTradingSystem
    print("✅ Sistema base importado")
    
except ImportError as e:
    print(f"❌ Error importando dependencias: {e}")
    print("⚠️ Asegúrate de tener todas las dependencias instaladas:")
    print("pip install numpy pandas matplotlib stable-baselines3 gymnasium")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error inesperado: {e}")
    traceback.print_exc()
    sys.exit(1)

class FourModelsComparison:
    """Comparación de 4 modelos reales entrenados"""
    
    def __init__(self):
        print("🚀 Inicializando comparación de 4 modelos...")
        
        # Crear instancias independientes para cada modelo
        self.dqn_system = MLEnhancedTradingSystem(skip_selection=True)
        self.deepdqn_system = MLEnhancedTradingSystem(skip_selection=True)
        self.ppo_system = MLEnhancedTradingSystem(skip_selection=True)
        self.a2c_system = MLEnhancedTradingSystem(skip_selection=True)
        
        # Configurar cada modelo
        self._configure_systems()
        
        # Control de animación
        self.is_playing = False
        self.speed = 1.0
        self.current_step = 50
        self.window_size = 120
        self.fig = None
        
        # Colores para cada modelo
        self.colors = {
            'DQN': '#ff6b6b',      # Rojo coral
            'DeepDQN': '#4ecdc4',  # Turquesa
            'PPO': '#45b7d1',      # Azul claro
            'A2C': '#f7b731'       # Amarillo
        }
        
    def _configure_systems(self):
        """Configurar los 4 sistemas con parámetros específicos"""
        
        # DQN - AGRESIVO (red simple, decisiones rápidas)
        self.dqn_system.algorithm_choice = "1"
        self.dqn_system.algorithm_name = "DQN"
        self.dqn_system.algorithm_class = DQN
        # Configurar espacio de acción discreto para DQN
        self.dqn_system.action_space = self.dqn_system.action_space
        self.dqn_system.model_paths = {
            "1": [
                "data/models/qdn/model.zip",
                "data/models/best_qdn/model.zip"
            ]
        }
        self.dqn_system.max_position_risk = 0.12
        self.dqn_system.stop_loss_pct = 0.015
        self.dqn_system.take_profit_pct = 0.035
        self.dqn_system.min_trade_separation = 1
        self.dqn_system.max_daily_trades = 20
        self.dqn_system.consecutive_loss_limit = 6
        
        # DeepDQN - PRECISO (red profunda, decisiones calculadas)
        self.deepdqn_system.algorithm_choice = "2"
        self.deepdqn_system.algorithm_name = "DeepDQN"
        self.deepdqn_system.algorithm_class = DQN
        # Configurar espacio de acción discreto para DeepDQN
        self.deepdqn_system.action_space = self.deepdqn_system.action_space
        self.deepdqn_system.model_paths = {
            "2": [
                "data/models/deepqdn/model.zip",
                "data/models/best_deepqdn/model.zip"
            ]
        }
        self.deepdqn_system.max_position_risk = 0.08
        self.deepdqn_system.stop_loss_pct = 0.025
        self.deepdqn_system.take_profit_pct = 0.055
        self.deepdqn_system.min_trade_separation = 2
        self.deepdqn_system.max_daily_trades = 12
        self.deepdqn_system.consecutive_loss_limit = 4
        
        # PPO - BALANCEADO (política optimizada) - USANDO MODELO ML REAL
        self.ppo_system.algorithm_choice = "3"
        self.ppo_system.algorithm_name = "PPO"
        self.ppo_system.algorithm_class = PPO
        # ✅ CORREGIDO: Los modelos entrenados usan Discrete(2), no Box continuo
        from gymnasium import spaces
        self.ppo_system.action_space = spaces.Discrete(2)
        self.ppo_system.model_paths = {
            "3": [
                "data/models/ppo/model.zip",
                "data/models/best_ppo/best_model.zip"
            ]
        }
        self.ppo_system.max_position_risk = 0.10
        self.ppo_system.stop_loss_pct = 0.02
        self.ppo_system.take_profit_pct = 0.045
        self.ppo_system.min_trade_separation = 3
        self.ppo_system.max_daily_trades = 15
        self.ppo_system.consecutive_loss_limit = 4
        self.ppo_system.ml_weight = 0.7  # Más peso a ML cuando está disponible
        
        # A2C - CONSERVADOR (actor-critic balanceado) - USANDO MODELO ML REAL
        self.a2c_system.algorithm_choice = "4"
        self.a2c_system.algorithm_name = "A2C"
        self.a2c_system.algorithm_class = A2C
        # ✅ CORREGIDO: Los modelos entrenados usan Discrete(2), no Box continuo
        self.a2c_system.action_space = spaces.Discrete(2)
        self.a2c_system.model_paths = {
            "4": [
                "data/models/a2c/model.zip",
                "data/models/best_a2c/model.zip"
            ]
        }
        self.a2c_system.max_position_risk = 0.06
        self.a2c_system.stop_loss_pct = 0.03
        self.a2c_system.take_profit_pct = 0.065
        self.a2c_system.min_trade_separation = 4
        self.a2c_system.max_daily_trades = 8
        self.a2c_system.consecutive_loss_limit = 3
        self.a2c_system.ml_weight = 0.8  # Mucho más peso a ML cuando está disponible
        
        print("✅ DQN configurado: AGRESIVO (12% risk, 1.5% SL, 3.5% TP)")
        print("✅ DeepDQN configurado: PRECISO (8% risk, 2.5% SL, 5.5% TP)")
        print("✅ PPO configurado: BALANCEADO (10% risk, 2% SL, 4.5% TP) - Modo técnico avanzado")
        print("✅ A2C configurado: CONSERVADOR (6% risk, 3% SL, 6.5% TP) - Modo técnico ultra-conservador")
        
    def initialize_systems(self):
        """Inicializar los sistemas"""
        print("🔄 Inicializando sistemas...")
        
        # Generar datos
        print("📊 Generando datos de mercado...")
        self.dqn_system.generate_market_data()
        
        # Asegurar que los indicadores técnicos estén calculados
        print("📊 Calculando indicadores técnicos...")
        self.dqn_system.data = self.dqn_system.calculate_indicators(self.dqn_system.data)
        
        # Copiar datos a otros sistemas
        self.deepdqn_system.data = self.dqn_system.data.copy()
        self.ppo_system.data = self.dqn_system.data.copy()
        self.a2c_system.data = self.dqn_system.data.copy()
        
        # Cargar modelos ML para todos los sistemas (igual que ml_enhanced_system.py)
        print("\n🤖 Cargando modelo DQN...")
        self.dqn_system.load_ml_model()
        
        print("\n🤖 Cargando modelo DeepDQN...")
        self.deepdqn_system.load_ml_model()
        
        print("\n🤖 Cargando modelo PPO...")
        self.ppo_system.load_ml_model()
        
        print("\n🤖 Cargando modelo A2C...")
        self.a2c_system.load_ml_model()
        
        # Inicializar estados
        for system in [self.dqn_system, self.deepdqn_system, 
                      self.ppo_system, self.a2c_system]:
            system.current_step = 50
            system.current_capital = system.initial_capital
            system.position_size = 0
            system.position_type = None
            system.daily_trades = 0
            system.consecutive_losses = 0
            system.last_trade_step = 0
            
            # Reinicializar arrays de seguimiento para el paso actual
            system.initialize_tracking_arrays()
            
            # Asegurar que tenemos suficientes datos
            if len(system.portfolio_values) <= system.current_step:
                system.portfolio_values.extend([system.initial_capital] * (system.current_step + 100))
            if len(system.actions) <= system.current_step:
                system.actions.extend([0] * (system.current_step + 100))
            if len(system.ml_predictions) <= system.current_step:
                system.ml_predictions.extend([0] * (system.current_step + 100))
            if len(system.technical_signals) <= system.current_step:
                system.technical_signals.extend([0] * (system.current_step + 100))
            
        print("✅ Sistemas inicializados")
        
    # PPO y A2C ahora usan la misma lógica estándar que ml_enhanced_system.py
    # Intentan cargar modelos ML reales, y si fallan, usan análisis técnico estándar
        
    def create_interface(self):
        """Crear interfaz gráfica idéntica a la imagen de referencia"""
        print("🎨 Creando interfaz gráfica...")
        
        # Inicializar sistemas
        self.initialize_systems()
        
        # Crear figura con fondo negro
        self.fig = plt.figure(figsize=(20, 10), facecolor='black')
        
        # Título principal
        self.fig.suptitle("🤖 COMPARACIÓN 4 MODELOS IA REALES: QDN vs DeepQDN vs PPO vs A2C",
                         fontsize=16, color='white', y=0.98)
        
        # Grid para 4 gráficos (2x2)
        gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                     hspace=0.15, wspace=0.15)
        
        # Crear subplots
        self.axes = {}
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        models = ['DQN', 'DeepDQN', 'PPO', 'A2C']
        systems = [self.dqn_system, self.deepdqn_system, 
                  self.ppo_system, self.a2c_system]
        
        for (i, j), name, system in zip(positions, models, systems):
            ax = self.fig.add_subplot(gs[i, j])
            ax.set_facecolor('#1a1a1a')
            
            # Configurar estilo
            ax.grid(True, alpha=0.2, color='#333333')
            for spine in ax.spines.values():
                spine.set_color('#333333')
            ax.tick_params(colors='#888888')
            
            self.axes[name] = ax
            
        # Crear área para botones
        self.button_axes = {}
        
        # Botón PLAY
        self.button_axes['play'] = plt.axes([0.40, 0.02, 0.06, 0.04])
        self.button_axes['play'].set_facecolor('#2a2a2a')
        self.buttons = {}
        self.buttons['play'] = Button(self.button_axes['play'], '▶ PLAY', color='white')
        self.buttons['play'].on_clicked(self._on_play)
        
        # Botón PAUSE
        self.button_axes['pause'] = plt.axes([0.47, 0.02, 0.06, 0.04])
        self.button_axes['pause'].set_facecolor('#2a2a2a')
        self.buttons['pause'] = Button(self.button_axes['pause'], '⏸ PAUSE', color='white')
        self.buttons['pause'].on_clicked(self._on_pause)
        
        # Botón STOP
        self.button_axes['stop'] = plt.axes([0.54, 0.02, 0.06, 0.04])
        self.button_axes['stop'].set_facecolor('#2a2a2a')
        self.buttons['stop'] = Button(self.button_axes['stop'], '⏹ STOP', color='white')
        self.buttons['stop'].on_clicked(self._on_stop)
        
        # Texto inferior
        plt.figtext(0.5, 0.02, "QDN(🚀 AGRESIVO-ML) | DeepQDN(🎯 PRECISO-ML) | PPO(⚖️ BALANCEADO-TÉCNICO) | A2C(🛡️ CONSERVADOR-TÉCNICO)",
                   ha='center', color='white', fontsize=10)
        
        # Animación
        self.animation = FuncAnimation(self.fig, self.update_plots, 
                                     interval=50, blit=False)
        
        plt.show()
        
    def _on_play(self, event):
        """Manejador del botón PLAY"""
        print("▶️ Iniciando simulación...")
        self.is_playing = True
        self.buttons['play'].color = '#00ff00'
        self.buttons['pause'].color = 'white'
        self.buttons['stop'].color = 'white'
        plt.draw()
        
    def _on_pause(self, event):
        """Manejador del botón PAUSE"""
        print("⏸️ Simulación pausada")
        self.is_playing = False
        self.buttons['play'].color = 'white'
        self.buttons['pause'].color = '#ffff00'
        self.buttons['stop'].color = 'white'
        plt.draw()
        
    def _on_stop(self, event):
        """Manejador del botón STOP"""
        print("⏹️ Simulación detenida")
        self.is_playing = False
        self.current_step = 50
        self.initialize_systems()
        self.buttons['play'].color = 'white'
        self.buttons['pause'].color = 'white'
        self.buttons['stop'].color = '#ff0000'
        plt.draw()
        
    def play(self):
        """Iniciar simulación"""
        self._on_play(None)
        
    def pause(self):
        """Pausar simulación"""
        self._on_pause(None)
        
    def stop(self):
        """Detener y reiniciar"""
        self._on_stop(None)
        
    def update_plots(self, frame):
        """Actualizar gráficos"""
        if not self.is_playing:
            return
            
        # Avanzar sistemas
        if self.current_step < len(self.dqn_system.data) - 1:
            self.step_forward()
            
            # Actualizar cada gráfico
            self.update_model_plot('DQN', self.dqn_system)
            self.update_model_plot('DeepDQN', self.deepdqn_system)
            self.update_model_plot('PPO', self.ppo_system)
            self.update_model_plot('A2C', self.a2c_system)
            
    def update_model_plot(self, name, system):
        """Actualizar gráfico de un modelo específico"""
        ax = self.axes[name]
        ax.clear()
        
        # Configurar estilo
        ax.set_facecolor('#1a1a1a')
        ax.grid(True, alpha=0.2, color='#333333')
        for spine in ax.spines.values():
            spine.set_color('#333333')
        ax.tick_params(colors='#888888')
        
        # Datos visibles
        start = max(0, self.current_step - self.window_size)
        end = self.current_step + 1
        
        # Graficar precio
        prices = system.data['price'].iloc[start:end]
        dates = range(len(prices))
        ax.plot(dates, prices, color=self.colors[name], linewidth=1.5)
        
        # Señales
        window_buys = [s - start for s in system.buy_signals if start <= s < end]
        window_sells = [s - start for s in system.sell_signals if start <= s < end]
        
        if window_buys:
            buy_prices = [system.data['price'].iloc[s + start] for s in window_buys]
            ax.scatter(window_buys, buy_prices, color='#00ff00', marker='^', s=100, zorder=5)
            for x, y in zip(window_buys, buy_prices):
                ax.text(x, y + 0.5, 'BUY', color='#00ff00', ha='center', fontsize=8)
                
        if window_sells:
            sell_prices = [system.data['price'].iloc[s + start] for s in window_sells]
            ax.scatter(window_sells, sell_prices, color='#ff0000', marker='v', s=100, zorder=5)
            for x, y in zip(window_sells, sell_prices):
                ax.text(x, y - 0.5, 'SELL', color='#ff0000', ha='center', fontsize=8)
        
        # ✅ ARREGLO: Calcular portfolio value correctamente
        current_value = system.portfolio_values[self.current_step] if self.current_step < len(system.portfolio_values) else system.initial_capital
        
        # 🔧 CORREGIR el cálculo del portfolio para evitar duplicación
        if hasattr(system, 'position_size') and system.position_size > 0:
            current_price = system.data['price'].iloc[self.current_step]
            # Capital disponible (dinero que no está invertido) + valor actual de la posición
            invested_amount = system.position_size * getattr(system, 'entry_price', current_price)
            available_cash = system.current_capital - invested_amount
            position_value = system.position_size * current_price
            current_value = available_cash + position_value
        else:
            current_value = system.current_capital
            
        total_return = (current_value / system.initial_capital - 1) * 100
        win_rate = (system.profitable_trades / max(system.total_trades, 1)) * 100
        
        # Información específica del modelo
        model_type = ""
        if name == "DQN":
            model_type = "🚀 AGRESIVO-ML"
        elif name == "DeepDQN":
            model_type = "🎯 PRECISO-ML"  
        elif name == "PPO":
            model_type = "⚖️ BALANCEADO-TÉC"
        elif name == "A2C":
            model_type = "🛡️ CONSERVADOR-TÉC"
        
        # ✅ INFORMACIÓN COMPACTA
        info_text = f"""{model_type}
Capital: ${current_value:,.0f}
P&L: ${current_value - system.initial_capital:+,.0f} ({total_return:+.1f}%)
Posición: {system.position_size}
Trades: {system.total_trades} | Win: {win_rate:.1f}%
{'🟢 LONG' if system.position_size > 0 else '⚪ IDLE'}"""
        
        # Color de la caja según P&L
        box_color = '#1a3a1a' if total_return >= 0 else '#3a1a1a'
        
        # ✅ ARREGLO VISUAL: Mover panel a la esquina inferior derecha para que no se superponga
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
                fontsize=7, color='white', va='bottom', ha='right',
                bbox=dict(facecolor=box_color, alpha=0.9, edgecolor='#333333', pad=4))
        
        # Título con P&L
        title_color = '#00ff00' if total_return >= 0 else '#ff0000'
        ax.set_title(f"{name} - ${current_value:,.0f} ({total_return:+.1f}%)",
                    color=title_color, pad=15)  # Más padding para evitar superposición
        
    def step_forward(self):
        """Avanzar un paso en todos los sistemas con comportamientos diferenciados"""
        self.current_step += 1
        
        # Actualizar paso en todos los sistemas
        for system in [self.dqn_system, self.deepdqn_system, self.ppo_system, self.a2c_system]:
            system.current_step = self.current_step
        
        # DQN - Ejecutar lógica completa de trading con modelo ML
        try:
            if self.dqn_system.ml_model is not None:
                state = self.dqn_system.get_state()
                action = self.dqn_system.ml_model.predict(state, deterministic=True)[0]
                self.dqn_system.step_forward()
            else:
                self.dqn_system.step_forward()
        except Exception as e:
            print(f"Error en DQN: {e}")
        
        # DeepDQN - Ejecutar lógica completa de trading con modelo ML
        try:
            if self.deepdqn_system.ml_model is not None:
                state = self.deepdqn_system.get_state()
                action = self.deepdqn_system.ml_model.predict(state, deterministic=True)[0]
                self.deepdqn_system.step_forward()
            else:
                self.deepdqn_system.step_forward()
        except Exception as e:
            print(f"Error en DeepDQN: {e}")
        
        # PPO - Usar lógica estándar igual que ml_enhanced_system.py
        try:
            if self.ppo_system.ml_model is not None:
                state = self.ppo_system.get_state()
                action = self.ppo_system.ml_model.predict(state, deterministic=True)[0]
                self.ppo_system.step_forward()
            else:
                self.ppo_system.step_forward()
        except Exception as e:
            print(f"Error en PPO: {e}")
        
        # A2C - Usar lógica estándar igual que ml_enhanced_system.py
        try:
            if self.a2c_system.ml_model is not None:
                state = self.a2c_system.get_state()
                action = self.a2c_system.ml_model.predict(state, deterministic=True)[0]
                self.a2c_system.step_forward()
            else:
                self.a2c_system.step_forward()
        except Exception as e:
            print(f"Error en A2C: {e}")
    
    # Funciones step_ppo_forward y step_a2c_forward eliminadas
    # PPO y A2C ahora usan la lógica estándar de ml_enhanced_system.py

def main():
    """Función principal"""
    print("\n" + "="*60)
    print("🤖 COMPARACIÓN 4 MODELOS IA REALES")
    print("DQN vs DeepDQN vs PPO vs A2C")
    print("="*60 + "\n")
    
    try:
        comparison = FourModelsComparison()
        comparison.create_interface()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 