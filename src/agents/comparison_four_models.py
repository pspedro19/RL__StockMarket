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
        
        # Colores profesionales para cada modelo
        self.colors = {
            'DQN': '#ff4757',      # Rojo profesional
            'DeepDQN': '#00d2d3',  # Cian profesional
            'PPO': '#3742fa',      # Azul profesional
            'A2C': '#ffa502'       # Naranja profesional
        }
        
    def _configure_systems(self):
        """Configurar los 4 sistemas con parámetros específicos"""
        
        # DQN - AGRESIVO (red simple, decisiones rápidas)
        self.dqn_system.algorithm_choice = "1"
        self.dqn_system.algorithm_name = "DQN"
        self.dqn_system.algorithm_class = DQN
        # Configurar espacio de acción discreto para DQN
        from gymnasium import spaces
        self.dqn_system.action_space = spaces.Discrete(2)
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
        self.deepdqn_system.action_space = spaces.Discrete(2)
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
        """Crear interfaz gráfica profesional"""
        print("🎨 Creando interfaz gráfica profesional...")
        
        # Inicializar sistemas
        self.initialize_systems()
        
        # Crear figura con fondo profesional
        self.fig = plt.figure(figsize=(22, 12), facecolor='#0a0a0a')
        
        # Título principal profesional
        self.fig.suptitle("SISTEMA DE TRADING - COMPARACIÓN DE 4 MODELOS DE INTELIGENCIA ARTIFICIAL",
                         fontsize=18, color='white', y=0.96, fontweight='bold', 
                         fontfamily='monospace')
        
        # Subtítulo
        plt.figtext(0.5, 0.93, "DQN vs DeepDQN vs PPO vs A2C | Análisis Técnico + Machine Learning",
                   ha='center', color='#cccccc', fontsize=12, fontweight='normal',
                   fontfamily='monospace')
        
        # Grid mejorado para 4 gráficos (2x2) con más espacio
        gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                     hspace=0.25, wspace=0.20, top=0.88, bottom=0.12, 
                     left=0.05, right=0.95)
        
        # Crear subplots con mejor estilo
        self.axes = {}
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        models = ['DQN', 'DeepDQN', 'PPO', 'A2C']
        systems = [self.dqn_system, self.deepdqn_system, 
                  self.ppo_system, self.a2c_system]
        
        for (i, j), name, system in zip(positions, models, systems):
            ax = self.fig.add_subplot(gs[i, j])
            ax.set_facecolor('#111111')
            
            # Configurar estilo profesional
            ax.grid(True, alpha=0.3, color='#333333', linewidth=0.5, linestyle='-')
            for spine in ax.spines.values():
                spine.set_color('#555555')
                spine.set_linewidth(1.5)
            ax.tick_params(colors='#aaaaaa', labelsize=9)
            ax.set_xlabel('Tiempo', color='#aaaaaa', fontweight='bold', fontsize=10)
            ax.set_ylabel('Precio USD', color='#aaaaaa', fontweight='bold', fontsize=10)
            
            self.axes[name] = ax
            
        # Crear área para controles profesionales
        self.button_axes = {}
        
        # Panel de control con fondo
        control_bg = plt.axes([0.02, 0.02, 0.96, 0.08])
        control_bg.set_facecolor('#1a1a1a')
        control_bg.set_xticks([])
        control_bg.set_yticks([])
        for spine in control_bg.spines.values():
            spine.set_color('#444444')
            spine.set_linewidth(2)
        
        # Botones con texto profesional (sin iconos)
        button_style = {
            'facecolor': '#2a2a2a',
            'edgecolor': '#555555',
            'linewidth': 2
        }
        
        # Botón PLAY
        self.button_axes['play'] = plt.axes([0.15, 0.04, 0.08, 0.04])
        self.button_axes['play'].set_facecolor(button_style['facecolor'])
        for spine in self.button_axes['play'].spines.values():
            spine.set_color(button_style['edgecolor'])
            spine.set_linewidth(button_style['linewidth'])
        self.buttons = {}
        self.buttons['play'] = Button(self.button_axes['play'], 'PLAY', 
                                     color='#2a2a2a', hovercolor='#3a3a3a')
        self.buttons['play'].label.set_fontweight('bold')
        self.buttons['play'].label.set_fontsize(11)
        self.buttons['play'].label.set_color('white')
        self.buttons['play'].on_clicked(self._on_play)
        
        # Botón PAUSE
        self.button_axes['pause'] = plt.axes([0.24, 0.04, 0.08, 0.04])
        self.button_axes['pause'].set_facecolor(button_style['facecolor'])
        for spine in self.button_axes['pause'].spines.values():
            spine.set_color(button_style['edgecolor'])
            spine.set_linewidth(button_style['linewidth'])
        self.buttons['pause'] = Button(self.button_axes['pause'], 'PAUSE', 
                                      color='#2a2a2a', hovercolor='#3a3a3a')
        self.buttons['pause'].label.set_fontweight('bold')
        self.buttons['pause'].label.set_fontsize(11)
        self.buttons['pause'].label.set_color('white')
        self.buttons['pause'].on_clicked(self._on_pause)
        
        # Botón STOP
        self.button_axes['stop'] = plt.axes([0.33, 0.04, 0.08, 0.04])
        self.button_axes['stop'].set_facecolor(button_style['facecolor'])
        for spine in self.button_axes['stop'].spines.values():
            spine.set_color(button_style['edgecolor'])
            spine.set_linewidth(button_style['linewidth'])
        self.buttons['stop'] = Button(self.button_axes['stop'], 'RESET', 
                                     color='#2a2a2a', hovercolor='#3a3a3a')
        self.buttons['stop'].label.set_fontweight('bold')
        self.buttons['stop'].label.set_fontsize(11)
        self.buttons['stop'].label.set_color('white')
        self.buttons['stop'].on_clicked(self._on_stop)
        
        # Etiquetas informativas profesionales
        plt.figtext(0.05, 0.02, "MODELOS:", ha='left', color='white', 
                   fontsize=11, fontweight='bold', fontfamily='monospace')
        
        plt.figtext(0.13, 0.02, "DQN (AGRESIVO) | DeepDQN (PRECISO) | PPO (BALANCEADO) | A2C (CONSERVADOR)",
                   ha='left', color='#cccccc', fontsize=10, fontweight='normal',
                   fontfamily='monospace')
        
        # Estado del sistema
        plt.figtext(0.70, 0.04, "SISTEMA:", ha='left', color='white', 
                   fontsize=11, fontweight='bold', fontfamily='monospace')
        plt.figtext(0.78, 0.04, "LISTO PARA TRADING", ha='left', color='#00ff00', 
                   fontsize=10, fontweight='bold', fontfamily='monospace')
        
        # Animación
        self.animation = FuncAnimation(self.fig, self.update_plots, 
                                     interval=50, blit=False)
        
        plt.show()
        
    def _on_play(self, event):
        """Manejador del botón PLAY con feedback visual profesional"""
        print("▶️ INICIANDO SIMULACIÓN...")
        self.is_playing = True
        
        # Actualizar colores de botones con estilo profesional
        self.buttons['play'].color = '#2d5016'  # Verde oscuro activo
        self.buttons['play'].hovercolor = '#3d6026'
        self.buttons['pause'].color = '#2a2a2a'  # Gris normal
        self.buttons['pause'].hovercolor = '#3a3a3a'
        self.buttons['stop'].color = '#2a2a2a'   # Gris normal
        self.buttons['stop'].hovercolor = '#3a3a3a'
        
        plt.draw()
        
    def _on_pause(self, event):
        """Manejador del botón PAUSE con feedback visual profesional"""
        print("⏸️ SIMULACIÓN PAUSADA")
        self.is_playing = False
        
        # Actualizar colores de botones con estilo profesional
        self.buttons['play'].color = '#2a2a2a'   # Gris normal
        self.buttons['play'].hovercolor = '#3a3a3a'
        self.buttons['pause'].color = '#5c4e00'  # Amarillo oscuro activo
        self.buttons['pause'].hovercolor = '#6c5e10'
        self.buttons['stop'].color = '#2a2a2a'   # Gris normal
        self.buttons['stop'].hovercolor = '#3a3a3a'
        
        plt.draw()
        
    def _on_stop(self, event):
        """Manejador del botón STOP con feedback visual profesional"""
        print("⏹️ SIMULACIÓN REINICIADA")
        self.is_playing = False
        self.current_step = 50
        self.initialize_systems()
        
        # Actualizar colores de botones con estilo profesional
        self.buttons['play'].color = '#2a2a2a'   # Gris normal
        self.buttons['play'].hovercolor = '#3a3a3a'
        self.buttons['pause'].color = '#2a2a2a'  # Gris normal
        self.buttons['pause'].hovercolor = '#3a3a3a'
        self.buttons['stop'].color = '#501616'   # Rojo oscuro activo
        self.buttons['stop'].hovercolor = '#602626'
        
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
        """Actualizar gráfico de un modelo específico con estilo profesional"""
        ax = self.axes[name]
        ax.clear()
        
        # Configurar estilo profesional
        ax.set_facecolor('#111111')
        ax.grid(True, alpha=0.3, color='#333333', linewidth=0.5, linestyle='-')
        for spine in ax.spines.values():
            spine.set_color('#555555')
            spine.set_linewidth(1.5)
        ax.tick_params(colors='#aaaaaa', labelsize=9)
        ax.set_xlabel('Tiempo', color='#aaaaaa', fontweight='bold', fontsize=10)
        ax.set_ylabel('Precio USD', color='#aaaaaa', fontweight='bold', fontsize=10)
        
        # Datos visibles
        start = max(0, self.current_step - self.window_size)
        end = self.current_step + 1
        
        # Graficar precio con línea más gruesa y profesional
        prices = system.data['price'].iloc[start:end]
        dates = range(len(prices))
        ax.plot(dates, prices, color=self.colors[name], linewidth=2.5, alpha=0.9)
        
        # Señales de compra/venta con mejor estilo
        window_buys = [s - start for s in system.buy_signals if start <= s < end]
        window_sells = [s - start for s in system.sell_signals if start <= s < end]
        
        if window_buys:
            buy_prices = [system.data['price'].iloc[s + start] for s in window_buys]
            ax.scatter(window_buys, buy_prices, color='#00ff41', marker='^', 
                      s=120, zorder=5, edgecolors='white', linewidths=1.5)
            for x, y in zip(window_buys, buy_prices):
                ax.text(x, y + 0.5, 'BUY', color='#00ff41', ha='center', 
                       fontsize=9, fontweight='bold', fontfamily='monospace')
                
        if window_sells:
            sell_prices = [system.data['price'].iloc[s + start] for s in window_sells]
            ax.scatter(window_sells, sell_prices, color='#ff4444', marker='v', 
                      s=120, zorder=5, edgecolors='white', linewidths=1.5)
            for x, y in zip(window_sells, sell_prices):
                ax.text(x, y - 0.5, 'SELL', color='#ff4444', ha='center', 
                       fontsize=9, fontweight='bold', fontfamily='monospace')
        
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
        
        # Información específica del modelo profesional
        model_descriptions = {
            "DQN": "AGRESIVO | ML",
            "DeepDQN": "PRECISO | ML",  
            "PPO": "BALANCEADO | TÉCNICO",
            "A2C": "CONSERVADOR | TÉCNICO"
        }
        
        model_type = model_descriptions.get(name, name)
        
        # Panel de información profesional
        info_text = f"""{model_type}
CAPITAL: ${current_value:,.0f}
P&L: ${current_value - system.initial_capital:+,.0f} ({total_return:+.1f}%)
POSICIÓN: {system.position_size if system.position_size > 0 else 0}
TRADES: {system.total_trades} | WIN: {win_rate:.1f}%
ESTADO: {'LONG' if system.position_size > 0 else 'IDLE'}"""
        
        # Color del panel según performance
        if total_return >= 5:
            box_color = '#1a4a1a'  # Verde oscuro para buena performance
            border_color = '#00ff41'
        elif total_return >= 0:
            box_color = '#2a3a1a'  # Verde muy oscuro para break-even
            border_color = '#88aa00'
        else:
            box_color = '#4a1a1a'  # Rojo oscuro para pérdidas
            border_color = '#ff4444'
        
        # Panel de información en esquina inferior derecha
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
                fontsize=8, color='white', va='bottom', ha='right',
                fontweight='bold', fontfamily='monospace',
                bbox=dict(facecolor=box_color, alpha=0.95, edgecolor=border_color, 
                         linewidth=2, pad=6))
        
        # Título profesional con indicador de performance
        performance_indicator = "↗" if total_return >= 0 else "↘"
        title_color = '#00ff41' if total_return >= 0 else '#ff4444'
        
        ax.set_title(f"{name} | ${current_value:,.0f} ({total_return:+.1f}%) {performance_indicator}",
                    color=title_color, pad=20, fontweight='bold', 
                    fontsize=12, fontfamily='monospace')
        
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