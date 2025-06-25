#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 GENERADOR DE MÚLTIPLES VISUALIZACIONES - SISTEMA AVANZADO DE TRADING
Crea múltiples PNGs separados para análisis detallado
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Agregar directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from src.agents.advanced_trading_analytics import AdvancedTradingAnalytics

class MultiVisualizationGenerator:
    """Generador de múltiples visualizaciones separadas"""
    
    def __init__(self, analytics_system):
        self.system = analytics_system
        self.output_dir = "visualizations"
        self._create_output_dir()
        
    def _create_output_dir(self):
        """Crear directorio de salida"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Directorio creado: {self.output_dir}/")
    
    def generate_all_visualizations(self):
        """Generar todas las visualizaciones por separado"""
        print("🎨 Generando visualizaciones múltiples...")
        
        # Configurar estilo
        plt.style.use('dark_background')
        
        visualizations = [
            ("01_precio_trades", self._viz_precio_trades, "Precio y Trades con IDs"),
            ("02_metricas_financieras", self._viz_metricas_panel, "Panel de Métricas Financieras"),
            ("03_distribuciones", self._viz_distribuciones, "Distribuciones y Estadísticas"),
            ("04_equity_drawdown", self._viz_equity_drawdown, "Equity y Drawdown"),
            ("05_performance_analysis", self._viz_performance_analysis, "Análisis de Performance"),
            ("06_trades_timeline", self._viz_trades_timeline, "Timeline de Trades"),
            ("07_risk_metrics", self._viz_risk_metrics, "Métricas de Riesgo"),
            ("08_technical_analysis", self._viz_technical_analysis, "Análisis Técnico"),
        ]
        
        successful = 0
        for filename, viz_func, title in visualizations:
            try:
                print(f"📊 Generando: {title}...")
                viz_func(filename, title)
                successful += 1
            except Exception as e:
                print(f"❌ Error en {title}: {e}")
        
        print(f"\n✅ Generadas {successful}/{len(visualizations)} visualizaciones")
        print(f"📁 Archivos guardados en: {self.output_dir}/")
        
        return successful
    
    def _viz_precio_trades(self, filename, title):
        """Visualización principal: Precio y Trades"""
        fig, ax = plt.subplots(figsize=(16, 10), facecolor='#0a0a0a')
        
        # Plotear precio
        ax.plot(self.system.data['price'], color='#00d2d3', linewidth=2, alpha=0.9, label='Precio SPY')
        
        # Plotear trades con información detallada
        for i, trade in enumerate(self.system.trades):
            if (trade.entry_time < len(self.system.data) and 
                trade.exit_time is not None and trade.exit_time < len(self.system.data)):
                
                entry_price = self.system.data.iloc[int(trade.entry_time)]['price']
                exit_price = self.system.data.iloc[int(trade.exit_time)]['price']
                
                # Color y estilo según resultado
                color = '#00ff41' if trade.return_pct > 0 else '#ff4444'
                alpha = 0.8 if trade.return_pct > 0 else 0.6
                linewidth = 3 if abs(trade.return_pct) > 0.5 else 2
                
                # Línea de conexión
                ax.plot([trade.entry_time, trade.exit_time], 
                       [entry_price, exit_price], 
                       color=color, linewidth=linewidth, alpha=alpha)
                
                # Marcadores de entrada (triángulos hacia arriba)
                ax.scatter(trade.entry_time, entry_price, 
                          color='#00ff41', marker='^', s=80, zorder=5, 
                          edgecolors='white', linewidths=1.5,
                          label='Entrada' if i == 0 else "")
                
                # Marcadores de salida (triángulos hacia abajo)
                ax.scatter(trade.exit_time, exit_price, 
                          color='#ff4444', marker='v', s=80, zorder=5,
                          edgecolors='white', linewidths=1.5,
                          label='Salida' if i == 0 else "")
                
                # Etiquetas con información del trade (cada 3 trades)
                if i % 3 == 0:
                    mid_time = (trade.entry_time + trade.exit_time) / 2
                    mid_price = (entry_price + exit_price) / 2
                    
                    ax.annotate(f'{trade.id[:8]}\n{trade.return_pct:.1f}%', 
                              xy=(mid_time, mid_price),
                              xytext=(10, 10), textcoords='offset points',
                              fontsize=9, color='white', alpha=0.8,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                              ha='center')
        
        # Configuración del gráfico
        ax.set_title(f"{title}\nTotal: {len(self.system.trades)} trades | Win Rate: {self.system.financial_metrics.get('win_rate', 0):.1f}%", 
                    fontweight='bold', color='white', fontsize=16, pad=20)
        ax.set_xlabel("Tiempo (steps)", fontweight='bold', color='white', fontsize=12)
        ax.set_ylabel("Precio USD", fontweight='bold', color='white', fontsize=12)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Guardar
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}.png', facecolor='#0a0a0a', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_metricas_panel(self, filename, title):
        """Panel de métricas financieras expandido"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0a0a0a')
        
        metrics = self.system.financial_metrics
        
        # Panel 1: Métricas Básicas
        basic_text = f"""
ESTADÍSTICAS BÁSICAS

Total de Trades: {metrics.get('total_trades', 0)}
Trades Ganadores: {metrics.get('winning_trades', 0)}
Trades Perdedores: {metrics.get('losing_trades', 0)}
Win Rate: {metrics.get('win_rate', 0):.2f}%

Mejor Trade: {max([t.return_pct for t in self.system.trades]) if self.system.trades else 0:.2f}%
Peor Trade: {min([t.return_pct for t in self.system.trades]) if self.system.trades else 0:.2f}%
        """
        
        ax1.text(0.05, 0.95, basic_text, transform=ax1.transAxes,
                fontsize=14, color='white', va='top', fontfamily='monospace',
                bbox=dict(facecolor='#1a3a1a', alpha=0.9, pad=15))
        ax1.set_title("ESTADÍSTICAS BÁSICAS", fontweight='bold', color='white', fontsize=14)
        ax1.axis('off')
        
        # Panel 2: Rendimiento
        return_text = f"""
RENDIMIENTO FINANCIERO

Retorno Total: ${metrics.get('total_return_abs', 0):.2f}
Retorno Porcentual: {metrics.get('total_return_pct', 0):.3f}%
Retorno Promedio: {metrics.get('avg_return', 0):.3f}%

Promedio Ganancia: {metrics.get('avg_winning_trade', 0):.2f}%
Promedio Pérdida: {metrics.get('avg_losing_trade', 0):.2f}%

Capital Inicial: $100,000
Capital Final: ${100000 + metrics.get('total_return_abs', 0):.2f}
        """
        
        ax2.text(0.05, 0.95, return_text, transform=ax2.transAxes,
                fontsize=14, color='white', va='top', fontfamily='monospace',
                bbox=dict(facecolor='#1a1a3a', alpha=0.9, pad=15))
        ax2.set_title("RENDIMIENTO", fontweight='bold', color='white', fontsize=14)
        ax2.axis('off')
        
        # Panel 3: Riesgo
        risk_text = f"""
MÉTRICAS DE RIESGO

Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}
Max Drawdown: ${metrics.get('max_drawdown', 0):.2f}
Profit Factor: {metrics.get('profit_factor', 0):.3f}

Volatilidad: {np.std([t.return_pct for t in self.system.trades]) if self.system.trades else 0:.2f}%

Ratio Ganancia/Pérdida: {abs(metrics.get('avg_winning_trade', 0)/metrics.get('avg_losing_trade', -1)) if metrics.get('avg_losing_trade', 0) != 0 else 'N/A'}
        """
        
        ax3.text(0.05, 0.95, risk_text, transform=ax3.transAxes,
                fontsize=14, color='white', va='top', fontfamily='monospace',
                bbox=dict(facecolor='#3a1a1a', alpha=0.9, pad=15))
        ax3.set_title("MÉTRICAS DE RIESGO", fontweight='bold', color='white', fontsize=14)
        ax3.axis('off')
        
        # Panel 4: ML y Timing
        ml_text = f"""
MACHINE LEARNING & TIMING

MAPE (Error Predicción): {metrics.get('mape', 0):.2f}%
Precisión ML: {'Alta ✅' if metrics.get('mape', 100) < 5 else 'Media ⚠️' if metrics.get('mape', 100) < 15 else 'Baja ❌'}

Duración Promedio: {metrics.get('avg_duration', 0):.1f} steps
Trade más Largo: {max([t.duration for t in self.system.trades if t.duration]) if self.system.trades else 0:.1f} steps
Trade más Corto: {min([t.duration for t in self.system.trades if t.duration]) if self.system.trades else 0:.1f} steps

Control PID: ACTIVO ✅
        """
        
        ax4.text(0.05, 0.95, ml_text, transform=ax4.transAxes,
                fontsize=14, color='white', va='top', fontfamily='monospace',
                bbox=dict(facecolor='#1a1a1a', alpha=0.9, pad=15))
        ax4.set_title("ML & TIMING", fontweight='bold', color='white', fontsize=14)
        ax4.axis('off')
        
        plt.suptitle(title, fontsize=18, fontweight='bold', color='white', y=0.95)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}.png', facecolor='#0a0a0a', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_distribuciones(self, filename, title):
        """Distribuciones y estadísticas"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0a0a0a')
        
        if not self.system.trades:
            fig.text(0.5, 0.5, "NO HAY DATOS SUFICIENTES", ha='center', va='center',
                    fontsize=20, color='white', fontweight='bold')
            plt.savefig(f'{self.output_dir}/{filename}.png', facecolor='#0a0a0a', dpi=300)
            plt.close()
            return
        
        returns = [trade.return_pct for trade in self.system.trades]
        durations = [trade.duration for trade in self.system.trades if trade.duration]
        
        # 1. Distribución de retornos
        ax1.hist(returns, bins=15, color='#3742fa', alpha=0.7, edgecolor='white', linewidth=1)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        ax1.axvline(mean_return, color='#ffa502', linestyle='--', linewidth=2, 
                   label=f'Media: {mean_return:.2f}%')
        ax1.axvline(0, color='white', linestyle='-', alpha=0.5, label='Break-even')
        ax1.set_title("DISTRIBUCIÓN DE RETORNOS", fontweight='bold', color='white')
        ax1.set_xlabel("Retorno %", color='white')
        ax1.set_ylabel("Frecuencia", color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot de retornos
        bp = ax2.boxplot([returns], patch_artist=True, labels=['Retornos'])
        bp['boxes'][0].set_facecolor('#3742fa')
        bp['boxes'][0].set_alpha(0.7)
        ax2.set_title("ESTADÍSTICAS DE RETORNOS", fontweight='bold', color='white')
        ax2.set_ylabel("Retorno %", color='white')
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribución de duraciones
        if durations:
            ax3.hist(durations, bins=10, color='#ff6b6b', alpha=0.7, edgecolor='white')
            ax3.axvline(np.mean(durations), color='#ffa502', linestyle='--', linewidth=2,
                       label=f'Media: {np.mean(durations):.1f}')
        ax3.set_title("DISTRIBUCIÓN DE DURACIONES", fontweight='bold', color='white')
        ax3.set_xlabel("Duración (steps)", color='white')
        ax3.set_ylabel("Frecuencia", color='white')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Scatter: Retorno vs Duración
        if durations and len(durations) == len(returns):
            colors = ['#00ff41' if r > 0 else '#ff4444' for r in returns]
            ax4.scatter(durations, returns, c=colors, alpha=0.7, s=60, edgecolors='white')
            ax4.axhline(0, color='white', linestyle='-', alpha=0.5)
            ax4.set_title("RETORNO vs DURACIÓN", fontweight='bold', color='white')
            ax4.set_xlabel("Duración (steps)", color='white')
            ax4.set_ylabel("Retorno %", color='white')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', color='white', y=0.95)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}.png', facecolor='#0a0a0a', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_equity_drawdown(self, filename, title):
        """Curva de equity y drawdown"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), facecolor='#0a0a0a')
        
        if not self.system.trades:
            fig.text(0.5, 0.5, "NO HAY DATOS SUFICIENTES", ha='center', va='center',
                    fontsize=20, color='white', fontweight='bold')
            plt.savefig(f'{self.output_dir}/{filename}.png', facecolor='#0a0a0a', dpi=300)
            plt.close()
            return
        
        # Calcular equity curve
        cumulative_returns = np.cumsum([trade.return_absolute for trade in self.system.trades])
        initial_capital = 100000
        equity_curve = initial_capital + cumulative_returns
        
        # 1. Equity Curve
        ax1.plot(equity_curve, color='#00ff41', linewidth=3, alpha=0.9, label='Equity')
        ax1.fill_between(range(len(equity_curve)), initial_capital, equity_curve, 
                        alpha=0.3, color='#00ff41')
        ax1.axhline(initial_capital, color='white', linestyle='--', alpha=0.7, 
                   label=f'Capital Inicial: ${initial_capital:,}')
        
        # Máximo
        max_equity = np.max(equity_curve)
        max_idx = np.argmax(equity_curve)
        ax1.scatter(max_idx, max_equity, color='#ffa502', s=100, zorder=5,
                   label=f'Máximo: ${max_equity:,.0f}')
        
        ax1.set_title("CURVA DE EQUITY", fontweight='bold', color='white', fontsize=14)
        ax1.set_ylabel("Capital USD", color='white', fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        
        ax2.fill_between(range(len(drawdown)), 0, drawdown, 
                        color='#ff4444', alpha=0.7, label='Drawdown')
        ax2.plot(drawdown, color='#ff4444', linewidth=2)
        
        # Máximo drawdown
        max_dd_idx = np.argmin(drawdown)
        max_dd_value = np.min(drawdown)
        ax2.scatter(max_dd_idx, max_dd_value, color='#ffa502', s=100, zorder=5,
                   label=f'Max DD: ${max_dd_value:.0f}')
        
        ax2.set_title("DRAWDOWN", fontweight='bold', color='white', fontsize=14)
        ax2.set_xlabel("Número de Trade", color='white', fontweight='bold')
        ax2.set_ylabel("Drawdown USD", color='white', fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', color='white', y=0.95)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}.png', facecolor='#0a0a0a', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_performance_analysis(self, filename, title):
        """Análisis de performance"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0a0a0a')
        
        if not self.system.trades:
            fig.text(0.5, 0.5, "NO HAY DATOS SUFICIENTES", ha='center', va='center',
                    fontsize=20, color='white', fontweight='bold')
            plt.savefig(f'{self.output_dir}/{filename}.png', facecolor='#0a0a0a', dpi=300)
            plt.close()
            return
        
        # 1. Win/Loss por trade
        results = [1 if trade.return_pct > 0 else -1 for trade in self.system.trades]
        colors = ['#00ff41' if r > 0 else '#ff4444' for r in results]
        
        ax1.bar(range(len(results)), results, color=colors, alpha=0.7, edgecolor='white')
        ax1.axhline(0, color='white', linestyle='-', alpha=0.5)
        ax1.set_title("WIN/LOSS POR TRADE", fontweight='bold', color='white')
        ax1.set_xlabel("Número de Trade", color='white')
        ax1.set_ylabel("Win(+1) / Loss(-1)", color='white')
        ax1.grid(True, alpha=0.3)
        
        # 2. Retornos acumulativos
        cumulative_returns_pct = np.cumsum([trade.return_pct for trade in self.system.trades])
        ax2.plot(cumulative_returns_pct, color='#3742fa', linewidth=3, alpha=0.9)
        ax2.fill_between(range(len(cumulative_returns_pct)), 0, cumulative_returns_pct,
                        alpha=0.3, color='#3742fa')
        ax2.axhline(0, color='white', linestyle='--', alpha=0.5)
        ax2.set_title("RETORNOS ACUMULATIVOS (%)", fontweight='bold', color='white')
        ax2.set_xlabel("Número de Trade", color='white')
        ax2.set_ylabel("Retorno Acumulativo %", color='white')
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe (si hay suficientes datos)
        if len(self.system.trades) >= 10:
            returns = [trade.return_pct for trade in self.system.trades]
            window = min(5, len(returns) // 3)
            rolling_sharpe = []
            
            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                mean_ret = np.mean(window_returns)
                std_ret = np.std(window_returns)
                sharpe = mean_ret / std_ret if std_ret > 0 else 0
                rolling_sharpe.append(sharpe)
            
            ax3.plot(rolling_sharpe, color='#ffa502', linewidth=3)
            ax3.axhline(0, color='white', linestyle='--', alpha=0.5)
            ax3.set_title("ROLLING SHARPE RATIO", fontweight='bold', color='white')
            ax3.set_xlabel("Período", color='white')
            ax3.set_ylabel("Sharpe Ratio", color='white')
        else:
            ax3.text(0.5, 0.5, "POCOS DATOS\nPARA ROLLING\nSHARPE", 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=14, color='white', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Distribución de retornos positivos vs negativos
        positive_returns = [r for r in [t.return_pct for t in self.system.trades] if r > 0]
        negative_returns = [r for r in [t.return_pct for t in self.system.trades] if r <= 0]
        
        ax4.hist([positive_returns, negative_returns], bins=10, 
                color=['#00ff41', '#ff4444'], alpha=0.7, 
                label=[f'Ganancias ({len(positive_returns)})', f'Pérdidas ({len(negative_returns)})'],
                edgecolor='white')
        ax4.set_title("DISTRIBUCIÓN: GANANCIAS vs PÉRDIDAS", fontweight='bold', color='white')
        ax4.set_xlabel("Retorno %", color='white')
        ax4.set_ylabel("Frecuencia", color='white')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', color='white', y=0.95)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}.png', facecolor='#0a0a0a', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_trades_timeline(self, filename, title):
        """Timeline detallado de trades"""
        fig, ax = plt.subplots(figsize=(18, 10), facecolor='#0a0a0a')
        
        if not self.system.trades:
            fig.text(0.5, 0.5, "NO HAY TRADES", ha='center', va='center',
                    fontsize=20, color='white', fontweight='bold')
            plt.savefig(f'{self.output_dir}/{filename}.png', facecolor='#0a0a0a', dpi=300)
            plt.close()
            return
        
        # Crear timeline de trades
        for i, trade in enumerate(self.system.trades):
            y_pos = i
            duration = trade.duration if trade.duration else 1
            
            # Color según resultado
            color = '#00ff41' if trade.return_pct > 0 else '#ff4444'
            
            # Barra horizontal representando la duración
            ax.barh(y_pos, duration, left=trade.entry_time, 
                   color=color, alpha=0.7, height=0.8,
                   edgecolor='white', linewidth=1)
            
            # Etiqueta con información
            label_text = f"{trade.id[:10]} | {trade.return_pct:.1f}% | {duration:.0f}s"
            ax.text(trade.entry_time + duration/2, y_pos, label_text,
                   ha='center', va='center', fontsize=8, color='white',
                   fontweight='bold')
        
        ax.set_title(f"{title}\nCada barra representa un trade (verde=ganancia, rojo=pérdida)", 
                    fontweight='bold', color='white', fontsize=14, pad=20)
        ax.set_xlabel("Tiempo (steps)", color='white', fontweight='bold')
        ax.set_ylabel("Número de Trade", color='white', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Invertir eje Y para que el primer trade esté arriba
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}.png', facecolor='#0a0a0a', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_risk_metrics(self, filename, title):
        """Métricas de riesgo visuales"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0a0a0a')
        
        metrics = self.system.financial_metrics
        
        # 1. Gauge del Sharpe Ratio
        sharpe = metrics.get('sharpe_ratio', 0)
        self._create_gauge(ax1, sharpe, "SHARPE RATIO", -1, 3, 
                          thresholds=[0, 1, 2], colors=['#ff4444', '#ffa502', '#00ff41'])
        
        # 2. Gauge del Win Rate
        win_rate = metrics.get('win_rate', 0)
        self._create_gauge(ax2, win_rate, "WIN RATE (%)", 0, 100,
                          thresholds=[40, 60, 80], colors=['#ff4444', '#ffa502', '#00ff41'])
        
        # 3. Gauge del Profit Factor
        profit_factor = min(metrics.get('profit_factor', 0), 5)  # Limitar para visualización
        self._create_gauge(ax3, profit_factor, "PROFIT FACTOR", 0, 3,
                          thresholds=[1, 1.5, 2], colors=['#ff4444', '#ffa502', '#00ff41'])
        
        # 4. Risk/Reward Metrics
        avg_win = metrics.get('avg_winning_trade', 0)
        avg_loss = abs(metrics.get('avg_losing_trade', 0))
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
        
        risk_metrics_text = f"""
ANÁLISIS DE RIESGO

Risk/Reward Ratio: {risk_reward:.2f}
Promedio Ganancia: {avg_win:.2f}%
Promedio Pérdida: {avg_loss:.2f}%

Max Drawdown: ${metrics.get('max_drawdown', 0):.2f}
Volatilidad: {np.std([t.return_pct for t in self.system.trades]) if self.system.trades else 0:.2f}%

MAPE (ML Error): {metrics.get('mape', 0):.2f}%
Precisión: {'ALTA' if metrics.get('mape', 100) < 5 else 'MEDIA' if metrics.get('mape', 100) < 15 else 'BAJA'}
        """
        
        ax4.text(0.05, 0.95, risk_metrics_text, transform=ax4.transAxes,
                fontsize=12, color='white', va='top', fontfamily='monospace',
                bbox=dict(facecolor='#1a1a1a', alpha=0.9, pad=15))
        ax4.set_title("ANÁLISIS DE RIESGO", fontweight='bold', color='white')
        ax4.axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', color='white', y=0.95)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}.png', facecolor='#0a0a0a', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_gauge(self, ax, value, title, min_val, max_val, thresholds, colors):
        """Crear un gauge circular"""
        # Normalizar valor
        norm_value = (value - min_val) / (max_val - min_val)
        norm_value = max(0, min(1, norm_value))
        
        # Crear semicírculo
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Fondo del gauge
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'white', alpha=0.3, linewidth=10)
        
        # Determinar color según valor
        if len(thresholds) >= 3:
            norm_thresholds = [(t - min_val) / (max_val - min_val) for t in thresholds]
            if norm_value <= norm_thresholds[0]:
                color = colors[0]
            elif norm_value <= norm_thresholds[1]:
                color = colors[1]
            else:
                color = colors[2]
        else:
            color = colors[0]
        
        # Arco del valor actual
        theta_value = theta[:int(norm_value * 100)]
        if len(theta_value) > 0:
            ax.plot(r * np.cos(theta_value), r * np.sin(theta_value), 
                   color, linewidth=10, alpha=0.9)
        
        # Aguja
        needle_angle = np.pi * (1 - norm_value)
        needle_x = 0.8 * np.cos(needle_angle)
        needle_y = 0.8 * np.sin(needle_angle)
        ax.arrow(0, 0, needle_x, needle_y, head_width=0.1, head_length=0.1, 
                fc='white', ec='white', linewidth=3)
        
        # Texto central
        ax.text(0, -0.3, f"{value:.2f}", ha='center', va='center',
               fontsize=16, color='white', fontweight='bold')
        ax.text(0, -0.5, title, ha='center', va='center',
               fontsize=12, color='white')
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.7, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _viz_technical_analysis(self, filename, title):
        """Análisis técnico con indicadores"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), facecolor='#0a0a0a',
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Plotear precio con medias móviles
        prices = self.system.data['price']
        ax1.plot(prices, color='#00d2d3', linewidth=2, alpha=0.9, label='Precio')
        
        # Calcular y plotear medias móviles
        if 'sma_20' in self.system.data.columns:
            ax1.plot(self.system.data['sma_20'], color='#ffa502', linewidth=1.5, 
                    alpha=0.8, label='SMA 20')
        if 'sma_50' in self.system.data.columns:
            ax1.plot(self.system.data['sma_50'], color='#ff6b6b', linewidth=1.5, 
                    alpha=0.8, label='SMA 50')
        
        # Bandas de Bollinger si están disponibles
        if 'bb_upper' in self.system.data.columns:
            ax1.plot(self.system.data['bb_upper'], color='#9b59b6', linestyle='--', 
                    alpha=0.6, label='BB Superior')
            ax1.plot(self.system.data['bb_lower'], color='#9b59b6', linestyle='--', 
                    alpha=0.6, label='BB Inferior')
            ax1.fill_between(range(len(self.system.data)), 
                           self.system.data['bb_upper'], self.system.data['bb_lower'],
                           alpha=0.1, color='#9b59b6')
        
        # Señales de trading
        for trade in self.system.trades:
            if trade.entry_time < len(self.system.data):
                entry_price = self.system.data.iloc[int(trade.entry_time)]['price']
                ax1.scatter(trade.entry_time, entry_price, 
                           color='#00ff41', marker='^', s=60, zorder=5)
            if trade.exit_time and trade.exit_time < len(self.system.data):
                exit_price = self.system.data.iloc[int(trade.exit_time)]['price']
                ax1.scatter(trade.exit_time, exit_price, 
                           color='#ff4444', marker='v', s=60, zorder=5)
        
        ax1.set_title("ANÁLISIS TÉCNICO CON INDICADORES", fontweight='bold', 
                     color='white', fontsize=14)
        ax1.set_ylabel("Precio USD", color='white', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # RSI o volumen en panel inferior
        if 'rsi' in self.system.data.columns:
            ax2.plot(self.system.data['rsi'], color='#e74c3c', linewidth=2)
            ax2.axhline(70, color='#ff4444', linestyle='--', alpha=0.7, label='Sobrecompra (70)')
            ax2.axhline(30, color='#00ff41', linestyle='--', alpha=0.7, label='Sobreventa (30)')
            ax2.fill_between(range(len(self.system.data)), 30, 70, alpha=0.1, color='gray')
            ax2.set_title("RSI (Relative Strength Index)", fontweight='bold', color='white')
            ax2.set_ylabel("RSI", color='white')
            ax2.legend()
        else:
            # Mostrar volumen si no hay RSI
            if 'volume' in self.system.data.columns:
                ax2.bar(range(len(self.system.data)), self.system.data['volume'], 
                       color='#3742fa', alpha=0.6)
                ax2.set_title("VOLUMEN", fontweight='bold', color='white')
                ax2.set_ylabel("Volumen", color='white')
        
        ax2.set_xlabel("Tiempo (steps)", color='white', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', color='white', y=0.95)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}.png', facecolor='#0a0a0a', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Función principal para generar múltiples visualizaciones"""
    print("🎨 GENERADOR DE MÚLTIPLES VISUALIZACIONES")
    print("="*60)
    
    # Crear y ejecutar sistema
    system = AdvancedTradingAnalytics(symbol='SPY', use_binance=False)
    
    if system.load_sp500_data(period='1y'):
        print("✅ Datos cargados")
        
        # Ejecutar backtest
        metrics = system.run_backtest()
        
        if metrics and metrics.get('total_trades', 0) > 0:
            # Generar múltiples visualizaciones
            viz_generator = MultiVisualizationGenerator(system)
            successful = viz_generator.generate_all_visualizations()
            
            print(f"\n🎉 ¡Visualizaciones completadas!")
            print(f"📁 Revisa la carpeta 'visualizations/' para ver {successful} archivos PNG")
        else:
            print("❌ No hay trades suficientes")
    else:
        print("❌ Error cargando datos")

if __name__ == "__main__":
    main() 