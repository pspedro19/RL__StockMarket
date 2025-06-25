#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 GENERADOR DE CHARTS MÚLTIPLES - VERSIÓN SIMPLE
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Agregar al path
sys.path.append('.')

try:
    from src.agents.advanced_trading_analytics import AdvancedTradingAnalytics
    
    def create_individual_charts():
        """Crear charts individuales"""
        print("🎨 Generando múltiples visualizaciones...")
        
        # Crear sistema
        system = AdvancedTradingAnalytics(symbol='SPY', use_binance=False)
        
        if system.load_sp500_data(period='1y'):
            print("✅ Datos cargados")
            
            # Ejecutar backtest
            metrics = system.run_backtest()
            
            if metrics and metrics.get('total_trades', 0) > 0:
                print("✅ Backtest completado")
                
                # Configurar estilo
                plt.style.use('dark_background')
                
                # 1. Precio y Trades
                fig, ax = plt.subplots(figsize=(16, 10), facecolor='#0a0a0a')
                
                # Plotear precio
                ax.plot(system.data['price'], color='#00d2d3', linewidth=2, alpha=0.9, label='Precio SPY')
                
                # Plotear trades
                for i, trade in enumerate(system.trades):
                    if (trade.entry_time < len(system.data) and 
                        trade.exit_time is not None and trade.exit_time < len(system.data)):
                        
                        entry_price = system.data.iloc[int(trade.entry_time)]['price']
                        exit_price = system.data.iloc[int(trade.exit_time)]['price']
                        
                        color = '#00ff41' if trade.return_pct > 0 else '#ff4444'
                        
                        # Línea de conexión
                        ax.plot([trade.entry_time, trade.exit_time], 
                               [entry_price, exit_price], 
                               color=color, linewidth=2, alpha=0.7)
                        
                        # Marcadores
                        ax.scatter(trade.entry_time, entry_price, 
                                  color='#00ff41', marker='^', s=60, zorder=5)
                        ax.scatter(trade.exit_time, exit_price, 
                                  color='#ff4444', marker='v', s=60, zorder=5)
                
                ax.set_title(f"PRECIO Y TRADES - SPY\nTotal: {len(system.trades)} trades | Win Rate: {metrics['win_rate']:.1f}%", 
                            fontweight='bold', color='white', fontsize=16)
                ax.set_xlabel("Tiempo (steps)", fontweight='bold', color='white')
                ax.set_ylabel("Precio USD", fontweight='bold', color='white')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('chart_01_precio_y_trades.png', facecolor='#0a0a0a', dpi=300, bbox_inches='tight')
                plt.close()
                print("📊 Chart 1: Precio y Trades guardado")
                
                # 2. Métricas Financieras
                fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0a0a0a')
                
                metrics_text = f"""
MÉTRICAS FINANCIERAS PRINCIPALES

📊 ESTADÍSTICAS:
• Total de Trades: {metrics['total_trades']}
• Trades Ganadores: {metrics['winning_trades']}
• Trades Perdedores: {metrics['losing_trades']}
• Win Rate: {metrics['win_rate']:.2f}%

💰 RENDIMIENTO:
• Retorno Total: ${metrics['total_return_abs']:.2f}
• Retorno Porcentual: {metrics['total_return_pct']:.3f}%
• Retorno Promedio: {metrics['avg_return']:.3f}%

📈 CALIDAD:
• Sharpe Ratio: {metrics['sharpe_ratio']:.4f}
• Max Drawdown: ${metrics['max_drawdown']:.2f}
• Profit Factor: {metrics['profit_factor']:.3f}

🔍 MACHINE LEARNING:
• MAPE (Error): {metrics.get('mape', 0):.2f}%
• Duración Promedio: {metrics['avg_duration']:.1f} steps

🎯 CONTROL PID: ACTIVO ✅
                """
                
                ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                        fontsize=14, color='white', va='top', fontfamily='monospace',
                        bbox=dict(facecolor='#1a1a3a', alpha=0.9, pad=20))
                ax.set_title("PANEL DE MÉTRICAS FINANCIERAS", fontweight='bold', color='white', fontsize=16)
                ax.axis('off')
                
                plt.tight_layout()
                plt.savefig('chart_02_metricas_financieras.png', facecolor='#0a0a0a', dpi=300, bbox_inches='tight')
                plt.close()
                print("📊 Chart 2: Métricas Financieras guardado")
                
                # 3. Distribuciones
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0a0a0a')
                
                returns = [trade.return_pct for trade in system.trades]
                durations = [trade.duration for trade in system.trades if trade.duration]
                
                # Distribución de retornos
                ax1.hist(returns, bins=15, color='#3742fa', alpha=0.7, edgecolor='white')
                ax1.axvline(np.mean(returns), color='#ffa502', linestyle='--', linewidth=2, 
                           label=f'Media: {np.mean(returns):.2f}%')
                ax1.axvline(0, color='white', linestyle='-', alpha=0.5)
                ax1.set_title("DISTRIBUCIÓN DE RETORNOS", fontweight='bold', color='white')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Curva de equity
                cumulative_returns = np.cumsum([trade.return_absolute for trade in system.trades])
                equity_curve = 100000 + cumulative_returns
                ax2.plot(equity_curve, color='#00ff41', linewidth=3)
                ax2.fill_between(range(len(equity_curve)), 100000, equity_curve, alpha=0.3, color='#00ff41')
                ax2.set_title("CURVA DE EQUITY", fontweight='bold', color='white')
                ax2.grid(True, alpha=0.3)
                
                # Win/Loss
                results = [1 if trade.return_pct > 0 else -1 for trade in system.trades]
                colors = ['#00ff41' if r > 0 else '#ff4444' for r in results]
                ax3.bar(range(len(results)), results, color=colors, alpha=0.7)
                ax3.set_title("WIN/LOSS POR TRADE", fontweight='bold', color='white')
                ax3.grid(True, alpha=0.3)
                
                # Duraciones
                if durations:
                    ax4.hist(durations, bins=10, color='#ff6b6b', alpha=0.7, edgecolor='white')
                    ax4.set_title("DISTRIBUCIÓN DE DURACIONES", fontweight='bold', color='white')
                    ax4.grid(True, alpha=0.3)
                
                plt.suptitle("ANÁLISIS ESTADÍSTICO COMPLETO", fontsize=16, fontweight='bold', color='white')
                plt.tight_layout()
                plt.savefig('chart_03_analisis_estadistico.png', facecolor='#0a0a0a', dpi=300, bbox_inches='tight')
                plt.close()
                print("📊 Chart 3: Análisis Estadístico guardado")
                
                # 4. Detalles de Trades
                if len(system.trades) >= 5:
                    fig, ax = plt.subplots(figsize=(18, 10), facecolor='#0a0a0a')
                    
                    # Timeline de trades
                    for i, trade in enumerate(system.trades[:20]):  # Máximo 20
                        y_pos = i
                        duration = trade.duration if trade.duration else 1
                        color = '#00ff41' if trade.return_pct > 0 else '#ff4444'
                        
                        ax.barh(y_pos, duration, left=trade.entry_time, 
                               color=color, alpha=0.7, height=0.8, edgecolor='white')
                        
                        # Etiqueta
                        label_text = f"{trade.id[:10]} | {trade.return_pct:.1f}%"
                        ax.text(trade.entry_time + duration/2, y_pos, label_text,
                               ha='center', va='center', fontsize=8, color='white', fontweight='bold')
                    
                    ax.set_title("TIMELINE DE TRADES INDIVIDUALES", fontweight='bold', color='white', fontsize=16)
                    ax.set_xlabel("Tiempo (steps)", color='white')
                    ax.set_ylabel("Número de Trade", color='white')
                    ax.invert_yaxis()
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    plt.tight_layout()
                    plt.savefig('chart_04_timeline_trades.png', facecolor='#0a0a0a', dpi=300, bbox_inches='tight')
                    plt.close()
                    print("📊 Chart 4: Timeline de Trades guardado")
                
                # Generar dashboard original también
                system.create_comprehensive_dashboard()
                print("📊 Dashboard completo regenerado")
                
                print(f"\n🎉 ¡Visualizaciones completadas!")
                print(f"📁 Archivos generados:")
                print(f"   • chart_01_precio_y_trades.png")
                print(f"   • chart_02_metricas_financieras.png") 
                print(f"   • chart_03_analisis_estadistico.png")
                print(f"   • chart_04_timeline_trades.png")
                print(f"   • dashboard_avanzado.png (completo)")
                
            else:
                print("❌ No hay trades suficientes")
        else:
            print("❌ Error cargando datos")

    if __name__ == "__main__":
        create_individual_charts()
        
except ImportError as e:
    print(f"❌ Error: {e}")
    print("💡 Asegúrate de estar en el directorio correcto") 