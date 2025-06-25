#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 GENERADOR DE VISUALIZACIONES LIMPIAS + EXPORTACIÓN CSV
Versión actualizada para nueva estructura del proyecto
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Agregar rutas del proyecto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Definir rutas de salida
VISUALIZATIONS_DIR = os.path.join(project_root, 'data', 'results', 'trading_analysis', 'visualizations')
CSV_EXPORTS_DIR = os.path.join(project_root, 'data', 'results', 'trading_analysis', 'csv_exports')

def ensure_directories():
    """Asegurar que existen los directorios de salida"""
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    os.makedirs(CSV_EXPORTS_DIR, exist_ok=True)

def generate_comprehensive_analysis():
    """Generar análisis completo con CSV y múltiples imágenes"""
    try:
        from src.agents.advanced_trading_analytics import AdvancedTradingAnalytics
        
        print("🚀 INICIANDO ANÁLISIS COMPLETO...")
        print("="*60)
        
        # Asegurar directorios
        ensure_directories()
        
        # Crear sistema
        system = AdvancedTradingAnalytics(symbol='SPY', use_binance=False)
        
        if not system.load_sp500_data(period='1y'):
            print("❌ Error cargando datos")
            return
            
        print("✅ Datos cargados")
        
        # Ejecutar backtest
        metrics = system.run_backtest()
        
        if not metrics or metrics.get('total_trades', 0) == 0:
            print("❌ No hay trades suficientes")
            return
            
        print("✅ Backtest completado")
        
        # 1. EXPORTAR CSV CON DATOS DE TRADES
        export_trades_to_csv(system.trades, metrics)
        
        # 2. GENERAR VISUALIZACIONES SEPARADAS
        generate_individual_charts(system, metrics)
        
        print(f"\n🎉 ¡ANÁLISIS COMPLETADO!")
        print(f"📁 Visualizaciones guardadas en: {VISUALIZATIONS_DIR}")
        print(f"📊 CSV exportados a: {CSV_EXPORTS_DIR}")
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")

def export_trades_to_csv(trades, metrics):
    """Exportar todos los datos de trades a CSV"""
    print("\n📊 EXPORTANDO DATOS A CSV...")
    
    # Crear DataFrame detallado con todos los trades
    trades_data = []
    
    for i, trade in enumerate(trades, 1):
        trades_data.append({
            'Numero_Trade': i,
            'ID_Unico': trade.id,
            'Tipo': trade.trade_type,
            'Tiempo_Entrada': trade.entry_time,
            'Tiempo_Salida': trade.exit_time if trade.exit_time else 'N/A',
            'Precio_Entrada': round(trade.entry_price, 2),
            'Precio_Salida': round(trade.exit_price, 2) if trade.exit_price else 'N/A',
            'Retorno_Porcentaje': round(trade.return_pct, 4),
            'Retorno_Absoluto_USD': round(trade.return_absolute, 2),
            'Duracion_Steps': round(trade.duration, 1) if trade.duration else 'N/A',
            'Estado': trade.status,
            'Ganancia_Perdida': 'GANANCIA' if trade.return_pct > 0 else 'PERDIDA',
            'Timestamp_Generacion': trade.id.split('_')[1] if '_' in trade.id else 'N/A'
        })
    
    trades_df = pd.DataFrame(trades_data)
    
    # Guardar CSV principal
    csv_filename = os.path.join(CSV_EXPORTS_DIR, 'trades_detallados.csv')
    trades_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"✅ Trades exportados a: {csv_filename}")
    
    # Crear resumen de métricas en CSV separado
    metrics_data = [
        ['Métrica', 'Valor', 'Unidad'],
        ['Total_Trades', metrics['total_trades'], 'cantidad'],
        ['Trades_Ganadores', metrics['winning_trades'], 'cantidad'],
        ['Trades_Perdedores', metrics['losing_trades'], 'cantidad'],
        ['Win_Rate', round(metrics['win_rate'], 2), 'porcentaje'],
        ['Retorno_Total_USD', round(metrics['total_return_abs'], 2), 'USD'],
        ['Retorno_Porcentual', round(metrics['total_return_pct'], 4), 'porcentaje'],
        ['Retorno_Promedio', round(metrics['avg_return'], 4), 'porcentaje'],
        ['Promedio_Ganancia', round(metrics['avg_winning_trade'], 2), 'porcentaje'],
        ['Promedio_Perdida', round(metrics['avg_losing_trade'], 2), 'porcentaje'],
        ['Sharpe_Ratio', round(metrics['sharpe_ratio'], 4), 'ratio'],
        ['Max_Drawdown_USD', round(metrics['max_drawdown'], 2), 'USD'],
        ['Profit_Factor', round(metrics['profit_factor'], 3), 'ratio'],
        ['Duracion_Promedio', round(metrics['avg_duration'], 1), 'steps'],
        ['MAPE_Error_ML', round(metrics.get('mape', 0), 2), 'porcentaje']
    ]
    
    metrics_df = pd.DataFrame(metrics_data[1:], columns=metrics_data[0])
    metrics_csv = os.path.join(CSV_EXPORTS_DIR, 'metricas_resumen.csv')
    metrics_df.to_csv(metrics_csv, index=False, encoding='utf-8-sig')
    print(f"✅ Métricas exportadas a: {metrics_csv}")
    
    # Estadísticas adicionales
    returns = [trade.return_pct for trade in trades]
    estadisticas_adicionales = pd.DataFrame([
        ['Mejor_Trade_Porcentaje', max(returns), '%'],
        ['Peor_Trade_Porcentaje', min(returns), '%'],
        ['Mediana_Retornos', np.median(returns), '%'],
        ['Volatilidad_Retornos', np.std(returns), '%'],
        ['Trades_Positivos_Consecutivos_Max', calculate_max_consecutive_wins(trades), 'cantidad'],
        ['Trades_Negativos_Consecutivos_Max', calculate_max_consecutive_losses(trades), 'cantidad']
    ], columns=['Estadistica', 'Valor', 'Unidad'])
    
    stats_csv = os.path.join(CSV_EXPORTS_DIR, 'estadisticas_adicionales.csv')
    estadisticas_adicionales.to_csv(stats_csv, index=False, encoding='utf-8-sig')
    print(f"✅ Estadísticas adicionales: {stats_csv}")

def calculate_max_consecutive_wins(trades):
    """Calcular máximo de trades ganadores consecutivos"""
    max_consecutive = 0
    current_consecutive = 0
    
    for trade in trades:
        if trade.return_pct > 0:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive

def calculate_max_consecutive_losses(trades):
    """Calcular máximo de trades perdedores consecutivos"""
    max_consecutive = 0
    current_consecutive = 0
    
    for trade in trades:
        if trade.return_pct <= 0:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive

def generate_individual_charts(system, metrics):
    """Generar múltiples charts separados y claros"""
    print("\n🎨 GENERANDO VISUALIZACIONES SEPARADAS...")
    
    # Configurar estilo
    plt.style.use('dark_background')
    
    # 1. GRÁFICO PRINCIPAL: PRECIO Y TRADES
    create_price_trades_chart(system, metrics)
    
    # 2. PANEL DE MÉTRICAS FINANCIERAS
    create_metrics_panel(metrics)
    
    # 3. DISTRIBUCIONES Y ESTADÍSTICAS
    create_distributions_chart(system)
    
    # 4. CURVA DE EQUITY Y DRAWDOWN
    create_equity_drawdown_chart(system)
    
    # 5. ANÁLISIS DE PERFORMANCE
    create_performance_analysis(system)
    
    # 6. TABLA DE TRADES (más legible)
    create_trades_table(system.trades[:15])  # Solo primeros 15 para legibilidad

def create_price_trades_chart(system, metrics):
    """Gráfico principal de precio y trades"""
    fig, ax = plt.subplots(figsize=(20, 12), facecolor='#0a0a0a')
    
    # Plotear precio
    ax.plot(system.data['price'], color='#00d2d3', linewidth=3, alpha=0.9, label='Precio SPY')
    
    # Plotear trades con mejor visibilidad
    for i, trade in enumerate(system.trades):
        if (trade.entry_time < len(system.data) and 
            trade.exit_time is not None and trade.exit_time < len(system.data)):
            
            entry_price = system.data.iloc[int(trade.entry_time)]['price']
            exit_price = system.data.iloc[int(trade.exit_time)]['price']
            
            # Color y estilo según resultado
            color = '#00ff41' if trade.return_pct > 0 else '#ff4444'
            alpha = 0.8 if trade.return_pct > 0 else 0.7
            linewidth = 4 if abs(trade.return_pct) > 1 else 3
            
            # Línea de conexión
            ax.plot([trade.entry_time, trade.exit_time], 
                   [entry_price, exit_price], 
                   color=color, linewidth=linewidth, alpha=alpha)
            
            # Marcadores más grandes
            ax.scatter(trade.entry_time, entry_price, 
                      color='#00ff41', marker='^', s=120, zorder=5, 
                      edgecolors='white', linewidths=2)
            ax.scatter(trade.exit_time, exit_price, 
                      color='#ff4444', marker='v', s=120, zorder=5,
                      edgecolors='white', linewidths=2)
            
            # Etiquetas cada 5 trades para no saturar
            if i % 5 == 0:
                mid_time = (trade.entry_time + trade.exit_time) / 2
                mid_price = (entry_price + exit_price) / 2
                
                ax.annotate(f'{trade.id[:8]}\n{trade.return_pct:.1f}%', 
                          xy=(mid_time, mid_price),
                          xytext=(15, 15), textcoords='offset points',
                          fontsize=11, color='white', fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.4),
                          ha='center')
    
    ax.set_title(f"PRECIO SPY Y TRADES EJECUTADOS\nTotal: {len(system.trades)} trades | Win Rate: {metrics['win_rate']:.1f}% | Retorno: ${metrics['total_return_abs']:.2f}", 
                fontweight='bold', color='white', fontsize=18, pad=30)
    ax.set_xlabel("Tiempo (steps)", fontweight='bold', color='white', fontsize=14)
    ax.set_ylabel("Precio USD", fontweight='bold', color='white', fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATIONS_DIR, '01_precio_y_trades_detallado.png')
    plt.savefig(save_path, facecolor='#0a0a0a', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 01_precio_y_trades_detallado.png guardado en {VISUALIZATIONS_DIR}")

def create_metrics_panel(metrics):
    """Panel de métricas financieras separado"""
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='#0a0a0a')
    
    # Crear texto formateado con mejor espaciado
    metrics_text = f"""
MÉTRICAS FINANCIERAS AVANZADAS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 ESTADÍSTICAS BÁSICAS:
    • Total de Trades:           {metrics['total_trades']:>8}
    • Trades Ganadores:          {metrics['winning_trades']:>8}
    • Trades Perdedores:         {metrics['losing_trades']:>8}
    • Win Rate:                  {metrics['win_rate']:>8.2f}%

💰 ANÁLISIS DE RENDIMIENTO:
    • Retorno Total:             ${metrics['total_return_abs']:>8.2f}
    • Retorno Porcentual:        {metrics['total_return_pct']:>8.3f}%
    • Retorno Promedio:          {metrics['avg_return']:>8.3f}%
    • Capital Inicial:           $100,000.00
    • Capital Final:             ${100000 + metrics['total_return_abs']:>8.2f}

📈 MÉTRICAS DE CALIDAD:
    • Sharpe Ratio:              {metrics['sharpe_ratio']:>8.4f}
    • Max Drawdown:              ${metrics['max_drawdown']:>8.2f}
    • Profit Factor:             {metrics['profit_factor']:>8.3f}
    • Promedio Ganancia:         {metrics['avg_winning_trade']:>8.2f}%
    • Promedio Pérdida:          {metrics['avg_losing_trade']:>8.2f}%

🔍 MACHINE LEARNING & TIMING:
    • MAPE (Error Predicción):   {metrics.get('mape', 0):>8.2f}%
    • Duración Promedio:         {metrics['avg_duration']:>8.1f} steps
    • Control PID:               {'ACTIVO ✅':>15}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=16, color='white', va='top', fontfamily='monospace',
            bbox=dict(facecolor='#1a1a3a', alpha=0.9, pad=30))
    
    ax.set_title("PANEL DE MÉTRICAS FINANCIERAS AVANZADAS", 
                fontweight='bold', color='white', fontsize=20, pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATIONS_DIR, '02_metricas_financieras_panel.png')
    plt.savefig(save_path, facecolor='#0a0a0a', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 02_metricas_financieras_panel.png guardado")

def create_distributions_chart(system):
    """Gráfico de distribuciones"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14), facecolor='#0a0a0a')
    
    returns = [trade.return_pct for trade in system.trades]
    durations = [trade.duration for trade in system.trades if trade.duration]
    
    # 1. Distribución de retornos
    ax1.hist(returns, bins=20, color='#3742fa', alpha=0.8, edgecolor='white', linewidth=1.5)
    ax1.axvline(np.mean(returns), color='#ffa502', linestyle='--', linewidth=3, 
               label=f'Media: {np.mean(returns):.2f}%')
    ax1.axvline(0, color='white', linestyle='-', alpha=0.7, linewidth=2, label='Break-even')
    ax1.set_title("DISTRIBUCIÓN DE RETORNOS", fontweight='bold', color='white', fontsize=14)
    ax1.set_xlabel("Retorno %", fontweight='bold', color='white')
    ax1.set_ylabel("Frecuencia", fontweight='bold', color='white')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot de retornos
    bp = ax2.boxplot([returns], patch_artist=True, labels=['Retornos %'])
    bp['boxes'][0].set_facecolor('#3742fa')
    bp['boxes'][0].set_alpha(0.8)
    ax2.set_title("ESTADÍSTICAS DE RETORNOS", fontweight='bold', color='white', fontsize=14)
    ax2.set_ylabel("Retorno %", fontweight='bold', color='white')
    ax2.grid(True, alpha=0.3)
    
    # 3. Win/Loss por trade
    results = [1 if trade.return_pct > 0 else -1 for trade in system.trades]
    colors = ['#00ff41' if r > 0 else '#ff4444' for r in results]
    ax3.bar(range(len(results)), results, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax3.axhline(0, color='white', linestyle='-', alpha=0.7, linewidth=2)
    ax3.set_title("WIN/LOSS POR TRADE", fontweight='bold', color='white', fontsize=14)
    ax3.set_xlabel("Número de Trade", fontweight='bold', color='white')
    ax3.set_ylabel("Win(+1) / Loss(-1)", fontweight='bold', color='white')
    ax3.grid(True, alpha=0.3)
    
    # 4. Duraciones
    if durations:
        ax4.hist(durations, bins=15, color='#ff6b6b', alpha=0.8, edgecolor='white', linewidth=1.5)
        ax4.axvline(np.mean(durations), color='#ffa502', linestyle='--', linewidth=3,
                   label=f'Media: {np.mean(durations):.1f}')
        ax4.set_title("DISTRIBUCIÓN DE DURACIONES", fontweight='bold', color='white', fontsize=14)
        ax4.set_xlabel("Duración (steps)", fontweight='bold', color='white')
        ax4.set_ylabel("Frecuencia", fontweight='bold', color='white')
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle("ANÁLISIS ESTADÍSTICO COMPLETO", fontsize=18, fontweight='bold', color='white', y=0.95)
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATIONS_DIR, '03_distribuciones_estadisticas.png')
    plt.savefig(save_path, facecolor='#0a0a0a', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 03_distribuciones_estadisticas.png guardado")

def create_equity_drawdown_chart(system):
    """Curva de equity y drawdown"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), facecolor='#0a0a0a')
    
    # Calcular datos
    cumulative_returns = np.cumsum([trade.return_absolute for trade in system.trades])
    initial_capital = 100000
    equity_curve = initial_capital + cumulative_returns
    
    # 1. Equity Curve
    ax1.plot(equity_curve, color='#00ff41', linewidth=4, alpha=0.9, label='Equity')
    ax1.fill_between(range(len(equity_curve)), initial_capital, equity_curve, 
                    alpha=0.3, color='#00ff41')
    ax1.axhline(initial_capital, color='white', linestyle='--', alpha=0.8, linewidth=2,
               label=f'Capital Inicial: ${initial_capital:,}')
    
    # Máximo y final
    max_equity = np.max(equity_curve)
    max_idx = np.argmax(equity_curve)
    final_equity = equity_curve[-1]
    
    ax1.scatter(max_idx, max_equity, color='#ffa502', s=150, zorder=5,
               label=f'Máximo: ${max_equity:,.0f}')
    ax1.scatter(len(equity_curve)-1, final_equity, color='#e74c3c', s=150, zorder=5,
               label=f'Final: ${final_equity:,.0f}')
    
    ax1.set_title("CURVA DE EQUITY - EVOLUCIÓN DEL CAPITAL", fontweight='bold', color='white', fontsize=16)
    ax1.set_ylabel("Capital USD", color='white', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    
    ax2.fill_between(range(len(drawdown)), 0, drawdown, 
                    color='#ff4444', alpha=0.7, label='Drawdown')
    ax2.plot(drawdown, color='#ff4444', linewidth=3)
    
    # Máximo drawdown
    max_dd_idx = np.argmin(drawdown)
    max_dd_value = np.min(drawdown)
    ax2.scatter(max_dd_idx, max_dd_value, color='#ffa502', s=150, zorder=5,
               label=f'Max Drawdown: ${max_dd_value:.0f}')
    
    ax2.set_title("DRAWDOWN - PÉRDIDAS DESDE MÁXIMO", fontweight='bold', color='white', fontsize=16)
    ax2.set_xlabel("Número de Trade", color='white', fontweight='bold', fontsize=14)
    ax2.set_ylabel("Drawdown USD", color='white', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATIONS_DIR, '04_equity_y_drawdown.png')
    plt.savefig(save_path, facecolor='#0a0a0a', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 04_equity_y_drawdown.png guardado")

def create_performance_analysis(system):
    """Análisis de performance"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14), facecolor='#0a0a0a')
    
    returns = [trade.return_pct for trade in system.trades]
    
    # 1. Retornos acumulativos
    cumulative_returns_pct = np.cumsum(returns)
    ax1.plot(cumulative_returns_pct, color='#3742fa', linewidth=4, alpha=0.9)
    ax1.fill_between(range(len(cumulative_returns_pct)), 0, cumulative_returns_pct,
                    alpha=0.3, color='#3742fa')
    ax1.axhline(0, color='white', linestyle='--', alpha=0.7, linewidth=2)
    ax1.set_title("RETORNOS ACUMULATIVOS (%)", fontweight='bold', color='white', fontsize=14)
    ax1.set_ylabel("Retorno Acumulativo %", fontweight='bold', color='white')
    ax1.grid(True, alpha=0.3)
    
    # 2. Ganancias vs Pérdidas
    positive_returns = [r for r in returns if r > 0]
    negative_returns = [r for r in returns if r <= 0]
    
    ax2.hist([positive_returns, negative_returns], bins=12, 
            color=['#00ff41', '#ff4444'], alpha=0.8, 
            label=[f'Ganancias ({len(positive_returns)})', f'Pérdidas ({len(negative_returns)})'],
            edgecolor='white', linewidth=1.5)
    ax2.set_title("DISTRIBUCIÓN: GANANCIAS vs PÉRDIDAS", fontweight='bold', color='white', fontsize=14)
    ax2.set_xlabel("Retorno %", fontweight='bold', color='white')
    ax2.set_ylabel("Frecuencia", fontweight='bold', color='white')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Rolling Sharpe (si hay suficientes datos)
    if len(returns) >= 10:
        window = min(5, len(returns) // 3)
        rolling_sharpe = []
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            mean_ret = np.mean(window_returns)
            std_ret = np.std(window_returns)
            sharpe = mean_ret / std_ret if std_ret > 0 else 0
            rolling_sharpe.append(sharpe)
        
        ax3.plot(rolling_sharpe, color='#ffa502', linewidth=4, alpha=0.9)
        ax3.axhline(0, color='white', linestyle='--', alpha=0.7, linewidth=2)
        ax3.set_title("ROLLING SHARPE RATIO", fontweight='bold', color='white', fontsize=14)
        ax3.set_ylabel("Sharpe Ratio", fontweight='bold', color='white')
        ax3.grid(True, alpha=0.3)
    
    # 4. Volatilidad Rolling
    if len(returns) >= 10:
        window = min(5, len(returns) // 3)
        rolling_vol = []
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            vol = np.std(window_returns)
            rolling_vol.append(vol)
        
        ax4.plot(rolling_vol, color='#e74c3c', linewidth=4, alpha=0.9)
        ax4.set_title("VOLATILIDAD ROLLING", fontweight='bold', color='white', fontsize=14)
        ax4.set_ylabel("Volatilidad %", fontweight='bold', color='white')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle("ANÁLISIS AVANZADO DE PERFORMANCE", fontsize=18, fontweight='bold', color='white', y=0.95)
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATIONS_DIR, '05_analisis_performance.png')
    plt.savefig(save_path, facecolor='#0a0a0a', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 05_analisis_performance.png guardado")

def create_trades_table(trades):
    """Crear tabla visual de trades más legible"""
    fig, ax = plt.subplots(figsize=(20, 12), facecolor='#0a0a0a')
    
    # Preparar datos para la tabla
    table_data = []
    headers = ['#', 'ID Trade', 'Tipo', 'Entrada $', 'Salida $', 'Retorno %', 'Retorno $', 'Duración', 'Estado']
    
    for i, trade in enumerate(trades, 1):
        table_data.append([
            str(i),
            trade.id[:12] + '...',
            trade.trade_type,
            f"${trade.entry_price:.2f}",
            f"${trade.exit_price:.2f}" if trade.exit_price else "N/A",
            f"{trade.return_pct:.2f}%",
            f"${trade.return_absolute:.2f}",
            f"{trade.duration:.1f}" if trade.duration else "N/A",
            trade.status
        ])
    
    # Crear tabla
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Configurar estilo de la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Colorear headers
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorear filas según ganancia/pérdida
    for i, trade in enumerate(trades, 1):
        color = '#2d5a2d' if trade.return_pct > 0 else '#5a2d2d'  # Verde oscuro o rojo oscuro
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_text_props(color='white')
    
    ax.set_title(f"TABLA DETALLADA DE TRADES (Primeros {len(trades)})", 
                fontweight='bold', color='white', fontsize=18, pad=30)
    ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATIONS_DIR, '06_tabla_trades_detallada.png')
    plt.savefig(save_path, facecolor='#0a0a0a', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 06_tabla_trades_detallada.png guardado")

if __name__ == "__main__":
    generate_comprehensive_analysis() 