# 🚀 RL Stock Market Trading System

Sistema avanzado de trading con Inteligencia Artificial para SP500 y Bitcoin.

## ✨ Características Principales

- 🎯 **IDs únicos para trades** con formato `T00001_20241201_123456_ABC12345`
- 📊 **Métricas financieras avanzadas**: Sharpe ratio, Drawdown, Profit Factor, MAPE
- 🤖 **Control PID** para optimización automática de señales
- 🔬 **Machine Learning** con evaluación MAPE de predicciones
- 📈 **Datos reales SP500** (Yahoo Finance) y preparación para Bitcoin (Binance API)
- 📋 **Exportación completa CSV** para análisis externos
- 🎨 **Visualizaciones separadas** y dashboard profesional
- 📓 **Jupyter Notebook** interactivo

## 🗂️ Estructura del Proyecto

```
RL__StockMarket/
├── 🚀 run_trading_analysis.py           # SCRIPT PRINCIPAL
├── 📁 src/                              # Código fuente
│   ├── agents/                         # Sistemas de trading IA
│   ├── collectors/                     # Recolectores de datos
│   ├── trading/                        # Lógica de trading
│   └── utils/                          # Utilidades
├── 📁 scripts/                          # Scripts auxiliares
│   ├── generate_clean_visualizations.py
│   └── generate_multiple_charts.py
├── 📁 notebooks/                        # Análisis interactivo
│   └── advanced_trading_notebook.ipynb # Notebook completo
├── 📁 data/                            # Datos y resultados
│   ├── models/                         # Modelos entrenados
│   ├── results/trading_analysis/       # Resultados organizados
│   │   ├── 📊 visualizations/          # PNG (15+ archivos)
│   │   └── 📋 csv_exports/             # CSV (3 archivos)
│   ├── processed/                      # Datos procesados
│   └── raw/                           # Datos en bruto
├── 📁 configs/                          # Configuraciones
├── 📁 docs/                            # Documentación
├── 📁 testing/                          # Pruebas y debug
├── 📁 utils/                           # Herramientas de instalación
├── 📁 deployment/                       # Docker y despliegue
├── 📁 monitoring/                       # Grafana y Prometheus
├── 📁 legacy/                          # Versiones anteriores
└── 📁 logs/                            # Archivos de log
```

## 🚀 Inicio Rápido

### 1. Ejecución Simple (Recomendado)

```bash
# Desde la raíz del proyecto
python run_trading_analysis.py
```

Este script te guiará con un menú interactivo:
- ✅ Análisis completo (recomendado)
- 📊 Solo visualizaciones  
- 📋 Solo exportar CSV
- 📓 Abrir notebook Jupyter
- 📁 Mostrar ubicaciones de archivos

### 2. Ejecución Manual

```bash
# Análisis completo con visualizaciones y CSV
python scripts/generate_clean_visualizations.py

# Notebook interactivo
jupyter notebook notebooks/advanced_trading_notebook.ipynb
```

## 📊 Archivos Generados

### 🎨 Visualizaciones (PNG)
**Ubicación:** `data/results/trading_analysis/visualizations/`

1. **01_precio_y_trades_detallado.png** - Precio SPY con trades marcados
2. **02_metricas_financieras_panel.png** - Panel de KPIs financieros
3. **03_distribuciones_estadisticas.png** - Distribuciones de retornos
4. **04_equity_y_drawdown.png** - Curva de equity y drawdown
5. **05_analisis_performance.png** - Análisis avanzado de performance
6. **06_tabla_trades_detallada.png** - Tabla visual de trades
7. **dashboard_avanzado.png** - Dashboard completo (18 paneles)

### 📋 Datos CSV
**Ubicación:** `data/results/trading_analysis/csv_exports/`

1. **trades_detallados.csv** - Datos completos de cada trade
2. **metricas_resumen.csv** - KPIs del sistema
3. **estadisticas_adicionales.csv** - Estadísticas adicionales

## 📈 Métricas Implementadas

### 💰 Financieras
- **Retorno Total** (absoluto y porcentual)
- **Win Rate** (porcentaje de trades ganadores)
- **Profit Factor** (ganancia total / pérdida total)
- **Sharpe Ratio** (retorno ajustado por riesgo)
- **Max Drawdown** (máxima pérdida desde pico)

### 🤖 Machine Learning
- **MAPE** (Mean Absolute Percentage Error)
- **Learning Curves** para modelos RL
- **Control PID** automático

### 📊 Operacionales
- **Duración promedio** de trades
- **Trades consecutivos** (rachas ganadoras/perdedoras)
- **Distribución de retornos**
- **Análisis de volatilidad**

## 🔧 Configuración

### SP500 (Activo por defecto)
No requiere configuración. Usa datos de Yahoo Finance.

### Bitcoin (Binance API)
1. Copia `configs/binance.env.example` a `configs/binance.env`
2. Agrega tus credenciales de Binance:
```env
BINANCE_API_KEY=tu_api_key
BINANCE_SECRET_KEY=tu_secret_key
BINANCE_TESTNET=True  # Para testnet
```

## 🛠️ Instalación de Dependencias

### 🚀 Instalación Rápida (Recomendado)
```bash
# Dependencias esenciales (más estable)
pip install -r requirements-minimal.txt

# O dependencias completas (más funciones)
pip install -r requirements.txt
```

### 🔧 Instalación con Scripts
```bash
# Básica
python utils/install_dependencies.py

# Avanzada (incluye ML completo)
python utils/install_advanced_dependencies.py
```

### 📦 Dependencias por Categoría

#### Esenciales (Siempre requeridas)
- `pandas, numpy` - Análisis de datos
- `matplotlib, seaborn` - Visualizaciones
- `yfinance` - Datos SP500
- `scikit-learn` - Métricas ML (Sharpe, MAPE)

#### Trading APIs
- `ccxt` - Binance API para Bitcoin
- `MetaTrader5` - Datos MT5 (opcional, solo Windows)

#### Machine Learning Completo
- `stable-baselines3, torch` - Algoritmos RL
- `tensorflow, keras` - Redes neuronales
- `optuna` - Optimización de hiperparámetros

#### Análisis Avanzado
- `ta, pandas-ta` - Indicadores técnicos
- `plotly, dash` - Gráficos interactivos
- `jupyter` - Notebooks interactivos

## 🎯 Casos de Uso

### 📊 Análisis de Backtesting
```python
from src.agents.advanced_trading_analytics import AdvancedTradingAnalytics

# Crear sistema
system = AdvancedTradingAnalytics(symbol='SPY')
system.load_sp500_data(period='1y')

# Ejecutar backtest
metrics = system.run_backtest()
print(f"Win Rate: {metrics['win_rate']:.1f}%")
print(f"Retorno Total: ${metrics['total_return_abs']:.2f}")
```

### 📈 Trading en Vivo (Preparado)
```python
# Configurar para Binance
system = AdvancedTradingAnalytics(symbol='BTCUSDT', use_binance=True)
# Requiere configuración de API keys
```

## 📋 Estructura de Trade ID

Cada trade tiene un ID único con formato:
```
T00001_20241201_123456_ABC12345
│  │      │       │       │
│  │      │       │       └── Hash único (8 chars)
│  │      │       └────────── Hora (HHMMSS)
│  │      └────────────────── Fecha (YYYYMMDD)
│  └───────────────────────── Número secuencial
└──────────────────────────── Prefijo 'T'
```

## 🔍 Monitoreo en Tiempo Real

El sistema incluye:
- ✅ **Control PID** para ajuste automático de señales
- ✅ **MAPE tracking** para calidad de predicciones ML
- ✅ **Real-time metrics** durante backtest
- ✅ **Learning curves** para modelos RL

## 🤝 Contribución

1. Fork del repositorio
2. Crear branch: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -am 'Agregar nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🆘 Soporte

- 📧 Crear un Issue en GitHub
- 📖 Revisar la documentación en `docs/`
- 🚀 Ejecutar `python run_trading_analysis.py` para guía interactiva

---

**🎯 Próximos Pasos:**
1. Ejecutar análisis con `python run_trading_analysis.py`
2. Revisar visualizaciones generadas
3. Analizar CSV con tus herramientas favoritas
4. Configurar Binance API para trading Bitcoin
5. Personalizar estrategias en el código fuente
