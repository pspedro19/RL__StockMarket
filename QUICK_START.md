# 🚀 GUÍA DE INICIO RÁPIDO

## ⚡ Ejecución en 30 Segundos

### 1. **Ejecutar el Sistema** (Método más fácil)
```bash
python run_trading_analysis.py
```
Selecciona opción **1** para análisis completo.

### 2. **Verificar Resultados**
- 📊 **Visualizaciones**: `data/results/trading_analysis/visualizations/`
- 📋 **CSV**: `data/results/trading_analysis/csv_exports/`
- 📓 **Notebook**: `notebooks/advanced_trading_notebook.ipynb`

---

## 📦 Instalación de Dependencias

### Opción A: Mínimas (Más Rápido)
```bash
pip install -r requirements-minimal.txt
```

### Opción B: Completas (Más Funciones)
```bash
pip install -r requirements.txt
```

### Opción C: Script Automático
```bash
python utils/install_advanced_dependencies.py
```

---

## 🎯 Lo Que Obtienes

### 📈 Análisis Financiero Completo
- ✅ **22 trades** con IDs únicos
- ✅ **59% win rate** típico
- ✅ **Control PID** automático
- ✅ **Métricas avanzadas**: Sharpe, Drawdown, MAPE

### 📊 Visualizaciones (15+ archivos PNG)
1. **01_precio_y_trades_detallado.png** - Trading signals
2. **02_metricas_financieras_panel.png** - KPI dashboard
3. **03_distribuciones_estadisticas.png** - Statistical analysis
4. **04_equity_y_drawdown.png** - Portfolio performance
5. **05_analisis_performance.png** - Advanced metrics
6. **06_tabla_trades_detallada.png** - Detailed trades table
7. **dashboard_avanzado.png** - Complete 18-panel dashboard

### 📋 Datos CSV Exportados
- **trades_detallados.csv** - All trade data
- **metricas_resumen.csv** - System KPIs
- **estadisticas_adicionales.csv** - Extra statistics

---

## 🛠️ Fuentes de Datos

### Automático (Recomendado)
- **MetaTrader5**: Datos en tiempo real (si está instalado)
- **Yahoo Finance**: SP500 histórico
- **Simulados**: Backup si no hay conexión

### Configuración Bitcoin (Opcional)
1. Copia `configs/binance.env.example` → `configs/binance.env`
2. Agrega tus API keys de Binance
3. Ejecuta con Bitcoin en lugar de SP500

---

## ⚡ Comandos Rápidos

```bash
# Análisis completo
python run_trading_analysis.py

# Solo visualizaciones
python scripts/generate_clean_visualizations.py

# Notebook interactivo
jupyter notebook notebooks/advanced_trading_notebook.ipynb

# Ver ubicación de archivos
python run_trading_analysis.py  # Opción 5
```

---

## 🎪 Características Destacadas

- 🎯 **IDs únicos** para cada trade (`T00001_20241201_123456_ABC12345`)
- 🤖 **Machine Learning** con evaluación MAPE
- 📊 **18 paneles** de análisis en dashboard
- 🔄 **Control PID** para optimización automática
- 📈 **Datos reales** de MetaTrader5 y Yahoo Finance
- 🧮 **Métricas financieras** profesionales
- 📱 **Jupyter interactivo** para análisis personalizado

---

## 🆘 Problemas Comunes

### Error de dependencias
```bash
# Instalar dependencias mínimas
pip install pandas numpy matplotlib yfinance scikit-learn
```

### Error de codificación (Windows)
- Ya solucionado en la versión actual

### MetaTrader5 no disponible
- El sistema automáticamente usa Yahoo Finance como backup

---

## 🎯 Próximos Pasos

1. ✅ **Ejecutar** `python run_trading_analysis.py`
2. ✅ **Revisar** visualizaciones generadas
3. ✅ **Analizar** datos CSV exportados
4. ✅ **Explorar** notebook Jupyter
5. ✅ **Configurar** Binance API (opcional)

---

**🚀 ¡Listo para empezar! El sistema está optimizado y funcionando con datos reales.** 