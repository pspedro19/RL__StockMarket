# 🛠️ Utilities

Herramientas y scripts de utilidad para la instalación y configuración del sistema.

## 📁 Contenido

### 📦 Scripts de Instalación
- `install_dependencies.py` - Instalador básico de dependencias
- `install_advanced_dependencies.py` - Instalador completo con dependencias avanzadas

## 🚀 Uso

### Instalación Básica
```bash
python utils/install_dependencies.py
```

### Instalación Completa
```bash
python utils/install_advanced_dependencies.py
```

### Instalación Manual
```bash
# Dependencias mínimas
pip install -r requirements-minimal.txt

# Dependencias completas
pip install -r requirements.txt
```

## 📋 Dependencias por Categoría

### Esenciales
- pandas, numpy, matplotlib
- yfinance (datos SP500)
- scikit-learn (métricas)

### Trading
- ccxt (Binance API)
- MetaTrader5 (opcional)

### Machine Learning
- stable-baselines3, torch
- tensorflow, keras

### Visualización
- seaborn, plotly, dash

## ⚠️ Problemas Comunes

### Windows
- TA-Lib requiere Visual Studio Build Tools
- numba puede fallar en algunos sistemas

### Linux/Mac
```bash
# Ubuntu
sudo apt-get install build-essential

# macOS
brew install gcc
``` 