# Trading RL System 🚀

Sistema de trading automatizado con Reinforcement Learning y MetaTrader 5.

## ✨ Características

- 🔌 Conexión directa con MetaTrader 5
- 🧠 Algoritmos de Reinforcement Learning (SAC)
- 📊 Features técnicas avanzadas
- 🗄️ Base de datos TimescaleDB
- 📈 Monitoreo en tiempo real
- 🛡️ Gestión de riesgo integrada

## 🛠️ Instalación

### 1. Configurar entorno
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Configurar variables de entorno
```bash
cp .env.example .env
nano .env  # Editar con tus credenciales MT5
```

### 3. Iniciar servicios (opcional)
```bash
docker-compose up -d
```

## 🚀 Uso Rápido

### 1. Probar conexión MT5
```bash
python src/collectors/mt5_connector.py
```

### 2. Entrenar modelo básico
```bash
python src/agents/train.py
```

### 3. Monitor en tiempo real
```bash
python src/utils/currency_monitor.py
```

## 📁 Estructura del Proyecto

```
trading-rl-system/
├── src/                    # Código fuente
│   ├── collectors/         # Recolección de datos MT5
│   ├── agents/            # Modelos RL
│   ├── trading/           # Ejecución de trades
│   ├── analysis/          # Backtesting y análisis
│   └── utils/             # Utilidades
├── data/                  # Datos y modelos
│   ├── raw/              # Datos crudos
│   ├── processed/        # Datos con features
│   ├── models/           # Modelos entrenados
│   └── results/          # Resultados
├── configs/              # Configuraciones
├── scripts/              # Scripts de utilidad
├── tests/                # Tests
└── monitoring/           # Grafana + Prometheus
```

## ⚠️ Advertencias

- 🧪 **Sistema educativo**: Para aprendizaje y experimentación
- 💰 **Solo cuenta demo**: Nunca usar dinero real sin pruebas extensas
- 📊 **Resultados pasados**: No garantizan rendimientos futuros
- ⚖️ **Riesgo**: El trading conlleva riesgos significativos

## 📞 Soporte

Si encuentras problemas:
1. Revisa que MT5 esté abierto y conectado
2. Verifica las credenciales en .env
3. Chequea los logs en logs/

¡Happy Trading! 📈
