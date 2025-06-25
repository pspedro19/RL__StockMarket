# 🚀 Guía de Instalación Rápida - Sistema de Trading IA

## ✅ **Opción 1: Instalación Automática (Recomendada)**

```bash
# Ejecutar el instalador automático
python install_dependencies.py
```

## ✅ **Opción 2: Instalación Manual**

```bash
# Instalar todas las dependencias
pip install -r requirements.txt
```

## ✅ **Opción 3: Instalación Mínima (Solo Dependencias Críticas)**

```bash
# Solo las dependencias esenciales para que funcione
pip install numpy>=1.26.4 pandas>=2.2.3 matplotlib>=3.10.1
pip install gymnasium==1.1.1 stable-baselines3==2.6.0
pip install MetaTrader5==5.0.5120 ta>=0.11.0
```

---

## 🎯 **Dependencias Críticas Verificadas**

Las siguientes dependencias son **OBLIGATORIAS** y han sido probadas:

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| `MetaTrader5` | 5.0.5120 | 📡 Conexión a datos en tiempo real |
| `gymnasium` | 1.1.1 | 🎮 Entornos de RL (reemplaza gym) |
| `stable-baselines3` | 2.6.0 | 🤖 Algoritmos de RL (DQN, A2C, etc.) |
| `matplotlib` | 3.10.1 | 📊 Interfaz gráfica del sistema |
| `pandas` | 2.2.3 | 📋 Procesamiento de datos |
| `numpy` | 1.26.4 | 🔢 Cálculos numéricos |
| `ta` | 0.11.0 | 📈 Indicadores técnicos |

---

## 🏃‍♂️ **Ejecutar el Sistema**

Una vez instaladas las dependencias:

### **Nueva Arquitectura (Recomendada)**
```bash
python main.py
```

### **Sistema Principal**
```bash
python src/agents/ml_enhanced_system.py
```

### **Sistema Legacy**
```bash
python ml_enhanced_trading_system.py
```

---

## 🔧 **Solución de Problemas**

### **Error: ModuleNotFoundError**
```bash
# Instalar el módulo faltante específico
pip install [nombre_del_modulo]
```

### **Error de MetaTrader5**
- ✅ Instalar MetaTrader5 desde [metaquotes.net](https://www.metaquotes.net/es/metatrader5)
- ✅ Configurar una cuenta demo
- ✅ Asegurarse de que MT5 esté ejecutándose

### **Error de GUI/Matplotlib**
```bash
# En sistemas Linux
sudo apt-get install python3-tk

# En Windows (usualmente no necesario)
pip install --upgrade matplotlib
```

---

## ✅ **Verificación de Instalación**

Para verificar que todo esté funcionando:

```python
# Probar imports críticos
python -c "import gymnasium, stable_baselines3, MetaTrader5, matplotlib, pandas, numpy, ta; print('✅ Todas las dependencias funcionan')"
```

---

## 🎉 **¡Listo para Trading!**

Si la instalación fue exitosa, deberías ver:
- ✅ MetaTrader5 disponible
- ✅ Componentes de RL disponibles
- ✅ Sistema iniciando correctamente
- ✅ Interfaz gráfica con 8 paneles
- ✅ Conexión a datos en tiempo real

**¡El sistema está listo para operar! 🚀** 