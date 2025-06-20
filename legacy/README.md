# 📦 Sistemas Legacy

Esta carpeta contiene las versiones originales del sistema de trading antes de la reorganización modular.

## 📁 Contenido

### 🔄 `ml_enhanced_trading_system_legacy.py`
- **Función**: Versión de compatibilidad hacia atrás
- **Descripción**: Sistema original adaptado con rutas actualizadas para la nueva estructura
- **Uso**: `python legacy/ml_enhanced_trading_system_legacy.py`
- **Tamaño**: ~1KB (wrapper/adaptador)

### 🎯 `ml_enhanced_trading_system.py` 
- **Función**: Sistema original completo e intacto
- **Descripción**: Tu sistema ML completo de 64KB sin modificaciones
- **Uso**: `python legacy/ml_enhanced_trading_system.py`
- **Tamaño**: 64KB (1506 líneas)
- **Estado**: 100% funcional, sin cambios

## 🚀 Formas de Ejecutar

### ✅ **Recomendado - Nueva Arquitectura**
```bash
python main.py
```

### ✅ **Legacy - Compatibilidad**
```bash
python legacy/ml_enhanced_trading_system_legacy.py
```

### ✅ **Legacy - Original Intacto**
```bash
python legacy/ml_enhanced_trading_system.py
```

## 📝 Notas

- **Modelos**: Todos los modelos entrenados están en `data/models/`
- **Resultados**: Todos los resultados están en `data/results/`
- **Funcionalidad**: Idéntica en todas las versiones
- **Rutas**: Los sistemas legacy usan rutas relativas actualizadas

## 🎯 Migración

Si quieres migrar completamente a la nueva arquitectura:
1. Usa `python main.py` como punto de entrada principal
2. Desarrolla nuevas funcionalidades en `src/`
3. Mantén estos archivos legacy como respaldo

**Los sistemas legacy están preservados para garantizar compatibilidad total.** 