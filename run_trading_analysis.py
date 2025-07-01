#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SCRIPT PRINCIPAL DE ANÁLISIS DE TRADING
Ejecuta el análisis completo con visualizaciones y CSV organizados

Estructura generada:
📁 data/results/trading_analysis/
   ├── 📊 visualizations/     (archivos PNG)
   ├── 📋 csv_exports/        (archivos CSV)
📁 notebooks/                 (Jupyter notebook)
📁 scripts/                   (scripts auxiliares)
"""

import os
import sys
import subprocess

def main():
    """Función principal para ejecutar el análisis completo"""
    print("🚀 SISTEMA AVANZADO DE TRADING CON IA")
    print("=" * 60)
    print("📊 Características:")
    print("   • IDs únicos para trades")
    print("   • Métricas financieras avanzadas (Sharpe, Drawdown, Profit Factor)")
    print("   • Control PID para optimización de señales")
    print("   • MAPE para evaluación de predicciones ML")
    print("   • Exportación completa a CSV")
    print("   • Visualizaciones separadas y dashboard")
    print("   • Datos SP500 (Yahoo Finance) + preparación Binance")
    print()
    
    # Verificar estructura del proyecto
    required_dirs = [
        'src/agents',
        'scripts',
        'data/results',
        'notebooks'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Error: Directorio {dir_path} no encontrado")
            print("💡 Asegúrate de ejecutar desde la raíz del proyecto")
            return False
    
    print("✅ Estructura del proyecto verificada")
    
    # Opciones disponibles
    print("\n🎯 OPCIONES DISPONIBLES:")
    print("1. 🔄 Ejecutar análisis completo (recomendado)")
    print("2. 📊 Solo generar visualizaciones")
    print("3. 📋 Solo exportar CSV")
    print("4. 📓 Abrir notebook Jupyter")
    print("5. 📁 Mostrar ubicación de archivos")
    print("0. ❌ Salir")
    
    try:
        opcion = input("\n👉 Selecciona una opción (1-5, 0 para salir): ").strip()
        
        if opcion == "1":
            ejecutar_analisis_completo()
        elif opcion == "2":
            generar_visualizaciones()
        elif opcion == "3":
            exportar_csv()
        elif opcion == "4":
            abrir_notebook()
        elif opcion == "5":
            mostrar_ubicaciones()
        elif opcion == "0":
            print("👋 ¡Hasta luego!")
            return True
        else:
            print("❌ Opción inválida")
            return False
            
    except KeyboardInterrupt:
        print("\n👋 Análisis cancelado por el usuario")
        return False
    
    return True

def ejecutar_analisis_completo():
    """Ejecutar análisis completo con visualizaciones y CSV"""
    print("\n🚀 EJECUTANDO ANÁLISIS COMPLETO...")
    print("⏳ Esto puede tomar 1-2 minutos...")
    
    try:
        script_path = os.path.join('scripts', 'generate_clean_visualizations.py')
        
        if not os.path.exists(script_path):
            print(f"❌ Script no encontrado: {script_path}")
            return
        
        # Ejecutar análisis
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
            print("\n📋 Resumen:")
            print("   • Visualizaciones generadas: 6+ archivos PNG")
            print("   • CSV exportados: 3 archivos")
            print("   • Dashboard completo creado")
            
            # Mostrar ubicaciones
            mostrar_ubicaciones()
            
            if result.stdout:
                print("\n📊 Detalles del análisis:")
                # Mostrar solo las líneas importantes
                for line in result.stdout.split('\n'):
                    if any(keyword in line for keyword in ['Total de Trades:', 'Win Rate:', 'Retorno Total:', 'guardado']):
                        print(f"   {line}")
        else:
            print("❌ Error en el análisis")
            if result.stderr:
                print(f"Error: {result.stderr}")
                
    except subprocess.TimeoutExpired:
        print("⏰ El análisis tardó demasiado, pero puede haberse completado")
    except Exception as e:
        print(f"❌ Error ejecutando análisis: {e}")

def generar_visualizaciones():
    """Solo generar visualizaciones"""
    print("\n🎨 GENERANDO SOLO VISUALIZACIONES...")
    # Reutilizar la función principal pero podríamos hacer una versión más específica
    ejecutar_analisis_completo()

def exportar_csv():
    """Solo exportar CSV"""
    print("\n📋 EXPORTANDO SOLO CSV...")
    print("💡 Para exportar solo CSV, usa el análisis completo (incluye CSV)")
    ejecutar_analisis_completo()

def abrir_notebook():
    """Abrir notebook Jupyter"""
    print("\n📓 PREPARANDO NOTEBOOK JUPYTER...")
    
    notebook_path = os.path.join('notebooks', 'advanced_trading_notebook.ipynb')
    
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook no encontrado: {notebook_path}")
        return
    
    print(f"📁 Notebook ubicado en: {notebook_path}")
    print("\n🚀 Para abrir el notebook, ejecuta uno de estos comandos:")
    print(f"   jupyter notebook {notebook_path}")
    print(f"   jupyter lab {notebook_path}")
    print(f"   code {notebook_path}  # Si usas VS Code")
    
    # Intentar abrir automáticamente
    try:
        if os.system("jupyter --version > nul 2>&1") == 0:  # Windows
            respuesta = input("\n❓ ¿Quieres abrir el notebook automáticamente? (y/n): ")
            if respuesta.lower() in ['y', 'yes', 'sí', 's']:
                os.system(f"jupyter notebook {notebook_path}")
        else:
            print("💡 Instala Jupyter con: pip install jupyter")
    except:
        pass

def mostrar_ubicaciones():
    """Mostrar ubicaciones de archivos generados"""
    print("\n📁 UBICACIONES DE ARCHIVOS:")
    
    # Visualizaciones
    viz_dir = os.path.join('data', 'results', 'trading_analysis', 'visualizations')
    if os.path.exists(viz_dir):
        png_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
        print(f"\n📊 VISUALIZACIONES ({len(png_files)} archivos):")
        print(f"   📂 {os.path.abspath(viz_dir)}")
        for file in sorted(png_files):
            size_mb = os.path.getsize(os.path.join(viz_dir, file)) / (1024*1024)
            print(f"   • {file} ({size_mb:.1f} MB)")
    
    # CSV
    csv_dir = os.path.join('data', 'results', 'trading_analysis', 'csv_exports')
    if os.path.exists(csv_dir):
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        print(f"\n📋 ARCHIVOS CSV ({len(csv_files)} archivos):")
        print(f"   📂 {os.path.abspath(csv_dir)}")
        for file in sorted(csv_files):
            size_kb = os.path.getsize(os.path.join(csv_dir, file)) / 1024
            print(f"   • {file} ({size_kb:.1f} KB)")
    
    # Notebook
    notebook_path = os.path.join('notebooks', 'advanced_trading_notebook.ipynb')
    if os.path.exists(notebook_path):
        size_kb = os.path.getsize(notebook_path) / 1024
        print(f"\n📓 NOTEBOOK JUPYTER:")
        print(f"   📂 {os.path.abspath(notebook_path)} ({size_kb:.1f} KB)")
    
    print(f"\n🎯 PRÓXIMOS PASOS:")
    print(f"   1. Revisar visualizaciones en la carpeta PNG")
    print(f"   2. Analizar datos en los archivos CSV")
    print(f"   3. Ejecutar el notebook para análisis interactivo")
    print(f"   4. Configurar Binance API para trading con Bitcoin")

if __name__ == "__main__":
    print("🎯 Iniciando sistema de análisis de trading...")
    
    # Verificar que estamos en la raíz del proyecto
    if not os.path.exists('src/agents/advanced_trading_analytics.py'):
        print("❌ Error: Debes ejecutar este script desde la raíz del proyecto")
        print("💡 Navega al directorio RL__StockMarket y ejecuta:")
        print("   python run_trading_analysis.py")
        sys.exit(1)
    
    # Ejecutar función principal
    success = main()
    
    if not success:
        print("\n❌ El análisis no se completó correctamente")
        sys.exit(1)
    else:
        print("\n✅ Operación completada exitosamente") 