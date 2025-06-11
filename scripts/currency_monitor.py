# currency_monitor_enhanced.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys
from pathlib import Path

# Cargar .env manualmente
def load_env_file():
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ Archivo .env no encontrado")
        print("Creando archivo .env con configuración...")
        
        # Crear .env con credenciales de ejemplo
        with open('.env', 'w') as f:
            f.write("""# MetaTrader 5 - CAMBIA ESTAS CREDENCIALES
MT5_LOGIN=tu_numero_de_cuenta
MT5_PASSWORD=tu_contraseña
MT5_SERVER=tu_servidor
""")
        print("✅ Archivo .env creado. DEBES CAMBIAR las credenciales por las tuyas.")
        return False
    
    # Cargar variables manualmente
    env_vars = {}
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
                    os.environ[key] = value
    
    print(f"✅ Variables cargadas: {list(env_vars.keys())}")
    return True

class CurrencyMonitor:
    def __init__(self):  # ✅ Corregido constructor
        self.running = False
        self.current_symbol = None
        self.max_price = 0
        self.min_price = float('inf')
        self.current_price = 0
        self.prev_price = 0
        
    def connect_mt5(self):
        """Conectar a MT5"""
        print("🔌 Conectando a MT5...")
        
        # Verificar variables
        login = os.getenv('MT5_LOGIN')
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')
        
        if not all([login, password, server]):
            print("❌ Faltan credenciales de MT5")
            print("Por favor configura el archivo .env con tus credenciales")
            return False
        
        if login in ['tu_numero_de_cuenta', 'ejemplo']:
            print("❌ Debes cambiar las credenciales en el archivo .env")
            return False
        
        print(f"Login: {login}")
        print(f"Server: {server}")
        print(f"Password: {'*' * len(password)}")
        
        if not mt5.initialize():
            print(f"❌ Error al inicializar MT5: {mt5.last_error()}")
            return False
        
        try:
            login_int = int(login)
        except ValueError:
            print(f"❌ Login debe ser numérico: {login}")
            return False
            
        if not mt5.login(login_int, password=password, server=server):
            print(f"❌ Error de login: {mt5.last_error()}")
            return False
            
        print("✅ Conectado a MT5")
        return True
    
    def validate_symbol(self, symbol):
        """Validar y activar símbolo si está disponible"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return False
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return False
        
        return True
    
    def get_available_symbols(self, show_all=False):
        """Obtener símbolos disponibles con opción de mostrar todos"""
        symbols = mt5.symbols_get()
        if not symbols:
            return []
        
        available_symbols = []
        
        # Si queremos ver TODOS los símbolos
        if show_all:
            for symbol in symbols:
                if symbol.visible:
                    available_symbols.append({
                        'name': symbol.name,
                        'description': symbol.description if hasattr(symbol, 'description') else '',
                        'type': self.get_symbol_type(symbol.name)
                    })
        else:
            # Filtrado normal
            major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'NZD']
            
            # Símbolos específicos que queremos incluir
            target_symbols = {
                # Divisas principales
                'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
                # Peso colombiano (múltiples variantes)
                'USDCOP', 'COPDOLLAR', 'COPUSD', 'COP', 'PESOS',
                # Índices (múltiples variantes)
                'US500', 'SPX500', 'SP500', 'S&P500', 'US500Cash', 'USTEC', 'US30', 
                'SPX', 'SPXUSD', 'USA500', 'NDX', 'NASDAQ', 'DOW30'
            }
            
            for symbol in symbols:
                if symbol.visible:
                    symbol_name = symbol.name.upper()
                    
                    # Verificar símbolos específicos
                    if symbol_name in target_symbols:
                        available_symbols.append({
                            'name': symbol.name,
                            'description': symbol.description if hasattr(symbol, 'description') else '',
                            'type': self.get_symbol_type(symbol.name)
                        })
                    
                    # Buscar COP en cualquier parte del nombre
                    elif 'COP' in symbol_name:
                        available_symbols.append({
                            'name': symbol.name,
                            'description': symbol.description if hasattr(symbol, 'description') else '',
                            'type': 'cop'
                        })
                    
                    # Buscar índices por palabras clave
                    elif any(idx in symbol_name for idx in ['500', 'SPX', 'DOW', 'NASDAQ', 'NDX']):
                        available_symbols.append({
                            'name': symbol.name,
                            'description': symbol.description if hasattr(symbol, 'description') else '',
                            'type': 'index'
                        })
                    
                    # Buscar otras divisas
                    elif any(curr in symbol_name for curr in major_currencies):
                        if len(symbol_name) <= 8:  # Filtrar símbolos muy largos
                            available_symbols.append({
                                'name': symbol.name,
                                'description': symbol.description if hasattr(symbol, 'description') else '',
                                'type': 'forex'
                            })
        
        # Eliminar duplicados y ordenar
        seen = set()
        unique_symbols = []
        for symbol in available_symbols:
            if symbol['name'] not in seen:
                seen.add(symbol['name'])
                unique_symbols.append(symbol)
        
        return sorted(unique_symbols, key=lambda x: (x['type'], x['name']))[:100 if show_all else 50]
    
    def get_symbol_type(self, symbol):
        """Determinar el tipo de símbolo"""
        symbol_upper = symbol.upper()
        
        if 'COP' in symbol_upper:
            return 'cop'
        elif any(idx in symbol_upper for idx in ['US500', 'SPX500', 'SP500', 'S&P', 'US30', 'USTEC']):
            return 'index'
        else:
            return 'forex'
    
    def select_currency(self):
        """Seleccionar instrumento mejorado"""
        symbols = self.get_available_symbols()
        
        if not symbols:
            print("❌ No se encontraron símbolos disponibles")
            return None
        
        print("\n💱 INSTRUMENTOS DISPONIBLES:")
        print("=" * 70)
        
        # Agrupar por categorías
        categories = {
            'cop': '🇨🇴 PESO COLOMBIANO',
            'index': '📈 ÍNDICES (S&P 500, etc.)',
            'forex': '💱 DIVISAS PRINCIPALES'
        }
        
        current_index = 1
        symbol_map = {}
        found_target_symbols = False
        
        for category, title in categories.items():
            category_symbols = [s for s in symbols if s['type'] == category]
            if category_symbols:
                found_target_symbols = True
                print(f"\n{title}:")
                print("-" * 50)
                
                for symbol in category_symbols[:10]:  # Limitar por categoría
                    print(f"  {current_index:2d}. {symbol['name']:<12} - {symbol['description']}")
                    symbol_map[current_index] = symbol['name']
                    current_index += 1
        
        # Si no encontramos USDCOP o índices, mostrar mensaje
        if not any(s['type'] in ['cop', 'index'] for s in symbols):
            print(f"\n⚠️  USDCOP y S&P 500 NO disponibles en {os.getenv('MT5_SERVER')}")
            print("💡 Opciones:")
            print("   • Escribe 'TODOS' para ver TODOS los símbolos disponibles")
            print("   • O selecciona una divisa principal disponible")
        
        print(f"\n{current_index}. 🔍 Ver TODOS los símbolos disponibles")
        symbol_map[current_index] = 'SHOW_ALL'
        
        print("\n" + "=" * 70)
        
        while True:
            try:
                choice = input(f"\n🎯 Selecciona (1-{len(symbol_map)}), escribe símbolo o 'TODOS': ").strip()
                
                # Mostrar todos los símbolos
                if choice.upper() in ['TODOS', 'ALL', 'SHOW_ALL'] or (choice.isdigit() and int(choice) == len(symbol_map) and symbol_map[int(choice)] == 'SHOW_ALL'):
                    return self.show_all_symbols()
                
                # Verificar si es un símbolo directo
                if choice.upper() in [s['name'].upper() for s in symbols]:
                    selected = next(s['name'] for s in symbols if s['name'].upper() == choice.upper())
                    if self.validate_symbol(selected):
                        return selected
                    else:
                        print(f"❌ {selected} no está disponible o no se pudo activar")
                        continue
                
                # Verificar si es un número
                if choice.isdigit():
                    idx = int(choice)
                    if idx in symbol_map and symbol_map[idx] != 'SHOW_ALL':
                        selected = symbol_map[idx]
                        if self.validate_symbol(selected):
                            return selected
                        else:
                            print(f"❌ {selected} no está disponible o no se pudo activar")
                            continue
                
                print("❌ Opción inválida. Intenta de nuevo.")
                    
            except ValueError:
                print("❌ Por favor ingresa un número válido o símbolo")
            except KeyboardInterrupt:
                return None
    
    def show_all_symbols(self):
        """Mostrar todos los símbolos disponibles para encontrar COP o índices"""
        print("\n🔍 BUSCANDO TODOS LOS SÍMBOLOS...")
        all_symbols = self.get_available_symbols(show_all=True)
        
        if not all_symbols:
            print("❌ No se encontraron símbolos")
            return None
        
        print(f"\n📋 TODOS LOS SÍMBOLOS DISPONIBLES ({len(all_symbols)}):")
        print("=" * 80)
        
        # Buscar símbolos que contengan palabras clave
        cop_symbols = [s for s in all_symbols if 'COP' in s['name'].upper()]
        index_symbols = [s for s in all_symbols if any(word in s['name'].upper() for word in ['500', 'SPX', 'DOW', 'NASDAQ', 'NDX', 'INDEX'])]
        
        if cop_symbols:
            print("\n🇨🇴 SÍMBOLOS CON 'COP':")
            for i, symbol in enumerate(cop_symbols, 1):
                print(f"  {i}. {symbol['name']:<15} - {symbol['description']}")
        
        if index_symbols:
            print("\n📈 POSIBLES ÍNDICES:")
            for i, symbol in enumerate(index_symbols, 1):
                print(f"  {i}. {symbol['name']:<15} - {symbol['description']}")
        
        # Mostrar otros símbolos en grupos
        other_symbols = [s for s in all_symbols if s not in cop_symbols and s not in index_symbols]
        
        print(f"\n💱 OTROS SÍMBOLOS DISPONIBLES:")
        print("-" * 60)
        
        for i, symbol in enumerate(other_symbols[:50], 1):  # Mostrar máximo 50
            print(f"  {i:2d}. {symbol['name']:<15} - {symbol['description'][:40]}")
            if i % 20 == 0:
                cont = input("\n📄 Presiona Enter para ver más o 'q' para seleccionar: ")
                if cont.lower() == 'q':
                    break
        
        # Permitir selección
        while True:
            choice = input(f"\n🎯 Escribe el símbolo que quieres monitorear: ").strip()
            
            if choice.upper() in [s['name'].upper() for s in all_symbols]:
                selected = next(s['name'] for s in all_symbols if s['name'].upper() == choice.upper())
                if self.validate_symbol(selected):
                    return selected
                else:
                    print(f"❌ {selected} no se pudo activar")
            else:
                print("❌ Símbolo no encontrado. Intenta de nuevo.")
                
        return None
    
    def get_historical_data(self, symbol, years=20):
        """Obtener datos históricos con mejor manejo de timeframes"""
        print(f"\n📊 Descargando {years} años de datos históricos para {symbol}...")
        
        date_to = datetime.now()
        date_from = date_to - timedelta(days=years*365)
        
        # Intentar diferentes timeframes según el tipo de instrumento
        timeframes = [
            (mt5.TIMEFRAME_D1, "diarios"),
            (mt5.TIMEFRAME_H4, "4 horas"),
            (mt5.TIMEFRAME_H1, "1 hora"),
            (mt5.TIMEFRAME_M30, "30 minutos")
        ]
        
        rates = None
        used_timeframe = None
        
        for timeframe, name in timeframes:
            print(f"⏱️  Intentando obtener datos {name}...")
            rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
            
            if rates is not None and len(rates) > 100:  # Mínimo 100 registros
                used_timeframe = name
                break
        
        if rates is None or len(rates) == 0:
            print("❌ No se pudieron obtener datos históricos")
            return None
            
        print(f"✅ Datos obtenidos: {len(rates)} registros ({used_timeframe})")
        
        # Convertir a DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Calcular máximo y mínimo
        self.max_price = df['high'].max()
        self.min_price = df['low'].min()
        
        # Encontrar fechas
        max_idx = df['high'].idxmax()
        min_idx = df['low'].idxmin()
        max_date = df.loc[max_idx, 'time']
        min_date = df.loc[min_idx, 'time']
        
        # Determinar número de decimales apropiado
        decimals = 5 if 'JPY' not in symbol else 3
        if any(idx in symbol.upper() for idx in ['US500', 'SPX500', 'SP500']):
            decimals = 2
        
        print(f"\n📈 ANÁLISIS HISTÓRICO ({years} AÑOS):")
        print("=" * 60)
        print(f"📊 Símbolo: {symbol}")
        print(f"⏱️  Timeframe: {used_timeframe}")
        print(f"📅 Período: {date_from.strftime('%Y-%m-%d')} a {date_to.strftime('%Y-%m-%d')}")
        print(f"📈 Máximo histórico: {self.max_price:.{decimals}f} ({max_date.strftime('%Y-%m-%d')})")
        print(f"📉 Mínimo histórico: {self.min_price:.{decimals}f} ({min_date.strftime('%Y-%m-%d')})")
        print(f"📊 Rango total: {((self.max_price/self.min_price - 1) * 100):.2f}%")
        print("=" * 60)
        
        return df
    
    def monitor_price(self):
        """Monitor en tiempo real mejorado"""
        print(f"\n💹 MONITOR EN TIEMPO REAL - {self.current_symbol}")
        print("Presiona Ctrl+C para detener\n")
        
        # Determinar decimales
        decimals = 5 if 'JPY' not in self.current_symbol else 3
        if any(idx in self.current_symbol.upper() for idx in ['US500', 'SPX500', 'SP500']):
            decimals = 2
        
        # Obtener precio inicial
        tick = mt5.symbol_info_tick(self.current_symbol)
        if tick:
            self.current_price = tick.bid if hasattr(tick, 'bid') else tick.last
            self.prev_price = self.current_price
        
        try:
            while self.running:
                tick = mt5.symbol_info_tick(self.current_symbol)
                if tick:
                    # Usar bid para forex, last para índices
                    if hasattr(tick, 'bid') and tick.bid > 0:
                        self.current_price = tick.bid
                    elif hasattr(tick, 'last') and tick.last > 0:
                        self.current_price = tick.last
                    else:
                        time.sleep(1)
                        continue
                    
                    # Determinar tendencia
                    if self.current_price > self.prev_price:
                        arrow = "📈"
                        trend = "SUBIENDO"
                    elif self.current_price < self.prev_price:
                        arrow = "📉" 
                        trend = "BAJANDO"
                    else:
                        arrow = "➡️"
                        trend = "LATERAL"
                    
                    # Calcular posición en rango
                    if self.max_price > self.min_price:
                        position = (self.current_price - self.min_price) / (self.max_price - self.min_price) * 100
                    else:
                        position = 50
                    
                    # Limpiar línea y mostrar
                    print(f"\r💰 {self.current_price:.{decimals}f} {arrow} | "
                          f"📊 {trend} | "
                          f"📍 {position:.1f}% | "
                          f"📉 {self.min_price:.{decimals}f} | "
                          f"📈 {self.max_price:.{decimals}f} | "
                          f"🕒 {datetime.now().strftime('%H:%M:%S')}    ", 
                          end='', flush=True)
                    
                    self.prev_price = self.current_price
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.running = False
            print("\n\n⏹ Monitor detenido")
    
    def run(self):
        """Ejecutar el monitor"""
        if not self.connect_mt5():
            return
        
        try:
            while True:
                self.current_symbol = self.select_currency()
                if not self.current_symbol:
                    break
                
                # Obtener datos históricos
                historical = self.get_historical_data(self.current_symbol)
                if historical is None:
                    continue
                
                # Iniciar monitor
                self.running = True
                self.monitor_price()
                
                # Preguntar si continuar
                choice = input("\n❓ ¿Monitorear otro instrumento? (s/n): ").lower()
                if choice != 's':
                    break
                    
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            mt5.shutdown()
            print("\n✅ Desconectado de MT5")

def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║     MONITOR DE DIVISAS E ÍNDICES MEJORADO        ║")
    print("║   📈 Incluye USDCOP y S&P 500 📈                 ║")
    print("║         Análisis histórico de 5 años            ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    
    # Cargar variables de entorno
    if not load_env_file():
        return
    
    # Verificar que MT5 esté instalado
    try:
        import MetaTrader5 as mt5
        print(f"✅ MetaTrader5 versión: {mt5.__version__}")
    except ImportError:
        print("❌ MetaTrader5 no está instalado")
        print("Instálalo con: pip install MetaTrader5")
        return
    
    monitor = CurrencyMonitor()
    monitor.run()

if __name__ == "__main__":
    main()