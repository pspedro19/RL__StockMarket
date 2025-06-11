# test_mt5_simple.py
import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

print("Inicializando MT5...")
if not mt5.initialize():
    print(f"Error: {mt5.last_error()}")
else:
    print("MT5 inicializado!")
    
    login = int(os.getenv('MT5_LOGIN', 0))
    password = os.getenv('MT5_PASSWORD', '')
    server = os.getenv('MT5_SERVER', '')
    
    print(f"Intentando login en {server}...")
    if mt5.login(login, password=password, server=server):
        print("✅ Login exitoso!")
        info = mt5.account_info()
        print(f"Balance: ${info.balance}")
    else:
        print(f"❌ Error de login: {mt5.last_error()}")
    
    mt5.shutdown()