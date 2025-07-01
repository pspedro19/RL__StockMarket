-- Script de Inicialización de Base de Datos
-- Crea las tablas necesarias para el sistema de trading

-- Tabla de precios históricos
CREATE TABLE IF NOT EXISTS historical_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    timestamp DATETIME NOT NULL,
    open_price DECIMAL(10,5) NOT NULL,
    high_price DECIMAL(10,5) NOT NULL,
    low_price DECIMAL(10,5) NOT NULL,
    close_price DECIMAL(10,5) NOT NULL,
    volume INTEGER,
    spread INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_timestamp (symbol, timestamp)
);

-- Tabla de operaciones/trades
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    trade_type VARCHAR(10) NOT NULL, -- 'BUY' or 'SELL'
    entry_price DECIMAL(10,5) NOT NULL,
    exit_price DECIMAL(10,5),
    entry_time DATETIME NOT NULL,
    exit_time DATETIME,
    quantity DECIMAL(10,3) NOT NULL,
    profit_loss DECIMAL(10,2),
    status VARCHAR(20) DEFAULT 'OPEN', -- 'OPEN', 'CLOSED'
    strategy VARCHAR(50),
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de métricas de performance
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    total_trades INTEGER DEFAULT 0,
    profitable_trades INTEGER DEFAULT 0,
    total_profit DECIMAL(10,2) DEFAULT 0,
    max_drawdown DECIMAL(5,2) DEFAULT 0,
    win_rate DECIMAL(5,2) DEFAULT 0,
    portfolio_value DECIMAL(15,2) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de configuración del sistema
CREATE TABLE IF NOT EXISTS system_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value TEXT NOT NULL,
    description TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insertar configuraciones por defecto
INSERT OR IGNORE INTO system_config (config_key, config_value, description) VALUES
('initial_capital', '100000', 'Capital inicial para trading'),
('max_position_risk', '0.03', 'Máximo riesgo por posición (3%)'),
('stop_loss_pct', '0.02', 'Porcentaje de stop loss (2%)'),
('take_profit_pct', '0.04', 'Porcentaje de take profit (4%)'),
('max_daily_trades', '5', 'Máximo de trades por día'),
('trading_symbol', 'US500', 'Símbolo principal de trading'),
('system_version', '2.0.0', 'Versión del sistema de trading');

-- Crear índices para optimizar consultas
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_metrics(date);

-- Comentarios
COMMENT ON TABLE historical_prices IS 'Almacena datos históricos de precios';
COMMENT ON TABLE trades IS 'Registro de todas las operaciones de trading';
COMMENT ON TABLE performance_metrics IS 'Métricas diarias de rendimiento';
COMMENT ON TABLE system_config IS 'Configuración del sistema'; 