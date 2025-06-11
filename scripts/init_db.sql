-- Crear extensiones necesarias
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Tabla principal de ticks
CREATE TABLE IF NOT EXISTS market_ticks (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    bid DECIMAL(20,5) NOT NULL,
    ask DECIMAL(20,5) NOT NULL,
    last DECIMAL(20,5),
    volume DECIMAL(20,2) DEFAULT 0,
    flags INTEGER DEFAULT 0,
    spread DECIMAL(10,5) GENERATED ALWAYS AS (ask - bid) STORED,
    mid_price DECIMAL(20,5) GENERATED ALWAYS AS ((bid + ask) / 2) STORED
);

-- Convertir a hypertable
SELECT create_hypertable('market_ticks', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Índices para performance
CREATE INDEX idx_ticks_symbol_time ON market_ticks (symbol, time DESC);
CREATE INDEX idx_ticks_time ON market_ticks (time DESC);

-- Tabla OHLCV (1 minuto)
CREATE TABLE IF NOT EXISTS ohlcv_1m (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DECIMAL(20,5) NOT NULL,
    high DECIMAL(20,5) NOT NULL,
    low DECIMAL(20,5) NOT NULL,
    close DECIMAL(20,5) NOT NULL,
    volume DECIMAL(20,2) DEFAULT 0,
    tick_volume INTEGER DEFAULT 0,
    spread DECIMAL(10,5),
    PRIMARY KEY (symbol, time)
);

SELECT create_hypertable('ohlcv_1m', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
