"""
Configuración de modelos de trading
"""

QDN_CONFIG = {
    "algorithm_choice": "1",
    "algorithm_name": "QDN",
    "model_paths": ["data/models/qdn/model.zip"],
    "policy_kwargs": {"net_arch": [64, 32]},
    "learning_rate": 0.001,
    "max_position_risk": 0.05,
    "stop_loss_pct": 0.01,
    "take_profit_pct": 0.02,
    "min_trade_separation": 1,
    "max_daily_trades": 10,
    "ml_weight": 0.4,
    "consecutive_losses": 3
}

DEEPQDN_CONFIG = {
    "algorithm_choice": "2",
    "algorithm_name": "DeepQDN",
    "model_paths": ["data/models/deepqdn/model.zip"],
    "policy_kwargs": {"net_arch": [256, 256, 128, 64]},
    "learning_rate": 0.0005,
    "max_position_risk": 0.04,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04,
    "min_trade_separation": 5,
    "max_daily_trades": 6,
    "ml_weight": 0.7,
    "consecutive_losses": 2
}

A2C_CONFIG = {
    "algorithm_choice": "3",
    "algorithm_name": "A2C",
    "model_paths": ["data/models/a2c/model.zip"],
    "max_position_risk": 0.03,
    "stop_loss_pct": 0.03,
    "take_profit_pct": 0.06,
    "min_trade_separation": 10,
    "max_daily_trades": 3,
    "ml_weight": 0.9,
    "consecutive_losses": 1
}
