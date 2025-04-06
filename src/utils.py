import yaml
import os
from dotenv import load_dotenv
from loguru import logger
import numpy as np
import pandas as pd

def load_config(config_path="config/config.yaml"):
    load_dotenv("config/.env")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config['api']['host'] = os.getenv('IB_HOST', config['api']['host'])
    config['api']['port'] = int(os.getenv('IB_PORT', config['api']['port']))
    config['api']['client_id'] = int(os.getenv('IB_CLIENT_ID', config['api']['client_id']))
    config['trading']['initial_balance'] = float(os.getenv('INITIAL_BALANCE', config['trading']['initial_balance']))

    return config

def setup_logging():
    logger.add("logs/trading.log", rotation="500 MB", level="INFO", format="{time} {level} {message}")
    return logger

def calculate_metrics(portfolio_values, initial_balance):
    final_value = portfolio_values[-1]
    annualized_roi = ((final_value / initial_balance) ** (252 / len(portfolio_values))) - 1
    max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / initial_balance
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    calmar_ratio = annualized_roi / max_drawdown if max_drawdown > 0 else float('inf')

    return {
        "Annualized ROI": annualized_roi,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "Calmar Ratio": calmar_ratio
    }