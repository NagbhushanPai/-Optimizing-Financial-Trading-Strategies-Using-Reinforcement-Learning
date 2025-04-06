import yaml
import os
from loguru import logger

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging():
    logger.add("logs/trading.log", rotation="500 MB")
    return logger

def calculate_metrics(portfolio_values, initial_balance):
    # Implement Sharpe Ratio, Calmar Ratio, etc.
    pass