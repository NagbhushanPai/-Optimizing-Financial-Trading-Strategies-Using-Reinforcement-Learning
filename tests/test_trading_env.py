import pytest
from src.trading_env import TradingEnv, load_market_data

def test_trading_env_init():
    data = load_market_data("AAPL")
    env = TradingEnv(data)
    assert env.action_space.n == 3
    assert env.observation_space.shape[0] == len(data.columns)

def test_reset():
    data = load_market_data("AAPL")
    env = TradingEnv(data)
    obs = env.reset()
    assert obs.shape == (len(data.columns),)
    assert env.balance == env.initial_balance
    assert env.position == 0
    assert env.current_step == 0

def test_step():
    data = load_market_data("AAPL")
    env = TradingEnv(data)
    obs = env.reset()
    obs, reward, done, info = env.step(0)  # Hold action
    assert not done  # Assuming short data, adjust if needed
    assert isinstance(reward, (int, float))
    assert "portfolio_value" in info