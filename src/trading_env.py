import numpy as np
import pandas as pd
import gym
from gym import spaces
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def load_market_data(symbol="AAPL", start_date="2018-01-01", end_date="2024-01-01"):
    df = yf.download(symbol, start=start_date, end=end_date)
    df['returns'] = df['Close'].pct_change()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    return df.dropna()

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(prices, slow=26, fast=12):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    return macd

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.transaction_cost = transaction_cost
        self.current_step = 0
        self.max_steps = len(data) - 1

        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(data)

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        return self.scaled_data[self.current_step]

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps

        prev_price = self.data.iloc[self.current_step - 1]['PC1'] if 'PC1' in self.data.columns else self.data.iloc[self.current_step - 1][0]
        current_price = self.data.iloc[self.current_step]['PC1'] if 'PC1' in self.data.columns else self.data.iloc[self.current_step][0]

        reward = 0
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            cost = current_price * (1 + self.transaction_cost)
            self.balance -= cost
            reward -= cost * 0.01

        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            profit = (current_price - prev_price) * self.position
            self.balance += current_price * (1 - self.transaction_cost)
            reward += profit - (current_price * self.transaction_cost)

        portfolio_value = self.balance + (self.position * current_price)
        info = {"portfolio_value": portfolio_value}

        # Risk-sensitive reward
        max_drawdown = self.calculate_max_drawdown()
        annualized_return = (portfolio_value / self.initial_balance) ** (252 / self.current_step) - 1
        reward += annualized_return - (max_drawdown * 0.5)

        return self.scaled_data[self.current_step], reward, done, info

    def calculate_max_drawdown(self):
        portfolio_values = [self.initial_balance]
        for i in range(self.current_step):
            price = self.data.iloc[i]['PC1'] if 'PC1' in self.data.columns else self.data.iloc[i][0]
            value = self.balance + (self.position * price)
            portfolio_values.append(value)
        return np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / self.initial_balance

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")

    def close(self):
        pass