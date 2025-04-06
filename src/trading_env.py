import numpy as np
import pandas as pd
import gym
from gym import spaces
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def load_market_data(symbol="AAPL", start_date="2018-01-01", end_date="2023-12-31"):  # Adjust end date
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data downloaded for {symbol}")
        df['returns'] = df['Close'].pct_change()
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'] = compute_macd(df['Close'])
        return df.dropna()
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()  # Return empty DataFrame if download fails

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
    

def compute_macd(prices, slow=26, fast=12, signal=9):
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
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
        # Get current price and portfolio value before action
        current_price = self.data.iloc[self.current_step]['Close']
        portfolio_value_before = self.balance + (self.position * current_price)
        
        # Initialize info dictionary
        info = {}
        
        # Process action (0: hold, 1: buy, 2: sell)
        if action == 1:  # Buy
            if self.balance > 0:
                # Calculate max shares we can buy
                max_shares = int(self.balance // (current_price * (1 + self.transaction_cost)))
                if max_shares > 0:
                    # Buy shares
                    cost = max_shares * current_price * (1 + self.transaction_cost)
                    self.balance -= cost
                    self.position += max_shares
                    info['action'] = 'buy'
                    info['shares'] = max_shares
                    info['cost'] = cost
        
        elif action == 2:  # Sell
            if self.position > 0:
                # Sell all shares
                sale_value = self.position * current_price * (1 - self.transaction_cost)
                self.balance += sale_value
                info['action'] = 'sell'
                info['shares'] = self.position
                info['value'] = sale_value
                self.position = 0
        
        else:  # Hold
            info['action'] = 'hold'
        
        # Move to the next time step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Calculate portfolio value after action and next step
        if not done:
            next_price = self.data.iloc[self.current_step]['Close']
            portfolio_value_after = self.balance + (self.position * next_price)
            next_observation = self.scaled_data[self.current_step]
        else:
            # If episode is done, use current price for final valuation
            portfolio_value_after = self.balance + (self.position * current_price)
            next_observation = self.scaled_data[self.current_step - 1]
        
        # Calculate reward as change in portfolio value
        reward = portfolio_value_after - portfolio_value_before
        
        info['portfolio_value'] = portfolio_value_after
        info['balance'] = self.balance
        info['position'] = self.position
        
        return next_observation, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass