from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from src.trading_env import TradingEnv
import numpy as np

def train_drl_model(env):
    env = DummyVecEnv([lambda: env])
    ppo_model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
    sac_model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    ppo_model.learn(total_timesteps=10000)
    sac_model.learn(total_timesteps=10000)

    return ppo_model, sac_model

def backtest(model, env, data):
    obs = env.reset()
    portfolio_values = [env.initial_balance]
    rewards = []

    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        rewards.append(reward)

        if done:
            break

    final_value = portfolio_values[-1]
    annualized_roi = ((final_value / env.initial_balance) ** (252 / len(data))) - 1
    max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / env.initial_balance
    sharpe_ratio = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252)  # Avoid division by zero
    calmar_ratio = annualized_roi / max_drawdown if max_drawdown > 0 else float('inf')

    return {
        "Annualized ROI": annualized_roi,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "Calmar Ratio": calmar_ratio
    }

def evaluate_model(model, env, episodes=10):
    for episode in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
        print(f"Episode {episode + 1} finished with portfolio value: {info['portfolio_value']}")
    env.close()