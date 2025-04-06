from src.trading_env import TradingEnv, load_market_data
from src.models import train_drl_model, backtest, evaluate_model
from src.data_processing import augment_data, apply_pca, wavelet_denoise
from src.api_integration import IBapi, connect_to_ib
from src.utils import load_config, setup_logging
import os
import pandas as pd

def main():
    # Load configuration
    config = load_config()
    logger = setup_logging()

    # Load and preprocess data
    logger.info("Loading market data...")
    raw_data = load_market_data(symbol=config['data']['symbol'], 
                               start_date=config['data']['start_date'], 
                               end_date=config['data']['end_date'])
    
    logger.info("Augmenting and processing data...")
    augmented_data = augment_data(raw_data[['Close', 'RSI', 'MACD']])
    denoised_data = wavelet_denoise(raw_data['Close'].values)
    reduced_features, pca_model = apply_pca(augmented_data)

    # Create trading environment
    env = TradingEnv(pd.DataFrame(reduced_features, columns=['PC1', 'PC2', 'PC3']), 
                     initial_balance=config['trading']['initial_balance'])

    # Train DRL models
    logger.info("Training DRL models...")
    ppo_model, sac_model = train_drl_model(env)

    # Backtest and evaluate
    logger.info("Backtesting models...")
    ppo_results = backtest(ppo_model, env, reduced_features)
    sac_results = backtest(sac_model, env, reduced_features)
    logger.info(f"PPO Results: {ppo_results}")
    logger.info(f"SAC Results: {sac_results}")

    # Evaluate models
    evaluate_model(ppo_model, env, episodes=5)

    # Deploy to live trading
    logger.info("Connecting to Interactive Brokers for live trading...")
    ib = connect_to_ib(host=os.getenv('IB_HOST'), port=int(os.getenv('IB_PORT')), client_id=int(os.getenv('IB_CLIENT_ID')))
    ib.run()

if __name__ == "__main__":
    main()