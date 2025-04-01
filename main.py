from src.trading_env import TradingEnv, load_market_data
from src.models import train_drl_model, backtest
from src.data_processing import augment_data, apply_pca, wavelet_denoise
from src.api_integration import IBapi
from src.utils import load_config

def main():
    # Load configuration
    config = load_config()

    # Load and preprocess data
    raw_data = load_market_data(symbol=config['data']['symbol'], 
                               start_date=config['data']['start_date'], 
                               end_date=config['data']['end_date'])
    
    # Apply data processing
    augmented_data = augment_data(raw_data[['Close', 'RSI', 'MACD']])
    denoised_data = wavelet_denoise(raw_data['Close'].values)
    reduced_features, _ = apply_pca(augmented_data)

    # Create trading environment
    env = TradingEnv(pd.DataFrame(reduced_features, columns=['PC1', 'PC2', 'PC3']), 
                     initial_balance=config['trading']['initial_balance'])

    # Train DRL models
    ppo_model, sac_model = train_drl_model(env)

    # Backtest
    results = backtest(ppo_model, env, reduced_features)
    print("Backtest Results:", results)

    # Deploy to live trading
    ib = IBapi()
    ib.connect(config['api']['host'], config['api']['port'], config['api']['client_id'])
    ib.run()

if __name__ == "__main__":
    main()