"""
Test script to verify signal generation and values.
"""
import logging
import pandas as pd
from trading_strategy import TradingStrategy
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_signal_generation():
    """Test signal generation and print results"""
    logger.info("Loading configuration...")
    # Load config directly from file
    config_path = os.path.join('config', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info("Initializing trading strategy...")
    trading_strategy = TradingStrategy(config)
    
    symbol = "ETH/USD"
    timeframe = "1H"
    
    logger.info(f"Getting market data for {symbol} on {timeframe} timeframe...")
    market_data = trading_strategy.get_market_data(symbol, timeframe)
    
    logger.info(f"Getting sentiment data for {symbol}...")
    sentiment_data = trading_strategy.get_sentiment_data(symbol, market_data)
    
    logger.info("Generating signals...")
    signals = trading_strategy.generate_signals(market_data, sentiment_data)
    
    # Print all signal keys and values
    logger.info("Signal data:")
    for key, value in signals.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
            
    logger.info("Running a test trading cycle...")
    cycle_result = trading_strategy.run_trading_cycle(symbol, timeframe)
    
    # Extract signals from cycle result
    if 'signals' in cycle_result:
        logger.info("Signal data from trading cycle:")
        for key, value in cycle_result['signals'].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    logger.info("Test completed.")

if __name__ == "__main__":
    logger.info("Starting signal test...")
    test_signal_generation() 