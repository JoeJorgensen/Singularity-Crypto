#!/usr/bin/env python3
"""
Script to force a refresh of the sentiment data 
and verify it's being picked up correctly.
"""
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from api.finnhub_api import FinnhubAPI
from trading_strategy import TradingStrategy
from api.alpaca_api import AlpacaAPI
from api.openai_api import OpenAIAPI

# Load environment variables
load_dotenv()

def log(message):
    """Simple logging function with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_config():
    """Load configuration from config files."""
    try:
        # Load trading config
        with open('config/trading_config.json', 'r') as f:
            trading_config = json.load(f)
        
        # Load main config
        with open('config/config.json', 'r') as f:
            main_config = json.load(f)
        
        # Merge configs
        merged_config = main_config.copy()
        merged_config.update(trading_config)
        
        return merged_config
    except Exception as e:
        log(f"Error loading config: {e}")
        return {}

def main():
    """Force a refresh of sentiment data and verify it's working."""
    log("Initializing APIs...")
    finnhub_api = FinnhubAPI()
    
    # Create alpaca config with credentials
    alpaca_config = {
        'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY', ''),
        'ALPACA_API_SECRET': os.getenv('ALPACA_API_SECRET', ''),
        'ALPACA_BASE_URL': os.getenv('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets'),
        'PAPER': True
    }
    alpaca_api = AlpacaAPI(config=alpaca_config)
    openai_api = OpenAIAPI()
    
    # Initialize trading strategy
    config = load_config()
    
    # Force config to have a very short sentiment TTL
    config['data_optimization'] = config.get('data_optimization', {})
    config['data_optimization']['cache_ttl'] = 1  # 1 second TTL to force refresh
    config['data_optimization']['sentiment_ttl'] = 1
    log(f"Set sentiment TTL to 1 second to force refreshes")
    
    strategy = TradingStrategy(config=config)
    
    # Test direct API call
    symbol = 'ETH/USD'
    base_symbol = symbol.split('/')[0]
    
    log(f"\nTesting direct API call for {symbol}...")
    direct_sentiment = finnhub_api.get_aggregate_sentiment(base_symbol)
    log(f"Direct FinnhubAPI sentiment for {base_symbol}:")
    log(f"  Score: {direct_sentiment.get('sentiment_score', 0)}")
    log(f"  News Count: {direct_sentiment.get('news_count', 0)}")
    log(f"  Source: {direct_sentiment.get('source', 'unknown')}")
    
    # Test through trading strategy
    log(f"\nTesting sentiment through TradingStrategy for {symbol}...")
    
    # Force clear the cache
    strategy.last_sentiment_cache_time = {}
    log(f"Cleared sentiment cache")
    
    # Get new sentiment data
    strategy_sentiment = strategy.get_sentiment_data(symbol)
    log(f"TradingStrategy sentiment for {symbol}:")
    log(f"  Score: {strategy_sentiment.get('sentiment_score', 0)}")
    log(f"  News Count: {strategy_sentiment.get('news_count', 0)}")
    log(f"  Source: {strategy_sentiment.get('source', 'unknown')}")
    
    # Test signal generation with sentiment
    log(f"\nTesting signal generation with sentiment...")
    market_data = strategy.get_market_data(symbol)
    
    # Force get sentiment with explicit log
    log(f"Explicitly fetching fresh sentiment data")
    fresh_sentiment = strategy.get_sentiment_data(symbol)
    log(f"Fresh sentiment data: {fresh_sentiment}")
    
    signals = strategy.generate_signals(market_data, fresh_sentiment)
    
    log(f"Generated signals for {symbol}:")
    log(f"  Signal: {signals.get('signal', 0)}")
    log(f"  Direction: {signals.get('signal_direction', 'unknown')}")
    log(f"  Strength: {signals.get('strength', 0)}")
    log(f"  Sentiment score: {signals.get('sentiment_score', 0)}")
    log(f"  Sentiment adjustment: {signals.get('sentiment_adjustment', 0)}")
    
    # Now check cached version
    log(f"\nTesting cached sentiment data...")
    cached_sentiment = strategy.get_sentiment_data(symbol)
    log(f"Cached sentiment for {symbol}:")
    log(f"  Score: {cached_sentiment.get('sentiment_score', 0)}")
    log(f"  News Count: {cached_sentiment.get('news_count', 0)}")
    
    # Compare to direct API again
    log(f"\nComparing direct API vs Strategy results:")
    log(f"  Direct API score: {direct_sentiment.get('sentiment_score', 0)}")
    log(f"  Strategy score: {strategy_sentiment.get('sentiment_score', 0)}")
    log(f"  Cached score: {cached_sentiment.get('sentiment_score', 0)}")
    
    log("\nRunning a direct trading cycle with the strategy to verify integration...")
    cycle_result = strategy.run_trading_cycle(symbol)
    log(f"Trading cycle result:")
    log(f"  Signal direction: {cycle_result.get('signals', {}).get('signal_direction', 'unknown')}")
    log(f"  Signal strength: {cycle_result.get('signals', {}).get('strength', 0)}")
    log(f"  Sentiment score: {cycle_result.get('signals', {}).get('sentiment_score', 0)}")
    log(f"  Sentiment adjustment: {cycle_result.get('signals', {}).get('sentiment_adjustment', 0)}")
    
    if strategy_sentiment.get('sentiment_score', 0) != 0:
        log("\nSUCCESS: Sentiment data is being properly utilized!")
    else:
        log("\nWARNING: Sentiment data may not be properly integrated.")
        log("Check logs/app.log for more details.")

if __name__ == "__main__":
    main() 