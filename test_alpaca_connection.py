#!/usr/bin/env python3
"""Test script to verify Alpaca API connection with the fixes.
"""
import os
import time
import logging
import json
from api.alpaca_api import AlpacaAPI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('alpaca_test')

def test_alpaca_connection():
    """Test Alpaca API connection with the correct feed settings."""
    # Create API instance
    logger.info("Initializing Alpaca API...")
    api = AlpacaAPI()
    
    # Test account connection
    logger.info("Testing account connection...")
    try:
        account = api.get_account()
        logger.info(f"Connected to Alpaca API. Account ID: {account.id}, Status: {account.status}")
    except Exception as e:
        logger.error(f"Failed to connect to account: {e}")
    
    # Test historical data
    logger.info("Testing historical data retrieval...")
    symbols = ['ETH/USD', 'BTC/USD']
    for symbol in symbols:
        try:
            logger.info(f"Fetching historical data for {symbol}...")
            bars = api.get_crypto_bars(symbol, timeframe='5Min', limit=10)
            if bars.empty:
                logger.warning(f"No data received for {symbol}")
            else:
                logger.info(f"Retrieved {len(bars)} bars for {symbol}")
                logger.info(f"Last bar: {bars.iloc[-1].to_dict() if not bars.empty else 'None'}")
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
    
    # Test websocket connection
    logger.info("Testing websocket connection...")
    try:
        # Start websocket in non-blocking mode
        logger.info("Starting websocket connection...")
        success = api.start_websocket(symbols=symbols, timeout=10.0, non_blocking=True)
        if success:
            logger.info("Successfully connected to websocket")
            
            # Wait a moment for data
            logger.info("Waiting for websocket data...")
            time.sleep(15)
            
            # Check if we received any data
            for symbol in symbols:
                # Strip '/' for data lookup if needed
                lookup_symbol = symbol.replace('/', '')
                data = api.get_latest_websocket_data(lookup_symbol)
                if data:
                    logger.info(f"Received data for {symbol}: {json.dumps(data, default=str)}")
                else:
                    logger.warning(f"No websocket data received for {symbol}")
            
            # Stop the websocket
            logger.info("Stopping websocket connection...")
            api.stop_websocket()
        else:
            logger.error("Failed to connect to websocket")
    except Exception as e:
        logger.error(f"Error in websocket test: {e}")
    
    logger.info("Alpaca API test completed")

if __name__ == "__main__":
    test_alpaca_connection() 