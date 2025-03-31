"""
Test script for Alpaca WebSocket functionality.
Helps diagnose issues with WebSocket connections.
"""
import time
import logging
import json
import threading
import os
from dotenv import load_dotenv
from api.alpaca_api import AlpacaAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('websocket_test')

# Load environment variables
load_dotenv()

def main():
    """Run the WebSocket test."""
    logger.info("Starting WebSocket connection test...")
    
    # Create a simple API instance with default config
    config = {
        'data_optimization': {'cache_ttl': 60},
        'trading': {'supported_pairs': ['ETH/USD']}
    }
    api = AlpacaAPI(config)
    logger.info("Created AlpacaAPI instance")
    
    # Set test parameters
    symbols = ['ETH/USD']
    timeout = 5.0  # 5 seconds timeout
    
    logger.info(f"Testing WebSocket connection for symbols: {symbols}")
    
    # Create a flag to track connection success
    connection_success = False
    
    def websocket_test():
        nonlocal connection_success
        try:
            # Start WebSocket with timeout
            logger.info(f"Starting WebSocket with {timeout}s timeout...")
            result = api.start_websocket(symbols, timeout=timeout, non_blocking=False)
            
            if result:
                logger.info("✅ WebSocket connection successful!")
                connection_success = True
            else:
                logger.error("❌ WebSocket connection failed")
        except Exception as e:
            logger.error(f"❌ WebSocket test error: {e}")
            logger.exception("Exception details:")
    
    # Create and start the test thread
    test_thread = threading.Thread(target=websocket_test, daemon=True)
    test_thread.start()
    
    # Wait for the thread to complete with a timeout
    wait_time = timeout + 2.0  # Allow a little extra time
    logger.info(f"Waiting up to {wait_time}s for test to complete...")
    test_thread.join(timeout=wait_time)
    
    # Check results
    if test_thread.is_alive():
        logger.error("❌ Test timed out! WebSocket connection is likely hanging")
    elif connection_success:
        logger.info("✅ Test completed successfully")
    else:
        logger.error("❌ Test completed with errors")
    
    # Clean up
    if hasattr(api, '_ws_client') and api._ws_client:
        logger.info("Stopping WebSocket connection...")
        api.stop_websocket()
    
    logger.info("Test complete")
    
if __name__ == "__main__":
    main() 