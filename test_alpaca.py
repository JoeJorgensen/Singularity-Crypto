"""
Test script for Alpaca API connection and data retrieval.
"""
import asyncio
import logging
from dotenv import load_dotenv
from api.alpaca_api import AlpacaAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_websocket_connection():
    """Test WebSocket connection to Alpaca."""
    logger.info("Testing WebSocket connection")
    
    # Initialize API with an empty config
    api = AlpacaAPI({})
    
    # Start WebSocket connection
    api.start_websocket(['ETH/USD'])
    
    # Wait for the connection to establish and data to flow
    logger.info("Waiting for WebSocket data...")
    await asyncio.sleep(5)
    
    # Check the data received from WebSocket
    data = api.get_latest_websocket_data('ETH/USD')
    logger.info(f"WebSocket data: {data}")
    
    return data

async def test_bars_retrieval():
    """Test retrieving historical bars data."""
    logger.info("Testing historical bars retrieval")
    
    # Initialize API with an empty config
    api = AlpacaAPI({})
    
    # Retrieve bars data
    logger.info("Fetching bars data...")
    bars = await api.get_crypto_bars_async('ETH/USD', '1Min', 10)
    
    # Log information about the bars data
    logger.info(f"Bars data shape: {bars.shape}")
    if not bars.empty:
        logger.info(f"Bars data columns: {bars.columns.tolist()}")
        logger.info(f"First few rows: {bars.head(2)}")
    else:
        logger.warning("Bars data is empty")
    
    return bars

async def main():
    """Main function to run the tests."""
    load_dotenv()
    
    # Test WebSocket connection
    ws_data = await test_websocket_connection()
    
    # Test bars retrieval
    bars_data = await test_bars_retrieval()
    
    # Print summary
    logger.info("Tests completed")
    logger.info(f"WebSocket connected: {bool(ws_data and ws_data.get('latest_bar'))}")
    logger.info(f"Bars data retrieved: {not bars_data.empty}")

if __name__ == "__main__":
    asyncio.run(main()) 