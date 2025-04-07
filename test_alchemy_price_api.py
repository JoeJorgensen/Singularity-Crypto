"""
Test script for Alchemy Price API integration.
This verifies that the Alchemy Price API is working correctly.
"""
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our API implementation
from api.alchemy_price_api import AlchemyPriceAPI
from on_chain.alchemy.client import AlchemyClient

def test_current_prices():
    """Test getting current cryptocurrency prices"""
    print("\n=== Testing Current Prices ===")
    
    # Initialize the API
    alchemy_price_api = AlchemyPriceAPI()
    
    # Test getting a single price
    print("\nTesting single price lookup:")
    eth_price = alchemy_price_api.get_current_price("ETH")
    print(f"ETH price: ${eth_price}")
    
    btc_price = alchemy_price_api.get_current_price("BTC")
    print(f"BTC price: ${btc_price}")
    
    # Test with USD suffix format
    eth_usd_price = alchemy_price_api.get_current_price("ETH/USD")
    print(f"ETH/USD price: ${eth_usd_price}")
    
    # Test batch price lookup
    print("\nTesting batch price lookup:")
    symbols = ["ETH", "BTC", "USDT", "USDC"]
    prices = alchemy_price_api.get_current_prices(symbols)
    
    for symbol, price in prices.items():
        print(f"{symbol} price: ${price}")
    
    # Test cache invalidation
    print("\nTesting cache invalidation:")
    start_time = time.time()
    alchemy_price_api.get_current_price("ETH")
    first_lookup_time = time.time() - start_time
    
    start_time = time.time()
    alchemy_price_api.get_current_price("ETH")  # Should use cache
    cached_lookup_time = time.time() - start_time
    
    print(f"First lookup time: {first_lookup_time:.6f} seconds")
    print(f"Cached lookup time: {cached_lookup_time:.6f} seconds")
    
    # Force refresh
    start_time = time.time()
    alchemy_price_api.get_current_price("ETH", force_refresh=True)
    force_refresh_time = time.time() - start_time
    print(f"Force refresh time: {force_refresh_time:.6f} seconds")

def test_historical_prices():
    """Test getting historical cryptocurrency prices"""
    print("\n=== Testing Historical Prices ===")
    
    # Initialize the API
    alchemy_price_api = AlchemyPriceAPI()
    
    # Test getting historical prices
    print("\nTesting historical price lookup:")
    eth_hist = alchemy_price_api.get_historical_prices("ETH", days=7, interval="1d")
    
    if eth_hist is not None and not eth_hist.empty:
        print(f"Retrieved {len(eth_hist)} historical price points for ETH")
        print("\nSample data:")
        print(eth_hist.head())
        
        # Plot the data
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(eth_hist.index, eth_hist['price'])
            plt.title('ETH Price History (7 Days)')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('eth_price_history.png')
            print("\nPlot saved to eth_price_history.png")
        except Exception as e:
            print(f"Error plotting data: {e}")
    else:
        print("No historical data received for ETH")
    
    # Test bars conversion
    print("\nTesting bars conversion:")
    eth_bars = alchemy_price_api.get_crypto_bars("ETH/USD", timeframe="1Hour", limit=24)
    
    if not eth_bars.empty:
        print(f"Retrieved {len(eth_bars)} bars for ETH/USD")
        print("\nSample bars data:")
        print(eth_bars.head())
        
        # Plot the candlestick chart if available
        try:
            plt.figure(figsize=(12, 6))
            
            # Create a date list for the x-axis
            dates = eth_bars['datetime'].astype(str)
            
            # Calculate middle points for candles
            width = 0.6
            up_indices = eth_bars['close'] >= eth_bars['open']
            down_indices = eth_bars['close'] < eth_bars['open']
            
            # Plot up candles
            plt.bar(
                dates[up_indices],
                eth_bars['close'][up_indices] - eth_bars['open'][up_indices],
                bottom=eth_bars['open'][up_indices],
                width=width,
                color='green'
            )
            
            # Plot down candles
            plt.bar(
                dates[down_indices],
                eth_bars['close'][down_indices] - eth_bars['open'][down_indices],
                bottom=eth_bars['open'][down_indices],
                width=width,
                color='red'
            )
            
            # Plot high-low lines
            for i, date in enumerate(dates):
                plt.plot(
                    [date, date],
                    [eth_bars['low'].iloc[i], eth_bars['high'].iloc[i]],
                    color='black',
                    linewidth=1
                )
            
            plt.title('ETH/USD Hourly Candles')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('eth_candles.png')
            print("Candle chart saved to eth_candles.png")
        except Exception as e:
            print(f"Error plotting candlestick chart: {e}")
    else:
        print("No bar data received for ETH/USD")

def test_direct_alchemy_client():
    """Test the direct Alchemy client for price data"""
    print("\n=== Testing Direct Alchemy Client ===")
    
    # Initialize the client
    alchemy_client = AlchemyClient()
    
    # Test getting price by symbol
    print("\nTesting price lookup by symbol:")
    symbols = ["ETH", "BTC"]
    price_data = alchemy_client.get_token_price_by_symbol(symbols)
    print(f"Response structure: {list(price_data.keys())}")
    
    if "data" in price_data:
        for token_data in price_data["data"]:
            symbol = token_data.get("symbol", "Unknown")
            if "prices" in token_data and token_data["prices"]:
                for price_info in token_data["prices"]:
                    if price_info["currency"] == "USD":
                        print(f"{symbol} price: ${price_info['value']}")
                        break
    
    # Test getting historical prices
    print("\nTesting historical price lookup:")
    try:
        end_time = int(time.time())
        start_time = end_time - (7 * 24 * 60 * 60)  # 7 days ago
        
        hist_data = alchemy_client.get_historical_token_prices(
            symbol_or_address="ETH",
            is_address=False,
            start_timestamp=start_time,
            end_timestamp=end_time,
            interval="1d"
        )
        
        if "data" in hist_data and hist_data["data"]:
            print(f"Retrieved {len(hist_data['data'])} historical price points")
            print("Sample data:")
            for i, point in enumerate(hist_data["data"][:3]):  # Show first 3 points
                timestamp = point.get("timestamp", 0)
                price = point.get("price", "Unknown")
                date_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                print(f"{date_str}: ${price}")
    except Exception as e:
        print(f"Error getting historical prices: {e}")

def run_all_tests():
    """Run all tests"""
    print("==== ALCHEMY PRICE API TESTS ====")
    
    # Check if ALCHEMY_API_KEY is set
    api_key = os.getenv("ALCHEMY_API_KEY")
    if not api_key:
        print("ERROR: ALCHEMY_API_KEY environment variable not set. Tests cannot run.")
        return
    
    # Run the tests
    try:
        test_current_prices()
        test_historical_prices()
        test_direct_alchemy_client()
        
        print("\n==== ALL TESTS COMPLETED ====")
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")

if __name__ == "__main__":
    run_all_tests() 