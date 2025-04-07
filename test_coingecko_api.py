"""
Test script for CoinGecko API integration.
This verifies that the CoinGecko API is working correctly.
"""
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our API implementation
from api.coingecko_api import CoinGeckoAPI

def test_current_prices():
    """Test getting current cryptocurrency prices"""
    print("\n=== Testing Current Prices ===")
    
    # Initialize the API
    coingecko_api = CoinGeckoAPI()
    
    # Test getting a single price
    print("\nTesting single price lookup:")
    eth_price = coingecko_api.get_current_price("ETH")
    print(f"ETH price: ${eth_price}")
    
    btc_price = coingecko_api.get_current_price("BTC")
    print(f"BTC price: ${btc_price}")
    
    # Test with USD suffix format
    eth_usd_price = coingecko_api.get_current_price("ETH/USD")
    print(f"ETH/USD price: ${eth_usd_price}")
    
    # Test batch price lookup
    print("\nTesting batch price lookup:")
    symbols = ["ETH", "BTC", "USDT", "USDC"]
    prices = coingecko_api.get_current_prices(symbols)
    
    for symbol, price in prices.items():
        print(f"{symbol} price: ${price}")
    
    # Test cache invalidation
    print("\nTesting cache invalidation:")
    start_time = time.time()
    coingecko_api.get_current_price("ETH")
    first_lookup_time = time.time() - start_time
    
    start_time = time.time()
    coingecko_api.get_current_price("ETH")  # Should use cache
    cached_lookup_time = time.time() - start_time
    
    print(f"First lookup time: {first_lookup_time:.6f} seconds")
    print(f"Cached lookup time: {cached_lookup_time:.6f} seconds")
    
    # Force refresh
    start_time = time.time()
    coingecko_api.get_current_price("ETH", force_refresh=True)
    force_refresh_time = time.time() - start_time
    print(f"Force refresh time: {force_refresh_time:.6f} seconds")

def test_historical_prices():
    """Test getting historical cryptocurrency prices"""
    print("\n=== Testing Historical Prices ===")
    
    # Initialize the API
    coingecko_api = CoinGeckoAPI()
    
    # Test getting historical prices
    print("\nTesting historical price lookup:")
    eth_hist = coingecko_api.get_historical_prices("ETH", days=7)
    
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
            plt.savefig('eth_price_history_coingecko.png')
            print("\nPlot saved to eth_price_history_coingecko.png")
        except Exception as e:
            print(f"Error plotting data: {e}")
    else:
        print("No historical data received for ETH")
    
    # Test bars conversion
    print("\nTesting bars conversion:")
    eth_bars = coingecko_api.get_crypto_bars("ETH/USD", timeframe="1Hour", limit=24)
    
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
            
            plt.title('ETH/USD Hourly Candles (CoinGecko)')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('eth_candles_coingecko.png')
            print("Candle chart saved to eth_candles_coingecko.png")
        except Exception as e:
            print(f"Error plotting candlestick chart: {e}")
    else:
        print("No bar data received for ETH/USD")

def test_market_data():
    """Test getting detailed market data for cryptocurrencies"""
    print("\n=== Testing Market Data ===")
    
    # Initialize the API
    coingecko_api = CoinGeckoAPI()
    
    # Test getting market data
    print("\nTesting market data lookup:")
    eth_market = coingecko_api.get_market_data("ETH")
    
    if eth_market:
        print("Market data for ETH:")
        # Print a selection of the market data
        print(f"Name: {eth_market.get('name')}")
        print(f"Price: ${eth_market.get('price_usd')}")
        print(f"Market Cap: ${eth_market.get('market_cap_usd')}")
        print(f"24h Volume: ${eth_market.get('volume_24h_usd')}")
        print(f"24h Change: {eth_market.get('price_change_percentage_24h')}%")
        print(f"7d Change: {eth_market.get('price_change_percentage_7d')}%")
        print(f"Market Cap Rank: #{eth_market.get('market_cap_rank')}")
        print(f"Circulating Supply: {eth_market.get('circulating_supply')} ETH")
    else:
        print("No market data received for ETH")

def run_all_tests():
    """Run all tests"""
    print("==== COINGECKO API TESTS ====")
    
    # Run the tests
    try:
        test_current_prices()
        test_historical_prices()
        test_market_data()
        
        print("\n==== ALL TESTS COMPLETED ====")
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")

if __name__ == "__main__":
    run_all_tests() 