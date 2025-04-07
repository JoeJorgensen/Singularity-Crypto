"""
Price data example script.
Demonstrates how to use the price factory to get price data from different providers.
"""
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import our price factory
from api.price_factory import (
    create_price_provider,
    create_price_aggregator,
    get_default_price_provider
)

def example_single_provider():
    """Example of using a single price provider"""
    print("\n=== Example: Single Price Provider ===")
    
    # Create an Alchemy price provider
    alchemy_provider = create_price_provider("alchemy")
    if not alchemy_provider:
        print("Failed to create Alchemy provider, check your API key")
        return
    
    # Get current price for ETH
    eth_price = alchemy_provider.get_current_price("ETH")
    print(f"ETH price from Alchemy: ${eth_price}")
    
    # Get multiple prices
    symbols = ["ETH", "BTC", "LINK"]
    prices = alchemy_provider.get_current_prices(symbols)
    
    print("\nCurrent prices from Alchemy:")
    for symbol, price in prices.items():
        print(f"{symbol}: ${price}")

def example_price_aggregator():
    """Example of using the price aggregator with multiple providers"""
    print("\n=== Example: Price Aggregator ===")
    
    # Create a price aggregator with all available providers
    aggregator = create_price_aggregator()
    
    # Get current price for ETH with different strategies
    eth_price_weighted = aggregator.get_current_price("ETH", strategy="weighted")
    eth_price_first = aggregator.get_current_price("ETH", strategy="first_available")
    eth_price_median = aggregator.get_current_price("ETH", strategy="median")
    
    print(f"ETH price (weighted): ${eth_price_weighted}")
    print(f"ETH price (first available): ${eth_price_first}")
    print(f"ETH price (median): ${eth_price_median}")
    
    # Get multiple prices
    symbols = ["ETH", "BTC", "LINK"]
    prices = aggregator.get_current_prices(symbols)
    
    print("\nCurrent prices from aggregator:")
    for symbol, price in prices.items():
        print(f"{symbol}: ${price}")

def example_historical_data():
    """Example of getting historical price data"""
    print("\n=== Example: Historical Price Data ===")
    
    # Get the default price provider (based on configuration)
    provider = get_default_price_provider()
    
    # Get historical prices for ETH
    eth_hist = provider.get_historical_prices("ETH", days=7, interval="1d")
    
    if eth_hist is not None and not eth_hist.empty:
        print(f"Retrieved {len(eth_hist)} historical price points for ETH")
        print("\nSample data:")
        print(eth_hist.head())
        
        # Try to plot the data
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

def example_custom_aggregator():
    """Example of creating a custom aggregator configuration"""
    print("\n=== Example: Custom Aggregator Configuration ===")
    
    # Create a custom aggregator with specific providers and weights
    custom_weights = {
        "alchemy": 1.0,  # Primary source
        "coingecko": 0.5  # Secondary source
    }
    
    # Only use Alchemy and CoinGecko, not Alpaca
    provider_types = ["alchemy", "coingecko"]
    
    custom_aggregator = create_price_aggregator(
        provider_types=provider_types,
        custom_weights=custom_weights
    )
    
    # Get BTC price
    btc_price = custom_aggregator.get_current_price("BTC")
    print(f"BTC price from custom aggregator: ${btc_price}")

def run_examples():
    """Run all examples"""
    print("==== PRICE DATA EXAMPLES ====")
    
    # Check if required API keys are set
    alchemy_api_key = os.getenv("ALCHEMY_API_KEY")
    if not alchemy_api_key:
        print("WARNING: ALCHEMY_API_KEY not set. Some examples may not work.")
    
    # Run examples
    try:
        example_single_provider()
        example_price_aggregator()
        example_historical_data()
        example_custom_aggregator()
        
        print("\n==== ALL EXAMPLES COMPLETED ====")
    except Exception as e:
        print(f"\nERROR: Example failed with exception: {e}")

if __name__ == "__main__":
    run_examples() 