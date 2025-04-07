"""
API module for trading application.
Contains integration with Alpaca API for stock market trading and price data APIs.
"""

# Removing direct import to avoid circular imports
# from api.alpaca_api import AlpacaAPI
# Adding price-related components to the exports

__all__ = [
    # APIs
    'AlpacaAPI',  # Keep in __all__ but don't import directly
    'AlchemyPriceAPI',  # Alchemy Price API
    'CoinGeckoAPI',  # CoinGecko API
    
    # Price providers
    'PriceProvider',
    'AlpacaPriceProvider',
    'CoinGeckoPriceProvider',
    'AlchemyPriceProvider',
    
    # Price aggregator
    'PriceAggregator'
] 