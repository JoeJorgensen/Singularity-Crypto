"""
Price Provider Interface for cryptocurrency price data.
Defines a common interface for different price data sources.
"""
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

import pandas as pd

from utils.logging_config import get_logger

# Get logger
logger = get_logger('price_provider')

class PriceProvider(ABC):
    """Abstract base class for price data providers"""
    
    @abstractmethod
    def get_current_price(self, symbol: str, force_refresh: bool = False) -> Optional[float]:
        """
        Get current price of a cryptocurrency in USD.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH", "BTC")
            force_refresh: If True, bypass cache
            
        Returns:
            Current price in USD or None if not available
        """
        pass
    
    @abstractmethod
    def get_current_prices(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, float]:
        """
        Get current prices for multiple cryptocurrencies in USD.
        
        Args:
            symbols: List of cryptocurrency symbols (e.g., ["ETH", "BTC"])
            force_refresh: If True, bypass cache
            
        Returns:
            Dictionary with symbol keys and price values
        """
        pass
    
    @abstractmethod
    def get_historical_prices(self, 
                            symbol: str, 
                            days: int = 30, 
                            interval: str = "1d", 
                            force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get historical prices for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH", "BTC")
            days: Number of days of historical data
            interval: Time interval (e.g., "1h", "1d")
            force_refresh: If True, bypass cache
            
        Returns:
            DataFrame with historical price data
        """
        pass
    
    @abstractmethod
    def get_crypto_bars(self, symbol: str, timeframe: str = "1Hour", limit: int = 100) -> pd.DataFrame:
        """
        Get historical bar data for a cryptocurrency, compatible with Alpaca API format.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH/USD", "BTC/USD")
            timeframe: Time interval (e.g., "1Min", "1Hour", "1Day")
            limit: Number of bars to return
            
        Returns:
            DataFrame with OHLCV bar data
        """
        pass
    
    @abstractmethod
    def clear_cache(self):
        """Clear all cached data"""
        pass

class AlpacaPriceProvider(PriceProvider):
    """Price provider implementation using Alpaca API"""
    
    def __init__(self, alpaca_api):
        """
        Initialize AlpacaPriceProvider.
        
        Args:
            alpaca_api: Initialized AlpacaAPI instance
        """
        self.alpaca_api = alpaca_api
        self.logger = get_logger('alpaca_price_provider')
        self.logger.info("Alpaca Price Provider initialized")
    
    def get_current_price(self, symbol: str, force_refresh: bool = False) -> Optional[float]:
        """
        Get current price using Alpaca API.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH/USD")
            force_refresh: If True, bypass cache
            
        Returns:
            Current price in USD or None if not available
        """
        try:
            # Ensure symbol has USD suffix
            if "/" not in symbol:
                symbol = f"{symbol}/USD"
                
            # Get the current price
            price = self.alpaca_api.get_current_price(symbol)
            return price
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol} from Alpaca: {e}")
            return None
    
    def get_current_prices(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, float]:
        """
        Get current prices for multiple cryptocurrencies using Alpaca API.
        
        Args:
            symbols: List of cryptocurrency symbols (e.g., ["ETH", "BTC"])
            force_refresh: If True, bypass cache
            
        Returns:
            Dictionary with symbol keys and price values
        """
        result = {}
        for symbol in symbols:
            price = self.get_current_price(symbol, force_refresh)
            if price is not None:
                # Store without USD suffix
                clean_symbol = symbol.split('/')[0] if '/' in symbol else symbol
                result[clean_symbol] = price
        return result
    
    def get_historical_prices(self, 
                            symbol: str, 
                            days: int = 30, 
                            interval: str = "1d", 
                            force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get historical prices using Alpaca API.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH")
            days: Number of days of historical data
            interval: Time interval (e.g., "1h", "1d")
            force_refresh: If True, bypass cache
            
        Returns:
            DataFrame with historical price data
        """
        try:
            # Convert days and interval to Alpaca-compatible format
            if interval == "1d":
                timeframe = "1Day"
            elif interval == "1h":
                timeframe = "1Hour"
            else:
                # Default to 1 hour
                timeframe = "1Hour"
                
            # Ensure symbol has USD suffix
            if "/" not in symbol:
                symbol = f"{symbol}/USD"
                
            # Calculate limit based on days and timeframe
            if timeframe == "1Day":
                limit = days
            elif timeframe == "1Hour":
                limit = days * 24
            else:
                limit = days * 24
                
            # Get historical bars
            bars = self.alpaca_api.get_crypto_bars(symbol, timeframe, limit)
            
            if bars is None or bars.empty:
                return None
                
            # Convert to the expected format
            prices_df = pd.DataFrame({
                'timestamp': bars['datetime'],
                'price': bars['close']
            })
            prices_df.set_index('timestamp', inplace=True)
            
            return prices_df
        except Exception as e:
            self.logger.error(f"Error getting historical prices for {symbol} from Alpaca: {e}")
            return None
    
    def get_crypto_bars(self, symbol: str, timeframe: str = "1Hour", limit: int = 100) -> pd.DataFrame:
        """
        Get historical bar data using Alpaca API.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH/USD")
            timeframe: Time interval (e.g., "1Min", "1Hour", "1Day")
            limit: Number of bars to return
            
        Returns:
            DataFrame with OHLCV bar data
        """
        try:
            # Ensure symbol has USD suffix
            if "/" not in symbol:
                symbol = f"{symbol}/USD"
                
            # Get bars directly from Alpaca
            bars = self.alpaca_api.get_crypto_bars(symbol, timeframe, limit)
            return bars if bars is not None else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error getting crypto bars for {symbol} from Alpaca: {e}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear all cached data"""
        # Alpaca API might not have a direct clear_cache method
        # We could implement this if Alpaca API has cache-clearing functionality
        pass

class CoinGeckoPriceProvider(PriceProvider):
    """Price provider implementation using CoinGecko API"""
    
    def __init__(self, coingecko_api):
        """
        Initialize CoinGeckoPriceProvider.
        
        Args:
            coingecko_api: Initialized CoinGeckoAPI instance
        """
        self.coingecko_api = coingecko_api
        self.logger = get_logger('coingecko_price_provider')
        self.logger.info("CoinGecko Price Provider initialized")
    
    def get_current_price(self, symbol: str, force_refresh: bool = False) -> Optional[float]:
        """
        Get current price using CoinGecko API.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH")
            force_refresh: If True, bypass cache
            
        Returns:
            Current price in USD or None if not available
        """
        try:
            return self.coingecko_api.get_current_price(symbol, force_refresh)
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol} from CoinGecko: {e}")
            return None
    
    def get_current_prices(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, float]:
        """
        Get current prices for multiple cryptocurrencies using CoinGecko API.
        
        Args:
            symbols: List of cryptocurrency symbols (e.g., ["ETH", "BTC"])
            force_refresh: If True, bypass cache
            
        Returns:
            Dictionary with symbol keys and price values
        """
        try:
            return self.coingecko_api.get_current_prices(symbols, force_refresh)
        except Exception as e:
            self.logger.error(f"Error getting current prices from CoinGecko: {e}")
            return {}
    
    def get_historical_prices(self, 
                            symbol: str, 
                            days: int = 30, 
                            interval: str = "1d", 
                            force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get historical prices using CoinGecko API.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH")
            days: Number of days of historical data
            interval: Time interval (ignored in CoinGecko)
            force_refresh: If True, bypass cache
            
        Returns:
            DataFrame with historical price data
        """
        try:
            return self.coingecko_api.get_historical_prices(symbol, days, interval, force_refresh)
        except Exception as e:
            self.logger.error(f"Error getting historical prices for {symbol} from CoinGecko: {e}")
            return None
    
    def get_crypto_bars(self, symbol: str, timeframe: str = "1Hour", limit: int = 100) -> pd.DataFrame:
        """
        Get historical bar data using CoinGecko API.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH/USD")
            timeframe: Time interval (e.g., "1Min", "1Hour", "1Day")
            limit: Number of bars to return
            
        Returns:
            DataFrame with OHLCV bar data
        """
        try:
            return self.coingecko_api.get_crypto_bars(symbol, timeframe, limit)
        except Exception as e:
            self.logger.error(f"Error getting crypto bars for {symbol} from CoinGecko: {e}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            self.coingecko_api.clear_cache()
        except Exception as e:
            self.logger.error(f"Error clearing CoinGecko cache: {e}")

class AlchemyPriceProvider(PriceProvider):
    """Price provider implementation using Alchemy API (if available)"""
    
    def __init__(self, alchemy_price_api):
        """
        Initialize AlchemyPriceProvider.
        
        Args:
            alchemy_price_api: Initialized AlchemyPriceAPI instance
        """
        self.alchemy_price_api = alchemy_price_api
        self.logger = get_logger('alchemy_price_provider')
        self.logger.info("Alchemy Price Provider initialized")
    
    def get_current_price(self, symbol: str, force_refresh: bool = False) -> Optional[float]:
        """
        Get current price using Alchemy API.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH")
            force_refresh: If True, bypass cache
            
        Returns:
            Current price in USD or None if not available
        """
        try:
            return self.alchemy_price_api.get_current_price(symbol, force_refresh)
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol} from Alchemy: {e}")
            return None
    
    def get_current_prices(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, float]:
        """
        Get current prices for multiple cryptocurrencies using Alchemy API.
        
        Args:
            symbols: List of cryptocurrency symbols (e.g., ["ETH", "BTC"])
            force_refresh: If True, bypass cache
            
        Returns:
            Dictionary with symbol keys and price values
        """
        try:
            return self.alchemy_price_api.get_current_prices(symbols, force_refresh)
        except Exception as e:
            self.logger.error(f"Error getting current prices from Alchemy: {e}")
            return {}
    
    def get_historical_prices(self, 
                            symbol: str, 
                            days: int = 30, 
                            interval: str = "1d", 
                            force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get historical prices using Alchemy API.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH")
            days: Number of days of historical data
            interval: Time interval (e.g., "1h", "1d")
            force_refresh: If True, bypass cache
            
        Returns:
            DataFrame with historical price data
        """
        try:
            return self.alchemy_price_api.get_historical_prices(symbol, days, interval, force_refresh)
        except Exception as e:
            self.logger.error(f"Error getting historical prices for {symbol} from Alchemy: {e}")
            return None
    
    def get_crypto_bars(self, symbol: str, timeframe: str = "1Hour", limit: int = 100) -> pd.DataFrame:
        """
        Get historical bar data using Alchemy API.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH/USD")
            timeframe: Time interval (e.g., "1Min", "1Hour", "1Day")
            limit: Number of bars to return
            
        Returns:
            DataFrame with OHLCV bar data
        """
        try:
            return self.alchemy_price_api.get_crypto_bars(symbol, timeframe, limit)
        except Exception as e:
            self.logger.error(f"Error getting crypto bars for {symbol} from Alchemy: {e}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            self.alchemy_price_api.clear_cache()
        except Exception as e:
            self.logger.error(f"Error clearing Alchemy cache: {e}") 