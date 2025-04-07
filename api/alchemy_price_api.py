"""
Alchemy Price API interface for cryptocurrency price data.
Uses Alchemy's Prices API for both current and historical price data.
"""
import os
import time
import json
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import requests
from dotenv import load_dotenv

from utils.logging_config import get_logger
from on_chain.alchemy.client import AlchemyClient

# Load environment variables
load_dotenv()

# Get logger
logger = get_logger('alchemy_price_api')

class AlchemyPriceAPI:
    """Wrapper for Alchemy's Prices API for cryptocurrency price data"""
    
    def __init__(self, config=None):
        """
        Initialize AlchemyPriceAPI client.
        
        Args:
            config: Configuration dictionary
        """
        # Set up logger
        self.logger = logger
        
        # Load configuration
        if config is None:
            config = {}
        
        # Store config
        self.config = config
        
        # Initialize Alchemy client
        try:
            self.alchemy_client = AlchemyClient()
            self.logger.info("Alchemy Price API initialized")
        except Exception as e:
            self.logger.error(f"Error initializing Alchemy client: {e}")
            raise
        
        # Cache settings
        self.cache_ttl = config.get('cache_ttl', 60)  # seconds for current price cache
        self.historical_cache_ttl = config.get('historical_cache_ttl', 3600)  # 1 hour for historical data
        
        # Rate limiting settings - 300 requests per hour = 12 seconds between requests on average
        rate_limit = config.get('rate_limit', {})
        self.requests_per_hour = rate_limit.get('requests_per_hour', 300)
        self.min_request_interval = rate_limit.get('min_interval', 12.0)  # seconds
        
        # Track request timestamps for rolling window rate limiting
        self.request_timestamps = []
        self.max_request_window = 3600  # 1 hour window in seconds
        self.last_request_time = 0  # Initialize last_request_time
        
        # Cache for last known valid prices
        self.last_known_prices = {}
        
        # Supported networks
        self.supported_networks = [
            "eth-mainnet", "polygon-mainnet", "optimism-mainnet", 
            "arbitrum-mainnet", "base-mainnet"
        ]
        
        # Token addresses for common cryptocurrencies on Ethereum mainnet
        self.token_addresses = {
            "ETH": {
                "network": "eth-mainnet",
                "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"  # WETH
            },
            "BTC": {
                "network": "eth-mainnet",
                "address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"  # WBTC
            },
            "USDT": {
                "network": "eth-mainnet",
                "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7"
            },
            "USDC": {
                "network": "eth-mainnet",
                "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
            }
        }
    
    def _rate_limit(self):
        """Implement rate limiting to avoid hitting the 300 requests per hour limit"""
        current_time = time.time()
        
        # Basic interval limiting
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            self.logger.debug(f"Rate limiting - sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        # Advanced rolling window rate limiting
        # Remove timestamps older than our window
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                 if current_time - ts <= self.max_request_window]
        
        # Check if we're at the rate limit
        if len(self.request_timestamps) >= self.requests_per_hour:
            # Calculate time until oldest request falls out of window
            oldest_timestamp = min(self.request_timestamps)
            wait_time = oldest_timestamp + self.max_request_window - current_time
            
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached ({self.requests_per_hour}/hour). "
                                   f"Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time + 0.1)  # Add a small buffer
        
        # Record this request timestamp
        self.last_request_time = time.time()
        self.request_timestamps.append(self.last_request_time)
    
    @lru_cache(maxsize=32)
    def _get_current_price_cached(self, symbol: str, cache_timestamp: int) -> Optional[float]:
        """
        Get current price with caching.
        
        Args:
            symbol: Cryptocurrency symbol
            cache_timestamp: Timestamp for cache busting
            
        Returns:
            Current price in USD
        """
        # This function shouldn't be called directly - use get_current_price instead
        # The cache_timestamp parameter is just for LRU cache invalidation
        return self.alchemy_client.get_current_crypto_price(symbol)
    
    def get_current_price(self, symbol: str, force_refresh: bool = False) -> Optional[float]:
        """
        Get current price of a cryptocurrency in USD.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH", "BTC")
            force_refresh: If True, bypass cache
            
        Returns:
            Current price in USD or None if not available
        """
        symbol = symbol.upper()
        
        # Remove USD suffix if present (e.g., "ETH/USD" -> "ETH")
        if "/" in symbol:
            symbol = symbol.split("/")[0]
        
        # Check if we need to force refresh
        if force_refresh:
            self._get_current_price_cached.cache_clear()
            self.logger.info(f"Price cache cleared for {symbol}")
        
        # Apply rate limiting
        self._rate_limit()
        
        # Round to nearest cache_ttl to provide cache stability
        cache_timestamp = int(time.time() / self.cache_ttl) * self.cache_ttl
        
        try:
            price = self._get_current_price_cached(symbol, cache_timestamp)
            
            if price is not None:
                # Update last known price
                self.last_known_prices[symbol] = price
                return price
            
            # If we can't get a current price, return the last known valid price if available
            if symbol in self.last_known_prices:
                self.logger.warning(f"Using last known price for {symbol}: {self.last_known_prices[symbol]}")
                return self.last_known_prices[symbol]
                
            return None
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            
            # Return last known price if available
            if symbol in self.last_known_prices:
                self.logger.warning(f"Using last known price for {symbol} after error: {self.last_known_prices[symbol]}")
                return self.last_known_prices[symbol]
                
            return None
    
    def get_current_prices(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, float]:
        """
        Get current prices for multiple cryptocurrencies in USD.
        
        Args:
            symbols: List of cryptocurrency symbols (e.g., ["ETH", "BTC"])
            force_refresh: If True, bypass cache
            
        Returns:
            Dictionary with symbol keys and price values
        """
        # Normalize symbols
        clean_symbols = []
        for symbol in symbols:
            sym = symbol.upper()
            if "/" in sym:
                sym = sym.split("/")[0]
            clean_symbols.append(sym)
        
        # Apply rate limiting
        self._rate_limit()
        
        if force_refresh:
            self._get_current_price_cached.cache_clear()
            self.logger.info("Price cache cleared for all symbols")
        
        try:
            # Get prices from Alchemy with a batch call
            price_data = self.alchemy_client.get_token_price_by_symbol(clean_symbols)
            
            # Process response
            result = {}
            if "data" in price_data and price_data["data"]:
                for token_data in price_data["data"]:
                    symbol = token_data["symbol"].upper()
                    if "prices" in token_data and token_data["prices"]:
                        for price_info in token_data["prices"]:
                            if price_info["currency"] == "USD":
                                price = float(price_info["value"])
                                result[symbol] = price
                                # Update last known price
                                self.last_known_prices[symbol] = price
                                break
            
            # For any missing prices, get individually (with cache)
            for symbol in clean_symbols:
                if symbol not in result:
                    price = self.get_current_price(symbol)
                    if price is not None:
                        result[symbol] = price
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error getting prices for multiple symbols: {e}")
            
            # Try to get prices individually with cache fallback
            result = {}
            for symbol in clean_symbols:
                price = self.get_current_price(symbol)
                if price is not None:
                    result[symbol] = price
            
            return result
    
    @lru_cache(maxsize=16)
    def _get_historical_prices_cached(self, 
                                     symbol: str, 
                                     days: int, 
                                     interval: str, 
                                     cache_timestamp: int) -> Optional[pd.DataFrame]:
        """
        Get historical prices with caching.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days of historical data
            interval: Time interval ("1h", "1d", etc.)
            cache_timestamp: Timestamp for cache busting
            
        Returns:
            DataFrame with historical price data
        """
        # This function shouldn't be called directly
        symbol = symbol.upper()
        
        # Remove USD suffix if present
        if "/" in symbol:
            symbol = symbol.split("/")[0]
        
        # Calculate timestamps
        end_time = int(time.time())
        start_time = end_time - (days * 24 * 60 * 60)
        
        # Try to get historical data by symbol first
        try:
            data = self.alchemy_client.get_historical_token_prices(
                symbol_or_address=symbol,
                is_address=False,
                start_timestamp=start_time,
                end_timestamp=end_time,
                interval=interval
            )
            
            # Process response into DataFrame
            if "data" in data and data["data"]:
                prices = []
                for price_point in data["data"]:
                    if "timestamp" in price_point and "price" in price_point:
                        prices.append({
                            "timestamp": datetime.fromtimestamp(price_point["timestamp"]),
                            "price": float(price_point["price"])
                        })
                
                if prices:
                    df = pd.DataFrame(prices)
                    df.set_index("timestamp", inplace=True)
                    return df
        
        except Exception as e:
            self.logger.warning(f"Failed to get historical prices by symbol for {symbol}: {e}")
            
            # Fall back to address-based lookup if symbol fails
            if symbol in self.token_addresses:
                try:
                    token_info = self.token_addresses[symbol]
                    
                    data = self.alchemy_client.get_historical_token_prices(
                        symbol_or_address=token_info["address"],
                        network=token_info["network"],
                        is_address=True,
                        start_timestamp=start_time,
                        end_timestamp=end_time,
                        interval=interval
                    )
                    
                    # Process response into DataFrame
                    if "data" in data and data["data"]:
                        prices = []
                        for price_point in data["data"]:
                            if "timestamp" in price_point and "price" in price_point:
                                prices.append({
                                    "timestamp": datetime.fromtimestamp(price_point["timestamp"]),
                                    "price": float(price_point["price"])
                                })
                        
                        if prices:
                            df = pd.DataFrame(prices)
                            df.set_index("timestamp", inplace=True)
                            return df
                
                except Exception as e:
                    self.logger.error(f"Failed to get historical prices by address for {symbol}: {e}")
        
        # Return None if all methods fail
        return None
    
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
            interval: Time interval ("1h", "1d", etc.)
            force_refresh: If True, bypass cache
            
        Returns:
            DataFrame with historical price data
        """
        # Clean the symbol
        symbol = symbol.upper()
        if "/" in symbol:
            symbol = symbol.split("/")[0]
        
        # Apply rate limiting
        self._rate_limit()
        
        # Clear cache if forced
        if force_refresh:
            self._get_historical_prices_cached.cache_clear()
            self.logger.info(f"Historical price cache cleared for {symbol}")
        
        # Round to nearest cache_ttl to provide cache stability
        cache_timestamp = int(time.time() / self.historical_cache_ttl) * self.historical_cache_ttl
        
        try:
            return self._get_historical_prices_cached(symbol, days, interval, cache_timestamp)
        except Exception as e:
            self.logger.error(f"Error getting historical prices for {symbol}: {e}")
            return None
    
    def convert_price_to_bars(self, 
                             df: pd.DataFrame, 
                             timeframe: str = "1Hour") -> pd.DataFrame:
        """
        Convert price data to OHLCV bar format compatible with the existing system.
        
        Args:
            df: DataFrame with timestamp index and price column
            timeframe: Timeframe for resampling
            
        Returns:
            DataFrame with Open, High, Low, Close, Volume columns
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Map timeframe to pandas resampling rule
        resample_rule = {
            "1Min": "1min",
            "5Min": "5min",
            "15Min": "15min",
            "30Min": "30min",
            "1Hour": "1h",
            "2Hour": "2h",
            "4Hour": "4h",
            "1Day": "1d",
        }.get(timeframe, "1h")
        
        # Create dummy volume (not available from Alchemy)
        df["volume"] = 0
        
        # Resample to desired timeframe
        resampled = df.resample(resample_rule).agg({
            "price": ["first", "max", "min", "last"],
            "volume": "sum"
        })
        
        # Flatten multi-index columns and rename
        resampled.columns = ["open", "high", "low", "close", "volume"]
        
        # Reset index to make timestamp a column
        resampled = resampled.reset_index()
        
        # Rename timestamp column to match existing format
        resampled = resampled.rename(columns={"timestamp": "datetime"})
        
        return resampled
    
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
        # Clean the symbol
        clean_symbol = symbol.upper()
        if "/" in clean_symbol:
            clean_symbol = clean_symbol.split("/")[0]
        
        # Map timeframe to Alchemy interval
        interval_mapping = {
            "1Min": "1m",
            "5Min": "5m",
            "15Min": "15m",
            "30Min": "30m",
            "1Hour": "1h",
            "2Hour": "2h",
            "4Hour": "4h",
            "1Day": "1d",
        }
        
        alchemy_interval = interval_mapping.get(timeframe, "1h")
        
        # Calculate days needed based on limit and timeframe
        days_mapping = {
            "1Min": 1,
            "5Min": 1,
            "15Min": 2,
            "30Min": 3,
            "1Hour": 5,
            "2Hour": 10,
            "4Hour": 20,
            "1Day": 100,
        }
        
        # Get base days from mapping or default to 30
        base_days = days_mapping.get(timeframe, 30)
        
        # Adjust days based on limit to ensure we get enough data
        # This is a rough estimation and may need tuning
        days_needed = max(base_days, limit // 24 + 1)
        
        # Get historical price data
        df = self.get_historical_prices(clean_symbol, days=days_needed, interval=alchemy_interval)
        
        if df is None or df.empty:
            self.logger.warning(f"No historical price data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to bars format
        bars = self.convert_price_to_bars(df, timeframe)
        
        # Limit the number of bars if necessary
        if len(bars) > limit:
            bars = bars.tail(limit)
        
        return bars
    
    def clear_cache(self):
        """Clear all cached data"""
        self._get_current_price_cached.cache_clear()
        self._get_historical_prices_cached.cache_clear()
        self.logger.info("All price caches cleared") 