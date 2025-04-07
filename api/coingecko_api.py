"""
CoinGecko API interface for cryptocurrency price data.
Provides a free alternative for price data when Alchemy Prices API is not available.
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

# Load environment variables
load_dotenv()

# Get logger
logger = get_logger('coingecko_api')

class CoinGeckoAPI:
    """Wrapper for CoinGecko API for cryptocurrency price data"""
    
    def __init__(self, config=None):
        """
        Initialize CoinGeckoAPI client.
        
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
        
        # API endpoint
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # Rate limiting configuration (free tier limits: 10-30 calls/minute)
        self.last_request_time = 0
        self.min_request_interval = config.get('min_request_interval', 2.0)  # 2 seconds between requests for free tier
        
        # Cache settings
        self.cache_ttl = config.get('cache_ttl', 60)  # seconds for current price cache
        self.historical_cache_ttl = config.get('historical_cache_ttl', 3600)  # 1 hour for historical data
        
        # Last known prices cache
        self.last_known_prices = {}
        
        # Coin ID mapping (symbol -> CoinGecko ID)
        self.coin_id_cache = {}
        self.coin_id_cache_timestamp = 0
        self.coin_list_cache_duration = 24 * 3600  # 24 hours
        
        # Common cryptocurrency mappings
        self.common_ids = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "USDT": "tether",
            "USDC": "usd-coin",
            "SOL": "solana",
            "BNB": "binancecoin",
            "XRP": "ripple",
            "ADA": "cardano",
            "AVAX": "avalanche-2",
            "DOGE": "dogecoin",
            "DOT": "polkadot",
            "MATIC": "matic-network",
            "LINK": "chainlink",
            "UNI": "uniswap",
            "LTC": "litecoin"
        }
        
        self.logger.info("CoinGecko API initialized")
    
    def _rate_limit(self):
        """
        Apply rate limiting to avoid hitting API limits.
        Adds delay if requests are made too quickly.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Any:
        """
        Make a request to the CoinGecko API with rate limiting.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            Parsed JSON response
        """
        if params is None:
            params = {}
            
        # Apply rate limiting
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            self.logger.debug(f"Making CoinGecko API request: {endpoint}")
            response = requests.get(url, params=params)
            
            # Handle 429 Too Many Requests
            if response.status_code == 429:
                self.logger.warning("CoinGecko API rate limit exceeded. Waiting before retry.")
                # Wait longer than the usual rate limit
                time.sleep(self.min_request_interval * 3)
                # Try again
                response = requests.get(url, params=params)
            
            # Raise for other errors
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error for {endpoint}: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error for {endpoint}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error for {endpoint}: {str(e)}")
            raise
    
    def get_coin_list(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get list of all coins supported by CoinGecko.
        
        Args:
            force_refresh: Force refresh the cache
            
        Returns:
            List of coin dictionaries
        """
        current_time = time.time()
        if (not force_refresh and 
            self.coin_id_cache and 
            current_time - self.coin_id_cache_timestamp < self.coin_list_cache_duration):
            return list(self.coin_id_cache.values())
        
        try:
            coin_list = self._make_request('coins/list')
            
            # Update cache
            self.coin_id_cache = {}
            for coin in coin_list:
                symbol = coin.get('symbol', '').upper()
                if symbol:
                    # If multiple coins have the same symbol, prefer the one with shorter ID
                    # This typically selects the "main" coin
                    if symbol not in self.coin_id_cache or len(coin['id']) < len(self.coin_id_cache[symbol]['id']):
                        self.coin_id_cache[symbol] = coin
            
            self.coin_id_cache_timestamp = current_time
            self.logger.info(f"Updated coin list cache with {len(self.coin_id_cache)} symbols")
            
            return list(self.coin_id_cache.values())
        
        except Exception as e:
            self.logger.error(f"Error fetching coin list: {e}")
            return []
    
    def get_coin_id(self, symbol: str) -> Optional[str]:
        """
        Get CoinGecko coin ID for a symbol.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            
        Returns:
            CoinGecko coin ID or None if not found
        """
        symbol = symbol.upper()
        
        # Handle USD suffix
        if "/" in symbol:
            symbol = symbol.split("/")[0]
        
        # Check common IDs first (no API call needed)
        if symbol in self.common_ids:
            return self.common_ids[symbol]
        
        # Check cache
        if symbol in self.coin_id_cache:
            return self.coin_id_cache[symbol]['id']
        
        # If cache is empty or expired, refresh it
        current_time = time.time()
        if not self.coin_id_cache or current_time - self.coin_id_cache_timestamp > self.coin_list_cache_duration:
            self.get_coin_list()
            
            # Check cache again after refresh
            if symbol in self.coin_id_cache:
                return self.coin_id_cache[symbol]['id']
        
        self.logger.warning(f"Coin ID not found for symbol: {symbol}")
        return None
    
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
        coin_id = self.get_coin_id(symbol)
        if not coin_id:
            return None
        
        try:
            response = self._make_request('simple/price', {
                'ids': coin_id,
                'vs_currencies': 'usd'
            })
            
            if coin_id in response and 'usd' in response[coin_id]:
                price = float(response[coin_id]['usd'])
                return price
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
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
        # Clean symbols
        clean_symbols = []
        for symbol in symbols:
            sym = symbol.upper()
            if "/" in sym:
                sym = sym.split("/")[0]
            clean_symbols.append(sym)
        
        # For few symbols, batch them in one request
        if len(clean_symbols) <= 20:
            coin_ids = []
            symbol_to_id_map = {}
            
            for symbol in clean_symbols:
                coin_id = self.get_coin_id(symbol)
                if coin_id:
                    coin_ids.append(coin_id)
                    symbol_to_id_map[coin_id] = symbol
            
            if not coin_ids:
                return {}
                
            try:
                response = self._make_request('simple/price', {
                    'ids': ','.join(coin_ids),
                    'vs_currencies': 'usd'
                })
                
                result = {}
                for coin_id, data in response.items():
                    if 'usd' in data and coin_id in symbol_to_id_map:
                        symbol = symbol_to_id_map[coin_id]
                        price = float(data['usd'])
                        result[symbol] = price
                        # Update cache
                        self.last_known_prices[symbol] = price
                
                return result
            except Exception as e:
                self.logger.error(f"Error getting batch prices: {e}")
                # Fall back to individual requests
        
        # For many symbols or if batch fails, get them individually
        result = {}
        for symbol in clean_symbols:
            price = self.get_current_price(symbol, force_refresh)
            if price is not None:
                result[symbol] = price
        
        return result
    
    @lru_cache(maxsize=16)
    def _get_historical_prices_cached(self, 
                                     coin_id: str, 
                                     days: int, 
                                     cache_timestamp: int) -> Optional[pd.DataFrame]:
        """
        Get historical prices with caching.
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days of historical data
            cache_timestamp: Timestamp for cache busting
            
        Returns:
            DataFrame with historical price data
        """
        # Limit days to valid CoinGecko values
        valid_days = [1, 7, 14, 30, 90, 180, 365, 'max']
        
        # Map requested days to closest valid option
        if days <= 1:
            days_param = 1
        elif days <= 7:
            days_param = 7
        elif days <= 14:
            days_param = 14
        elif days <= 30:
            days_param = 30
        elif days <= 90:
            days_param = 90 
        elif days <= 180:
            days_param = 180
        elif days <= 365:
            days_param = 365
        else:
            days_param = 'max'
            
        try:
            response = self._make_request(f'coins/{coin_id}/market_chart', {
                'vs_currency': 'usd',
                'days': days_param,
                'interval': 'daily'
            })
            
            # Extract price data (CoinGecko returns [timestamp, price] pairs)
            if 'prices' in response and response['prices']:
                prices = []
                for timestamp_ms, price in response['prices']:
                    # Convert milliseconds to seconds
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                    prices.append({
                        'timestamp': timestamp,
                        'price': float(price)
                    })
                
                if prices:
                    df = pd.DataFrame(prices)
                    df.set_index('timestamp', inplace=True)
                    return df
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting historical prices for {coin_id}: {e}")
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
            interval: Time interval (ignored, CoinGecko determines based on days)
            force_refresh: If True, bypass cache
            
        Returns:
            DataFrame with historical price data
        """
        symbol = symbol.upper()
        
        # Remove USD suffix if present
        if "/" in symbol:
            symbol = symbol.split("/")[0]
        
        # Get coin ID
        coin_id = self.get_coin_id(symbol)
        if not coin_id:
            self.logger.warning(f"Coin ID not found for {symbol}, cannot fetch historical data")
            return None
        
        # Clear cache if forced
        if force_refresh:
            self._get_historical_prices_cached.cache_clear()
            self.logger.info(f"Historical price cache cleared for {symbol}")
        
        # Round to nearest cache_ttl to provide cache stability
        cache_timestamp = int(time.time() / self.historical_cache_ttl) * self.historical_cache_ttl
        
        try:
            return self._get_historical_prices_cached(coin_id, days, cache_timestamp)
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
        
        # Create dummy volume (CoinGecko has volume data but we're not retrieving it)
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
        days_needed = max(base_days, limit // 24 + 1)
        
        # Get historical price data
        df = self.get_historical_prices(clean_symbol, days=days_needed)
        
        if df is None or df.empty:
            self.logger.warning(f"No historical price data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to bars format
        bars = self.convert_price_to_bars(df, timeframe)
        
        # Limit the number of bars if necessary
        if len(bars) > limit:
            bars = bars.tail(limit)
        
        return bars
    
    def get_market_data(self, symbol: str) -> Dict:
        """
        Get detailed market data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH", "BTC")
            
        Returns:
            Dictionary with market data
        """
        coin_id = self.get_coin_id(symbol)
        if not coin_id:
            return {}
        
        try:
            response = self._make_request(f'coins/{coin_id}', {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false'
            })
            
            if 'market_data' in response:
                market_data = response['market_data']
                result = {
                    'symbol': symbol,
                    'name': response.get('name', ''),
                    'price_usd': market_data.get('current_price', {}).get('usd', 0),
                    'market_cap_usd': market_data.get('market_cap', {}).get('usd', 0),
                    'volume_24h_usd': market_data.get('total_volume', {}).get('usd', 0),
                    'price_change_24h': market_data.get('price_change_24h', 0),
                    'price_change_percentage_24h': market_data.get('price_change_percentage_24h', 0),
                    'price_change_percentage_7d': market_data.get('price_change_percentage_7d', 0),
                    'price_change_percentage_30d': market_data.get('price_change_percentage_30d', 0),
                    'market_cap_rank': market_data.get('market_cap_rank', 0),
                    'circulating_supply': market_data.get('circulating_supply', 0),
                    'total_supply': market_data.get('total_supply', 0),
                    'max_supply': market_data.get('max_supply', 0),
                    'ath': market_data.get('ath', {}).get('usd', 0),
                    'ath_change_percentage': market_data.get('ath_change_percentage', {}).get('usd', 0),
                    'atl': market_data.get('atl', {}).get('usd', 0),
                    'atl_change_percentage': market_data.get('atl_change_percentage', {}).get('usd', 0),
                    'last_updated': market_data.get('last_updated', '')
                }
                return result
            
            return {}
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return {}
    
    def clear_cache(self):
        """Clear all cached data"""
        self._get_current_price_cached.cache_clear()
        self._get_historical_prices_cached.cache_clear()
        self.coin_id_cache = {}
        self.coin_id_cache_timestamp = 0
        self.logger.info("All CoinGecko API caches cleared") 