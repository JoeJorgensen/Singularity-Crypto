"""
Alpaca API interface for cryptocurrency trading.
Uses Alpaca's US feed for market data as required by project rules.
"""
import os
import time
import threading
import json
import asyncio
import pandas as pd
import requests
from datetime import datetime, timedelta
import aiohttp
import alpaca
import logging
from dotenv import load_dotenv
from functools import lru_cache
import websockets
from alpaca.data.timeframe import TimeFrameUnit
import numpy as np
import random


# Alpaca-py imports
from alpaca.trading.client import TradingClient, RESTClient as REST
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import CryptoDataStream
# Don't import CryptoFeed enum since it causes issues in Streamlit cloud
# from alpaca.data.enums import CryptoFeed 
from alpaca.data.models import Bar, Trade, Quote

# Load environment variables
load_dotenv()

class AlpacaAPI:
    def __init__(self, config=None):
        """
        Initialize the Alpaca API client.
        
        Args:
            config: Configuration dictionary with Alpaca API credentials
        """
        # Set up logging first thing to avoid attribute errors
        self.logger = logging.getLogger('alpaca_api')
        
        # Load configuration
        if config is None:
            config = {}
        
        # Store config
        self.config = config
        
        # Set up credentials
        self.api_key = config.get('ALPACA_API_KEY', os.getenv('ALPACA_API_KEY', ''))
        # Try multiple possible secret key names (ALPACA_SECRET_KEY and ALPACA_API_SECRET)
        self.api_secret = config.get('ALPACA_SECRET_KEY', 
                            config.get('ALPACA_API_SECRET', 
                                os.getenv('ALPACA_SECRET_KEY', 
                                    os.getenv('ALPACA_API_SECRET', ''))))
        self.base_url = config.get('ALPACA_BASE_URL', os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'))
        
        # Debug credential information (without revealing the actual keys)
        self.logger.info(f"API Key length: {len(self.api_key)} chars, Secret Key length: {len(self.api_secret)} chars")
        if not self.api_key or not self.api_secret:
            self.logger.error("API key or secret key is empty - check your configuration and environment variables")
        
        # Configure paper/live mode
        self.paper = bool(config.get('PAPER', True))
        if self.paper:
            self.base_url = 'https://paper-api.alpaca.markets'
        
        # Use specified log level, default to INFO
        log_level = config.get('LOG_LEVEL', 'INFO')
        numeric_level = getattr(logging, log_level.upper(), None)
        
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO
        
        # Remove handlers setup which would bypass the global configuration
        # Let the global configuration handle the logging setup
        
        # Initialize REST API client
        self.api = REST(
            api_key=self.api_key,
            secret_key=self.api_secret,
            base_url=self.base_url,
            api_version='v2'
        )
        
        # Initialize variables for websocket data
        self.websocket_data = {}  # Store latest data for each symbol
        self.ws_lock = threading.Lock()  # Lock for thread-safe websocket data access
        self.ws_connected = False  # Flag for websocket connection status
        self.ws_subscribed_symbols = []  # Symbols that are currently subscribed
        self.ws_connection_time = None  # Timestamp of when websocket was connected
        self.ws_thread = None  # Thread for websocket connection
        self._ws_client = None  # The websocket client instance
        
        # Variables to track connection attempts and prevent excessive reconnections
        self.ws_last_connection_attempt = None  # When the last connection attempt was made
        self.ws_connection_attempts = 0  # Number of consecutive connection attempts
        self.ws_connection_cooldown = False  # Flag to indicate if we're in a connection cooldown period
        self.ws_cooldown_until = None  # Time until cooldown ends
        
        self.logger.info(f"Alpaca API client initialized with paper mode: {self.paper}")
        
        # Test connection - we'll test with trading_client below after it's initialized
        # Rather than testing here with the REST client which doesn't have get_account()
        
        # Cache for historical data requests
        self.cache_timestamp = time.time()
        
        # Fix for handling timezone-aware datetimes in pandas
        # Removed deprecated use_inf_as_na option
        
        # Dictionary to track orders placed
        self.open_orders = {}
        
        # Dictionary to track positions
        self.positions = {}
        
        # If these flags are part of a larger test mode infrastructure
        self.use_mock_broker = config.get('USE_MOCK_BROKER', False)
        self.generate_mock_data = config.get('GENERATE_MOCK_DATA', False)
        
        # Set client parameters
        self.max_retries = 3
        self.retry_delay = 1  # Default retry delay in seconds
        
        # Initialize Alpaca clients
        if self.paper:
            base_url = 'https://paper-api.alpaca.markets'
        else:
            base_url = 'https://api.alpaca.markets'
        
        # Create trading client for Alpaca
        try:
            if not self.api_key or not self.api_secret:
                raise ValueError("Missing API credentials - cannot initialize TradingClient")
                
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret, 
                paper=self.paper
            )
            self.logger.info("Successfully initialized TradingClient")
            
            # Test account access with trading_client
            try:
                account = self.trading_client.get_account()
                self.logger.info(f"Connected to Alpaca API. Account ID: {account.id}, Status: {account.status}")
            except Exception as e:
                self.logger.warning(f"Could not retrieve account info: {e}")
        except Exception as e:
            self.logger.error(f"Error initializing TradingClient: {e}")
            # Create a placeholder to avoid NoneType errors later
            self.trading_client = None
            
        # Create data client for market data
        try:
            if not self.api_key or not self.api_secret:
                raise ValueError("Missing API credentials - cannot initialize CryptoHistoricalDataClient")
                
            self.data_client = CryptoHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret
            )
            self.logger.info("Successfully initialized CryptoHistoricalDataClient")
        except Exception as e:
            self.logger.error(f"Error initializing CryptoHistoricalDataClient: {e}")
            # Create a placeholder to avoid NoneType errors later
            self.data_client = None
            
        # We won't create the streaming client here, we'll create it when needed
        # This prevents any startup errors with the websocket
        self._ws_client = None
        
        # Cache settings
        self.cache_ttl = self.config.get('data_optimization', {}).get('cache_ttl', 60)  # seconds
        
        # Cache for last known valid prices
        self.last_known_prices = {}
        self.last_price_update_time = {}
        self.price_cache_updater_running = False
        self.price_cache_updater_thread = None
        self.price_update_interval = 15  # seconds
        
        # Rate limiting settings to prevent hitting API limits
        self.api_calls = 0
        self.api_call_reset_time = time.time() + 60
        self.max_calls_per_minute = 175  # Alpaca limit is 200/min, use 175 for safety
        self.rate_limit_lock = threading.Lock()
        
        # Track recent API calls for advanced rate limiting
        self.recent_api_calls = []
        self.max_burst_calls = 50  # Maximum calls to allow in a 10-second window
        
    # Use LRU cache for account info (refreshes every minute)
    @lru_cache(maxsize=1)
    def _get_account_cached(self, timestamp):
        """Get account information with caching."""
        return self.trading_client.get_account()
    
    def get_account(self, force_refresh=False):
        """Get account information with time-based cache busting.
        
        Args:
            force_refresh: If True, bypass cache and get fresh data
        """
        if force_refresh:
            self._get_account_cached.cache_clear()
            self.logger.info("Account cache cleared for fresh data")
            
        # Round to nearest minute to provide cache stability
        cache_timestamp = int(time.time() / self.cache_ttl) * self.cache_ttl
        return self._get_account_cached(cache_timestamp)
    
    async def get_crypto_bars_async(self, symbol='ETH/USD', timeframe='1H', limit=100):
        """Async version of get_crypto_bars method."""
        retries = 0
        while retries < self.max_retries:
            try:
                # Keep the original symbol format for the API request
                api_symbol = symbol
                
                # Get base timeframe
                if timeframe.endswith('Min'):
                    tf_str = f"{timeframe[:-3]}Min"
                    multiplier = int(timeframe[:-3])
                    seconds_per_unit = 60
                elif timeframe.endswith('H'):
                    tf_str = f"{timeframe[:-1]}Hour"
                    multiplier = int(timeframe[:-1])
                    seconds_per_unit = 3600
                else:  # Day
                    tf_str = "1Day"
                    multiplier = 1
                    seconds_per_unit = 86400
                
                # Calculate start time based on limit and timeframe
                end = datetime.now()
                start = end - timedelta(seconds=multiplier * seconds_per_unit * limit)
                
                # Format timestamps for API request
                start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
                end_str = end.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                # Create direct API request
                url = f"https://data.alpaca.markets/v1beta3/crypto/us/bars"
                headers = {
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.api_secret
                }
                params = {
                    "symbols": api_symbol,
                    "timeframe": tf_str,
                    "start": start_str,
                    "end": end_str,
                    "limit": limit
                    # Note: The crypto/us endpoint automatically uses IEX feed
                    # so we don't need to specify it
                }
                
                # Make the async API request
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params, timeout=10) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"API request failed with status {response.status}: {error_text}")
                            # Check for rate limit errors specifically
                            if response.status == 429:
                                print("Rate limit exceeded, waiting longer before retry")
                                await asyncio.sleep(self.retry_delay * 5)  # Wait longer for rate limit errors
                            else:
                                await asyncio.sleep(self.retry_delay)
                            retries += 1
                            continue
                        
                        try:
                            bars_data = await response.json()
                        except Exception as e:
                            print(f"Failed to parse API response as JSON: {e}")
                            retries += 1
                            await asyncio.sleep(self.retry_delay)
                            continue
                        
                        if not bars_data or 'bars' not in bars_data or not bars_data['bars'] or api_symbol not in bars_data['bars']:
                            print(f"No data received from Alpaca API (attempt {retries + 1}/{self.max_retries})")
                            print(f"Response content: {bars_data}")
                            retries += 1
                            await asyncio.sleep(self.retry_delay)
                            continue
                        
                        # Convert to DataFrame
                        bars_list = bars_data['bars'][api_symbol]
                        df = pd.DataFrame(bars_list)
                        
                        if df.empty:
                            print(f"Empty dataframe received from Alpaca API (attempt {retries + 1}/{self.max_retries})")
                            retries += 1
                            await asyncio.sleep(self.retry_delay)
                            continue
                        
                        # Rename columns to match the SDK's format
                        column_mapping = {
                            't': 'timestamp',
                            'o': 'open',
                            'h': 'high',
                            'l': 'low',
                            'c': 'close',
                            'v': 'volume',
                            'n': 'trade_count',
                            'vw': 'vwap'
                        }
                        
                        for orig_col, new_col in column_mapping.items():
                            if orig_col in df.columns:
                                df[new_col] = df[orig_col]
                                df.drop(orig_col, axis=1, inplace=True)
                        
                        # Convert timestamp to datetime
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df.set_index('timestamp', inplace=True)
                        
                        # Ensure all required columns exist
                        required_columns = ['open', 'high', 'low', 'close', 'volume']
                        for col in required_columns:
                            if col not in df.columns:
                                print(f"Warning: '{col}' column missing from API response. Adding default values.")
                                if col == 'volume':
                                    # Use a small default volume value to prevent division by zero errors
                                    df[col] = 1.0
                                elif col in ['open', 'high', 'low']:
                                    # Use close price if available, otherwise 0
                                    df[col] = df['close'] if 'close' in df.columns else 0.0
                                else:
                                    df[col] = 0.0
                        
                        # Ensure numeric data types
                        for col in df.columns:
                            if col in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                # Fill NaN values with appropriate defaults
                                if col == 'volume' and df[col].isna().any():
                                    df[col].fillna(1.0, inplace=True)
                                elif col in ['open', 'high', 'low'] and df[col].isna().any():
                                    df[col].fillna(df['close'], inplace=True)
                                elif df[col].isna().any():
                                    df[col].fillna(0.0, inplace=True)
                        
                        # Debug info
                        if 'volume' in df.columns:
                            print(f"Volume data summary: min={df['volume'].min()}, max={df['volume'].max()}, mean={df['volume'].mean()}")
                        else:
                            print("Warning: Volume data still missing after processing.")
                        
                        return self._replace_inf_with_nan(df)
                        
            except aiohttp.ClientError as e:
                print(f"Network error fetching crypto bars: {str(e)}")
                retries += 1
                await asyncio.sleep(self.retry_delay)
            except asyncio.TimeoutError:
                print(f"Timeout error fetching crypto bars (attempt {retries + 1}/{self.max_retries})")
                retries += 1
                await asyncio.sleep(self.retry_delay)
            except Exception as e:
                print(f"Error fetching crypto bars asynchronously: {str(e)}")
                retries += 1
                if retries < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                
        # If all retries failed, return a minimal valid DataFrame
        print("Failed to fetch crypto bars after all retries, returning minimal DataFrame")
        # Try to use polling data as fallback before returning empty DataFrame
        try:
            ws_data = self.get_latest_websocket_data(symbol)
            if ws_data is not None and ws_data['close'] is not None:
                # Use the data from our polling method
                return self._replace_inf_with_nan(pd.DataFrame({
                    'open': [ws_data['open']],
                    'high': [ws_data['high']],
                    'low': [ws_data['low']],
                    'close': [ws_data['close']],
                    'volume': [ws_data['volume'] if ws_data['volume'] > 0 else 1.0]
                }, index=[ws_data['timestamp'] if ws_data['timestamp'] else pd.Timestamp.now()]))
        except Exception as e:
            print(f"Failed to use polling data as fallback: {e}")
        
        # Return minimal empty DataFrame as last resort
        empty_df = pd.DataFrame({
            'open': [0.0],
            'high': [0.0],
            'low': [0.0],
            'close': [0.0],
            'volume': [1.0]
        }, index=[pd.Timestamp.now()])
        
        return empty_df
    
    # Cached version of get_crypto_bars for frequently accessed timeframes
    @lru_cache(maxsize=32)
    def _get_crypto_bars_cached(self, symbol, timeframe, limit, cache_timestamp):
        """Cached version of get_crypto_bars."""
        return self.get_crypto_bars(symbol, timeframe, limit)
    
    def get_crypto_bars(self, symbol: str, timeframe: str = '1Min', limit: int = 100) -> pd.DataFrame:
        """
        Get cryptocurrency bars from Alpaca for ETH/USD only.
        
        Args:
            symbol: The cryptocurrency symbol (only ETH/USD supported)
            timeframe: The timeframe for the bars
            limit: Number of bars to return
            
        Returns:
            DataFrame with the bars
        """
        # Only support ETH/USD - skip other symbols
        if symbol != 'ETH/USD':
            self.logger.debug(f"Only ETH/USD is supported, but {symbol} was requested")
            return pd.DataFrame()
            
        try:
            # Parse the timeframe
            if timeframe.endswith('Min'):
                tf_str = f"{timeframe[:-3]}Min"
                multiplier = int(timeframe[:-3])
                seconds_per_unit = 60
            elif timeframe.endswith('H'):
                tf_str = f"{timeframe[:-1]}Hour"
                multiplier = int(timeframe[:-1])
                seconds_per_unit = 3600
            else:  # Day
                tf_str = "1Day"
                multiplier = 1
                seconds_per_unit = 86400
            
            # Calculate start time based on limit and timeframe
            end = datetime.now()
            start = end - timedelta(seconds=multiplier * seconds_per_unit * limit)
            
            # Format timestamps for API request
            start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_str = end.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Create direct API request
            url = f"https://data.alpaca.markets/v1beta3/crypto/us/bars"
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret
            }
            params = {
                "symbols": symbol,
                "timeframe": tf_str,
                "start": start_str,
                "end": end_str,
                "limit": limit
                # Note: The crypto/us endpoint automatically uses IEX feed
            }
            
            # Make the API request directly
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code != 200:
                self.logger.error(f"Error getting crypto bars: {response.text}")
                return pd.DataFrame()
            
            bars_data = response.json()
            
            if not bars_data or 'bars' not in bars_data or not bars_data['bars'] or symbol not in bars_data['bars']:
                self.logger.error(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            bars_list = bars_data['bars'][symbol]
            df = pd.DataFrame(bars_list)
            
            if df.empty:
                self.logger.error(f"Empty dataframe received from Alpaca API")
                return pd.DataFrame()
            
            # Rename columns to match the SDK's format
            column_mapping = {
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'n': 'trade_count',
                'vw': 'vwap'
            }
            
            for orig_col, new_col in column_mapping.items():
                if orig_col in df.columns:
                    df[new_col] = df[orig_col]
                    df.drop(orig_col, axis=1, inplace=True)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # Ensure all required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'volume':
                        # Use a small default volume value to prevent division by zero errors
                        df[col] = 1.0
                    elif col in ['open', 'high', 'low']:
                        # Use close price if available, otherwise 0
                        df[col] = df['close'] if 'close' in df.columns else 0.0
                    else:
                        df[col] = 0.0
            
            return self._replace_inf_with_nan(df)
            
        except Exception as e:
            self.logger.error(f"Error getting crypto bars: {str(e)}")
            
            # Implement retry with exponential backoff
            for i in range(3):
                try:
                    time.sleep(2 ** i)
                    self.logger.info(f"Retrying get_crypto_bars (attempt {i+1})...")
                    
                    # Make the API request directly
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                    
                    if response.status_code != 200:
                        self.logger.error(f"Retry {i+1} failed: {response.text}")
                        continue
                    
                    bars_data = response.json()
                    
                    if not bars_data or 'bars' not in bars_data or not bars_data['bars'] or symbol not in bars_data['bars']:
                        self.logger.error(f"No data found for symbol {symbol} in retry {i+1}")
                        continue
                    
                    # Convert to DataFrame
                    bars_list = bars_data['bars'][symbol]
                    df = pd.DataFrame(bars_list)
                    
                    if df.empty:
                        self.logger.error(f"Empty dataframe received from Alpaca API in retry {i+1}")
                        continue
                    
                    # Rename columns to match the SDK's format
                    for orig_col, new_col in column_mapping.items():
                        if orig_col in df.columns:
                            df[new_col] = df[orig_col]
                            df.drop(orig_col, axis=1, inplace=True)
                    
                    # Convert timestamp to datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    
                    # Ensure all required columns exist
                    for col in required_columns:
                        if col not in df.columns:
                            if col == 'volume':
                                # Use a small default volume value to prevent division by zero errors
                                df[col] = 1.0
                            elif col in ['open', 'high', 'low']:
                                # Use close price if available, otherwise 0
                                df[col] = df['close'] if 'close' in df.columns else 0.0
                            else:
                                df[col] = 0.0
                    
                    return self._replace_inf_with_nan(df)
                
                except Exception as retry_e:
                    self.logger.error(f"Error in retry {i+1}: {str(retry_e)}")
                    continue
            
            # If all retries fail, return empty DataFrame
            self.logger.warning(f"No data available for {symbol}")
            return pd.DataFrame()
    
    # Websocket methods for real-time market data
    async def _ws_connect(self):
        """
        Connect to Alpaca websocket for real-time data.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        try:
            self.logger.info("Connecting to Alpaca Crypto Websocket (US feed)...")
            
            # Try to import CryptoFeed enum for more explicit feed selection
            try:
                from alpaca.data.enums import CryptoFeed
                # Create the crypto data stream client with US feed (IEX)
                self._ws_client = CryptoDataStream(
                    api_key=self.api_key,
                    secret_key=self.api_secret,
                    feed=CryptoFeed.US  # Explicitly use IEX feed via enum
                )
                self.logger.info("Using CryptoFeed.US enum for IEX feed")
            except ImportError:
                # Fallback to string if enum is not available
                self._ws_client = CryptoDataStream(
                    api_key=self.api_key,
                    secret_key=self.api_secret,
                    feed="us"  # Use string value as fallback
                )
                self.logger.info("Using 'us' string for IEX feed (CryptoFeed enum not available)")
            
            # Add explicit debug log to confirm websocket URL
            if hasattr(self._ws_client, '_ws_url'):
                self.logger.info(f"Using websocket URL: {self._ws_client._ws_url}")
            else:
                self.logger.info("Using Alpaca US feed (IEX) for crypto websocket")
            
            # Set connected flag
            self.ws_connected = True
            self.logger.info("Connected to Alpaca Crypto Websocket (US feed)")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to Alpaca websocket: {str(e)}")
            self.ws_connected = False
            return False

    async def _ws_subscribe(self, symbols, channels=["bars"]):
        """
        Subscribe to websocket channels for specified symbols.
        
        Args:
            symbols: List of symbols to subscribe to
            channels: List of channels to subscribe to (default: ["bars"])
            
        Returns:
            bool: True if subscription successful, False otherwise
        """
        if not self.ws_connected or not self._ws_client:
            self.logger.error("Cannot subscribe: websocket not connected")
            return False
        
        try:
            self.logger.info(f"Subscribing to {channels} for {symbols}")
            
            # Define handler functions for different data types
            async def on_bar(bar):
                symbol = bar.symbol.replace("/", "")  # Convert ETH/USD to ETHUSD format if needed
                with self.ws_lock:
                    if symbol not in self.websocket_data:
                        self.websocket_data[symbol] = {}
                        
                    # Store the bar data
                    self.websocket_data[symbol] = {
                        'timestamp': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                        'vwap': getattr(bar, 'vwap', 0),
                        'trade_count': getattr(bar, 'trade_count', 0)
                    }
                    self.logger.debug(f"Updated {symbol} bar data: {bar.close}")
            
            # Subscribe to the requested channels
            if "bars" in channels:
                self._ws_client.subscribe_bars(on_bar, *symbols)
                self.logger.info(f"Subscribed to bars for {symbols}")
                
            # Add handlers for other channels if needed (trades, quotes, etc.)
            
            # Store subscribed symbols
            self.ws_subscribed_symbols = symbols
            self.ws_connection_time = datetime.now()
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing to websocket channels: {str(e)}")
            return False

    def _ws_listener(self):
        """
        Start the websocket listener to receive updates.
        This method starts the Alpaca CryptoDataStream run method.
        """
        if not hasattr(self, '_ws_client') or not self._ws_client:
            self.logger.error("Cannot start listener: websocket not connected")
            return
        
        try:
            self.logger.info("Starting websocket listener")
            # Run the websocket client - this is a blocking method that will run until connection is closed
            self._ws_client.run()
        except KeyboardInterrupt:
            self.logger.info("Websocket listener stopped by user")
            self.ws_connected = False
        except ConnectionRefusedError:
            self.logger.error("Websocket connection refused by server")
            self.ws_connected = False
        except TimeoutError:
            self.logger.error("Websocket connection timeout")
            self.ws_connected = False
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error in websocket listener: {error_msg}")
            
            # Handle specific error types
            if "connection limit exceeded" in error_msg.lower():
                self.logger.warning("Connection limit error detected in listener. Setting cooldown.")
                self.ws_connection_cooldown = True
                self.ws_cooldown_until = datetime.now() + timedelta(minutes=15)
                self.ws_connection_attempts += 2
            elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                self.logger.error("Authentication error with websocket. Check API credentials.")
            elif "network" in error_msg.lower() or "connect" in error_msg.lower():
                self.logger.warning("Network error in websocket connection. Will retry later.")
                
            self.ws_connected = False

    def start_websocket(self, symbols=None, timeout=3.0, non_blocking=False):
        """
        Start a websocket connection for real-time market data.
        
        Args:
            symbols: List of symbols to subscribe to
            timeout: Maximum time to wait for connection in seconds
            non_blocking: If True, start websocket in background thread
            
        Returns:
            bool: True if connection started successfully, False otherwise
        """
        # Default symbols if not provided
        if not symbols:
            symbols = ['ETH/USD']  # Default to only ETH/USD
        
        # Check if we're in a cooldown period
        current_time = datetime.now()
        if self.ws_connection_cooldown and self.ws_cooldown_until and current_time < self.ws_cooldown_until:
            time_left = (self.ws_cooldown_until - current_time).total_seconds()
            self.logger.warning(f"Connection in cooldown for {time_left:.1f} more seconds. Skipping connection attempt.")
            return False
        
        # Update connection attempt tracking
        if self.ws_last_connection_attempt:
            # If last attempt was more than 5 minutes ago, reset counter
            if (current_time - self.ws_last_connection_attempt).total_seconds() > 300:
                self.ws_connection_attempts = 0
            else:
                self.ws_connection_attempts += 1
        
        self.ws_last_connection_attempt = current_time
        
        # If too many connection attempts in a short period, enter cooldown
        if self.ws_connection_attempts >= 5:  # 5 consecutive attempts
            cooldown_minutes = min(5 * (self.ws_connection_attempts - 4), 60)  # Increasing cooldown, max 60 minutes
            self.ws_cooldown_until = current_time + timedelta(minutes=cooldown_minutes)
            self.ws_connection_cooldown = True
            self.logger.warning(f"Too many connection attempts. Entering cooldown for {cooldown_minutes} minutes.")
            return False
        
        # If there's already a connection, close it first
        if hasattr(self, '_ws_client') and self._ws_client:
            self.logger.info("Stopping existing websocket connection before starting a new one")
            self.stop_websocket()
        
        self.logger.info(f"Starting websocket connection for {symbols}")
        
        try:
            # Import CryptoFeed if available for more explicit feed selection
            try:
                from alpaca.data.enums import CryptoFeed
                # Create the crypto data stream client with US feed (IEX)
                self._ws_client = CryptoDataStream(
                    api_key=self.api_key,
                    secret_key=self.api_secret,
                    feed=CryptoFeed.US  # Explicitly use IEX feed via enum
                )
                self.logger.info("Using CryptoFeed.US enum for IEX feed")
            except ImportError:
                # Fallback to string if enum is not available
                self._ws_client = CryptoDataStream(
                    api_key=self.api_key,
                    secret_key=self.api_secret,
                    feed="us"  # Use string value as fallback
                )
                self.logger.info("Using 'us' string for IEX feed (CryptoFeed enum not available)")
            
            # Configure authentication parameters to be more aggressive
            if hasattr(self._ws_client, '_conn_options'):
                # Reduce handshake timeout if the property exists
                if hasattr(self._ws_client._conn_options, 'handshake_timeout'):
                    self._ws_client._conn_options.handshake_timeout = 2.0
                # Reduce max_connection_queue_size if exists to prioritize authentication
                if hasattr(self._ws_client, 'max_connection_queue_size'):
                    self._ws_client.max_connection_queue_size = 10
            
            # Add explicit debug log to confirm websocket URL
            if hasattr(self._ws_client, '_ws_url'):
                self.logger.info(f"Using websocket URL: {self._ws_client._ws_url}")
            else:
                self.logger.info("Using Alpaca US feed (IEX) for crypto websocket")
            
            # Set up subscription handlers before starting the connection
            async def on_bar(bar):
                try:
                    # Convert symbol format if needed (ETH/USD -> ETHUSD)
                    symbol = bar.symbol.replace("/", "")
                    with self.ws_lock:
                        if symbol not in self.websocket_data:
                            self.websocket_data[symbol] = {}
                        
                        # Store the bar data
                        self.websocket_data[symbol] = {
                            'timestamp': bar.timestamp,
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume,
                            'vwap': getattr(bar, 'vwap', 0),
                            'trade_count': getattr(bar, 'trade_count', 0)
                        }
                        # Log the price update more clearly for debugging
                        self.logger.info(f"[WebSocket] Updated {symbol} price: ${bar.close} at {bar.timestamp}")
                        
                        # Also update the price cache for backup
                        self.last_known_prices[symbol.replace('USD', '/USD')] = bar.close
                        self.last_price_update_time[symbol.replace('USD', '/USD')] = datetime.now()
                except Exception as e:
                    self.logger.error(f"Error processing bar data: {e}")

            # Define function to run websocket in a thread
            def run_websocket():
                try:
                    import asyncio
                    
                    # Create an async function to handle the setup
                    async def setup_and_run():
                        # Pre-authenticate quickly
                        self.logger.info("Pre-authenticating websocket...")
                        
                        # Subscribe to bars immediately after connection
                        for symbol in symbols:
                            try:
                                self._ws_client.subscribe_bars(on_bar, symbol)
                                self.logger.info(f"Subscribed to bars for {symbol}")
                            except Exception as sub_err:
                                self.logger.error(f"Error subscribing to {symbol}: {sub_err}")
                        
                        # Update connection status
                        self.ws_connected = True
                        self.ws_connection_time = datetime.now()
                        self.ws_subscribed_symbols = symbols
                        
                        # Start the client with a timeout for connection
                        self.logger.info("Starting websocket listener in thread")
                        await asyncio.wait_for(self._ws_client._run_forever(), timeout=30.0)
                    
                    # Run the async function in the event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(setup_and_run())
                    except asyncio.TimeoutError:
                        self.logger.error("Websocket connection timed out")
                    except Exception as e:
                        self.logger.error(f"Error in websocket setup: {e}")
                    finally:
                        loop.close()
                        
                except Exception as e:
                    self.logger.error(f"Error in websocket thread: {e}")
                    self.ws_connected = False
                    
            # Start the websocket based on blocking mode
            if non_blocking:
                # Start the websocket thread
                self.logger.info("Starting websocket in non-blocking mode")
                self.ws_thread = threading.Thread(target=run_websocket, daemon=True)
                self.ws_thread.start()
                
                # Give it a moment to establish connection
                time.sleep(0.5)  # Short wait for the connection to start
                
                self.logger.info(f"Successfully started websocket connection for {symbols}")
                return True
            else:
                # For blocking mode, we'll run the setup directly
                self.logger.info("Starting websocket in blocking mode")
                
                # Run websocket setup in a separate thread but wait for it
                ws_thread = threading.Thread(target=run_websocket, daemon=True)
                ws_thread.start()
                
                # Wait for timeout or successful connection
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if self.ws_connected:
                        self.logger.info(f"Successfully started websocket connection for {symbols}")
                        return True
                    time.sleep(0.1)
                
                # If we're here, the connection timed out
                self.logger.warning(f"Failed to start websocket connection for {symbols}")
                return False
                
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error starting websocket: {error_msg}")
            self.ws_connected = False
            return False

    def stop_websocket(self):
        """
        Stop the websocket connection.
        
        Returns:
            bool: True if connection stopped successfully, False otherwise
        """
        try:
            if hasattr(self, '_ws_client') and self._ws_client:
                # Create a new thread to handle the async closing
                def close_websocket():
                    try:
                        # Create a new event loop for this thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Run the stop function in this loop
                        loop.run_until_complete(self._stop_websocket_async())
                        
                        # Close the loop
                        loop.close()
                    except Exception as e:
                        self.logger.error(f"Error in websocket closing thread: {str(e)}")
                
                # Create and start the thread
                close_thread = threading.Thread(target=close_websocket, daemon=True)
                close_thread.start()
                close_thread.join(timeout=5.0)  # Wait for up to 5 seconds for closing
                
                return True
            else:
                self.logger.info("No websocket connection to stop")
                return False
        except Exception as e:
            self.logger.error(f"Error stopping websocket: {str(e)}")
            return False

    def get_latest_websocket_data(self, symbol):
        """
        Get the latest data for a symbol from websocket.
        Falls back to REST API if websocket data is not available.
        
        Args:
            symbol: The trading symbol to get data for
            
        Returns:
            dict: Dictionary with latest price data or None values if unavailable
        """
        # Convert symbol format if needed (ETH/USD -> ETHUSD)
        symbol_key = symbol.replace('/', '')
        
        try:
            # First try to get data from websocket
            with self.ws_lock:
                if symbol_key in self.websocket_data and self.websocket_data[symbol_key].get('close') is not None:
                    return self.websocket_data[symbol_key]
            
            # If we don't have websocket data, fall back to REST API polling
            self.logger.info(f"No websocket data available for {symbol}, falling back to REST API")
            current_price = self.get_current_price(symbol)
            
            if current_price:
                # We only have the current price, not full bar data
                now = datetime.now()
                return {
                    'timestamp': now,
                    'open': current_price,
                    'high': current_price,
                    'low': current_price, 
                    'close': current_price,
                    'volume': 0,
                    'vwap': 0,
                    'trade_count': 0
                }
            else:
                self.logger.warning(f"No data available for {symbol}")
                # Return None for all values if we couldn't get the data
                return {
                    'timestamp': None,
                    'open': None,
                    'high': None,
                    'low': None,
                    'close': None,
                    'volume': None,
                    'vwap': None,
                    'trade_count': None
                }
        except Exception as e:
            self.logger.error(f"Error getting latest data for {symbol}: {e}")
            # Return None for all values if an exception occurred
            return {
                'timestamp': None,
                'open': None,
                'high': None,
                'low': None,
                'close': None,
                'volume': None,
                'vwap': None,
                'trade_count': None
            }
    
    def get_current_price(self, symbol='ETH/USD'):
        """
        Get the current price for a given symbol directly from the API.
        
        Args:
            symbol: Trading symbol to get price for (default: 'ETH/USD')
            
        Returns:
            float: Current price or None if unavailable
        """
        try:
            # Get the latest bar data
            bars = self.get_crypto_bars(symbol, '1Min', 1)
            
            if not bars.empty:
                current_price = bars['close'].iloc[-1]
                self.logger.info(f"Current price for {symbol}: ${current_price}")
                
                # Update the price cache
                self.last_known_prices[symbol] = current_price
                self.last_price_update_time[symbol] = datetime.now()
                
                return current_price
            else:
                self.logger.warning(f"No price data available for {symbol}")
                
                # Check if we have a cached price
                if symbol in self.last_known_prices:
                    cached_price = self.last_known_prices[symbol]
                    cache_age = (datetime.now() - self.last_price_update_time[symbol]).total_seconds()
                    self.logger.warning(f"Using cached price for {symbol}: ${cached_price} (age: {cache_age:.1f}s)")
                    return cached_price
                    
                return None
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            
            # Check if we have a cached price
            if symbol in self.last_known_prices:
                cached_price = self.last_known_prices[symbol]
                cache_age = (datetime.now() - self.last_price_update_time[symbol]).total_seconds()
                self.logger.warning(f"Using cached price after error for {symbol}: ${cached_price} (age: {cache_age:.1f}s)")
                return cached_price
                
            return None
    
    def get_position(self, symbol='ETH/USD'):
        """Get current position for a symbol."""
        try:
            # Convert symbol format for Alpaca API
            # The Trading API needs ETHUSD format (without slash)
            symbol_without_slash = symbol.replace('/', '')
            
            # Get position
            return self.trading_client.get_open_position(symbol_without_slash)
        except Exception as e:
            # Handle the "position does not exist" error gracefully
            if 'position does not exist' in str(e).lower():
                # Instead of printing an error, we'll return None with proper logging
                # Since this is an expected case when no position exists
                return None
            else:
                # For other errors, log it but still return None
                print(f"Error getting position for {symbol}: {str(e)}")
                return None
    
    def submit_order(self, symbol='ETH/USD', qty=0, side='buy', type='market', limit_price=None, stop_price=None):
        """
        Submit a new order.
        
        Args:
            symbol: Symbol to trade
            qty: Quantity to trade
            side: 'buy' or 'sell' (for crypto, 'sell' is only for closing existing long positions as 
                  short selling is not supported)
            type: Order type ('market', 'limit', 'stop_limit')
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Order object
        """
        try:
            # Convert symbol format for Trading API (remove slash)
            trading_symbol = symbol.replace('/', '')
            
            # Sanity check - make sure qty is a positive number
            if qty <= 0:
                raise ValueError(f"Invalid quantity: {qty}. Must be a positive number.")
            
            # For crypto, if side is 'sell', verify that a position exists to avoid unintended errors
            # since crypto accounts don't support short selling
            if side.lower() == 'sell':
                try:
                    position = self.get_position(symbol)
                    if position is None or float(position.qty) <= 0:
                        raise ValueError(
                            f"Cannot sell {symbol} - no position exists. "
                            "Cryptocurrency accounts do not support short selling. "
                            "You must first have a long position to sell."
                        )
                except Exception as pos_error:
                    if 'position does not exist' in str(pos_error).lower():
                        raise ValueError(
                            f"Cannot sell {symbol} - no position exists. "
                            "Cryptocurrency accounts do not support short selling."
                        )
                    # If it's some other error checking the position, log but continue
                    print(f"Warning: Could not verify position existence: {str(pos_error)}")
            
            # Double-check buying power for buy orders to prevent frequent errors
            if side.lower() == 'buy' and type.lower() in ['market', 'limit']:
                # Get fresh account data (bypass cache)
                account = self.get_account(force_refresh=True)
                buying_power = float(account.buying_power)
                
                # Log clear account information for debugging
                self.logger.info(f"Account status: Cash: ${float(account.cash):.2f}, Buying Power: ${buying_power:.2f}")
                
                # Estimate order value
                if type.lower() == 'limit':
                    order_value = qty * limit_price
                else:
                    # For market orders, we need to fetch the current price
                    current_price = self.get_current_price(symbol)  # Use symbol with slash
                    
                    # If current price is not available, try alternative approaches
                    if not current_price:
                        self.logger.warning(f"Could not determine current price for {symbol}, trying alternative methods")
                        
                        # Try to get recent bars with longer timeframe
                        try:
                            # Try using larger timeframes
                            for tf in ['5Min', '15Min', '1H']:
                                bars = self.get_crypto_bars(symbol, tf, 1)
                                if not bars.empty:
                                    current_price = bars['close'].iloc[-1]
                                    self.logger.info(f"Using {tf} timeframe price for {symbol}: ${current_price}")
                                    
                                    # Update the price cache
                                    self.last_known_prices[symbol] = current_price
                                    self.last_price_update_time[symbol] = datetime.now()
                                    
                                    break
                        except Exception as e:
                            self.logger.error(f"Error getting alternative timeframe price: {e}")
                            
                        # If still no price, try websocket data
                        if not current_price:
                            try:
                                ws_data = self.get_latest_websocket_data(symbol)
                                if ws_data and ws_data['close'] is not None:
                                    current_price = ws_data['close']
                                    self.logger.info(f"Using websocket price for {symbol}: ${current_price}")
                                    
                                    # Update the price cache
                                    self.last_known_prices[symbol] = current_price
                                    self.last_price_update_time[symbol] = datetime.now()
                            except Exception as e:
                                self.logger.error(f"Error getting websocket price: {e}")
                        
                        # If still no price, check if we have a cached price
                        if not current_price and symbol in self.last_known_prices:
                            current_price = self.last_known_prices[symbol]
                            cache_age = (datetime.now() - self.last_price_update_time[symbol]).total_seconds()
                            
                            # Only use cached price if it's reasonably recent (less than 1 hour old)
                            if cache_age < 3600:  # 1 hour in seconds
                                self.logger.warning(f"Using cached price from {cache_age:.1f}s ago: ${current_price}")
                            else:
                                # Cache is too old, don't use it
                                self.logger.warning(f"Cached price is too old ({cache_age:.1f}s), skipping buying power check")
                                current_price = None
                    
                    # At this point, if we still don't have a price, skip the buying power check
                    if current_price:
                        order_value = qty * current_price
                        
                        # Add a safety margin for market price fluctuations (10%)
                        if type.lower() == 'market':
                            order_value *= 1.1
                        
                        # Check if we have sufficient buying power
                        if order_value > buying_power:
                            raise ValueError(
                                f"Insufficient buying power for order: ${order_value:.2f} required, but only ${buying_power:.2f} available. "
                                f"Reduce quantity to approximately {(buying_power / order_value * qty * 0.9):.6f} or less."
                            )
                    else:
                        self.logger.warning("Skipping buying power check due to unavailable price data")
            
            # Prepare order parameters
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Validate order type - for crypto, we can only use market, limit, and stop_limit
            order_type = type.lower()
            if order_type not in ['market', 'limit', 'stop_limit']:
                raise ValueError(f"Unsupported order type for crypto: {order_type}. Must be one of: market, limit, stop_limit")
            
            # Check rate limit before submitting order
            if not self.check_rate_limit():
                raise Exception("Rate limit protection activated. Order submission aborted to prevent API ban.")
            
            # Build the appropriate order based on type
            if order_type == 'market':
                # Market order - we don't need the price for a market order
                # Just create the order
                order_data = MarketOrderRequest(
                    symbol=trading_symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.GTC
                )
                
                # Log the order details
                self.logger.info(f"Submitting market {side} order for {qty} {trading_symbol}")
            elif order_type == 'limit':
                # Ensure limit_price is provided for limit orders
                if limit_price is None:
                    raise ValueError("Limit price is required for limit orders")
                    
                # Limit order
                order_data = LimitOrderRequest(
                    symbol=trading_symbol,
                    qty=qty,
                    side=order_side,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.GTC
                )
            elif order_type == 'stop_limit':
                # Ensure both stop_price and limit_price are provided for stop_limit orders
                if stop_price is None:
                    raise ValueError("Stop price is required for stop_limit orders")
                if limit_price is None:
                    raise ValueError("Limit price is required for stop_limit orders")
                
                # Validate the stop and limit prices based on order side
                if side.lower() == 'sell' and stop_price <= limit_price:
                    # For sell stop limit, the stop price should be lower than the limit price
                    # But some platforms use the opposite convention, so we'll provide a clear error
                    raise ValueError(
                        "For a sell stop_limit order, stop_price must be HIGHER than limit_price. "
                        f"You provided stop_price={stop_price} and limit_price={limit_price}."
                    )
                elif side.lower() == 'buy' and stop_price >= limit_price:
                    # For buy stop limit, the stop price should be higher than the limit price
                    raise ValueError(
                        "For a buy stop_limit order, stop_price must be LOWER than limit_price. "
                        f"You provided stop_price={stop_price} and limit_price={limit_price}."
                    )
                
                # We need to create a direct API request because the Python SDK might not support stop_limit directly
                base_url = "https://paper-api.alpaca.markets/v2/orders"
                headers = {
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.api_secret,
                    "Content-Type": "application/json"
                }
                
                order_data = {
                    "symbol": trading_symbol,
                    "qty": str(qty),
                    "side": side.lower(),
                    "type": "stop_limit",
                    "time_in_force": "gtc",
                    "limit_price": str(limit_price),
                    "stop_price": str(stop_price)
                }
                
                # Log the order details for debugging
                print(f"Submitting stop_limit order: {order_data}")
                
                # Make the direct API request
                response = requests.post(base_url, headers=headers, json=order_data)
                if response.status_code != 200:
                    error_data = response.text
                    raise Exception(f"Order submission failed (status {response.status_code}): {error_data}")
                
                # Return the order data parsed from the response
                return response.json()
            else:
                raise ValueError(f"Unsupported order type: {type}. Supported types are: market, limit, stop_limit")
            
            try:
                # Submit the order
                order = self.trading_client.submit_order(order_data)
                return order
            except alpaca.common.exceptions.APIError as e:
                # Extract error details for better error handling
                error_data = str(e)
                
                # Special handling for rate limit errors
                if "rate limit exceeded" in error_data.lower() or "429" in error_data:
                    retry_attempts = 3
                    base_delay = 1.0
                    
                    # Attempt to retry with exponential backoff
                    for retry in range(retry_attempts):
                        # Calculate backoff delay with jitter
                        delay = base_delay * (2 ** retry) + random.uniform(0, 0.5)
                        self.logger.warning(f"Rate limit exceeded. Retrying in {delay:.2f}s (attempt {retry+1}/{retry_attempts})")
                        
                        # Sleep before retry
                        time.sleep(delay)
                        
                        try:
                            # Retry the order submission
                            order = self.trading_client.submit_order(order_data)
                            self.logger.info(f"Order retry successful after rate limit error")
                            return order
                        except alpaca.common.exceptions.APIError as retry_error:
                            # If this is still a rate limit error, continue the retry loop
                            if "rate limit exceeded" in str(retry_error).lower() or "429" in str(retry_error):
                                continue
                            else:
                                # If it's a different error, raise it
                                raise retry_error
                    
                    # If we've exhausted retries, raise the original error
                    raise Exception(f"Order failed after multiple retries: {error_data}")
                
                # Special handling for insufficient balance errors
                if "insufficient balance" in error_data.lower():
                    # Get available balance from the error message if possible
                    import re
                    import json
                    
                    # Try to parse the JSON error message to extract available balance
                    try:
                        # The error message might be JSON or a string that contains JSON
                        json_start = error_data.find('{')
                        json_end = error_data.rfind('}') + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            json_str = error_data[json_start:json_end]
                            error_json = json.loads(json_str)
                            
                            # Extract values from the JSON
                            available = float(error_json.get('available', 0))
                            message = error_json.get('message', '')
                            
                            # Try to extract requested amount from message
                            requested_match = re.search(r'requested: ([\d.]+)', message)
                            requested = float(requested_match.group(1)) if requested_match else 0
                            
                            # Calculate the percentage difference
                            shortfall = requested - available
                            shortfall_pct = (shortfall / requested) * 100 if requested > 0 else 0
                            
                            # Calculate a safe reduction percentage (add 5% safety margin)
                            safe_reduction_pct = min(shortfall_pct + 5, 95)
                            
                            # Calculate a recommended quantity
                            recommended_qty = qty * (1 - (safe_reduction_pct / 100))
                            
                            # Provide a more helpful error message
                            raise Exception(
                                f"Insufficient balance: requested ${requested:.2f}, available ${available:.2f}, "
                                f"shortfall ${shortfall:.2f} ({shortfall_pct:.1f}%). "
                                f"Try reducing position size to {recommended_qty:.6f} or less."
                            )
                            
                        else:
                            # If we can't find JSON, look for key values in the message
                            available_match = re.search(r'available: ([\d.]+)', error_data)
                            requested_match = re.search(r'requested: ([\d.]+)', error_data)
                            
                            if available_match and requested_match:
                                available = float(available_match.group(1))
                                requested = float(requested_match.group(1))
                                
                                shortfall = requested - available
                                shortfall_pct = (shortfall / requested) * 100 if requested > 0 else 0
                                
                                # Calculate a safe reduction percentage 
                                safe_reduction_pct = min(shortfall_pct + 5, 95)
                                
                                # Calculate a recommended quantity
                                recommended_qty = qty * (1 - (safe_reduction_pct / 100))
                                
                                raise Exception(
                                    f"Insufficient balance: requested ${requested:.2f}, available ${available:.2f}, "
                                    f"shortfall ${shortfall:.2f} ({shortfall_pct:.1f}%). "
                                    f"Try reducing position size to {recommended_qty:.6f} or less."
                                )
                                
                    except Exception as parse_error:
                        # Fall back to the original error if parsing fails
                        raise Exception(f"Insufficient balance error: {error_data}. Try reducing position size.")
                else:
                    # Re-raise other API errors
                    raise Exception(f"Order submission failed: {error_data}")
                
        except Exception as e:
            # For any other exceptions
            raise Exception(f"Order submission failed: {str(e)}")
    
    def close_position(self, symbol='ETH/USD'):
        """Close a position."""
        try:
            # Convert symbol format for Trading API (remove slash)
            trading_symbol = symbol.replace('/', '')  
            
            # Check rate limit before proceeding
            if not self.check_rate_limit():
                raise Exception("Rate limit protection activated. Position close aborted to prevent API ban.")
            
            # Direct API approach - more reliable in edge cases
            # Step 1: Cancel all orders for this symbol using direct API
            try:
                print(f"Direct API: Canceling all orders for {trading_symbol}")
                
                # Get all open orders
                base_url = "https://paper-api.alpaca.markets/v2/orders"
                headers = {
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.api_secret
                }
                
                # Make the request to get all orders
                response = requests.get(base_url, headers=headers)
                if response.status_code != 200:
                    print(f"Failed to get orders: {response.text}")
                else:
                    orders = response.json()
                    for order in orders:
                        # Check if this order is for our symbol (handle both formats)
                        order_symbol = order.get('symbol', '').replace('/', '')
                        if order_symbol == trading_symbol:
                            order_id = order.get('id')
                            if order_id:
                                print(f"Canceling order {order_id} for {trading_symbol}")
                                cancel_url = f"{base_url}/{order_id}"
                                cancel_response = requests.delete(cancel_url, headers=headers)
                                if cancel_response.status_code != 200:
                                    print(f"Failed to cancel order {order_id}: {cancel_response.text}")
                                else:
                                    print(f"Order {order_id} canceled successfully")
                
                # Wait to ensure cancellations process
                time.sleep(1.5)
            except Exception as cancel_error:
                print(f"Error during direct API order cancellation: {str(cancel_error)}")
            
            # Step 2: Close the position using direct API
            try:
                print(f"Direct API: Closing position for {trading_symbol}")
                close_url = f"https://paper-api.alpaca.markets/v2/positions/{trading_symbol}"
                close_response = requests.delete(close_url, headers=headers)
                
                if close_response.status_code != 200:
                    error_text = close_response.text
                    print(f"Direct API position close failed: {error_text}")
                    
                    # If we get an insufficient balance error, try market order
                    if "insufficient balance" in error_text.lower():
                        print("Insufficient balance error, trying market order approach")
                        
                        # Get current position size
                        pos_url = f"https://paper-api.alpaca.markets/v2/positions/{trading_symbol}"
                        pos_response = requests.get(pos_url, headers=headers)
                        
                        if pos_response.status_code == 200:
                            position_data = pos_response.json()
                            position_qty = float(position_data.get('qty', 0))
                            
                            if position_qty > 0:
                                print(f"Creating market order to sell {position_qty} {trading_symbol}")
                                # Create market order
                                order_data = {
                                    "symbol": trading_symbol,
                                    "qty": str(position_qty),
                                    "side": "sell",
                                    "type": "market",
                                    "time_in_force": "gtc"
                                }
                                
                                order_response = requests.post(base_url, headers=headers, json=order_data)
                                if order_response.status_code == 200:
                                    print(f"Market order placed successfully: {order_response.text}")
                                    return order_response.json()
                                else:
                                    print(f"Market order failed: {order_response.text}")
                        else:
                            print(f"Failed to get position data: {pos_response.text}")
                    
                    # If we've reached here, both approaches failed
                    raise Exception(f"Failed to close position: {error_text}")
                else:
                    # Success!
                    print(f"Position closed successfully via direct API")
                    return close_response.json()
                    
            except Exception as close_error:
                print(f"Error during direct API position close: {str(close_error)}")
                raise
                    
        except Exception as e:
            raise Exception(f"Failed to close position: {str(e)}")
    
    def close_all_positions(self):
        """Close all open positions."""
        try:
            return self.trading_client.close_all_positions()
        except Exception as e:
            raise Exception(f"Failed to close all positions: {str(e)}")
    
    # Cached version of get_trades
    @lru_cache(maxsize=16)
    def _get_trades_cached(self, symbol, limit, cache_timestamp):
        """Cached version of get_trades."""
        try:
            # Symbol should already be in ETHUSD format at this point
            # Create request parameters with explicit status to get all orders
            request_params = GetOrdersRequest(
                status=QueryOrderStatus.ALL,  # Get orders with any status
                limit=limit,
                symbols=[symbol]
            )
            
            # Get orders using the request parameters
            trades = self.trading_client.get_orders(request_params)
            
            # Improve logging for debugging
            print(f"Retrieved {len(trades) if trades else 0} trades for {symbol}")
            
            # Return only the specified number of trades
            return trades[:limit] if trades else []
            
        except Exception as e:
            print(f"Error fetching trades for {symbol}: {str(e)}")
            return []
    
    def clear_cache(self):
        """Clear all cached data to ensure fresh API calls."""
        # Clear the LRU caches
        self._get_trades_cached.cache_clear()
        self._get_account_cached.cache_clear()
        print("API cache cleared")
        
    def get_trades(self, symbol='ETH/USD', limit=5, force_fresh=False):
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Number of trades to return
            force_fresh: Whether to bypass the cache
            
        Returns:
            List of trade objects
        """
        try:
            # Convert the symbol to the format expected by Alpaca API (ETHUSD)
            api_symbol = symbol.replace('/', '')
            
            if force_fresh:
                # Clear the cache for this specific query
                self._get_trades_cached.cache_clear()
                print("Trade cache cleared - forcing fresh data")
                
            # Normalize cache timestamp for consistent caching
            cache_timestamp = int(time.time() / self.cache_ttl) * self.cache_ttl
            
            # Call the cached version with the properly formatted symbol
            return self._get_trades_cached(api_symbol, limit, cache_timestamp)
        except Exception as e:
            print(f"Error in get_trades for {symbol}: {str(e)}")
            return []
    
    def get_portfolio_history(self, period="1M", timeframe="1D"):
        """
        Get portfolio history for the specified period and timeframe.
        
        Args:
            period: Time period to retrieve (e.g., '1D', '5D', '1M', '3M', '1Y')
            timeframe: Resolution of data points (e.g., '1Min', '15Min', '1H', '1D')
            
        Returns:
            Dictionary with portfolio history data
        """
        try:
            # Convert period string to API params
            period_map = {
                "1D": "1D",
                "5D": "5D", 
                "1M": "1M",
                "3M": "3M",
                "6M": "6M",
                "1Y": "1Y",
                "5Y": "5Y"
            }
            
            # Convert timeframe to API params
            timeframe_map = {
                "1Min": "1Min",
                "5Min": "5Min",
                "15Min": "15Min",
                "1H": "1H",
                "1D": "1D"
            }
            
            api_period = period_map.get(period, "1M")
            api_timeframe = timeframe_map.get(timeframe, "1D")
            
            # Call Alpaca API directly (account_portfolio_history is not exposed in SDK)
            url = f"https://paper-api.alpaca.markets/v2/account/portfolio/history"
            params = {
                "period": api_period,
                "timeframe": api_timeframe,
                "extended_hours": "true"
            }
            
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            data = response.json()
            
            # Validate data has expected keys
            if 'timestamp' not in data or 'equity' not in data or len(data['timestamp']) == 0:
                print("Portfolio history API returned empty or invalid data, using mock data")
                return self._generate_mock_portfolio_history(period, timeframe)
                
            return data
            
        except Exception as e:
            print(f"Error fetching portfolio history: {str(e)}")
            # Return mock data for visualization
            return self._generate_mock_portfolio_history(period, timeframe)
    
    def _generate_mock_portfolio_history(self, period="1M", timeframe="1D"):
        """
        Generate mock portfolio history data for visualization.
        
        Args:
            period: Time period to generate
            timeframe: Resolution of data points
            
        Returns:
            Dictionary with mock portfolio history data
        """
        import random
        import numpy as np
        import pandas as pd  # Ensure pandas is imported locally within the method
        from datetime import datetime, timedelta
        
        # Determine number of data points based on period and timeframe
        periods = {
            "1D": 1,
            "5D": 5,
            "1M": 30,
            "3M": 90,
            "6M": 180,
            "1Y": 365,
            "5Y": 1825
        }
        
        timeframes = {
            "1Min": 1/1440,
            "5Min": 5/1440,
            "15Min": 15/1440,
            "1H": 1/24,
            "1D": 1
        }
        
        # Get number of days
        days = periods.get(period, 30)
        
        # Get points per day
        points_per_day = int(1 / timeframes.get(timeframe, 1))
        
        # Calculate total points
        total_points = days * points_per_day
        
        # Limit to reasonable number
        total_points = min(total_points, 500)
        
        # Generate timestamps
        end_time = datetime.now()
        timestamps = [(end_time - timedelta(days=days) + timedelta(days=days*i/total_points)).timestamp() for i in range(total_points)]
        
        # Generate equity values
        # Start with account value and apply random walk with slight upward bias
        account = self.get_account()
        try:
            start_value = float(account.portfolio_value) * 0.8  # Start at 80% of current value
        except:
            start_value = 10000.0
            
        # Generate a random walk with upward drift
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(0.001, 0.02, total_points)
        equity_values = [start_value]
        
        for ret in daily_returns:
            equity_values.append(equity_values[-1] * (1 + ret))
        
        # Remove the first element (it was just a starting point)
        equity_values = equity_values[1:]
        
        # Calculate profit/loss
        profit_loss = [equity - start_value for equity in equity_values]
        profit_loss_pct = [(equity - start_value) / start_value * 100 for equity in equity_values]
        
        return {
            "timestamp": timestamps,
            "equity": equity_values,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss_pct
        }
    
    def get_bars(self, symbol='ETH/USD', timeframe='1H', limit=100):
        """Get historical price bars for a symbol."""
        return self.get_crypto_bars(symbol, timeframe, limit)

    def _parse_timeframe(self, timeframe: str):
        """Parse timeframe string into a format suitable for Alpaca API requests.
        
        Args:
            timeframe: String timeframe like '1Min', '5Min', '1H', etc.
            
        Returns:
            String for direct API usage
        """
        # Remove any spaces
        tf = timeframe.replace(' ', '')
        
        try:
            # Handle minute timeframes
            if tf.endswith('Min'):
                minutes = int(tf[:-3])
                return f"{minutes}Min"
                
            # Handle hour timeframes
            elif tf.endswith('H'):
                hours = int(tf[:-1])
                return f"{hours}Hour"
                
            # Handle day timeframes
            elif tf.endswith('D'):
                days = int(tf[:-1]) if len(tf) > 1 else 1
                return f"{days}Day"
                
            # Default to 1 Hour if unrecognized
            else:
                self.logger.warning(f"Unrecognized timeframe format '{timeframe}', defaulting to 1Hour")
                return "1Hour"
                
        except Exception as e:
            # Catch any unexpected errors and provide a safe default
            self.logger.error(f"Error parsing timeframe '{timeframe}': {e}. Using default 1Hour")
            return "1Hour"

    async def _stop_websocket_async(self):
        """Async helper to stop the websocket connection."""
        try:
            if hasattr(self, '_ws_client') and self._ws_client:
                self.logger.info("Stopping websocket connection...")
                
                # Check what kind of client we have and use appropriate method to stop it
                if hasattr(self._ws_client, 'close'):
                    try:
                        await self._ws_client.close()
                        self.logger.info("Closed websocket connection with close() method")
                    except Exception as e:
                        self.logger.warning(f"Error calling close() on websocket client: {e}")
                
                elif hasattr(self._ws_client, 'stop'):
                    try:
                        self._ws_client.stop()
                        self.logger.info("Stopped websocket connection with stop() method")
                        # Wait a moment for it to process
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        self.logger.warning(f"Error calling stop() on websocket client: {e}")
                
                else:
                    # If no standard method exists, try to disconnect or cleanup
                    self.logger.warning("No standard close/stop method found on websocket client")
                    if hasattr(self._ws_client, 'disconnect'):
                        try:
                            self._ws_client.disconnect()
                            self.logger.info("Disconnected websocket with disconnect() method")
                        except Exception as e:
                            self.logger.warning(f"Error disconnecting websocket: {e}")
                
                # Clear the client reference
                self._ws_client = None
                
                # Cancel monitor thread if it exists
                if hasattr(self, '_ws_monitor_thread') and self._ws_monitor_thread:
                    try:
                        self._ws_monitor_thread.cancel()
                        self.logger.info("Stopped websocket monitor thread")
                    except Exception as e:
                        self.logger.warning(f"Error stopping monitor thread: {e}")
                    self._ws_monitor_thread = None
                
                # Reset connection state
                self.ws_connected = False
                self.logger.info("Websocket connection stopped")
                return True
            else:
                self.logger.info("No websocket connection to stop")
                return False
        except Exception as e:
            self.logger.error(f"Error stopping websocket: {str(e)}")
            self.ws_connected = False
            return False

    def stop_websocket(self):
        """Stop the websocket connection."""
        try:
            if hasattr(self, '_ws_client') and self._ws_client:
                # Create a new thread to handle the async closing
                def close_websocket():
                    try:
                        # Create a new event loop for this thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Run the stop function in this loop
                        loop.run_until_complete(self._stop_websocket_async())
                        
                        # Close the loop
                        loop.close()
                    except Exception as e:
                        self.logger.error(f"Error in websocket closing thread: {str(e)}")
                
                # Create and start the thread
                close_thread = threading.Thread(target=close_websocket, daemon=True)
                close_thread.start()
                close_thread.join(timeout=5.0)  # Wait for up to 5 seconds for closing
                
                return True
            else:
                self.logger.info("No websocket connection to stop")
                return False
        except Exception as e:
            self.logger.error(f"Error stopping websocket: {str(e)}")
            return False

    def start_price_cache_updater(self, symbols=None):
        """
        Start a background thread to periodically update the price cache for specified symbols.
        
        Args:
            symbols: List of symbols to cache prices for (default: ETH/USD only)
            
        Returns:
            bool: True if started, False if already running
        """
        if self.price_cache_updater_running:
            self.logger.info("Price cache updater already running")
            return False
            
        # Default to only ETH/USD if not provided
        if not symbols:
            symbols = ['ETH/USD']
            
        self.logger.info(f"Starting price cache updater for symbols: {symbols}")
        
        def update_price_cache():
            self.price_cache_updater_running = True
            while self.price_cache_updater_running:
                try:
                    # Update price for each symbol
                    for symbol in symbols:
                        try:
                            # Try different timeframes if 1Min fails
                            price = None
                            for timeframe in ['1Min', '5Min', '15Min', '1H']:
                                try:
                                    bars = self.get_crypto_bars(symbol, timeframe, 1)
                                    if not bars.empty:
                                        price = bars['close'].iloc[-1]
                                        # Update the price cache
                                        self.last_known_prices[symbol] = price
                                        self.last_price_update_time[symbol] = datetime.now()
                                        self.logger.debug(f"Updated price cache for {symbol}: ${price} using {timeframe} timeframe")
                                        break
                                except Exception as tf_error:
                                    continue
                                    
                            # Also try websocket data if available
                            if price is None:
                                try:
                                    ws_data = self.get_latest_websocket_data(symbol)
                                    if ws_data and ws_data['close'] is not None:
                                        price = ws_data['close']
                                        # Update the price cache
                                        self.last_known_prices[symbol] = price
                                        self.last_price_update_time[symbol] = datetime.now()
                                        self.logger.debug(f"Updated price cache for {symbol}: ${price} from websocket")
                                except Exception:
                                    pass
                                    
                        except Exception as symbol_error:
                            self.logger.warning(f"Error updating price cache for {symbol}: {symbol_error}")
                    
                    # Sleep before next update
                    time.sleep(self.price_update_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in price cache updater: {e}")
                    time.sleep(5)  # Wait a bit before retrying
                    
        # Start the thread
        self.price_cache_updater_thread = threading.Thread(
            target=update_price_cache,
            daemon=True
        )
        self.price_cache_updater_thread.start()
        return True
    
    def stop_price_cache_updater(self):
        """Stop the price cache updater thread."""
        if self.price_cache_updater_running:
            self.logger.info("Stopping price cache updater")
            self.price_cache_updater_running = False
            if self.price_cache_updater_thread:
                # Wait for thread to finish
                self.price_cache_updater_thread.join(timeout=2.0)
            return True
        else:
            self.logger.info("Price cache updater not running")
            return False

    @staticmethod
    def _replace_inf_with_nan(df):
        """Replace inf values with NaN in a DataFrame."""
        if df is not None and not df.empty:
            return df.replace([np.inf, -np.inf], np.nan)
        return df

    def check_rate_limit(self):
        """
        Check if we're at our rate limit and wait if necessary.
        Returns False if we should abort due to excessive rate limiting.
        """
        with self.rate_limit_lock:
            current_time = time.time()
            
            # Clean up old API call timestamps
            self.recent_api_calls = [t for t in self.recent_api_calls if current_time - t < 10]
            
            # Check if we're exceeding our burst limit (too many calls in 10 seconds)
            if len(self.recent_api_calls) >= self.max_burst_calls:
                # We're making too many calls too quickly, need to back off
                oldest_call = min(self.recent_api_calls)
                wait_time = 10 - (current_time - oldest_call) + 1  # Add 1 second buffer
                self.logger.warning(f"Rate limit burst protection: waiting {wait_time:.2f}s to avoid exceeding limits")
                time.sleep(wait_time)
                
                # After waiting, clean up the list again
                current_time = time.time()
                self.recent_api_calls = [t for t in self.recent_api_calls if current_time - t < 10]
            
            # Reset counter if we've passed the reset time
            if current_time > self.api_call_reset_time:
                self.api_calls = 0
                self.api_call_reset_time = current_time + 60
            
            # Check if we're at the rate limit
            if self.api_calls >= self.max_calls_per_minute:
                # Calculate wait time until reset
                wait_time = self.api_call_reset_time - current_time
                
                if wait_time > 0:
                    self.logger.warning(f"Rate limit reached ({self.api_calls} calls in this minute). Waiting {wait_time:.2f}s for limit reset.")
                    time.sleep(wait_time)
                    
                    # Reset after waiting
                    self.api_calls = 0
                    self.api_call_reset_time = time.time() + 60
            
            # Track this API call
            self.api_calls += 1
            self.recent_api_calls.append(time.time())
            
            return True
