"""
Core trading strategy for cryptocurrency trading.
Implements the main trading logic and signal processing.
"""
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import time
import json
import logging
import asyncio
import aiohttp
import threading
from functools import lru_cache
import os

import pandas as pd
import numpy as np

# Import APIs and utilities
from api.alpaca_api import AlpacaAPI
from api.finnhub_api import FinnhubAPI
from api.openai_api import OpenAIAPI
from api.coinlore_api import CoinloreAPI
from technical.indicators import TechnicalIndicators
from technical.signal_generator import SignalGenerator
from utils.risk_manager import RiskManager
from utils.position_calculator import PositionCalculator
from utils.order_manager import OrderManager
from utils.logging_config import get_logger

# Configure logging
from utils.logging_config import setup_logging
logger = get_logger('trading_strategy')

class TradingStrategy:
    """
    Core trading strategy for cryptocurrency trading.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trading strategy.

        Args:
            config: Configuration dictionary
        """
        # Setup logging
        self.logger = logging.getLogger('CryptoTrader.trading_strategy')
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            
        self.logger.info("Initializing Trading Strategy with config")
        
        # Initialize from provided config or default
        self.config = config or {}
        
        # Flag to track websocket initialization
        self.ws_initialization_attempted = False
        
        # Initialize API connections
        self.apis = self._initialize_apis()
        
        # Default trading parameters
        self.default_pair = self.config.get('trading', {}).get('default_pair', 'ETH/USD')
        self.default_timeframe = self.config.get('trading', {}).get('timeframe', '1Hour')
        
        # Initialize cache
        self.signal_cache = {}
        self.sentiment_cache = {}
        self.sentiment_ttl = 60  # Default TTL for sentiment data in seconds
        
        # Initialize trade tracking
        self.last_trade_time = datetime.now() - timedelta(hours=1)  # Start with allowing trades
        self.min_time_between_trades = timedelta(minutes=self.config.get('risk_management', {}).get('min_time_between_trades_min', 15))
        
        # Trading state
        self.is_trading_active = False
        
        logger.info("Initializing Trading Strategy with config")
        self.signal_generator = SignalGenerator(self.config)
        self.risk_manager = RiskManager(self.config)
        self.position_calculator = PositionCalculator(self.risk_manager, self.config)
        self.order_manager = OrderManager(self.apis['alpaca'], self.config)
        
        # Trading state
        self.is_active = False
        self.last_signal_time = None
        self.current_signal = None
        self.cycle_count = 0
        
        # Data caching attributes
        self.cached_sentiment_data = {}
        self.sentiment_cache_ttl = self.config.get('data_optimization', {}).get('cache_ttl', 60) * 3  # Default TTL (3x regular data)
        self.last_sentiment_update = {}
        self.market_volatility = {}  # Track volatility for dynamic TTL calculation
        
        # Load trading rules
        self.max_trades_per_day = self.config.get('trading', {}).get('max_trades_per_day', 5)
        self.risk_per_trade = self.config.get('risk_management', {}).get('risk_per_trade', 0.02)
        logger.info(f"Trading Strategy initialized with default pair: {self.default_pair}, timeframe: {self.default_timeframe}")
        
        # Track last price cache time for controlling updates
        self.last_price_cache_time = {}
        self.last_sentiment_cache_time = {}  # Add missing initialization
        
        # Configure market data caching
        self.market_data_ttl = self.config.get('data_optimization', {}).get('cache_ttl', 60)  # seconds
        self.sentiment_data_ttl = self.config.get('data_optimization', {}).get('sentiment_ttl', 300)  # seconds
    
    def _initialize_apis(self) -> Dict:
        """
        Initialize API clients.
        
        Returns:
            Dictionary with API clients
        """
        apis = {}
        
        # Initialize Alpaca API
        try:
            self.logger.info("Initializing Alpaca API with config")
            alpaca_api = AlpacaAPI(self.config)
            apis['alpaca'] = alpaca_api
            
            # Check if websocket initialization has already been attempted
            if not hasattr(self, 'ws_initialization_attempted') or not self.ws_initialization_attempted:
                # Mark as attempted to prevent duplicate initializations
                self.ws_initialization_attempted = True
                
                # Start websocket connection for real-time data
                # We start this in a thread to avoid blocking initialization
                # List of symbols to monitor
                symbols_to_monitor = self.config.get('trading', {}).get(
                    'supported_pairs', 
                    ['ETH/USD', 'BTC/USD', 'LTC/USD', 'BCH/USD', 'XRP/USD']
                )
                
                # Define background function for starting websocket
                def start_ws_in_background():
                    try:
                        self.logger.info(f"Starting websocket connection for {symbols_to_monitor} in background")
                        # Start with a short timeout for initial connection
                        result = alpaca_api.start_websocket(
                            symbols=symbols_to_monitor,
                            timeout=5.0, 
                            non_blocking=True
                        )
                        
                        if result:
                            self.logger.info(f"Websocket connection started successfully for {symbols_to_monitor}")
                        else:
                            self.logger.warning(f"Websocket connection failed to start for {symbols_to_monitor}, using REST API polling as fallback")
                    except Exception as e:
                        self.logger.error(f"Error starting websocket: {str(e)}")
                        self.logger.info("Continuing with REST API polling as fallback")
                
                # Start websocket thread - only done once per application instance
                self.logger.info(f"Starting websocket connection for {symbols_to_monitor} in background")
                ws_thread = threading.Thread(target=start_ws_in_background, daemon=True)
                ws_thread.start()
                self.logger.info("Websocket initialization triggered in background thread")
            else:
                self.logger.info("Websocket initialization already attempted, skipping redundant initialization")
        except Exception as e:
            self.logger.error(f"Error initializing Alpaca API: {str(e)}")
            apis['alpaca'] = None
        
        # Initialize Finnhub API
        try:
            self.logger.info("Initializing Finnhub API")
            finnhub_api = FinnhubAPI()
            apis['finnhub'] = finnhub_api
        except Exception as e:
            self.logger.error(f"Error initializing Finnhub API: {str(e)}")
            apis['finnhub'] = None
            
        # Initialize OpenAI API
        try:
            self.logger.info("Initializing OpenAI API")
            openai_api = OpenAIAPI()
            apis['openai'] = openai_api
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI API: {str(e)}")
            apis['openai'] = None
            
        # Initialize CoinLore API
        try:
            self.logger.info("Initializing CoinLore API")
            coinlore_api = CoinloreAPI()
            apis['coinlore'] = coinlore_api
        except Exception as e:
            self.logger.error(f"Error initializing CoinLore API: {str(e)}")
            apis['coinlore'] = None
        
        # Return the dictionary of API connections
        return apis
    
    async def get_market_data_async(self, symbol: str = None, timeframe: str = None, limit: int = 100) -> pd.DataFrame:
        """
        Get historical market data asynchronously.
        
        Args:
            symbol: Trading pair symbol (default: use default_pair)
            timeframe: Time interval (default: use default_timeframe)
            limit: Number of bars to retrieve
            
        Returns:
            DataFrame with market data
        """
        symbol = symbol or self.default_pair
        timeframe = timeframe or self.default_timeframe
        
        # If we have an async method in the API, use it
        if hasattr(self.apis['alpaca'], 'get_bars_async'):
            return await self.apis['alpaca'].get_bars_async(symbol, timeframe, limit)
        else:
            # Otherwise, run the synchronous method in a separate thread
            return await asyncio.to_thread(self.get_market_data, symbol, timeframe, limit)
    
    def get_market_data(
        self,
        symbol: str = None,
        timeframe: str = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading pair symbol (default: use default_pair)
            timeframe: Time interval (default: use default_timeframe)
            limit: Number of bars to retrieve
            
        Returns:
            DataFrame with market data
        """
        symbol = symbol or self.default_pair
        timeframe = timeframe or self.default_timeframe
        
        return self.apis['alpaca'].get_bars(symbol, timeframe, limit)
    
    def get_sentiment_data(self, symbol: str = None, market_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Get sentiment data for the specified trading symbol.
        
        Args:
            symbol: Trading symbol to get sentiment for (default: class default)
            market_data: Market data to use for volatility calculation
            
        Returns:
            Dictionary with sentiment data
        """
        # Use default symbol if not provided
        if symbol is None:
            symbol = self.default_pair
        
        try:
            # Initialize empty result dictionary with default values
            sentiment_data = {
                'sentiment_score': 0.0,
                'news_count': 0,
                'source': 'default',
                'timestamp': datetime.now()
            }
            
            # Check if we have a valid cache time and TTL
            current_time = time.time()
            last_cache_time = self.last_sentiment_cache_time.get(symbol, 0)
            
            # Calculate a dynamic TTL based on market volatility if we have market data
            sentiment_ttl = self.sentiment_data_ttl
            if market_data is not None and not market_data.empty:
                sentiment_ttl = self.calculate_dynamic_ttl(market_data)
                
            # If cache is still valid, get the real cached sentiment data instead of default
            if current_time - last_cache_time < sentiment_ttl and hasattr(self, 'sentiment_cache'):
                if symbol in getattr(self, 'sentiment_cache', {}):
                    self.logger.info(f"Using cached sentiment data for {symbol} ({int(current_time - last_cache_time)}s old, TTL: {sentiment_ttl}s)")
                    return self.sentiment_cache[symbol]
                
            # Try to get data from available APIs
            tries = 0
            max_tries = 3
            api_data_found = False
            
            while tries < max_tries and not api_data_found:
                try:
                    # Try Finnhub API first if available
                    if 'finnhub' in self.apis and self.apis['finnhub'] is not None:
                        finnhub_data = self.apis['finnhub'].get_aggregate_sentiment(symbol)
                        if finnhub_data and finnhub_data.get('sentiment_score', 0) != 0:
                            sentiment_data['sentiment_score'] = finnhub_data.get('sentiment_score', 0)
                            sentiment_data['news_count'] = finnhub_data.get('news_count', 0)
                            sentiment_data['source'] = 'finnhub'
                            api_data_found = True
                            break
                    
                    # Try OpenAI API as fallback
                    if not api_data_found and 'openai' in self.apis and self.apis['openai'] is not None:
                        openai_data = self.apis['openai'].get_sentiment(symbol)
                        if openai_data and openai_data.get('sentiment_score', 0) != 0:
                            sentiment_data['sentiment_score'] = openai_data.get('sentiment_score', 0)
                            sentiment_data['news_count'] = openai_data.get('news_count', 0)
                            sentiment_data['source'] = 'openai'
                            api_data_found = True
                            break
                    
                    # Use fallback values if neither API works
                    if not api_data_found:
                        sentiment_data['sentiment_score'] = -0.1  # Slightly bearish default
                        sentiment_data['news_count'] = 50  # Reasonable default
                        sentiment_data['source'] = 'fallback'
                        break
                        
                except Exception as inner_e:
                    self.logger.error(f"Error in attempt {tries+1} for sentiment data: {str(inner_e)}")
                    tries += 1
                    time.sleep(0.5)  # Short delay before retry
            
            # Update cache timestamp and store the sentiment data
            self.last_sentiment_cache_time[symbol] = current_time
            
            # Create sentiment_cache attribute if it doesn't exist
            if not hasattr(self, 'sentiment_cache'):
                self.sentiment_cache = {}
                
            # Store the new sentiment data in the cache
            self.sentiment_cache[symbol] = sentiment_data
            
            # Return the sentiment data
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment data: {str(e)}")
            # Always return a valid dictionary even on error
            return {
                'sentiment_score': -0.1,  # Slightly bearish default
                'news_count': 50,  # Reasonable default
                'source': 'error_fallback',
                'timestamp': datetime.now()
            }
    
    async def get_sentiment_data_async(self, symbol: str = None, market_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Get sentiment data asynchronously.
        Uses caching to reduce API calls.
        
        Args:
            symbol: Trading pair symbol (default: use default_pair)
            market_data: Optional market data for dynamic TTL calculation
            
        Returns:
            Dictionary with sentiment data
        """
        symbol = symbol or self.default_pair
        current_time = time.time()
        
        # Calculate dynamic TTL if market data is provided
        if market_data is not None and not market_data.empty:
            dynamic_ttl = self.calculate_dynamic_ttl(market_data)
            if dynamic_ttl != self.sentiment_cache_ttl:
                logger.info(f"Adjusting sentiment cache TTL from {self.sentiment_cache_ttl}s to {dynamic_ttl}s based on market conditions")
                self.sentiment_cache_ttl = dynamic_ttl
        
        # Check if we have cached data that's still valid
        if symbol in self.cached_sentiment_data and symbol in self.last_sentiment_update:
            time_since_update = current_time - self.last_sentiment_update[symbol]
            if time_since_update < self.sentiment_cache_ttl:
                logger.info(f"Using cached sentiment data for {symbol} ({time_since_update:.0f}s old, TTL: {self.sentiment_cache_ttl}s)")
                return self.cached_sentiment_data[symbol]
            else:
                logger.info(f"Cached sentiment data for {symbol} expired ({time_since_update:.0f}s old, TTL: {self.sentiment_cache_ttl}s)")
        
        # Extract base symbol (e.g., "ETH" from "ETH/USD")
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
        
        # If we don't have async methods in the APIs, use the synchronous version
        if not hasattr(self.apis['finnhub'], 'get_aggregate_sentiment_async') or \
           not hasattr(self.apis['coinlore'], 'get_on_chain_metrics_async'):
            return self.get_sentiment_data(symbol, market_data)
        
        try:
            # Run sentiment and on-chain metrics API calls concurrently
            async with asyncio.TaskGroup() as tg:
                finnhub_task = tg.create_task(
                    self.apis['finnhub'].get_aggregate_sentiment_async(base_symbol)
                )
                coinlore_task = tg.create_task(
                    self.apis['coinlore'].get_on_chain_metrics_async(base_symbol)
                )
            
            # Get results
            finnhub_sentiment = finnhub_task.result()
            coinlore_metrics = coinlore_task.result()
            
            # Combine data
            sentiment_data = {
                "symbol": symbol,
                "sentiment_score": finnhub_sentiment.get('sentiment_score', 0),
                "news_count": finnhub_sentiment.get('news_count', 0),
                "market_cap_usd": coinlore_metrics.get('market_cap_usd', 0),
                "volume_24h_usd": coinlore_metrics.get('volume_24h_usd', 0),
                "percent_change_24h": coinlore_metrics.get('percent_change_24h', 0),
                "percent_change_7d": coinlore_metrics.get('percent_change_7d', 0),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update cache
            self.cached_sentiment_data[symbol] = sentiment_data
            self.last_sentiment_update[symbol] = current_time
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error getting sentiment data asynchronously: {e}")
            
            # Return cached data if available, even if expired
            if symbol in self.cached_sentiment_data:
                logger.warning(f"Returning expired cached sentiment data for {symbol} due to API error")
                return self.cached_sentiment_data[symbol]
                
            # Return minimal data structure if no cached data
            return {
                "symbol": symbol,
                "sentiment_score": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        sentiment_data: Optional[Dict] = None
    ) -> Dict:
        """
        Generate trading signals from market data and sentiment.
        
        Args:
            market_data: Dataframe with market data
            sentiment_data: Dictionary with sentiment data
            
        Returns:
            Dictionary with signal information
        """
        try:
            if market_data.empty:
                logger.warning("Cannot generate signals: Market data is empty")
                return {"signal": 0, "signal_direction": "neutral", "signal_strength": 0}
            
            logger.info("Generating signals for " + self.default_pair)
            
            # Create copy of market data to avoid modifying original
            data = market_data.copy()
            
            # Calculate technical indicators using the static method
            data = TechnicalIndicators.add_all_indicators(
                data,
                self.config.get('technical_indicators', {})
            )
            
            # Calculate trend signal
            trend_signal = self.calculate_trend_signal(data)
            
            # Calculate momentum signal
            momentum_signal = self.calculate_momentum_signal(data)
            
            # Calculate volatility signal
            volatility_signal = self.calculate_volatility_signal(data)
            
            # Calculate volume signal
            volume_signal = self.calculate_volume_signal(data)
            
            # Log component signals
            logger.info(f"Component signals: trend={trend_signal:.4f}, momentum={momentum_signal:.4f}, volatility={volatility_signal:.4f}, volume={volume_signal:.4f}")
            
            # Combine signals with weights (default weights from config)
            weights = self.config.get('trading', {}).get('signal_weights', {})
            trend_weight = weights.get('trend', 0.4)
            momentum_weight = weights.get('momentum', 0.3)
            volatility_weight = weights.get('volatility', 0.1)
            volume_weight = weights.get('volume', 0.2)
            
            # Calculate composite signal from technical indicators
            composite_signal = (
                trend_signal * trend_weight +
                momentum_signal * momentum_weight +
                volatility_signal * volatility_weight +
                volume_signal * volume_weight
            )
            
            # Determine signal direction and initial strength
            if composite_signal > 0:
                signal_direction = "buy"
                trade_direction = "long"
                signal_strength = min(composite_signal, 1.0)  # Cap at 1.0
            elif composite_signal < 0:
                signal_direction = "sell"
                trade_direction = "short"
                signal_strength = min(abs(composite_signal), 1.0)  # Cap at 1.0
            else:
                signal_direction = "neutral"
                trade_direction = "none"
                signal_strength = 0.0
                
            logger.info(f"Generated {signal_direction} signal with strength {signal_strength:.4f} and direction {trade_direction}")
            
            # Add sentiment adjustment if available
            sentiment_adjustment = 0
            if sentiment_data:
                sentiment_score = sentiment_data.get('sentiment_score', 0)
                news_count = sentiment_data.get('news_count', 0)
                
                # Sentiment adjustment between -0.2 and 0.2 based on sentiment score
                sentiment_adjustment = sentiment_score * 0.2
                
                # Log sentiment data
                logger.info(f"Adding sentiment data - score: {sentiment_score:.4f}, news count: {news_count}")
                
                # Adjust signal based on sentiment
                adjusted_signal = composite_signal + sentiment_adjustment
                
                # Recalculate direction and strength
                if adjusted_signal > 0:
                    signal_direction = "buy"
                    trade_direction = "long"
                    signal_strength = min(adjusted_signal, 1.0)
                elif adjusted_signal < 0:
                    signal_direction = "sell"
                    trade_direction = "short"
                    signal_strength = min(abs(adjusted_signal), 1.0)
                else:
                    signal_direction = "neutral"
                    trade_direction = "none"
                    signal_strength = 0.0
            
            # Final signal values
            final_signal = signal_strength if signal_direction == "buy" else -signal_strength if signal_direction == "sell" else 0
            
            logger.info(f"Signal generated: {signal_direction} with strength {signal_strength:.4f}")
            
            # Create signals dictionary
            signals = {
                "signal": composite_signal,
                "signal_direction": signal_direction,
                "signal_strength": signal_strength,
                "timestamp": datetime.now().isoformat(),
                "symbol": self.default_pair,
                "components": {
                    "trend": trend_signal,
                    "momentum": momentum_signal,
                    "volatility": volatility_signal,
                    "volume": volume_signal,
                    "direction": trade_direction
                }
            }
            
            # Add sentiment data if available
            if sentiment_data:
                signals["sentiment_score"] = sentiment_data.get('sentiment_score', 0)
                signals["sentiment_adjustment"] = sentiment_adjustment
                signals["news_count"] = sentiment_data.get('news_count', 0)
            
            # Add market data summary
            try:
                latest_price = data['close'].iloc[-1]
                signals["price"] = latest_price
                signals["price_change_pct"] = data['close'].pct_change().iloc[-1] * 100
                
                # Add entry point if signal is strong enough
                if abs(final_signal) >= self.config.get('trading', {}).get('min_signal_strength', 0.2):
                    signals["entry_point"] = latest_price
            except:
                pass
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}", exc_info=True)
            return {"signal": 0, "signal_direction": "neutral", "signal_strength": 0}
    
    def should_trade(self, signals: Dict) -> bool:
        """
        Determine if we should execute a trade based on signals.
        
        Args:
            signals: Dictionary with signal information
            
        Returns:
            Boolean indicating whether to trade
        """
        # Extract the necessary values from signals
        signal = signals.get('signal', 0)
        signal_direction = signals.get('signal_direction', 'neutral')
        signal_strength = signals.get('signal_strength', 0)
        trade_direction = signals.get('components', {}).get('direction', 'none')
        
        # Get min signal strength from config
        min_signal_strength = self.config.get('trading', {}).get('min_signal_strength', 0.2)
        
        # Debug output to verify the threshold
        logger.warning(f"DEBUG: Using min_signal_strength={min_signal_strength} (config value)")
        
        # Check minimum strength requirement
        if signal_strength < min_signal_strength:
            logger.info(f"Signal strength {signal_strength:.4f} is below minimum threshold {min_signal_strength}")
            return False
        
        # Check for neutral signal
        if signal_direction == 'neutral':
            logger.info("Neutral signal - no trade")
            return False
            
        # Check if we have recent trades
        current_time = datetime.now()
        min_time_between_trades = self.config.get('trading', {}).get('min_time_between_trades', 15) # minutes
        
        if self.last_trade_time:
            time_since_last_trade = (current_time - self.last_trade_time).total_seconds() / 60.0
            if time_since_last_trade < min_time_between_trades:
                logger.info(f"Not enough time since last trade ({time_since_last_trade:.1f} min < {min_time_between_trades} min)")
                return False
        
        # Check if we already have a position for this symbol
        symbol = signals.get('symbol', self.default_pair)
        try:
            position = self.apis['alpaca'].get_position(symbol)
            
            if position:
                # Calculate position value
                position_qty = float(position.qty)
                current_price = float(position.current_price)
                position_value = position_qty * current_price
                position_is_long = position_qty > 0
                position_type = "LONG" if position_is_long else "SHORT"
                
                logger.info(f"Checking existing {position_type} position: {position_qty} {symbol} (value: ${abs(position_value):.2f})")
                
                # If we already have a position in the same direction, don't trade
                # Only skip if it's the same direction (long position and buy signal, or short position and sell signal)
                if (signal_direction == 'buy' and position_is_long) or (signal_direction == 'sell' and not position_is_long):
                    logger.info(f"Already have a {position_type} position for {symbol} matching the {signal_direction} signal - skipping trade")
                    return False
                
                # If position value is significant and signal is in opposite direction, we should trade to reverse the position
                min_position_value = 10  # Minimum position value to consider
                if abs(position_value) > min_position_value:
                    # If signal is in opposite direction of current position, we should trade
                    if (signal_direction == 'buy' and not position_is_long) or (signal_direction == 'sell' and position_is_long):
                        logger.info(f"Signal {signal_direction} is opposite to current {position_type} position - will execute trade to reverse position")
                        return True
                
        except Exception as e:
            error_message = str(e).lower()
            if "position does not exist" not in error_message:
                logger.warning(f"Error checking position for {symbol}: {str(e)}")
            else:
                logger.info(f"No existing position for {symbol} - will proceed if signal is strong enough")
        
        # If we got this far, we should trade
        logger.warning(f"Trade opportunity detected: {signal_direction} with strength {signal_strength:.4f}")
        return True
    
    def execute_trade(self, signals: Dict, symbol: str = None) -> Dict:
        """
        Execute a trade based on signals.
        
        Args:
            signals: Dictionary with signal information
            symbol: Symbol to trade (default: use default_pair)
            
        Returns:
            Dictionary with trade results
        """
        symbol = symbol or self.default_pair
        side = signals.get('signal_direction', '').lower()
        trade_direction = signals.get('components', {}).get('direction', 'long' if side == 'buy' else 'short' if side == 'sell' else 'none')
        
        if not side or side not in ['buy', 'sell']:
            logger.error(f"Trade execution failed: Invalid side '{side}'")
            return {
                "executed": False,
                "error": f"Invalid side '{side}'"
            }
            
        try:
            # Get account information
            account = self.apis['alpaca'].get_account()
            account_balance = float(account.cash)
            available_balance = float(account.buying_power)
            portfolio_value = float(account.portfolio_value)
            
            # Log account status info for debugging
            logger.info(f"Account status - Cash: ${account_balance:.2f}, Buying Power: ${available_balance:.2f}, Portfolio Value: ${portfolio_value:.2f}")
            
            # Check if buying power is critically low
            if available_balance < 20 and side == 'buy':
                logger.error(f"Buying power (${available_balance:.2f}) is too low for meaningful trades.")
                return {
                    "executed": False,
                    "error": f"Buying power (${available_balance:.2f}) is too low for meaningful trades."
                }
            
            # Check existing position
            has_existing_position = False
            existing_position_size = 0
            existing_position_value = 0
            existing_avg_entry = 0
            existing_current_price = 0
            position_is_long = False
            
            try:
                # Get detailed position information
                position = self.apis['alpaca'].get_position(symbol)
                if position:
                    has_existing_position = True
                    existing_position_size = float(position.qty)
                    existing_position_value = float(position.market_value)
                    existing_avg_entry = float(position.avg_entry_price) if hasattr(position, 'avg_entry_price') else 0
                    existing_current_price = float(position.current_price) if hasattr(position, 'current_price') else 0
                    position_is_long = existing_position_size > 0
                    
                    # Calculate what percentage of portfolio this position represents
                    portfolio_allocation = existing_position_value / portfolio_value * 100 if portfolio_value > 0 else 0
                    
                    position_type = "LONG" if position_is_long else "SHORT"
                    logger.info(f"Existing {position_type} position found: {existing_position_size} {symbol} at avg entry ${existing_avg_entry:.2f}")
                    logger.info(f"Position value: ${existing_position_value:.2f} ({portfolio_allocation:.1f}% of portfolio)")
                    
                    # Check if we need to close the existing position first due to opposite signal
                    need_to_close_first = (side == 'buy' and not position_is_long) or (side == 'sell' and position_is_long)
                    
                    if need_to_close_first:
                        close_side = 'buy' if not position_is_long else 'sell'
                        logger.info(f"Closing existing {position_type} position before opening new {'LONG' if side == 'buy' else 'SHORT'} position")
                        
                        # Close the existing position with a market order
                        close_order = self.order_manager.execute_trade(
                            symbol=symbol,
                            side=close_side,
                            qty=abs(existing_position_size),
                            order_type="market"
                        )
                        
                        if close_order.get('error'):
                            logger.error(f"Failed to close existing position: {close_order.get('error')}")
                            return {
                                "executed": False,
                                "error": f"Failed to close existing position: {close_order.get('error')}"
                            }
                        
                        logger.info(f"Position closed: {close_order}")
                        
                        # Small delay to ensure the close order is processed
                        time.sleep(0.5)
                        
                        # Reset position flags since we've closed it
                        has_existing_position = False
                        existing_position_size = 0
                        existing_position_value = 0
                    elif (side == 'buy' and position_is_long) or (side == 'sell' and not position_is_long):
                        # If we already have a position in the same direction with significant allocation, don't add more
                        if portfolio_allocation > 90:
                            logger.warning(f"Position already represents {portfolio_allocation:.1f}% of portfolio - skipping additional {side}.")
                            return {
                                "executed": False,
                                "error": f"Position already represents {portfolio_allocation:.1f}% of portfolio - adding more would be too concentrated."
                            }
            except Exception as e:
                error_msg = str(e).lower()
                # Only log as warning if it's a 'position does not exist' error
                if 'position does not exist' in error_msg:
                    logger.info(f"No existing position for {symbol} - proceeding with {side} order")
                else:
                    logger.warning(f"Error checking existing position: {str(e)}")
            
            # Use available buying power for orders and respect position sizing limits
            # For buy (long) orders
            if side == 'buy':
                # Ensure we use no more than available buying power
                position_sizing_balance = available_balance
                # Add more conservative safety buffer (10% below available)
                safe_buy_amount = position_sizing_balance * 0.9
                
                # For extremely low buying power, use an even more conservative approach
                if position_sizing_balance < 100:
                    # Use only 70% of available buying power for very small amounts
                    safe_buy_amount = position_sizing_balance * 0.7
                    logger.info(f"Very low buying power (${position_sizing_balance:.2f}). Using conservative position sizing (70%)")
                
                logger.info(f"Using buy amount of ${safe_buy_amount:.2f} (from buying power ${position_sizing_balance:.2f})")
            else:
                # For sell (short) orders
                if has_existing_position and position_is_long:
                    # This is closing an existing long position
                    position_sizing_balance = existing_position_value
                    safe_buy_amount = position_sizing_balance
                    logger.info(f"Selling entire position: {existing_position_size} {symbol} worth ${existing_position_value:.2f}")
                else:
                    # This is opening a new short position
                    logger.info(f"Opening new short position for {symbol} with direction: {trade_direction}")
                    
                    # Use position sizing similar to buy orders but for shorts
                    position_sizing_balance = available_balance
                    # Add more conservative safety buffer (10% below available)
                    safe_buy_amount = position_sizing_balance * 0.9
                    
                    # For extremely low buying power, use an even more conservative approach
                    if position_sizing_balance < 100:
                        # Use only 70% of available buying power for very small amounts
                        safe_buy_amount = position_sizing_balance * 0.7
                        logger.info(f"Very low buying power (${position_sizing_balance:.2f}). Using conservative position sizing (70%) for short")
                    
                    logger.info(f"Using short position amount of ${safe_buy_amount:.2f} (from buying power ${position_sizing_balance:.2f})")
            
            # Check if user settings are available (look for a state file)
            strategy_mode = "Moderate"  # Default value
            try:
                state_file = 'config/app_state.json'
                if os.path.exists(state_file):
                    with open(state_file, 'r') as f:
                        state_data = json.load(f)
                        if 'strategy_mode' in state_data:
                            strategy_mode = state_data['strategy_mode']
                            logger.info(f"Using strategy mode from settings: {strategy_mode}")
            except Exception as e:
                logger.warning(f"Could not load strategy mode from settings: {str(e)}")
            
            # Decide whether to use market price
            if signals.get('entry_point'):
                price = signals.get('entry_point')
            elif signals.get('price'):
                price = signals.get('price')
            elif signals.get('current_price'):
                price = signals.get('current_price')
            else:
                # Get current price from websocket if available
                if hasattr(self.apis['alpaca'], 'get_current_price'):
                    price = self.apis['alpaca'].get_current_price(symbol)
                else:
                    # Fallback to getting the latest market data
                    latest_data = self.get_market_data(symbol, '1Min', 1)
                    price = latest_data['close'].iloc[-1] if not latest_data.empty else None
            
            # Skip the trade if we couldn't determine a price
            if not price:
                logger.error("Trade execution failed: Could not determine current price")
                return {
                    "executed": False,
                    "error": "Could not determine current price"
                }
            
            # Calculate position size in quantity
            if side == 'buy':
                # Long position
                # Add a safety margin to account for potential price fluctuations
                adjusted_price = price * 1.005  # Add 0.5% buffer for potential price fluctuations
                position_size = safe_buy_amount / adjusted_price
                
                # Make sure the position size is not too small
                if position_size * price < 10 and position_size * price < safe_buy_amount * 0.9:
                    logger.warning(f"Position value (${position_size * price:.2f}) is below minimum. Skipping trade.")
                    return {
                        "executed": False,
                        "error": f"Position value (${position_size * price:.2f}) is below minimum threshold of $10"
                    }
            else:
                # Short position or close long
                if has_existing_position and position_is_long:
                    # Closing an existing long position
                    position_size = existing_position_size
                else:
                    # Opening a new short position
                    adjusted_price = price * 0.995  # Subtract 0.5% buffer for potential price fluctuations
                    position_size = safe_buy_amount / adjusted_price
                    
                    # Make sure the position size is not too small
                    if position_size * price < 10 and position_size * price < safe_buy_amount * 0.9:
                        logger.warning(f"Short position value (${position_size * price:.2f}) is below minimum. Skipping trade.")
                        return {
                            "executed": False,
                            "error": f"Short position value (${position_size * price:.2f}) is below minimum threshold of $10"
                        }
            
            # Log position sizing details
            if side == 'buy':
                logger.info(f"Buy position: {position_size:.6f} {symbol} at ${price:.2f} (value: ${position_size * price:.2f})")
            else:
                if has_existing_position and position_is_long:
                    logger.info(f"Sell existing long position: {position_size:.6f} {symbol} at ${price:.2f} (value: ${position_size * price:.2f})")
                else:
                    logger.info(f"Open short position: {position_size:.6f} {symbol} at ${price:.2f} (value: ${position_size * price:.2f})")
            
            logger.info(f"Strategy mode: {strategy_mode}")
            
            # Try to execute the order, with retry logic for size issues
            order = None
            max_retries = 3
            retry_count = 0
            current_position_size = position_size
            
            while retry_count < max_retries:
                try:
                    # Execute the order
                    order = self.order_manager.execute_trade(
                        symbol=symbol,
                        side=side,
                        qty=current_position_size,
                        order_type="market"
                    )
                    
                    # Check if the order was rejected
                    if order.get('status') == 'rejected' or order.get('error'):
                        error_msg = order.get('error', 'Unknown error')
                        
                        # Check if it's an insufficient buying power error
                        if 'insufficient' in error_msg.lower() and retry_count < max_retries - 1:
                            # Reduce position size by 15% and try again
                            retry_count += 1
                            current_position_size *= 0.85
                            logger.warning(f"Retrying with reduced position size: {current_position_size:.6f} (attempt {retry_count}/{max_retries})")
                            continue
                        else:
                            # Non-retryable error or out of retries
                            logger.error(f"Order rejected: {error_msg}")
                            return {
                                "executed": False,
                                "error": error_msg,
                                "attempted_qty": current_position_size,
                                "price": price
                            }
                    else:
                        # Order accepted, break out of retry loop
                        break
                    
                except Exception as e:
                    # Handle unexpected exceptions
                    error_message = str(e)
                    logger.error(f"Unexpected error executing trade: {error_message}")
                    
                    # Check if it might be a sizing issue we can retry
                    if 'insufficient' in error_message.lower() and retry_count < max_retries - 1:
                        retry_count += 1
                        current_position_size *= 0.85
                        logger.warning(f"Error suggests sizing issue. Retrying with reduced size: {current_position_size:.6f} (attempt {retry_count}/{max_retries})")
                        continue
                    else:
                        # Non-retryable error or out of retries
                        return {
                            "executed": False,
                            "error": f"Trade execution error: {error_message}",
                            "attempted_qty": current_position_size,
                            "price": price
                        }
            
            # Check if the order was successful
            if not order or order.get('status') == 'rejected' or order.get('error'):
                error_msg = order.get('error', 'Unknown error') if order else "Failed to place order"
                logger.error(f"Trade execution failed: {error_msg}")
                return {
                    "executed": False,
                    "error": error_msg,
                    "attempted_qty": current_position_size,
                    "price": price
                }
            
            # Update last trade time ONLY on successful orders
            self.last_trade_time = datetime.now()
            
            # Log the executed trade with specific action type
            if side == 'buy':
                trade_action = "BUY (LONG)"
            else:
                if has_existing_position and position_is_long:
                    trade_action = "SELL (CLOSE LONG)"
                else:
                    trade_action = "SELL (SHORT)"
            
            logger.info(f"Trade executed: {trade_action} {current_position_size} {symbol} at ~${price:.2f}")
            
            return {
                "executed": True,
                "trade": order,
                "side": side,
                "direction": trade_direction,
                "qty": current_position_size,
                "price": price,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "strategy_mode": strategy_mode,
                "order_id": order.get('id', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}", exc_info=True)
            return {
                "executed": False,
                "error": str(e)
            }
    
    async def run_trading_cycle_async(self, symbol: str = None, timeframe: str = None) -> Dict:
        """
        Run a complete trading cycle asynchronously.
        
        Args:
            symbol: Trading pair symbol (default: use default_pair)
            timeframe: Time interval (default: use default_timeframe)
            
        Returns:
            Dictionary with cycle results
        """
        symbol = symbol or self.default_pair
        timeframe = timeframe or self.default_timeframe
        
        # Generate cycle identifier
        self.cycle_count += 1
        cycle_id = f"Cycle-{self.cycle_count}"
        cycle_start_time = datetime.now()
        
        logger.info(f"[{cycle_id}] Starting async trading cycle for {symbol} on {timeframe} timeframe")
        
        try:
            # Try to use current price from websocket first
            current_price = None
            if hasattr(self.apis['alpaca'], 'get_current_price'):
                current_price = self.apis['alpaca'].get_current_price(symbol)
                if current_price:
                    logger.info(f"[{cycle_id}] Using real-time price: ${current_price:.2f}")
            
            # Fetch market data first
            logger.info(f"[{cycle_id}] Fetching market data for {symbol}")
            market_data = await asyncio.to_thread(self.get_market_data, symbol, timeframe)
            
            # If we got a real-time price, add it to the market data
            if current_price and not market_data.empty:
                # Update the latest close price with real-time data
                market_data.iloc[-1, market_data.columns.get_loc('close')] = current_price
            
            # Now fetch sentiment data with market data for dynamic TTL
            logger.info(f"[{cycle_id}] Fetching sentiment data for {symbol}")
            sentiment_data = await self.get_sentiment_data_async(symbol, market_data)
            
            # Generate signals
            signals = self.generate_signals(market_data, sentiment_data)
            
            # If we have a current price from websocket, add it to signals
            if current_price:
                signals['current_price'] = current_price
            
            # Ensure signal strength is consistently reported
            signal_strength = signals.get('signal_strength', 0)
            signal_direction = signals.get('signal_direction', 'none')
            
            # Check if we should trade
            should_trade = self.should_trade(signals)
            
            # Execute trade if conditions are met
            trade_result = None
            trade_status = "no_trade"
            trade_message = "No trade executed"
            trade_qty = 0
            trade_price = current_price or 0
            order_id = "none"
            
            if should_trade:
                logger.info(f"[{cycle_id}] Trade conditions met for {symbol}, proceeding with execution")
                trade_result = self.execute_trade(signals, symbol)
                
                # Process and record trade outcome
                if trade_result.get('executed', False):
                    trade_status = "executed"
                    trade_message = f"Trade executed: {trade_result.get('side', 'unknown')} {trade_result.get('qty', 0)} {symbol}"
                    trade_qty = trade_result.get('qty', 0)
                    trade_price = trade_result.get('price', current_price or 0)
                    order_id = trade_result.get('order_id', 'unknown')
                    
                    # Record successful trade in log with clear details
                    logger.info(f"[{cycle_id}] {trade_message} at ${trade_price:.2f}, Order ID: {order_id}")
                else:
                    # Trade was attempted but failed
                    trade_status = "failed"
                    error_message = trade_result.get('error', 'Unknown error')
                    trade_message = f"Trade failed: {error_message}"
                    trade_qty = trade_result.get('attempted_qty', 0)
                    
                    # Log failed trade clearly
                    logger.warning(f"[{cycle_id}] {trade_message}")
            else:
                logger.info(f"[{cycle_id}] Trade conditions not met for {symbol}, skipping execution")
                logger.info(f"[{cycle_id}] No trade executed, continuing to monitor markets")
            
            # Calculate cycle duration
            cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
            logger.info(f"[{cycle_id}] Async trading cycle completed in {cycle_duration:.2f}s for {symbol}")
            
            # Create cycle result with consistent signal information
            cycle_result = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": current_price or (market_data['close'].iloc[-1] if not market_data.empty else 0),
                "signals": signals,
                "signal_strength": signal_strength,
                "signal_direction": signal_direction,
                "should_trade": should_trade,
                "trade_result": trade_result,
                "trade_status": trade_status,
                "trade_message": trade_message,
                "trade_qty": trade_qty,
                "trade_price": trade_price,
                "order_id": order_id,
                "cycle_id": cycle_id,
                "duration_seconds": cycle_duration
            }
            
            # Append additional logging for debugging (similar to non-async version)
            logger.info(f"[{cycle_id}-ASYNC-{datetime.now().strftime('%Y%m%d%H%M%S')}] Completed trading cycle")
            logger.info(f"[{cycle_id}-ASYNC-{datetime.now().strftime('%Y%m%d%H%M%S')}] Current {symbol} price: {cycle_result['current_price']}")
            logger.info(f"[{cycle_id}-ASYNC-{datetime.now().strftime('%Y%m%d%H%M%S')}] Strategy is looking for a {signal_direction.upper()} opportunity")
            logger.info(f"[{cycle_id}-ASYNC-{datetime.now().strftime('%Y%m%d%H%M%S')}] Signal: {signal_direction} ({signal_strength:.4f}), Direction: {signals.get('components', {}).get('direction', 'unknown')}, Strength: {signal_strength:.4f}")
            
            if should_trade:
                if trade_status == "executed":
                    logger.info(f"[{cycle_id}-ASYNC-{datetime.now().strftime('%Y%m%d%H%M%S')}] Trade executed: {trade_result.get('side')} {trade_qty} {symbol} at ${trade_price:.2f}, Order ID: {order_id}")
                else:
                    error_message = trade_result.get('error', 'Unknown error') if trade_result else 'Unknown error'
                    logger.warning(f"[{cycle_id}-ASYNC-{datetime.now().strftime('%Y%m%d%H%M%S')}] Trade opportunity detected: {signal_direction} with strength {signal_strength:.4f}")
                    logger.warning(f"[{cycle_id}-ASYNC-{datetime.now().strftime('%Y%m%d%H%M%S')}] Trade failed: {error_message}")
            else:
                logger.info(f"[{cycle_id}-ASYNC-{datetime.now().strftime('%Y%m%d%H%M%S')}] No trade executed, continuing to monitor markets")
                
            logger.info(f"[{cycle_id}-ASYNC-{datetime.now().strftime('%Y%m%d%H%M%S')}] Trading cycle completed in {cycle_duration:.2f} seconds.")
            
            return cycle_result
            
        except Exception as e:
            cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
            logger.error(f"[{cycle_id}] Error in async trading cycle for {symbol}: {str(e)}", exc_info=True)
            return {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "error": str(e),
                "cycle_id": cycle_id,
                "duration_seconds": cycle_duration,
                "trade_status": "error"
            }
    
    def run_trading_cycle(self, symbol: str = None, timeframe: str = None) -> Dict:
        """
        Run a complete trading cycle (data -> signals -> trade decision).
        
        Args:
            symbol: Trading pair symbol (default: use default_pair)
            timeframe: Time interval (default: use default_timeframe)
            
        Returns:
            Dictionary with cycle results
        """
        symbol = symbol or self.default_pair
        timeframe = timeframe or self.default_timeframe
        
        # Generate cycle identifier
        self.cycle_count += 1
        cycle_id = f"Cycle-{self.cycle_count}"
        cycle_start_time = datetime.now()
        
        logger.info(f"[{cycle_id}] Starting trading cycle for {symbol} on {timeframe} timeframe")
        
        try:
            # Try to use current price from websocket first
            current_price = None
            if hasattr(self.apis['alpaca'], 'get_current_price'):
                current_price = self.apis['alpaca'].get_current_price(symbol)
                if current_price:
                    logger.info(f"[{cycle_id}] Using real-time price: ${current_price:.2f}")
            
            # Get market data
            logger.info(f"[{cycle_id}] Fetching market data for {symbol}")
            market_data = self.get_market_data(symbol, timeframe)
            
            # If we got a real-time price, add it to the market data
            if current_price and not market_data.empty:
                # Update the latest close price with real-time data
                market_data.iloc[-1, market_data.columns.get_loc('close')] = current_price
            
            # Get sentiment data (utilizing cache if available)
            logger.info(f"[{cycle_id}] Fetching sentiment data for {symbol}")
            sentiment_data = self.get_sentiment_data(symbol, market_data)
            
            # Generate signals
            signals = self.generate_signals(market_data, sentiment_data)
            
            # If we have a current price from websocket, add it to signals
            if current_price:
                signals['current_price'] = current_price
            
            # Ensure signal strength is consistently reported
            signal_strength = signals.get('signal_strength', 0)
            signal_direction = signals.get('signal_direction', 'none')
            
            # Check if we should trade
            should_trade = self.should_trade(signals)
            if should_trade:
                logger.info(f"[{cycle_id}] Trade conditions met for {symbol}, proceeding with execution")
            else:
                logger.info(f"[{cycle_id}] Trade conditions not met for {symbol}, skipping execution")
            
            # Execute trade if conditions are met
            trade_result = None
            trade_status = "no_trade"
            trade_message = "No trade executed"
            trade_qty = 0
            trade_price = current_price or 0
            order_id = "none"
            
            if should_trade:
                trade_result = self.execute_trade(signals, symbol)
                
                # Process and record trade outcome
                if trade_result.get('executed', False):
                    trade_status = "executed"
                    trade_message = f"Trade executed: {trade_result.get('side', 'unknown')} {trade_result.get('qty', 0)} {symbol}"
                    trade_qty = trade_result.get('qty', 0)
                    trade_price = trade_result.get('price', current_price or 0)
                    order_id = trade_result.get('order_id', 'unknown')
                    
                    # Record successful trade in log with clear details
                    logger.info(f"[{cycle_id}] {trade_message} at ${trade_price:.2f}, Order ID: {order_id}")
                else:
                    # Trade was attempted but failed
                    trade_status = "failed"
                    error_message = trade_result.get('error', 'Unknown error')
                    trade_message = f"Trade failed: {error_message}"
                    trade_qty = trade_result.get('attempted_qty', 0)
                    
                    # Log failed trade clearly
                    logger.warning(f"[{cycle_id}] {trade_message}")
            else:
                # Log that no trade was executed
                logger.info(f"[{cycle_id}] No trade executed, continuing to monitor markets")
            
            # Calculate cycle duration
            cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
            logger.info(f"[{cycle_id}] Trading cycle completed in {cycle_duration:.2f}s for {symbol}")
            
            # Create cycle result with consistent signal information
            cycle_result = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": current_price or (market_data['close'].iloc[-1] if not market_data.empty else 0),
                "signals": signals,
                "signal_strength": signal_strength,
                "signal_direction": signal_direction,
                "should_trade": should_trade,
                "trade_result": trade_result,
                "trade_status": trade_status,
                "trade_message": trade_message,
                "trade_qty": trade_qty,
                "trade_price": trade_price,
                "order_id": order_id,
                "cycle_id": cycle_id,
                "duration_seconds": cycle_duration
            }
            
            # Append additional logging for debugging
            logger.info(f"[{cycle_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}] Completed trading cycle")
            logger.info(f"[{cycle_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}] Current {symbol} price: {cycle_result['current_price']}")
            logger.info(f"[{cycle_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}] Strategy is looking for a {signal_direction.upper()} opportunity")
            logger.info(f"[{cycle_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}] Signal: {signal_direction} ({signal_strength:.4f}), Direction: {signals.get('components', {}).get('direction', 'unknown')}, Strength: {signal_strength:.4f}")
            
            if should_trade:
                if trade_status == "executed":
                    logger.info(f"[{cycle_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}] Trade executed: {trade_result.get('side')} {trade_qty} {symbol} at ${trade_price:.2f}, Order ID: {order_id}")
                else:
                    error_message = trade_result.get('error', 'Unknown error') if trade_result else 'Unknown error'
                    logger.warning(f"[{cycle_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}] Trade opportunity detected: {signal_direction} with strength {signal_strength:.4f}")
                    logger.warning(f"[{cycle_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}] Trade failed: {error_message}")
            else:
                logger.info(f"[{cycle_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}] No trade executed, continuing to monitor markets")
                
            logger.info(f"[{cycle_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}] Trading cycle completed in {cycle_duration:.2f} seconds. Next check at {(datetime.now() + timedelta(seconds=15)).strftime('%H:%M:%S')}")
            
            return cycle_result
            
        except Exception as e:
            cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
            logger.error(f"[{cycle_id}] Error in trading cycle for {symbol}: {str(e)}", exc_info=True)
            return {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "error": str(e),
                "cycle_id": cycle_id,
                "duration_seconds": cycle_duration,
                "trade_status": "error"
            }
    
    def start_trading(self) -> bool:
        """
        Start the trading strategy.
        
        Returns:
            True if started successfully, False otherwise
        """
        self.is_active = True
        logger.info(f"Trading started with default pair: {self.default_pair}, timeframe: {self.default_timeframe}")
        return True
    
    def stop_trading(self) -> bool:
        """
        Stop the trading strategy.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        self.is_active = False
        logger.info("Trading stopped")
        return True
    
    def get_trading_status(self) -> Dict:
        """
        Get current trading status.
        
        Returns:
            Dictionary with trading status
        """
        # Get account information
        account = self.order_manager.get_account_info()
        
        # Get positions
        positions = self.order_manager.get_positions()
        
        # Get recent trades
        recent_trades = self.order_manager.get_recent_trades()
        
        return {
            "is_active": self.is_active,
            "account": account,
            "positions": positions,
            "recent_trades": recent_trades,
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "current_signal": self.current_signal,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_dynamic_ttl(self, market_data: pd.DataFrame) -> int:
        """
        Calculate a dynamic TTL for sentiment data based on market conditions.
        
        During high volatility periods, we want a shorter TTL to update more frequently.
        During stable market conditions, we can use a longer TTL.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            TTL in seconds
        """
        # Extract symbol from market data if available, otherwise use default
        symbol = market_data['symbol'].iloc[0] if 'symbol' in market_data.columns else self.default_pair
        
        try:
            # Calculate a simple volatility measure (standard deviation of returns)
            if len(market_data) >= 10:
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.std()
                
                # Update tracked volatility for this symbol
                self.market_volatility[symbol] = volatility
                
                # Base TTL from config
                base_ttl = self.config.get('data_optimization', {}).get('cache_ttl', 60)
                
                # In highly volatile markets (volatility > 0.01 or 1%), reduce TTL
                if volatility > 0.01:
                    # Inverse relationship: higher volatility -> lower TTL
                    # Minimum TTL is 30 seconds
                    ttl = max(30, int(base_ttl * (0.01 / volatility)))
                    logger.info(f"High market volatility ({volatility:.4f}), setting shorter sentiment TTL: {ttl}s")
                    return ttl
                else:
                    # In stable markets, use a longer TTL (up to 3x base)
                    ttl = min(base_ttl * 3, int(base_ttl * (0.01 / max(0.001, volatility))))
                    logger.info(f"Stable market ({volatility:.4f}), setting longer sentiment TTL: {ttl}s")
                    return ttl
        except Exception as e:
            logger.error(f"Error calculating dynamic TTL: {str(e)}")
        
        # Default TTL (3x base TTL) if calculation fails
        return self.config.get('data_optimization', {}).get('cache_ttl', 60) * 3
    
    def calculate_trend_signal(self, data: pd.DataFrame) -> float:
        """
        Calculate trend signal from price data.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Signal value between -1 and 1 (positive = bullish)
        """
        try:
            # Default signal if no indicators available
            if data.empty:
                return 0
                
            signal = 0
            count = 0
            
            # Check for EMA crossovers (EMA20 vs EMA50)
            if 'ema_20' in data.columns and 'ema_50' in data.columns:
                # Get the most recent values
                ema_20 = data['ema_20'].iloc[-1]
                ema_50 = data['ema_50'].iloc[-1]
                
                # Previous values (to check for crossovers)
                prev_ema_20 = data['ema_20'].iloc[-2] if len(data) > 2 else ema_20
                prev_ema_50 = data['ema_50'].iloc[-2] if len(data) > 2 else ema_50
                
                # Current relation
                if ema_20 > ema_50:
                    # Bullish trend
                    signal += 0.3
                    
                    # Check for recent crossover (stronger signal)
                    if prev_ema_20 <= prev_ema_50:
                        signal += 0.2  # Recent bullish crossover
                else:
                    # Bearish trend
                    signal -= 0.3
                    
                    # Check for recent crossover (stronger signal)
                    if prev_ema_20 >= prev_ema_50:
                        signal -= 0.2  # Recent bearish crossover
                        
                count += 1
            
            # Check price relative to moving averages
            if 'close' in data.columns and 'sma_50' in data.columns:
                current_price = data['close'].iloc[-1]
                sma_50 = data['sma_50'].iloc[-1]
                
                # Price above/below SMA50
                if current_price > sma_50:
                    signal += 0.2
                else:
                    signal -= 0.2
                    
                count += 1
            
            # Add Bollinger Bands signal
            if 'close' in data.columns and 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                current_price = data['close'].iloc[-1]
                bb_upper = data['bb_upper'].iloc[-1]
                bb_lower = data['bb_lower'].iloc[-1]
                bb_middle = data['bb_middle'].iloc[-1] if 'bb_middle' in data.columns else None
                
                # Calculate distance from bands
                band_width = bb_upper - bb_lower
                if band_width > 0:
                    # Normalize position within bands (-1 to +1)
                    if bb_middle is not None:
                        # Use position relative to middle band
                        band_position = (current_price - bb_middle) / (band_width / 2)
                        # Cap at -1 to +1
                        band_position = max(-1, min(1, band_position))
                        signal += band_position * 0.2
                    else:
                        # Fallback to simple band position
                        if current_price > bb_upper:
                            signal += 0.3  # Overbought
                        elif current_price < bb_lower:
                            signal -= 0.3  # Oversold
                count += 1
            
            # Normalize signal by count of indicators used
            if count > 0:
                normalized_signal = signal / count
                # Cap at -1 to +1
                return max(-1, min(1, normalized_signal))
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error calculating trend signal: {str(e)}")
            return 0
    
    def calculate_momentum_signal(self, data: pd.DataFrame) -> float:
        """
        Calculate momentum signal from price data.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Signal value between -1 and 1 (positive = bullish momentum)
        """
        try:
            # Default signal if no indicators available
            if data.empty:
                return 0
                
            signal = 0
            count = 0
            
            # RSI signal
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                
                # Convert RSI to a -1 to +1 signal
                # RSI 70+ = overbought (-0.5)
                # RSI 30- = oversold (+0.5)
                # RSI 50 = neutral (0)
                if rsi >= 70:
                    signal -= 0.5  # Overbought
                elif rsi <= 30:
                    signal += 0.5  # Oversold
                else:
                    # Linear interpolation between 30-70
                    signal += (50 - rsi) / 40  # -0.5 to +0.5
                
                count += 1
            
            # MACD signal
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd = data['macd'].iloc[-1]
                macd_signal = data['macd_signal'].iloc[-1]
                
                # MACD crossing above signal line = bullish
                # MACD crossing below signal line = bearish
                if macd > macd_signal:
                    signal += 0.3
                    
                    # Check for recent crossover
                    if len(data) > 2 and data['macd'].iloc[-2] <= data['macd_signal'].iloc[-2]:
                        signal += 0.2  # Recent bullish crossover
                else:
                    signal -= 0.3
                    
                    # Check for recent crossover
                    if len(data) > 2 and data['macd'].iloc[-2] >= data['macd_signal'].iloc[-2]:
                        signal -= 0.2  # Recent bearish crossover
                
                count += 1
            
            # Stochastic oscillator
            if 'stoch_k' in data.columns and 'stoch_d' in data.columns:
                k = data['stoch_k'].iloc[-1]
                d = data['stoch_d'].iloc[-1]
                
                # Overbought/oversold
                if k > 80 and d > 80:
                    signal -= 0.4  # Overbought
                elif k < 20 and d < 20:
                    signal += 0.4  # Oversold
                
                # Crossover signal
                if k > d:
                    signal += 0.2
                    
                    # Recent crossover check
                    if len(data) > 2 and data['stoch_k'].iloc[-2] <= data['stoch_d'].iloc[-2]:
                        signal += 0.2  # Recent bullish crossover
                else:
                    signal -= 0.2
                    
                    # Recent crossover check
                    if len(data) > 2 and data['stoch_k'].iloc[-2] >= data['stoch_d'].iloc[-2]:
                        signal -= 0.2  # Recent bearish crossover
                
                count += 1
            
            # Normalize signal by count of indicators used
            if count > 0:
                normalized_signal = signal / count
                # Cap at -1 to +1
                return max(-1, min(1, normalized_signal))
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error calculating momentum signal: {str(e)}")
            return 0
    
    def calculate_volatility_signal(self, data: pd.DataFrame) -> float:
        """
        Calculate volatility signal from price data.
        Used primarily for risk assessment rather than direction.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Signal value between -1 and 1 (higher absolute value = higher volatility)
        """
        try:
            # Default signal if no indicators available
            if data.empty:
                return 0
                
            # Calculate historical volatility
            if 'close' in data.columns and len(data) > 5:
                # Calculate returns
                returns = data['close'].pct_change().dropna()
                
                # Calculate rolling standard deviation (volatility)
                volatility = returns.rolling(window=14).std().iloc[-1] if len(returns) >= 14 else returns.std()
                
                # Annualized volatility
                annualized_volatility = volatility * (252 ** 0.5)  # 252 trading days 
                
                # Convert to signal (-1 to +1)
                # For volatility, we use 0 to represent normal volatility
                # Negative values indicate unusually low volatility (coiling for a move)
                # Positive values indicate unusually high volatility (potential reversals)
                
                # Baseline volatility for crypto (adjust as needed)
                baseline_volatility = 0.6  # 60% annual volatility as baseline
                
                if annualized_volatility < baseline_volatility * 0.5:
                    # Unusually low volatility - potential for breakout
                    return -0.5
                elif annualized_volatility > baseline_volatility * 2:
                    # Unusually high volatility - potential for reversal
                    return 0.5
                else:
                    # Normal volatility range - scale between -0.3 and +0.3
                    normalized = ((annualized_volatility / baseline_volatility) - 1) * 0.3
                    return max(-0.3, min(0.3, normalized))
            
            # ATR can also be used if present
            if 'atr' in data.columns and 'close' in data.columns:
                # Calculate ATR as percentage of price
                atr_pct = data['atr'].iloc[-1] / data['close'].iloc[-1] * 100  # ATR as percentage of price
                
                # Baseline ATR percentage for crypto
                baseline_atr_pct = 3.0  # 3% daily ATR as baseline
                
                if atr_pct < baseline_atr_pct * 0.5:
                    # Unusually low volatility
                    return -0.5
                elif atr_pct > baseline_atr_pct * 2:
                    # Unusually high volatility
                    return 0.5
                else:
                    # Normal volatility range
                    normalized = ((atr_pct / baseline_atr_pct) - 1) * 0.3
                    return max(-0.3, min(0.3, normalized))
            
            # If no volatility indicators available
            return 0
                
        except Exception as e:
            logger.error(f"Error calculating volatility signal: {str(e)}")
            return 0
    
    def calculate_volume_signal(self, data: pd.DataFrame) -> float:
        """
        Calculate volume signal from price data.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Signal value between -1 and 1 (positive = bullish volume)
        """
        try:
            # Default signal if no indicators available
            if data.empty:
                return 0
                
            signal = 0
            count = 0
            
            # Volume change
            if 'volume' in data.columns and len(data) > 20:
                # Get current volume
                current_volume = data['volume'].iloc[-1]
                
                # Calculate average volume (20 periods)
                avg_volume = data['volume'].iloc[-20:].mean()
                
                # Check if volume is significantly higher/lower than average
                if current_volume > 0 and avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    
                    # Volume significantly higher than average
                    if volume_ratio > 2.0:
                        # High volume - direction depends on price movement
                        price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
                        
                        if price_change > 0:
                            signal += 0.7  # Strong bullish signal
                        else:
                            signal -= 0.7  # Strong bearish signal
                    elif volume_ratio > 1.5:
                        # Moderately high volume
                        price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
                        
                        if price_change > 0:
                            signal += 0.4  # Moderate bullish signal
                        else:
                            signal -= 0.4  # Moderate bearish signal
                    elif volume_ratio < 0.5:
                        # Very low volume
                        signal += 0.1  # Slight bullish bias (low volume pullbacks)
                
                count += 1
            
            # On-Balance Volume (OBV) trend
            if 'obv' in data.columns and len(data) > 20:
                # Calculate short-term OBV trend (5 periods)
                short_obv_trend = data['obv'].iloc[-1] - data['obv'].iloc[-5]
                
                # Calculate longer-term OBV trend (20 periods)
                long_obv_trend = data['obv'].iloc[-1] - data['obv'].iloc[-20]
                
                # Check for OBV divergence with price
                price_change_5 = data['close'].iloc[-1] - data['close'].iloc[-5]
                price_change_20 = data['close'].iloc[-1] - data['close'].iloc[-20]
                
                # Short-term OBV trend
                if short_obv_trend > 0:
                    signal += 0.3  # Bullish OBV
                else:
                    signal -= 0.3  # Bearish OBV
                
                # OBV divergence (bullish)
                if price_change_20 < 0 and long_obv_trend > 0:
                    signal += 0.5  # Bullish divergence
                
                # OBV divergence (bearish)
                if price_change_20 > 0 and long_obv_trend < 0:
                    signal -= 0.5  # Bearish divergence
                
                count += 1
            
            # Volume-weighted MACD if available
            if 'volume_weighted_macd' in data.columns and 'volume_weighted_macd_signal' in data.columns:
                vw_macd = data['volume_weighted_macd'].iloc[-1]
                vw_macd_signal = data['volume_weighted_macd_signal'].iloc[-1]
                
                if vw_macd > vw_macd_signal:
                    signal += 0.4
                    
                    # Check for recent crossover
                    if len(data) > 2 and data['volume_weighted_macd'].iloc[-2] <= data['volume_weighted_macd_signal'].iloc[-2]:
                        signal += 0.3  # Recent bullish crossover with volume confirmation
                else:
                    signal -= 0.4
                    
                    # Check for recent crossover
                    if len(data) > 2 and data['volume_weighted_macd'].iloc[-2] >= data['volume_weighted_macd_signal'].iloc[-2]:
                        signal -= 0.3  # Recent bearish crossover with volume confirmation
                
                count += 1
            
            # Normalize signal by count of indicators used
            if count > 0:
                normalized_signal = signal / count
                # Cap at -1 to +1
                return max(-1, min(1, normalized_signal))
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error calculating volume signal: {str(e)}")
            return 0
    
    def _initialize_websocket(self):
        """Initialize the websocket connection for real-time market data"""
        try:
            # Check if websocket has already been initialized
            if hasattr(self, 'ws_initialization_attempted') and self.ws_initialization_attempted:
                self.logger.info("Websocket initialization already attempted, skipping redundant initialization")
                return
            
            # Mark as attempted
            self.ws_initialization_attempted = True
            
            self.logger.info(f"Initializing websocket for {self.default_pair}")
            
            # Get all crypto pairs we might trade
            supported_pairs = self.config.get('trading', {}).get(
                'supported_pairs', 
                ['ETH/USD', 'BTC/USD', 'LTC/USD', 'BCH/USD', 'XRP/USD']
            )
            
            # Make sure default pair is included
            if self.default_pair not in supported_pairs:
                supported_pairs.append(self.default_pair)
            
            websocket_started = self.apis['alpaca'].start_websocket(
                symbols=supported_pairs, 
                timeout=5.0,
                non_blocking=True
            )
            
            if websocket_started:
                self.logger.info(f"Websocket initialized successfully for {supported_pairs}")
            else:
                self.logger.warning(f"Failed to initialize websocket for {supported_pairs}. Will use REST API.")
        except Exception as e:
            self.logger.error(f"Error initializing websocket: {e}")
            self.logger.warning("Will use REST API for market data.")
    
    def cleanup(self):
        """Cleanup resources like websocket connections"""
        try:
            # Stop the websocket connection if it exists
            if hasattr(self.apis['alpaca'], 'stop_websocket'):
                self.logger.info("Stopping websocket connection")
                self.apis['alpaca'].stop_websocket()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}") 