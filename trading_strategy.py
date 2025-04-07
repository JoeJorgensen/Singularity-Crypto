"""
Core trading strategy for cryptocurrency trading.
Implements the main trading logic and signal processing.
"""
from datetime import datetime, timedelta
import json
import logging
import os
import threading
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import aiohttp
import asyncio
import numpy as np
import pandas as pd

# Import APIs and utilities
from api.alpaca_api import AlpacaAPI
from api.coinlore_api import CoinloreAPI
from api.finnhub_api import FinnhubAPI
from api.openai_api import OpenAIAPI
from technical.indicators import TechnicalIndicators
from technical.signal_generator import SignalGenerator
from utils.logging_config import get_logger
from utils.order_manager import OrderManager
from utils.position_calculator import PositionCalculator
from utils.risk_manager import RiskManager

# Get logger
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
                # Only monitor ETH/USD
                symbols_to_monitor = ['ETH/USD']
                
                # Define background function for starting websocket
                def start_ws_in_background():
                    try:
                        self.logger.info("Starting websocket connection for ETH/USD in background")
                        # Start with a short timeout for initial connection
                        result = alpaca_api.start_websocket(
                            symbols=symbols_to_monitor,
                            timeout=5.0, 
                            non_blocking=True
                        )
                        
                        if result:
                            self.logger.info("Websocket connection started successfully for ETH/USD")
                        else:
                            self.logger.warning("Websocket connection failed to start for ETH/USD, using REST API polling as fallback")
                    except Exception as e:
                        self.logger.error(f"Error starting websocket: {str(e)}")
                        self.logger.info("Continuing with REST API polling as fallback")
                
                # Start websocket thread - only done once per application instance
                self.logger.info("Starting websocket connection for ETH/USD in background")
                ws_thread = threading.Thread(target=start_ws_in_background, daemon=True)
                ws_thread.start()
                self.logger.info("Websocket initialization triggered in background thread")
                
                # Start the price cache updater to maintain price data even during API disruptions
                try:
                    self.logger.info("Starting price cache updater for ETH/USD")
                    alpaca_api.start_price_cache_updater(symbols=symbols_to_monitor)
                    self.logger.info("Price cache updater started successfully")
                except Exception as e:
                    self.logger.warning(f"Error starting price cache updater: {str(e)}")
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
            
            # Use the SignalGenerator to get all component signals
            trend_signal = self.signal_generator.generate_trend_signal(data)
            momentum_signal = self.signal_generator.generate_momentum_signal(data)
            volatility_signal = self.signal_generator.generate_volatility_signal(data)
            volume_signal = self.signal_generator.generate_volume_signal(data)
            
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
                trade_direction = "bullish"  # Consistent terminology
                signal_strength = min(abs(composite_signal), 1.0)  # Cap at 1.0
            elif composite_signal < 0:
                signal_direction = "sell"
                trade_direction = "bearish"  # Consistent terminology
                signal_strength = min(abs(composite_signal), 1.0)  # Cap at 1.0
            else:
                signal_direction = "neutral"
                trade_direction = "neutral"  # Consistent terminology
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
                    trade_direction = "bullish"
                    signal_strength = min(abs(adjusted_signal), 1.0)
                elif adjusted_signal < 0:
                    signal_direction = "sell"
                    trade_direction = "bearish"
                    signal_strength = min(abs(adjusted_signal), 1.0)
                else:
                    signal_direction = "neutral"
                    trade_direction = "neutral"
                    signal_strength = 0.0
            
            # Final signal value with sign
            final_signal = adjusted_signal if sentiment_data else composite_signal
            
            # Ensure signal strength is always positive (magnitude)
            signal_strength = min(abs(final_signal), 1.0)
            
            logger.info(f"Signal generated: {signal_direction} with strength {signal_strength:.4f}")
            
            # Create signals dictionary
            signals = {
                "signal": final_signal,  # Keep sign for raw signal value
                "signal_direction": signal_direction,
                "signal_strength": signal_strength,  # Always positive (magnitude)
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
                if signal_strength >= self.config.get('trading', {}).get('min_signal_strength', 0.2):
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
        logger.warning(f"DEBUG: Current signal strength={signal_strength}, abs value={abs(signal_strength)}")
        
        # Check minimum strength requirement - ensure we use absolute value
        if abs(signal_strength) < min_signal_strength:
            logger.info(f"Signal strength {signal_strength:.4f} is below minimum threshold {min_signal_strength}")
            return False
        else:
            logger.info(f"Signal strength {signal_strength:.4f} exceeds minimum threshold {min_signal_strength}")
        
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
        trade_direction = signals.get('components', {}).get('direction', 'long' if side == 'buy' else 'none')
        
        # Crypto accounts are non-marginable and don't support short selling
        # We only support long positions (buy to open, sell to close)
        if side == 'sell' and trade_direction == 'short':
            logger.warning(f"Short selling not supported for crypto as accounts are non-marginable. Skipping trade.")
            return {
                "executed": False,
                "error": "Short selling not supported for crypto. Only long positions are allowed."
            }
        
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
            
            try:
                # Get detailed position information
                position = self.apis['alpaca'].get_position(symbol)
                if position:
                    has_existing_position = True
                    existing_position_size = float(position.qty)
                    existing_position_value = float(position.market_value)
                    existing_avg_entry = float(position.avg_entry_price) if hasattr(position, 'avg_entry_price') else 0
                    existing_current_price = float(position.current_price) if hasattr(position, 'current_price') else 0
                    
                    # Calculate what percentage of portfolio this position represents
                    portfolio_allocation = existing_position_value / portfolio_value * 100 if portfolio_value > 0 else 0
                    
                    logger.info(f"Existing LONG position found: {existing_position_size} {symbol} at avg entry ${existing_avg_entry:.2f}")
                    logger.info(f"Position value: ${existing_position_value:.2f} ({portfolio_allocation:.1f}% of portfolio)")
                    
                    # If we're trying to sell but don't have a position to sell, report an error
                    if side == 'sell' and existing_position_size <= 0:
                        logger.warning(f"Cannot sell {symbol} - no long position exists to close.")
                        return {
                            "executed": False,
                            "error": "No long position exists to close"
                        }
                    
                    # If we already have a position and are trying to buy more, check portfolio allocation limits
                    if side == 'buy' and portfolio_allocation > 90:
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
                    
                    # If trying to sell but no position exists, return error
                    if side == 'sell':
                        logger.warning(f"Cannot sell {symbol} - no position exists")
                        return {
                            "executed": False,
                            "error": "No position exists to sell"
                        }
                else:
                    logger.warning(f"Error checking existing position: {str(e)}")
            
            # Use available buying power for orders and respect position sizing limits
            # For buy (long) orders
            if side == 'buy':
                # Use a fixed percentage of available buying power (95%)
                safe_buy_amount = available_balance * 0.95
                logger.info(f"Using 95% of buying power: ${safe_buy_amount:.2f} (from ${available_balance:.2f})")
            else:
                # For sell orders - this is selling an existing long position
                if has_existing_position:
                    # This is closing an existing long position
                    position_sizing_balance = existing_position_value
                    safe_buy_amount = position_sizing_balance
                    logger.info(f"Selling entire position: {existing_position_size} {symbol} worth ${existing_position_value:.2f}")
                else:
                    # Cannot sell if no position exists
                    logger.warning(f"Cannot sell {symbol} - no position exists")
                    return {
                        "executed": False,
                        "error": "No position exists to sell"
                    }
            
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
            price = None  # Initialize price variable
            
            # Try to get price from various sources
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
                    try:
                        latest_data = self.get_market_data(symbol, '1Min', 1)
                        if not latest_data.empty and 'close' in latest_data.columns:
                            price = latest_data['close'].iloc[-1]
                    except Exception as e:
                        logger.warning(f"Failed to get price from market data: {str(e)}")
            
            # Skip the trade if we couldn't determine a price
            if price is None:
                logger.error("Trade execution failed: Could not determine current price")
                return {
                    "executed": False,
                    "error": "Could not determine current price"
                }
            
            # Make sure we're using fresh account data
            try:
                account = self.apis['alpaca'].get_account(force_refresh=True)
                available_balance = float(account.buying_power)
                portfolio_value = float(account.portfolio_value)
                
                # Log updated account information
                logger.info(f"Using fresh account data - Buying Power: ${available_balance:.2f}, Portfolio Value: ${portfolio_value:.2f}")
            except Exception as e:
                logger.warning(f"Could not refresh account data: {str(e)}")
            
            # Calculate position size in quantity
            if side == 'buy':
                # Long position
                # Add a small safety margin for price fluctuations
                adjusted_price = price * 1.005  # Add 0.5% buffer
                position_size = safe_buy_amount / adjusted_price
                
                # Make sure the position size is not too small
                if position_size * price < 10:
                    logger.warning(f"Position value (${position_size * price:.2f}) is below minimum. Skipping trade.")
                    return {
                        "executed": False,
                        "error": f"Position value (${position_size * price:.2f}) is below minimum threshold of $10"
                    }
            else:
                # Selling an existing long position - use exact position size
                position_size = existing_position_size
            
            # Log position sizing details
            if side == 'buy':
                logger.info(f"Buy position: {position_size:.6f} {symbol} at ${price:.2f} (value: ${position_size * price:.2f})")
            else:
                logger.info(f"Sell existing long position: {position_size:.6f} {symbol} at ${price:.2f} (value: ${position_size * price:.2f})")
            
            logger.info(f"Strategy mode: {strategy_mode}")
            
            # Try to execute the order, with retry logic for size issues
            order = None
            max_retries = 3
            retry_count = 0
            current_position_size = position_size
            
            # Always initialize current_price to a valid value or 0
            current_price = price if price is not None else 0
            
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
                            # Reduce position size by 25% on each retry
                            retry_count += 1
                            current_position_size *= 0.75
                            
                            # Log with more details about the retry
                            logger.warning(f"Insufficient balance error. Retrying with reduced position size: {current_position_size:.6f} (value: ${current_position_size * price:.2f}) (attempt {retry_count}/{max_retries})")
                            continue
                        else:
                            # Non-retryable error or out of retries
                            logger.error(f"Order rejected: {error_msg}")
                            return {
                                "executed": False,
                                "error": error_msg,
                                "attempted_qty": current_position_size,
                                "price": current_price
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
                        
                        # More aggressive reduction for exception errors
                        current_position_size *= 0.70
                        
                        logger.warning(f"Error suggests insufficient balance. Retrying with reduced size: {current_position_size:.6f} (value: ${current_position_size * price:.2f}) (attempt {retry_count}/{max_retries})")
                        continue
                    else:
                        # Non-retryable error or out of retries
                        return {
                            "executed": False,
                            "error": f"Trade execution error: {error_message}",
                            "attempted_qty": current_position_size,
                            "price": current_price
                        }
            
            # Check if the order was successful
            if not order or order.get('status') == 'rejected' or order.get('error'):
                error_msg = order.get('error', 'Unknown error') if order else "Failed to place order"
                logger.error(f"Trade execution failed: {error_msg}")
                return {
                    "executed": False,
                    "error": error_msg,
                    "attempted_qty": current_position_size,
                    "price": current_price
                }
            
            # Update last trade time ONLY on successful orders
            self.last_trade_time = datetime.now()
            
            # Log the executed trade with specific action type
            if side == 'buy':
                trade_action = "BUY (LONG)"
            else:
                trade_action = "SELL (CLOSE LONG)"
            
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
        Calculate dynamic TTL (time-to-live) for cache data based on market volatility.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            TTL in seconds
        """
        # Base TTL from config
        base_ttl = self.config.get('data_optimization', {}).get('cache_ttl', 60)  # seconds
        
        try:
            # Calculate recent volatility (standard deviation of returns)
            if len(market_data) > 10:
                recent_returns = market_data['close'].pct_change().dropna().tail(10)
                volatility = recent_returns.std()
                
                # Scale TTL inversely with volatility
                # Higher volatility = lower TTL
                if volatility > 0:
                    # Default max_volatility is 0.03 (3% std dev per bar)
                    max_volatility = self.config.get('data_optimization', {}).get('max_volatility', 0.03)
                    min_ttl_factor = self.config.get('data_optimization', {}).get('min_ttl_factor', 0.2)
                    
                    # Calculate TTL factor between min_ttl_factor and 1.0
                    ttl_factor = max(min_ttl_factor, 1.0 - (volatility / max_volatility))
                    
                    # Apply TTL factor
                    adjusted_ttl = int(base_ttl * ttl_factor)
                    
                    # Log the adjustment
                    logger.debug(f"Dynamic TTL adjusted based on volatility {volatility:.4f}: {adjusted_ttl}s (base: {base_ttl}s)")
                    
                    return adjusted_ttl
        except Exception as e:
            logger.debug(f"Error calculating dynamic TTL, using base value: {str(e)}")
        
        return base_ttl
        
    def _initialize_websocket(self):
        """
        Initialize and start websocket connection for real-time data.
        Separated to avoid redundant initializations.
        """
        alpaca_api = self.apis['alpaca']
        
        # Only monitor ETH/USD
        symbols_to_monitor = ['ETH/USD']
        
        try:
            logger.info("Starting websocket connection for ETH/USD")
            # Start with a short timeout for initial connection
            result = alpaca_api.start_websocket(
                symbols=symbols_to_monitor,
                timeout=5.0, 
                non_blocking=True
            )
            
            if result:
                logger.info("Websocket connection started successfully for ETH/USD")
            else:
                logger.warning("Websocket connection failed to start for ETH/USD, using REST API polling as fallback")
        except Exception as e:
            logger.error(f"Error starting websocket: {str(e)}")
            logger.info("Continuing with REST API polling as fallback")
        
        # Start the price cache updater to maintain price data
        try:
            logger.info("Starting price cache updater for ETH/USD")
            alpaca_api.start_price_cache_updater(symbols=symbols_to_monitor)
            logger.info("Price cache updater started successfully")
        except Exception as e:
            logger.warning(f"Error starting price cache updater: {str(e)}")
            
    def cleanup(self):
        """
        Clean up resources when shutting down the trading strategy.
        """
        logger.info("Cleaning up trading strategy resources")
        try:
            # Stop websocket connection if it exists
            if self.apis.get('alpaca'):
                self.apis['alpaca'].stop_websocket()
                self.apis['alpaca'].stop_price_cache_updater()
                
            logger.info("Trading strategy cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")