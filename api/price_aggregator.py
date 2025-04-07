"""
Price Aggregator Service for cryptocurrency price data.
Combines multiple price data sources with fallback and weighting options.
"""
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np

from api.price_provider import PriceProvider
from utils.logging_config import get_logger

# Get logger
logger = get_logger('price_aggregator')

class PriceAggregator:
    """
    Price aggregator service that combines multiple price data sources.
    Provides a unified interface with fallback capabilities.
    """
    
    def __init__(self, providers: List[Tuple[PriceProvider, float]] = None):
        """
        Initialize PriceAggregator with a list of price providers and their weights.
        
        Args:
            providers: List of tuples with (provider, weight) pairs
                       Weight determines priority (higher = more important)
        """
        self.providers = providers or []
        self.normalize_weights()
        self.logger = get_logger('price_aggregator')
        self.logger.info(f"Price Aggregator initialized with {len(self.providers)} providers")
    
    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        if not self.providers:
            return
            
        total_weight = sum(weight for _, weight in self.providers)
        if total_weight > 0:
            self.providers = [(provider, weight/total_weight) 
                             for provider, weight in self.providers]
    
    def add_provider(self, provider: PriceProvider, weight: float = 1.0):
        """
        Add a price provider to the aggregator.
        
        Args:
            provider: PriceProvider instance
            weight: Weight for this provider (higher = more important)
        """
        self.providers.append((provider, weight))
        self.normalize_weights()
        self.logger.info(f"Added provider {provider.__class__.__name__} with weight {weight}")
    
    def remove_provider(self, provider_type: type):
        """
        Remove a price provider from the aggregator by type.
        
        Args:
            provider_type: The type of provider to remove
        """
        self.providers = [(p, w) for p, w in self.providers 
                         if not isinstance(p, provider_type)]
        self.normalize_weights()
        self.logger.info(f"Removed providers of type {provider_type.__name__}")
    
    def get_current_price(self, symbol: str, 
                         force_refresh: bool = False,
                         strategy: str = 'weighted') -> Optional[float]:
        """
        Get current price from providers using specified strategy.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH", "BTC")
            force_refresh: If True, bypass cache
            strategy: 'first_available', 'weighted', or 'median'
            
        Returns:
            Current price in USD or None if not available
        """
        if not self.providers:
            self.logger.warning("No price providers available")
            return None
            
        prices = []
        weights = []
        
        for provider, weight in self.providers:
            try:
                start_time = time.time()
                price = provider.get_current_price(symbol, force_refresh)
                elapsed = time.time() - start_time
                
                if price is not None:
                    prices.append(price)
                    weights.append(weight)
                    self.logger.debug(f"Got price for {symbol} from {provider.__class__.__name__}: "
                                     f"${price:.2f} (took {elapsed:.3f}s)")
                    
                    # If using first available strategy, return the first valid price
                    if strategy == 'first_available' and price is not None:
                        return price
                        
            except Exception as e:
                self.logger.error(f"Error getting {symbol} price from {provider.__class__.__name__}: {e}")
        
        # Return None if no prices were found
        if not prices:
            self.logger.warning(f"No prices found for {symbol} from any provider")
            return None
            
        # Apply aggregation strategy
        if strategy == 'weighted' and len(prices) > 0:
            # Calculate weighted average
            weighted_price = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
            self.logger.info(f"Weighted price for {symbol}: ${weighted_price:.2f} "
                           f"from {len(prices)} providers")
            return weighted_price
            
        elif strategy == 'median' and len(prices) > 0:
            # Use median price
            median_price = sorted(prices)[len(prices) // 2]
            self.logger.info(f"Median price for {symbol}: ${median_price:.2f} "
                           f"from {len(prices)} providers")
            return median_price
            
        # Default to first price if strategy not recognized
        return prices[0] if prices else None
    
    def get_current_prices(self, symbols: List[str], 
                         force_refresh: bool = False,
                         strategy: str = 'weighted') -> Dict[str, float]:
        """
        Get current prices for multiple cryptocurrencies.
        
        Args:
            symbols: List of cryptocurrency symbols (e.g., ["ETH", "BTC"])
            force_refresh: If True, bypass cache
            strategy: 'first_available', 'weighted', or 'median'
            
        Returns:
            Dictionary with symbol keys and price values
        """
        result = {}
        for symbol in symbols:
            price = self.get_current_price(symbol, force_refresh, strategy)
            if price is not None:
                result[symbol] = price
        
        return result
    
    def get_historical_prices(self, 
                            symbol: str, 
                            days: int = 30, 
                            interval: str = "1d", 
                            force_refresh: bool = False,
                            strategy: str = 'first_available') -> Optional[pd.DataFrame]:
        """
        Get historical prices for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH", "BTC")
            days: Number of days of historical data
            interval: Time interval (e.g., "1h", "1d")
            force_refresh: If True, bypass cache
            strategy: 'first_available', 'combined' or 'most_complete'
            
        Returns:
            DataFrame with historical price data
        """
        if not self.providers:
            self.logger.warning("No price providers available")
            return None
            
        all_data = []
        weights = []
        
        for provider, weight in self.providers:
            try:
                start_time = time.time()
                data = provider.get_historical_prices(symbol, days, interval, force_refresh)
                elapsed = time.time() - start_time
                
                if data is not None and not data.empty:
                    all_data.append(data)
                    weights.append(weight)
                    row_count = len(data)
                    self.logger.debug(f"Got historical data for {symbol} from "
                                     f"{provider.__class__.__name__}: {row_count} rows "
                                     f"(took {elapsed:.3f}s)")
                    
                    # If using first available strategy, return the first valid dataset
                    if strategy == 'first_available' and not data.empty:
                        return data
                        
            except Exception as e:
                self.logger.error(f"Error getting historical data for {symbol} from "
                                 f"{provider.__class__.__name__}: {e}")
        
        # Return None if no data was found
        if not all_data:
            self.logger.warning(f"No historical data found for {symbol} from any provider")
            return None
            
        # Apply aggregation strategy
        if strategy == 'most_complete' and all_data:
            # Find the dataset with the most rows
            most_complete = max(all_data, key=len)
            self.logger.info(f"Using most complete dataset for {symbol}: "
                           f"{len(most_complete)} rows")
            return most_complete
            
        elif strategy == 'combined' and all_data:
            # Combine all datasets
            try:
                combined = pd.concat(all_data)
                combined = combined[~combined.index.duplicated(keep='first')]
                combined.sort_index(inplace=True)
                self.logger.info(f"Combined historical data for {symbol}: "
                               f"{len(combined)} rows from {len(all_data)} providers")
                return combined
            except Exception as e:
                self.logger.error(f"Error combining historical data: {e}")
                # Fall back to the first dataset
                return all_data[0]
                
        # Default to first dataset if strategy not recognized
        return all_data[0] if all_data else None
    
    def get_crypto_bars(self, symbol: str, 
                       timeframe: str = "1Hour", 
                       limit: int = 100,
                       strategy: str = 'first_available') -> pd.DataFrame:
        """
        Get historical bar data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH/USD", "BTC/USD")
            timeframe: Time interval (e.g., "1Min", "1Hour", "1Day")
            limit: Number of bars to return
            strategy: 'first_available', 'most_complete', or 'combined'
            
        Returns:
            DataFrame with OHLCV bar data
        """
        if not self.providers:
            self.logger.warning("No price providers available")
            return pd.DataFrame()
            
        all_bars = []
        weights = []
        
        for provider, weight in self.providers:
            try:
                start_time = time.time()
                bars = provider.get_crypto_bars(symbol, timeframe, limit)
                elapsed = time.time() - start_time
                
                if bars is not None and not bars.empty:
                    all_bars.append(bars)
                    weights.append(weight)
                    row_count = len(bars)
                    self.logger.debug(f"Got bars for {symbol} from {provider.__class__.__name__}: "
                                     f"{row_count} bars (took {elapsed:.3f}s)")
                    
                    # If using first available strategy, return the first valid dataset
                    if strategy == 'first_available' and not bars.empty:
                        return bars
                        
            except Exception as e:
                self.logger.error(f"Error getting bars for {symbol} from "
                                 f"{provider.__class__.__name__}: {e}")
        
        # Return empty DataFrame if no data was found
        if not all_bars:
            self.logger.warning(f"No bar data found for {symbol} from any provider")
            return pd.DataFrame()
            
        # Apply aggregation strategy
        if strategy == 'most_complete' and all_bars:
            # Find the dataset with the most rows
            most_complete = max(all_bars, key=len)
            self.logger.info(f"Using most complete bar data for {symbol}: "
                           f"{len(most_complete)} bars")
            return most_complete
            
        elif strategy == 'combined' and all_bars:
            # Combine all datasets
            try:
                # Make sure all DataFrames have the same structure
                common_columns = set.intersection(*[set(df.columns) for df in all_bars])
                standardized_bars = [df[list(common_columns)] for df in all_bars]
                
                combined = pd.concat(standardized_bars)
                combined = combined[~combined.index.duplicated(keep='first')]
                combined.sort_index(inplace=True)
                self.logger.info(f"Combined bar data for {symbol}: "
                               f"{len(combined)} bars from {len(all_bars)} providers")
                return combined
            except Exception as e:
                self.logger.error(f"Error combining bar data: {e}")
                # Fall back to the first dataset
                return all_bars[0]
                
        # Default to first dataset if strategy not recognized
        return all_bars[0] if all_bars else pd.DataFrame()
    
    def clear_all_caches(self):
        """Clear caches in all providers"""
        for provider, _ in self.providers:
            try:
                provider.clear_cache()
            except Exception as e:
                self.logger.error(f"Error clearing cache for {provider.__class__.__name__}: {e}") 