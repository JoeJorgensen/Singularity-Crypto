"""
Technical indicators for cryptocurrency trading.
Implements various technical indicators such as RSI, MACD, EMA, etc.
"""
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging

class TechnicalIndicators:
    """
    Technical indicators for cryptocurrency trading.
    All methods are static to allow for functional usage.
    """
    
    @staticmethod
    def add_rsi(
        df: pd.DataFrame,
        period: int = 14,
        column: str = 'close',
        new_column: str = 'rsi'
    ) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) to DataFrame.
        
        Args:
            df: DataFrame with price data
            period: RSI period
            column: Column to use for calculation
            new_column: Column name for RSI values
            
        Returns:
            DataFrame with RSI column added
        """
        if len(df) >= period:
            delta = df[column].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            df[new_column] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) to DataFrame.
        
        Args:
            df: DataFrame with price data
            fast_period: Fast period
            slow_period: Slow period
            signal_period: Signal period
            column: Column to use for calculation
            
        Returns:
            DataFrame with MACD columns added
        """
        if len(df) >= slow_period:
            # Calculate EMA values
            ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line
            df['macd'] = ema_fast - ema_slow
            
            # Calculate signal line
            df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
            
            # Calculate histogram
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def add_ema(
        df: pd.DataFrame,
        periods: List[int] = [9, 21],
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add Exponential Moving Average (EMA) to DataFrame.
        
        Args:
            df: DataFrame with price data
            periods: List of periods for different EMAs
            column: Column to use for calculation
            
        Returns:
            DataFrame with EMA columns added
        """
        for period in periods:
            if len(df) >= period:
                df[f'ema_{period}'] = df[column].ewm(span=period, adjust=False).mean()
        
        return df
    
    @staticmethod
    def add_sma(
        df: pd.DataFrame,
        periods: List[int] = [50, 200],
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add Simple Moving Average (SMA) to DataFrame.
        
        Args:
            df: DataFrame with price data
            periods: List of periods for different SMAs
            column: Column to use for calculation
            
        Returns:
            DataFrame with SMA columns added
        """
        for period in periods:
            if len(df) >= period:
                df[f'sma_{period}'] = df[column].rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands to DataFrame.
        
        Args:
            df: DataFrame with price data
            period: Bollinger Bands period
            std_dev: Number of standard deviations
            column: Column to use for calculation
            
        Returns:
            DataFrame with Bollinger Bands columns added
        """
        if len(df) >= period:
            # Calculate middle band (SMA)
            df['bb_middle'] = df[column].rolling(window=period).mean()
            
            # Calculate standard deviation
            rolling_std = df[column].rolling(window=period).std()
            
            # Calculate upper and lower bands
            df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
            df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
            
            # Calculate bandwidth and %B
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_percent'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    @staticmethod
    def add_atr(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Add Average True Range (ATR) to DataFrame.
        
        Args:
            df: DataFrame with price data
            period: ATR period
            
        Returns:
            DataFrame with ATR column added
        """
        if len(df) >= period and all(col in df.columns for col in ['high', 'low', 'close']):
            df['atr'] = ta.atr(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                length=period
            )
        
        return df
    
    @staticmethod
    def add_stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3
    ) -> pd.DataFrame:
        """
        Add Stochastic Oscillator to DataFrame.
        
        Args:
            df: DataFrame with price data
            k_period: K period
            d_period: D period
            smooth_k: K smoothing
            
        Returns:
            DataFrame with Stochastic columns added
        """
        if len(df) >= k_period and all(col in df.columns for col in ['high', 'low', 'close']):
            stoch = ta.stoch(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                k=k_period,
                d=d_period,
                smooth_k=smooth_k
            )
            
            # Rename columns to simpler names
            stoch.columns = ['stoch_k', 'stoch_d']
            
            # Add to original DataFrame
            for col in stoch.columns:
                df[col] = stoch[col]
        
        return df
    
    @staticmethod
    def add_adx(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Add Average Directional Index (ADX) to DataFrame.
        
        Args:
            df: DataFrame with price data
            period: ADX period
            
        Returns:
            DataFrame with ADX columns added
        """
        if len(df) >= period and all(col in df.columns for col in ['high', 'low', 'close']):
            adx = ta.adx(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                length=period
            )
            
            # Rename columns to simpler names
            adx.columns = ['adx', 'dmp', 'dmn']
            
            # Add to original DataFrame
            for col in adx.columns:
                df[col] = adx[col]
        
        return df
    
    @staticmethod
    def add_ichimoku(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add Ichimoku Cloud to DataFrame.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with Ichimoku columns added
        """
        if len(df) >= 52 and all(col in df.columns for col in ['high', 'low', 'close']):
            ichimoku = ta.ichimoku(
                high=df['high'],
                low=df['low'],
                close=df['close']
            )
            
            # Add to original DataFrame
            for col in ichimoku.columns:
                df[col] = ichimoku[col]
        
        return df
    
    @staticmethod
    def add_volume_indicators(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add volume-based technical indicators.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with volume indicator columns added
        """
        logger = logging.getLogger('CryptoTrader.technical.indicators')
        
        logger.debug(f"Adding volume indicators to DataFrame with shape {df.shape}")
        
        if 'volume' not in df.columns:
            logger.warning("Cannot add volume indicators: 'volume' column missing from DataFrame")
            # Add a default volume column with 0 values
            df['volume'] = 0
            # Add placeholder zero values for volume indicators
            df['obv'] = 0
            df['vwap'] = df['close'] if 'close' in df.columns else 0
            df['mfi'] = 50  # Neutral value
            df['volume_sma20'] = 0
            df['volume_osc'] = 0
            logger.info("Added placeholder volume indicators with neutral values")
            return df
            
        if 'close' not in df.columns:
            logger.warning("Cannot add volume indicators: 'close' column missing from DataFrame")
            return df
            
        try:
            # On-Balance Volume (OBV)
            df['obv'] = ta.obv(df['close'], df['volume'])
            logger.debug(f"Added OBV indicator, values: min={df['obv'].min()}, max={df['obv'].max()}, last={df['obv'].iloc[-1]}")
            
            # Volume Weighted Average Price (VWAP)
            if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
                # Create a copy of the dataframe with a naive datetime index to avoid timezone warnings
                df_naive = df.copy()
                if hasattr(df_naive.index, 'tz') and df_naive.index.tz is not None:
                    df_naive.index = df_naive.index.tz_localize(None)
                
                # Calculate VWAP on the naive timestamp dataframe
                df['vwap'] = ta.vwap(
                    high=df_naive['high'],
                    low=df_naive['low'],
                    close=df_naive['close'],
                    volume=df_naive['volume']
                )
                logger.debug(f"Added VWAP indicator, values: min={df['vwap'].min()}, max={df['vwap'].max()}, last={df['vwap'].iloc[-1]}")
            else:
                logger.warning("Missing required columns for VWAP calculation")
            
            # Money Flow Index (MFI)
            if len(df) >= 14 and all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
                df['mfi'] = ta.mfi(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    volume=df['volume'],
                    length=14
                )
                logger.debug(f"Added MFI indicator, values: min={df['mfi'].min()}, max={df['mfi'].max()}, last={df['mfi'].iloc[-1]}")
            else:
                logger.warning(f"Cannot add MFI: insufficient data (length: {len(df)}) or missing columns")
            
            # Volume SMA - add simple moving average of volume for reference
            if len(df) >= 20:
                df['volume_sma20'] = df['volume'].rolling(window=20).mean()
                logger.debug(f"Added Volume SMA(20), last value: {df['volume_sma20'].iloc[-1]}")
            
            # Volume oscillator (percentage difference between fast and slow volume MAs)
            if len(df) >= 20:
                vol_fast = df['volume'].rolling(window=5).mean()
                vol_slow = df['volume'].rolling(window=20).mean()
                df['volume_osc'] = 100 * (vol_fast - vol_slow) / vol_slow
                logger.debug(f"Added Volume Oscillator, last value: {df['volume_osc'].iloc[-1]}")
                
            logger.debug(f"Successfully added volume indicators to DataFrame")
            return df
            
        except Exception as e:
            logger.error(f"Error adding volume indicators: {str(e)}")
            # Add placeholder values if an error occurs
            for indicator in ['obv', 'vwap', 'mfi', 'volume_sma20', 'volume_osc']:
                if indicator not in df.columns:
                    df[indicator] = 0 if indicator != 'mfi' else 50
            return df
    
    @staticmethod
    def add_all_indicators(
        df: pd.DataFrame,
        config: Dict = None
    ) -> pd.DataFrame:
        """
        Add all configured technical indicators to DataFrame.
        
        Args:
            df: DataFrame with price data
            config: Configuration dictionary
            
        Returns:
            DataFrame with all indicator columns added
        """
        if config is None:
            config = {}
        
        # Make a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Add RSI
        rsi_config = config.get('rsi', {})
        if rsi_config.get('enabled', True):
            df_copy = TechnicalIndicators.add_rsi(
                df_copy,
                period=rsi_config.get('period', 14)
            )
        
        # Add MACD
        macd_config = config.get('macd', {})
        if macd_config.get('enabled', True):
            df_copy = TechnicalIndicators.add_macd(
                df_copy,
                fast_period=macd_config.get('fast_period', 12),
                slow_period=macd_config.get('slow_period', 26),
                signal_period=macd_config.get('signal_period', 9)
            )
        
        # Add EMA
        ema_config = config.get('ema', {})
        if ema_config.get('enabled', True):
            periods = []
            if 'fast_period' in ema_config:
                periods.append(ema_config['fast_period'])
            if 'slow_period' in ema_config:
                periods.append(ema_config['slow_period'])
            if not periods:
                periods = [9, 21]  # Default periods
            
            df_copy = TechnicalIndicators.add_ema(
                df_copy,
                periods=periods
            )
        
        # Add SMA
        sma_config = config.get('sma', {})
        if sma_config.get('enabled', True):
            periods = sma_config.get('periods', [50, 200])
            df_copy = TechnicalIndicators.add_sma(
                df_copy,
                periods=periods
            )
        
        # Add Bollinger Bands
        bb_config = config.get('bollinger_bands', {})
        if bb_config.get('enabled', True):
            df_copy = TechnicalIndicators.add_bollinger_bands(
                df_copy,
                period=bb_config.get('period', 20),
                std_dev=bb_config.get('std_dev', 2.0)
            )
        
        # Add ATR
        atr_config = config.get('atr', {})
        if atr_config.get('enabled', True):
            df_copy = TechnicalIndicators.add_atr(
                df_copy,
                period=atr_config.get('period', 14)
            )
        
        # Add Stochastic
        stoch_config = config.get('stochastic', {})
        if stoch_config.get('enabled', True):
            df_copy = TechnicalIndicators.add_stochastic(
                df_copy,
                k_period=stoch_config.get('k_period', 14),
                d_period=stoch_config.get('d_period', 3),
                smooth_k=stoch_config.get('smooth_k', 3)
            )
        
        # Add ADX
        adx_config = config.get('adx', {})
        if adx_config.get('enabled', True):
            df_copy = TechnicalIndicators.add_adx(
                df_copy,
                period=adx_config.get('period', 14)
            )
        
        # Add Ichimoku
        ichimoku_config = config.get('ichimoku', {})
        if ichimoku_config.get('enabled', False):  # Disabled by default
            df_copy = TechnicalIndicators.add_ichimoku(df_copy)
        
        # Add volume indicators
        volume_config = config.get('volume_indicators', {})
        if volume_config.get('enabled', True):
            df_copy = TechnicalIndicators.add_volume_indicators(df_copy)
        
        return df_copy
    
    @staticmethod
    def calculate_support_resistance(
        df: pd.DataFrame,
        window: int = 10,
        min_touches: int = 2,
        percent_threshold: float = 0.2
    ) -> Dict:
        """
        Calculate support and resistance levels from historical data.
        
        Args:
            df: DataFrame with price data
            window: Window size for finding local extrema
            min_touches: Minimum number of touches required for a valid level
            percent_threshold: Threshold for clustering price levels (% of price)
            
        Returns:
            Dictionary with support and resistance levels
        """
        df = df.copy()
        
        # Get the high and low prices
        highs = df['high'].values
        lows = df['low'].values
        close = df['close'].values
        
        # Find local maxima and minima
        local_maxima = []
        local_minima = []
        
        for i in range(window, len(df) - window):
            # Check if a local maximum
            if highs[i] == max(highs[i-window:i+window+1]):
                local_maxima.append((i, highs[i]))
            
            # Check if a local minimum
            if lows[i] == min(lows[i-window:i+window+1]):
                local_minima.append((i, lows[i]))
        
        # Cluster levels that are within percent_threshold of each other
        def cluster_levels(levels):
            if not levels:
                return []
            
            # Sort by price
            levels = sorted(levels, key=lambda x: x[1])
            
            # Cluster levels
            clusters = [[levels[0]]]
            
            for i in range(1, len(levels)):
                idx, price = levels[i]
                last_cluster = clusters[-1]
                last_price = last_cluster[-1][1]
                
                # If price is within threshold, add to current cluster
                if abs(price - last_price) / last_price <= percent_threshold / 100:
                    last_cluster.append((idx, price))
                # Otherwise, start a new cluster
                else:
                    clusters.append([(idx, price)])
            
            # Calculate average price for each cluster
            return [
                (len(cluster), sum(p for _, p in cluster) / len(cluster))
                for cluster in clusters
            ]
        
        resistance_clusters = cluster_levels(local_maxima)
        support_clusters = cluster_levels(local_minima)
        
        # Filter clusters with at least min_touches
        resistance_levels = [price for touches, price in resistance_clusters if touches >= min_touches]
        support_levels = [price for touches, price in support_clusters if touches >= min_touches]
        
        # Sort by price
        resistance_levels = sorted(resistance_levels)
        support_levels = sorted(support_levels)
        
        # Get current price
        current_price = close[-1]
        
        # Find nearest levels
        def find_nearest_levels(levels, price, n=3):
            # Find levels above and below current price
            above = [l for l in levels if l > price]
            below = [l for l in levels if l < price]
            
            # Sort by distance to current price
            above = sorted(above, key=lambda x: abs(x - price))
            below = sorted(below, key=lambda x: abs(x - price))
            
            # Return n nearest levels or all if fewer than n
            return {
                'above': above[:n] if len(above) >= n else above,
                'below': below[:n] if len(below) >= n else below
            }
        
        # Return structured result
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'nearest_support': find_nearest_levels(support_levels, current_price),
            'nearest_resistance': find_nearest_levels(resistance_levels, current_price),
            'current_price': current_price
        } 