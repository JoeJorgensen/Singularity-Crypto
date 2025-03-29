"""
Technical analysis module for generating trading signals.
"""
import numpy as np
import pandas as pd
# Remove TA-Lib dependency
import ta
from ta.trend import ema_indicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

class TechnicalAnalysis:
    def __init__(self, config):
        """Initialize technical analysis with configuration."""
        self.config = config
        
        # Default parameters
        self.ema_short = 20
        self.ema_long = 50
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2
        
        # Load custom parameters from config if available
        if 'technical_analysis' in config:
            ta_config = config['technical_analysis']
            self.ema_short = ta_config.get('ema_short', self.ema_short)
            self.ema_long = ta_config.get('ema_long', self.ema_long)
            self.rsi_period = ta_config.get('rsi_period', self.rsi_period)
            self.macd_fast = ta_config.get('macd_fast', self.macd_fast)
            self.macd_slow = ta_config.get('macd_slow', self.macd_slow)
            self.macd_signal = ta_config.get('macd_signal', self.macd_signal)
            self.bb_period = ta_config.get('bb_period', self.bb_period)
            self.bb_std = ta_config.get('bb_std', self.bb_std)
    
    def add_indicators(self, df):
        """Add technical indicators to the dataframe."""
        # Make sure we have OHLCV data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("DataFrame must contain OHLCV data")
        
        # Add EMAs using TA library
        df['ema_20'] = ema_indicator(df['close'], window=self.ema_short)
        df['ema_50'] = ema_indicator(df['close'], window=self.ema_long)
        
        # Add RSI
        rsi_indicator = RSIIndicator(df['close'], window=self.rsi_period)
        df['rsi'] = rsi_indicator.rsi()
        
        # Add MACD
        macd_indicator = MACD(
            df['close'],
            window_slow=self.macd_slow, 
            window_fast=self.macd_fast, 
            window_sign=self.macd_signal
        )
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()
        
        # Add Bollinger Bands
        bb_indicator = BollingerBands(
            df['close'],
            window=self.bb_period, 
            window_dev=self.bb_std
        )
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        
        # Add ATR for volatility
        atr_indicator = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        df['atr'] = atr_indicator.average_true_range()
        
        return df
    
    def generate_signals(self, df):
        """Generate trading signals from technical indicators."""
        if df.empty:
            return {
                'trend': 0,
                'momentum': 0,
                'volume': 0,
                'volatility': 0,
                'risk_score': 0,
                'risk_reward': 0
            }
        
        # Add indicators if they don't exist
        if 'ema_20' not in df.columns:
            df = self.add_indicators(df)
        
        # Get latest values
        current_close = df['close'].iloc[-1]
        current_ema_20 = df['ema_20'].iloc[-1]
        current_ema_50 = df['ema_50'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_macd = df['macd'].iloc[-1]
        current_macd_signal = df['macd_signal'].iloc[-1]
        current_bb_upper = df['bb_upper'].iloc[-1]
        current_bb_lower = df['bb_lower'].iloc[-1]
        
        # Safely handle NaN values
        if np.isnan(current_close) or np.isnan(current_ema_20) or np.isnan(current_ema_50):
            trend_signal = 0
        else:
            # Trend signals (-1 to 1)
            ema_trend = np.clip((current_close - current_ema_50) / current_ema_50 * 5, -1, 1) if current_ema_50 != 0 else 0
            ema_cross = np.clip((current_ema_20 - current_ema_50) / current_ema_50 * 5, -1, 1) if current_ema_50 != 0 else 0
            trend_signal = (ema_trend + ema_cross) / 2
        
        if np.isnan(current_rsi) or np.isnan(current_macd) or np.isnan(current_macd_signal):
            momentum_signal = 0
        else:
            # Momentum signals (-1 to 1)
            rsi_signal = (current_rsi - 50) / 50 if current_rsi is not None else 0  # Normalize to -1 to 1
            
            # Safely handle division by zero for MACD signal
            if current_macd_signal != 0 and not np.isnan(current_macd_signal):
                macd_signal_value = np.clip((current_macd - current_macd_signal) / abs(current_macd_signal), -1, 1)
            else:
                macd_signal_value = 0
            
            momentum_signal = (rsi_signal + macd_signal_value) / 2
        
        # Volume signal (-1 to 1)
        volume_sma = df['volume'].rolling(20).mean()
        if not df['volume'].empty and not volume_sma.empty and not np.isnan(volume_sma.iloc[-1]) and volume_sma.iloc[-1] > 0:
            volume_signal = np.clip((df['volume'].iloc[-1] - volume_sma.iloc[-1]) / volume_sma.iloc[-1], -1, 1)
        else:
            volume_signal = 0
        
        # Volatility calculation
        if not np.isnan(df['atr'].iloc[-1]) and current_close > 0:
            atr_pct = (df['atr'].iloc[-1] / current_close) * 100
        else:
            atr_pct = 0
        
        # Risk calculation
        if not np.isnan(trend_signal) and not np.isnan(momentum_signal) and not np.isnan(volume_signal):
            risk_score = abs(trend_signal * 0.4 + momentum_signal * 0.3 + volume_signal * 0.3)
        else:
            risk_score = 0
        
        # Risk/Reward calculation
        if not np.isnan(current_bb_upper) and not np.isnan(current_close) and not np.isnan(current_bb_lower):
            potential_reward = abs(current_bb_upper - current_close)
            potential_risk = abs(current_close - current_bb_lower)
            risk_reward = potential_reward / potential_risk if potential_risk > 0 else 0
        else:
            risk_reward = 0
        
        return {
            'trend': trend_signal,
            'momentum': momentum_signal,
            'volume': volume_signal,
            'volatility': atr_pct,
            'risk_score': risk_score,
            'risk_reward': risk_reward
        } 