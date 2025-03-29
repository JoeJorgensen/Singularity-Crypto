"""
Signal generator for cryptocurrency trading.
Implements signal generation from technical indicators.
"""
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import logging

# Get logger
logger = logging.getLogger('CryptoTrader.technical.signal_generator')

class SignalGenerator:
    """
    Signal generator for cryptocurrency trading.
    Generates trading signals from technical indicators.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize SignalGenerator with configuration.
        
        Args:
            config: Configuration dictionary with signal parameters
        """
        self.config = config or {}
        
        # Load signal thresholds from config
        self.signal_thresholds = self.config.get('signal_thresholds', {})
        self.strong_buy = self.signal_thresholds.get('strong_buy', 0.5)
        self.buy = self.signal_thresholds.get('buy', 0.2)
        self.neutral_band = self.signal_thresholds.get('neutral_band', 0.05)
        self.sell = self.signal_thresholds.get('sell', -0.2)
        self.strong_sell = self.signal_thresholds.get('strong_sell', -0.5)
        
        # Load indicator weights from config
        self.indicator_weights = self.config.get('indicator_weights', {})
        self.trend_weight = self.indicator_weights.get('trend', 0.4)
        self.momentum_weight = self.indicator_weights.get('momentum', 0.3)
        self.volatility_weight = self.indicator_weights.get('volatility', 0.15)
        self.volume_weight = self.indicator_weights.get('volume', 0.15)
        
        logger.info(f"SignalGenerator initialized with weights: trend={self.trend_weight}, momentum={self.momentum_weight}, volatility={self.volatility_weight}, volume={self.volume_weight}")
    
    def generate_trend_signal(self, df: pd.DataFrame) -> float:
        """
        Generate trend signal from moving averages.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Trend signal value (-1.0 to 1.0)
        """
        signals = []
        
        # Current price
        if 'close' not in df.columns:
            logger.warning("No 'close' column found in dataframe for trend signal calculation")
            return 0
        
        current_price = df['close'].iloc[-1]
        logger.debug(f"Current price for trend signal: {current_price}")
        
        # EMA signals
        if 'ema_9' in df.columns and 'ema_21' in df.columns:
            ema_short = df['ema_9'].iloc[-1]
            ema_long = df['ema_21'].iloc[-1]
            
            # EMA crossover
            if not pd.isna(ema_short) and not pd.isna(ema_long):
                ema_diff = (ema_short - ema_long) / ema_long
                ema_signal = min(max(ema_diff * 10, -1), 1)  # Scale and clamp
                signals.append(ema_signal)
                logger.debug(f"EMA signal: {ema_signal} (EMA9={ema_short}, EMA21={ema_long})")
        else:
            logger.debug("EMA9 or EMA21 not available for trend calculation")
        
        # SMA signals
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            sma_50 = df['sma_50'].iloc[-1]
            sma_200 = df['sma_200'].iloc[-1]
            
            # Golden/Death cross
            if not pd.isna(sma_50) and not pd.isna(sma_200):
                sma_diff = (sma_50 - sma_200) / sma_200
                sma_signal = min(max(sma_diff * 5, -1), 1)  # Scale and clamp
                signals.append(sma_signal)
                logger.debug(f"SMA signal: {sma_signal} (SMA50={sma_50}, SMA200={sma_200})")
        else:
            logger.debug("SMA50 or SMA200 not available for trend calculation")
        
        # Price position relative to moving averages
        ma_count = 0
        ma_above_count = 0
        
        for col in df.columns:
            if col.startswith('ema_') or col.startswith('sma_'):
                ma_value = df[col].iloc[-1]
                if not pd.isna(ma_value):
                    ma_count += 1
                    if current_price > ma_value:
                        ma_above_count += 1
        
        if ma_count > 0:
            ma_position_signal = (ma_above_count / ma_count) * 2 - 1  # Convert to -1 to 1
            signals.append(ma_position_signal)
            logger.debug(f"MA position signal: {ma_position_signal} (above {ma_above_count}/{ma_count} MAs)")
        else:
            logger.debug("No moving averages available for position calculation")
        
        # Bollinger Bands position
        if all(col in df.columns for col in ['bb_lower', 'bb_middle', 'bb_upper']):
            bb_lower = df['bb_lower'].iloc[-1]
            bb_middle = df['bb_middle'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            
            if not pd.isna(bb_lower) and not pd.isna(bb_middle) and not pd.isna(bb_upper):
                # Normalize position within the bands
                band_width = bb_upper - bb_lower
                if band_width > 0:
                    normalized_pos = (current_price - bb_lower) / band_width
                    # Convert to signal: 0-0.3 (strong buy), 0.3-0.45 (buy), 0.45-0.55 (neutral),
                    # 0.55-0.7 (sell), 0.7-1.0 (strong sell)
                    if normalized_pos < 0.3:
                        bb_signal = -1 + normalized_pos / 0.3
                    elif normalized_pos < 0.45:
                        bb_signal = -0.5 * (0.45 - normalized_pos) / 0.15
                    elif normalized_pos < 0.55:
                        bb_signal = 0
                    elif normalized_pos < 0.7:
                        bb_signal = 0.5 * (normalized_pos - 0.55) / 0.15
                    else:
                        bb_signal = 1 - (1 - normalized_pos) / 0.3
                    
                    signals.append(bb_signal)
                    logger.debug(f"BB signal: {bb_signal} (position: {normalized_pos}, lower: {bb_lower}, middle: {bb_middle}, upper: {bb_upper})")
        else:
            logger.debug("Bollinger Bands not available for signal calculation")
        
        # Average all signals if we have any
        if signals:
            trend_signal = sum(signals) / len(signals)
            logger.debug(f"Final trend signal: {trend_signal} (from {len(signals)} components)")
            return trend_signal
        
        logger.warning("No trend signals were generated, returning 0")
        return 0
    
    def generate_momentum_signal(self, df: pd.DataFrame) -> float:
        """
        Generate momentum signal from oscillators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Momentum signal value (-1.0 to 1.0)
        """
        signals = []
        
        # RSI signal
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if not pd.isna(rsi):
                # RSI interpretation: <30 oversold (buy), >70 overbought (sell)
                if rsi < 30:
                    rsi_signal = -1 + rsi / 30  # -1 to 0
                elif rsi > 70:
                    rsi_signal = (rsi - 70) / 30  # 0 to 1
                else:
                    # Convert mid-range to small signal
                    rsi_signal = (rsi - 50) / 40  # -0.5 to 0.5
                signals.append(rsi_signal)
                logger.debug(f"RSI signal: {rsi_signal} (RSI value: {rsi})")
        else:
            logger.debug("RSI not available for momentum calculation")
        
        # MACD signal
        if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_hist = df['macd_histogram'].iloc[-1]
            
            if not pd.isna(macd) and not pd.isna(macd_signal):
                # MACD crossover signal
                if macd > macd_signal:
                    cross_signal = min(abs(macd - macd_signal) * 5, 1) * -1  # Bullish (negative)
                else:
                    cross_signal = min(abs(macd - macd_signal) * 5, 1)  # Bearish (positive)
                
                signals.append(cross_signal)
                logger.debug(f"MACD crossover signal: {cross_signal} (MACD: {macd}, Signal: {macd_signal})")
                
                # MACD histogram direction
                if len(df) >= 3 and not pd.isna(df['macd_histogram'].iloc[-2]):
                    hist_prev = df['macd_histogram'].iloc[-2]
                    if macd_hist > hist_prev:
                        hist_signal = -0.5  # Bullish momentum
                    else:
                        hist_signal = 0.5  # Bearish momentum
                    signals.append(hist_signal)
                    logger.debug(f"MACD histogram signal: {hist_signal} (Current: {macd_hist}, Previous: {hist_prev})")
        else:
            logger.debug("MACD components not available for momentum calculation")
        
        # Stochastic signal
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            k = df['stoch_k'].iloc[-1]
            d = df['stoch_d'].iloc[-1]
            
            if not pd.isna(k) and not pd.isna(d):
                # Oversold/overbought signal
                if k < 20 and d < 20:
                    stoch_signal = -1  # Strongly oversold (buy)
                elif k < 30 and d < 30:
                    stoch_signal = -0.5  # Oversold (buy)
                elif k > 80 and d > 80:
                    stoch_signal = 1  # Strongly overbought (sell)
                elif k > 70 and d > 70:
                    stoch_signal = 0.5  # Overbought (sell)
                else:
                    stoch_signal = 0  # Neutral
                
                signals.append(stoch_signal)
                logger.debug(f"Stochastic signal: {stoch_signal} (K: {k}, D: {d})")
                
                # Stochastic crossover
                if k > d:
                    cross_signal = -0.3  # Bullish
                else:
                    cross_signal = 0.3  # Bearish
                
                signals.append(cross_signal)
                logger.debug(f"Stochastic crossover signal: {cross_signal} (K: {k}, D: {d})")
        else:
            logger.debug("Stochastic oscillator not available for momentum calculation")
        
        # ADX signal (trend strength)
        if 'adx' in df.columns and 'dmp' in df.columns and 'dmn' in df.columns:
            adx = df['adx'].iloc[-1]
            dmp = df['dmp'].iloc[-1]
            dmn = df['dmn'].iloc[-1]
            
            if not pd.isna(adx) and not pd.isna(dmp) and not pd.isna(dmn):
                # ADX trend strength
                trend_strength = min(adx / 50, 1)
                
                # Direction from DMI
                if dmp > dmn:
                    direction = -1  # Bullish
                else:
                    direction = 1  # Bearish
                
                adx_signal = trend_strength * direction
                signals.append(adx_signal)
                logger.debug(f"ADX signal: {adx_signal} (ADX: {adx}, +DI: {dmp}, -DI: {dmn})")
        else:
            logger.debug("ADX components not available for momentum calculation")
        
        # Average all signals if we have any
        if signals:
            momentum_signal = sum(signals) / len(signals)
            logger.debug(f"Final momentum signal: {momentum_signal} (from {len(signals)} components)")
            return momentum_signal
        
        logger.warning("No momentum signals were generated, returning 0")
        return 0
    
    def generate_volatility_signal(self, df: pd.DataFrame) -> float:
        """
        Generate volatility signal from ATR and Bollinger Bands.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Volatility signal value (-1.0 to 1.0)
        """
        signals = []
        
        # ATR-based signal
        if 'atr' in df.columns and 'close' in df.columns:
            atr = df['atr'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if not pd.isna(atr) and not pd.isna(current_price) and current_price > 0:
                # Convert ATR to percentage
                atr_percent = atr / current_price
                
                # Higher ATR suggests higher volatility
                # Normalize to typical crypto volatility of around 3-5%
                normal_volatility = 0.04  # 4% daily volatility as baseline
                volatility_ratio = atr_percent / normal_volatility
                
                # Higher volatility slightly bearish, lower volatility slightly bullish
                volatility_signal = -min(max(volatility_ratio - 1, -1), 1) * 0.5
                signals.append(volatility_signal)
                logger.debug(f"ATR volatility signal: {volatility_signal} (ATR: {atr}, ATR%: {atr_percent*100:.2f}%, ratio: {volatility_ratio:.2f})")
        else:
            logger.debug("ATR not available for volatility signal calculation")
        
        # Bollinger Bands width signal
        if 'bb_width' in df.columns:
            bb_width = df['bb_width'].iloc[-1]
            
            if not pd.isna(bb_width):
                # Calculate average BB width over last 20 periods
                avg_bb_width = df['bb_width'].rolling(20).mean().iloc[-1]
                
                if not pd.isna(avg_bb_width) and avg_bb_width > 0:
                    # Normalize width relative to average
                    relative_width = bb_width / avg_bb_width
                    
                    # Expanding bands (increasing volatility) - slightly bearish
                    # Contracting bands (decreasing volatility) - slightly bullish
                    if relative_width > 1.2:
                        bb_signal = 0.2  # Expanding bands - slight sell
                    elif relative_width < 0.8:
                        bb_signal = -0.2  # Contracting bands - slight buy
                    else:
                        bb_signal = 0  # Normal volatility - neutral
                    
                    signals.append(bb_signal)
                    logger.debug(f"BB width signal: {bb_signal} (Width: {bb_width:.4f}, Avg Width: {avg_bb_width:.4f}, Relative: {relative_width:.2f})")
        else:
            logger.debug("Bollinger Band width not available for volatility signal calculation")
        
        # Average all signals if we have any
        if signals:
            volatility_signal = sum(signals) / len(signals)
            logger.debug(f"Final volatility signal: {volatility_signal} (from {len(signals)} components)")
            return volatility_signal
        
        logger.warning("No volatility signals were generated, returning 0")
        return 0
    
    def generate_volume_signal(self, df: pd.DataFrame) -> float:
        """
        Generate volume signal.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Volume signal value (-1.0 to 1.0)
        """
        signals = []
        
        if 'volume' not in df.columns or len(df) < 20:
            logger.warning(f"Volume data not available or insufficient for signal calculation. Columns: {df.columns.tolist()}")
            # Force a small neutral signal to avoid exactly zero
            return 0.01
            
        try:
            # Get current and previous volumes
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-20:].mean()
            
            if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume <= 0:
                logger.debug(f"Invalid volume data: current={current_volume}, avg={avg_volume}")
                return 0.01
                
            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume
            logger.debug(f"Volume ratio: {volume_ratio:.2f} (Current: {current_volume}, Avg: {avg_volume})")
            
            # Always generate a base volume signal based on the ratio
            if volume_ratio > 1.0:
                # Above average volume (moderately bullish)
                base_signal = -0.1 * min(volume_ratio - 1.0, 2.0)
            else:
                # Below average volume (moderately bearish)
                base_signal = 0.1 * min(1.0 - volume_ratio, 0.5)
                
            signals.append(base_signal)
            logger.debug(f"Base volume signal: {base_signal:.4f} (ratio: {volume_ratio:.2f})")
            
            # Volume trend with price
            if 'close' in df.columns and len(df) >= 2:
                price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
                price_change_pct = price_change / df['close'].iloc[-2] if df['close'].iloc[-2] > 0 else 0
                
                # High volume with price increase (bullish)
                if price_change > 0:
                    # Scale signal based on both volume and price change
                    volume_signal = -0.3 * min(volume_ratio / 2, 1.5) * min(price_change_pct * 100, 2)
                    signals.append(volume_signal)
                    logger.debug(f"Bullish volume signal: {volume_signal:.4f} (Price change: +{price_change:.2f}, {price_change_pct*100:.2f}%, Volume ratio: {volume_ratio:.2f})")
                
                # High volume with price decrease (bearish)
                elif price_change < 0:
                    # Scale signal based on both volume and price change
                    volume_signal = 0.3 * min(volume_ratio / 2, 1.5) * min(abs(price_change_pct) * 100, 2)
                    signals.append(volume_signal)
                    logger.debug(f"Bearish volume signal: {volume_signal:.4f} (Price change: {price_change:.2f}, {price_change_pct*100:.2f}%, Volume ratio: {volume_ratio:.2f})")
                else:
                    logger.debug(f"No price change signal (Price change: {price_change}, Volume ratio: {volume_ratio:.2f})")
            else:
                logger.debug("Price data not available for volume signal calculation")
                
            # Add OBV signal if available
            if 'obv' in df.columns and len(df) >= 20:
                current_obv = df['obv'].iloc[-1]
                prev_obv = df['obv'].iloc[-2]
                avg_obv_change = abs(np.mean(np.diff(df['obv'].iloc[-20:])))
                
                if not pd.isna(current_obv) and not pd.isna(prev_obv) and avg_obv_change > 0:
                    # Calculate OBV change as a signal
                    obv_change = current_obv - prev_obv
                    normalized_obv_change = obv_change / avg_obv_change
                    
                    # Cap and convert to signal (-0.25 to 0.25)
                    obv_signal = -np.clip(normalized_obv_change / 2, -0.25, 0.25)
                    signals.append(obv_signal)
                    logger.debug(f"OBV signal: {obv_signal:.4f} (OBV change: {obv_change}, normalized: {normalized_obv_change:.2f})")
                    
            # Add MFI signal if available
            if 'mfi' in df.columns:
                mfi = df['mfi'].iloc[-1]
                if not pd.isna(mfi):
                    # MFI interpretation: <20 oversold (buy), >80 overbought (sell)
                    if mfi < 20:
                        mfi_signal = -0.4  # Strong buy
                    elif mfi < 30:
                        mfi_signal = -0.2  # Buy
                    elif mfi > 80:
                        mfi_signal = 0.4  # Strong sell
                    elif mfi > 70:
                        mfi_signal = 0.2  # Sell
                    else:
                        mfi_signal = (mfi - 50) / 100  # Small signal (-0.3 to 0.3)
                        
                    signals.append(mfi_signal)
                    logger.debug(f"MFI signal: {mfi_signal:.4f} (MFI value: {mfi:.2f})")
                
        except Exception as e:
            logger.error(f"Error calculating volume signals: {str(e)}", exc_info=True)
            return 0.01
            
        # Average all signals if we have any
        if signals:
            volume_signal = sum(signals) / len(signals)
            logger.debug(f"Final volume signal: {volume_signal:.4f} (from {len(signals)} components)")
            return volume_signal
        
        # Small non-zero value in case of no signals to avoid exactly zero
        logger.warning("No volume signals were generated, returning small non-zero value")
        return 0.01
    
    def generate_complete_signal(self, df: pd.DataFrame) -> Dict:
        """
        Generate complete signal with all components.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dictionary with comprehensive signal information
        """
        # Define a default neutral signal response
        neutral_signal = {
            'signal': 0,
            'signal_name': 'neutral',
            'trend': 0,
            'momentum': 0,
            'volatility': 0,
            'volume': 0,
            'signal_strength': 0,
            'entry_point': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward_ratio': 0,
            'action': 'hold',
            'direction': 'none',
            'current_price': 0
        }

        if df is None or df.empty:
            logger.warning("Empty dataframe provided to generate_complete_signal, returning neutral signal")
            return neutral_signal
        
        # Ensure we have at least the close price
        if 'close' not in df.columns:
            logger.warning("DataFrame missing 'close' column, returning neutral signal")
            return neutral_signal
        
        # Check if we have enough data points for technical analysis
        if len(df) < 14:  # Most basic indicators need at least 14 periods
            logger.warning(f"Insufficient data points ({len(df)}) for technical analysis, returning neutral signal")
            return neutral_signal
        
        # Log the columns available in the dataframe for debugging
        logger.debug(f"DataFrame columns for signal generation: {df.columns.tolist()}")
        logger.debug(f"DataFrame shape: {df.shape}")
        
        # Generate component signals
        try:
            trend_signal = self.generate_trend_signal(df)
        except Exception as e:
            logger.error(f"Error generating trend signal: {str(e)}", exc_info=True)
            trend_signal = 0
            
        try:
            momentum_signal = self.generate_momentum_signal(df)
        except Exception as e:
            logger.error(f"Error generating momentum signal: {str(e)}", exc_info=True)
            momentum_signal = 0
            
        try:
            volatility_signal = self.generate_volatility_signal(df)
        except Exception as e:
            logger.error(f"Error generating volatility signal: {str(e)}", exc_info=True)
            volatility_signal = 0
            
        try:
            volume_signal = self.generate_volume_signal(df)
        except Exception as e:
            logger.error(f"Error generating volume signal: {str(e)}", exc_info=True)
            volume_signal = 0
        
        # Log the individual component signals
        logger.info(f"Component signals: trend={trend_signal:.4f}, momentum={momentum_signal:.4f}, volatility={volatility_signal:.4f}, volume={volume_signal:.4f}")
        
        # Combine signals with weights
        combined_signal = (
            trend_signal * self.trend_weight +
            momentum_signal * self.momentum_weight +
            volatility_signal * self.volatility_weight +
            volume_signal * self.volume_weight
        )
        
        # Determine signal name based on thresholds
        if combined_signal >= self.strong_buy:
            signal_name = 'strong_buy'
        elif combined_signal >= self.buy:
            signal_name = 'buy'
        elif combined_signal <= self.strong_sell:
            signal_name = 'strong_sell'
        elif combined_signal <= self.sell:
            signal_name = 'sell'
        else:
            signal_name = 'neutral'
        
        # Determine direction and action
        if signal_name in ['buy', 'strong_buy']:
            direction = 'long'
            action = 'buy'
        elif signal_name in ['sell', 'strong_sell']:
            direction = 'short'
            action = 'sell'
        else:
            direction = 'none'
            action = 'hold'
        
        # Calculate signal strength (absolute value)
        signal_strength = abs(combined_signal)
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Calculate stop loss and take profit levels
        # Default values based on signal strength if no better method is available
        stop_loss_percent = 0.03 * (1 - signal_strength * 0.5)  # 1.5-3% stop loss
        take_profit_percent = 0.06 * (1 + signal_strength * 0.5)  # 6-9% take profit
        
        if direction == 'long':
            stop_loss = current_price * (1 - stop_loss_percent)
            take_profit = current_price * (1 + take_profit_percent)
        elif direction == 'short':
            stop_loss = current_price * (1 + stop_loss_percent)
            take_profit = current_price * (1 - take_profit_percent)
        else:
            stop_loss = current_price * (1 - 0.03)  # Default 3% stop loss
            take_profit = current_price * (1 + 0.06)  # Default 6% take profit
        
        # Calculate risk-reward ratio
        if abs(current_price - stop_loss) > 0:
            risk_reward_ratio = abs(take_profit - current_price) / abs(current_price - stop_loss)
        else:
            risk_reward_ratio = 0
        
        logger.info(f"Generated {signal_name} signal with strength {signal_strength:.4f} and direction {direction}")
        
        # Return complete signal information
        return {
            'signal': combined_signal,
            'signal_name': signal_name,
            'trend': trend_signal,
            'momentum': momentum_signal,
            'volatility': volatility_signal,
            'volume': volume_signal,
            'signal_strength': signal_strength,
            'entry_point': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'action': action,
            'direction': direction,
            'current_price': current_price
        } 