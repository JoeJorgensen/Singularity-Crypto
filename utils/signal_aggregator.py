"""
Signal Aggregator for CryptoTrading.

This module combines signals from multiple sources (technical indicators, sentiment analysis,
on-chain metrics, and machine learning models) to produce a unified trading signal with
weighting according to the configuration.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union

class SignalAggregator:
    """
    Aggregates various signals from different sources to produce a unified trading signal.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SignalAggregator with configuration settings.
        
        Args:
            config: Dictionary containing configuration settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Define default weights if not specified in config
        self.technical_weight = config.get('signal_weights', {}).get('technical', 0.4)
        self.sentiment_weight = config.get('signal_weights', {}).get('sentiment', 0.2)
        self.on_chain_weight = config.get('signal_weights', {}).get('on_chain', 0.2)
        self.ml_weight = config.get('signal_weights', {}).get('ml', 0.2)
        
        # Ensure weights sum to 1.0
        total = self.technical_weight + self.sentiment_weight + self.on_chain_weight + self.ml_weight
        if total != 1.0:
            self.logger.warning(f"Signal weights do not sum to 1.0 (sum: {total}). Normalizing weights.")
            self.technical_weight /= total
            self.sentiment_weight /= total
            self.on_chain_weight /= total
            self.ml_weight /= total
        
        # Signal thresholds from config
        self.thresholds = self.config.get('signal_thresholds', {})
        self.strong_buy = self.thresholds.get('strong_buy', 0.5)
        self.buy = self.thresholds.get('buy', 0.2)
        self.neutral_band = self.thresholds.get('neutral_band', 0.05)
        self.sell = self.thresholds.get('sell', -0.2)
        self.strong_sell = self.thresholds.get('strong_sell', -0.5)
    
    def aggregate_signals(self, 
                         technical_signal: Dict[str, Any],
                         sentiment_signal: Optional[Dict[str, Any]] = None,
                         on_chain_signal: Optional[Dict[str, Any]] = None,
                         ml_signal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Aggregate signals from different sources with their respective weights.
        
        Args:
            technical_signal: Dictionary containing technical analysis signals
            sentiment_signal: Optional dictionary containing sentiment analysis signals
            on_chain_signal: Optional dictionary containing on-chain metrics signals
            ml_signal: Optional dictionary containing machine learning model signals
            
        Returns:
            Dictionary with aggregated signal information including:
                - signal_strength: Combined weighted signal (-1 to 1)
                - action: String indicating the recommended action
                - direction: 'long' or 'short' if applicable
                - confidence: Confidence level in the signal (0-1)
                - components: Dictionary with individual signal contributions
                - entry_point: Suggested entry price if available
                - stop_loss: Suggested stop loss price if available
                - take_profit: Suggested take profit price if available
        """
        # Initialize component signals
        tech_strength = technical_signal.get('signal_strength', 0)
        sent_strength = sentiment_signal.get('signal_strength', 0) if sentiment_signal else 0
        chain_strength = on_chain_signal.get('signal_strength', 0) if on_chain_signal else 0
        ml_strength = ml_signal.get('signal_strength', 0) if ml_signal else 0
        
        # Log individual signal strengths
        self.logger.info(f"Signal components: Technical={tech_strength:.3f}, Sentiment={sent_strength:.3f}, "
                         f"On-Chain={chain_strength:.3f}, ML={ml_strength:.3f}")
        
        # Calculate weighted signal
        weighted_signal = (
            self.technical_weight * tech_strength +
            self.sentiment_weight * sent_strength +
            self.on_chain_weight * chain_strength +
            self.ml_weight * ml_strength
        )
        
        # Calculate signal consistency/confidence (how aligned are all signals?)
        signal_components = [
            (tech_strength, self.technical_weight),
            (sent_strength, self.sentiment_weight),
            (chain_strength, self.on_chain_weight),
            (ml_strength, self.ml_weight)
        ]
        
        # Only consider components with weight > 0
        signal_components = [s for s in signal_components if s[1] > 0]
        
        # Calculate weighted variance to assess alignment (lower variance = more aligned signals)
        if len(signal_components) > 1:
            weighted_variance = np.sum([weight * (strength - weighted_signal)**2 
                                      for strength, weight in signal_components])
            # Convert variance to confidence score (1 = perfectly aligned, 0 = completely contradictory)
            confidence = max(0, 1 - (weighted_variance / 2))  # Scale appropriately
        else:
            confidence = 1.0  # Only one signal component
        
        # Determine action based on signal strength and thresholds
        if weighted_signal >= self.strong_buy:
            action = 'strong_buy'
            direction = 'long'
        elif weighted_signal >= self.buy:
            action = 'buy'
            direction = 'long'
        elif weighted_signal <= self.strong_sell:
            action = 'strong_sell'
            direction = 'short'
        elif weighted_signal <= self.sell:
            action = 'sell'
            direction = 'short'
        else:
            action = 'neutral'
            direction = None
            
        # Get price levels from technical signals (most reliable source for these)
        entry_point = technical_signal.get('entry_point')
        stop_loss = technical_signal.get('stop_loss')
        take_profit = technical_signal.get('take_profit')
        
        # If ML signals provide price targets, blend them with technical signals
        if ml_signal and 'entry_point' in ml_signal and entry_point is not None:
            entry_point = (entry_point + ml_signal['entry_point']) / 2
            
        if ml_signal and 'stop_loss' in ml_signal and stop_loss is not None:
            stop_loss = (stop_loss + ml_signal['stop_loss']) / 2
            
        if ml_signal and 'take_profit' in ml_signal and take_profit is not None:
            take_profit = (take_profit + ml_signal['take_profit']) / 2
        
        # Ensure we have a valid risk/reward ratio if all price levels are present
        if entry_point is not None and stop_loss is not None and take_profit is not None:
            min_rr_ratio = self.config.get('trading', {}).get('min_risk_reward_ratio', 1.5)
            
            if direction == 'long':
                risk = entry_point - stop_loss
                reward = take_profit - entry_point
            else:  # short
                risk = stop_loss - entry_point
                reward = entry_point - take_profit
                
            if risk > 0:  # Avoid division by zero
                actual_rr = reward / risk
                if actual_rr < min_rr_ratio:
                    self.logger.warning(f"Risk/reward ratio ({actual_rr:.2f}) below minimum ({min_rr_ratio})")
                    
                    # Adjust take_profit to meet minimum R:R if the signal is still valid
                    if action in ['buy', 'strong_buy', 'sell', 'strong_sell']:
                        if direction == 'long':
                            take_profit = entry_point + (risk * min_rr_ratio)
                        else:  # short
                            take_profit = entry_point - (risk * min_rr_ratio)
                        self.logger.info(f"Adjusted take_profit to {take_profit} to meet minimum R:R ratio")
        
        return {
            'signal_strength': weighted_signal,
            'action': action,
            'direction': direction,
            'confidence': confidence,
            'components': {
                'technical': {
                    'strength': tech_strength,
                    'weight': self.technical_weight,
                    'contribution': tech_strength * self.technical_weight
                },
                'sentiment': {
                    'strength': sent_strength,
                    'weight': self.sentiment_weight,
                    'contribution': sent_strength * self.sentiment_weight
                },
                'on_chain': {
                    'strength': chain_strength,
                    'weight': self.on_chain_weight,
                    'contribution': chain_strength * self.on_chain_weight
                },
                'ml': {
                    'strength': ml_strength,
                    'weight': self.ml_weight,
                    'contribution': ml_strength * self.ml_weight
                }
            },
            'entry_point': entry_point,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def evaluate_signals_history(self, signal_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the history of signals to identify trends and consistency.
        
        Args:
            signal_history: List of historical signal dictionaries
            
        Returns:
            Dictionary with signal trend analysis including consistency and momentum
        """
        if not signal_history or len(signal_history) < 2:
            return {
                'signal_trend': 'insufficient_data',
                'consistency': 0,
                'momentum': 0
            }
        
        # Extract signal strengths and timestamps
        strengths = [signal['signal_strength'] for signal in signal_history]
        timestamps = [pd.Timestamp(signal['timestamp']) for signal in signal_history]
        
        # Calculate signal trend (linear regression slope)
        df = pd.DataFrame({
            'timestamp': timestamps,
            'strength': strengths
        })
        df['timestamp_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        
        if len(df) > 1:
            # Simple linear regression to get slope
            x = df['timestamp_numeric'].values
            y = df['strength'].values
            
            slope = np.cov(x, y)[0, 1] / np.var(x)
            momentum = slope * 3600  # Normalized to hourly change for better interpretability
            
            # Consistency: inverse of standard deviation
            consistency = 1.0 / (1.0 + np.std(strengths))
            
            # Determine trend description
            if abs(momentum) < 0.01:
                trend = 'flat'
            elif momentum > 0:
                trend = 'strengthening' if momentum > 0.05 else 'slightly_strengthening'
            else:
                trend = 'weakening' if momentum < -0.05 else 'slightly_weakening'
        else:
            trend = 'insufficient_data'
            consistency = 0
            momentum = 0
            
        return {
            'signal_trend': trend,
            'consistency': consistency,
            'momentum': momentum,
            'recent_values': strengths[-5:] if len(strengths) > 5 else strengths
        }

    def get_signal_description(self, signal: Dict[str, Any]) -> str:
        """
        Generate a human-readable description of the signal.
        
        Args:
            signal: Signal dictionary from aggregate_signals
            
        Returns:
            String with a description of the signal
        """
        if not signal:
            return "No signal data available."
            
        strength = signal.get('signal_strength', 0)
        action = signal.get('action', 'unknown')
        confidence = signal.get('confidence', 0)
        
        description = f"Signal strength: {strength:.3f} ({action}), Confidence: {confidence:.2f}"
        
        if signal.get('direction'):
            description += f"\nDirection: {signal.get('direction')}"
        
        if signal.get('entry_point'):
            description += f"\nSuggested entry: {signal.get('entry_point'):.2f}"
        
        if signal.get('stop_loss'):
            description += f"\nStop loss: {signal.get('stop_loss'):.2f}"
            
        if signal.get('take_profit'):
            description += f"\nTake profit: {signal.get('take_profit'):.2f}"
            
        return description
        
    def get_all_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate all signals from a dataframe with price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with all signals combined
        """
        if df is None or df.empty:
            return {}
            
        try:
            # Get the latest price data
            latest = df.iloc[-1]
            
            # Create a basic technical signal using the latest price
            technical_signal = {
                'signal_strength': 0,  # Neutral by default
                'entry_point': latest.get('close', 0) if hasattr(latest, 'get') else latest['close']
            }
            
            # Add trend indicators if available in the DataFrame
            if 'ema_20' in df.columns and 'ema_50' in df.columns:
                ema_20 = latest['ema_20']
                ema_50 = latest['ema_50']
                if ema_20 > ema_50:
                    technical_signal['signal_strength'] = 0.2  # Slight bullish
                elif ema_20 < ema_50:
                    technical_signal['signal_strength'] = -0.2  # Slight bearish
                    
            # Add RSI if available
            if 'rsi' in df.columns:
                rsi = latest['rsi']
                if rsi < 30:
                    technical_signal['signal_strength'] += 0.3  # Oversold
                elif rsi > 70:
                    technical_signal['signal_strength'] -= 0.3  # Overbought
            
            # For demo purposes, create placeholder signals for other components
            sentiment_signal = {'signal_strength': 0.1}  # Slightly positive sentiment
            on_chain_signal = {'signal_strength': 0}     # Neutral on-chain metrics
            ml_signal = {'signal_strength': 0.05}        # Slightly positive ML prediction
            
            # Aggregate all signals
            result = self.aggregate_signals(
                technical_signal=technical_signal,
                sentiment_signal=sentiment_signal,
                on_chain_signal=on_chain_signal,
                ml_signal=ml_signal
            )
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                return {}
                
            return result
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {} 