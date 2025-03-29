"""
PatternDetector - Detect technical chart patterns in cryptocurrency price data.
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import time
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PatternDetector:
    """
    Detect technical chart patterns in cryptocurrency price data.
    This implementation uses simplified pattern detection algorithms with
    some randomness for demonstration. In production, more sophisticated
    techniques would be used.
    """
    
    def __init__(self):
        """Initialize pattern detector."""
        self.cache = {}
        self.cache_expiry = 3600  # 1 hour
        self.supported_patterns = [
            'double_top', 'double_bottom', 'head_and_shoulders', 'inverse_head_and_shoulders',
            'triangle', 'wedge', 'channel', 'flag', 'pennant', 'cup_and_handle'
        ]
        
        # Pattern characteristics for interpretation
        self.pattern_characteristics = {
            'double_top': {
                'type': 'reversal',
                'trend_before': 'bullish',
                'signal': 'bearish',
                'reliability': 0.65,
                'description': 'A bearish reversal pattern showing two peaks at approximately the same price level'
            },
            'double_bottom': {
                'type': 'reversal',
                'trend_before': 'bearish',
                'signal': 'bullish',
                'reliability': 0.65,
                'description': 'A bullish reversal pattern showing two troughs at approximately the same price level'
            },
            'head_and_shoulders': {
                'type': 'reversal',
                'trend_before': 'bullish',
                'signal': 'bearish',
                'reliability': 0.7,
                'description': 'A bearish reversal pattern with three peaks, the middle one higher than the others'
            },
            'inverse_head_and_shoulders': {
                'type': 'reversal',
                'trend_before': 'bearish',
                'signal': 'bullish',
                'reliability': 0.7,
                'description': 'A bullish reversal pattern with three troughs, the middle one lower than the others'
            },
            'triangle': {
                'type': 'continuation',
                'trend_before': 'any',
                'signal': 'neutral',
                'reliability': 0.6,
                'description': 'A consolidation pattern where price range narrows, often continuing the previous trend'
            },
            'wedge': {
                'type': 'reversal',
                'trend_before': 'any',
                'signal': 'reversal',
                'reliability': 0.55,
                'description': 'A pattern where price range narrows with sloping trend lines, often leading to reversal'
            },
            'channel': {
                'type': 'continuation',
                'trend_before': 'any',
                'signal': 'continuation',
                'reliability': 0.6,
                'description': 'Price moving between parallel support and resistance lines'
            },
            'flag': {
                'type': 'continuation',
                'trend_before': 'any',
                'signal': 'continuation',
                'reliability': 0.65,
                'description': 'A short-term consolidation pattern after a strong move, usually continuing the trend'
            },
            'pennant': {
                'type': 'continuation',
                'trend_before': 'any',
                'signal': 'continuation',
                'reliability': 0.65,
                'description': 'Similar to a flag but with converging trend lines in a symmetrical triangle'
            },
            'cup_and_handle': {
                'type': 'continuation',
                'trend_before': 'bullish',
                'signal': 'bullish',
                'reliability': 0.7,
                'description': 'A bullish continuation pattern resembling a cup followed by a small downward move'
            }
        }
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect chart patterns in price data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with detected patterns
        """
        if len(df) < 30:
            return {
                'patterns': [],
                'count': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        # Create a cache key based on the last 50 rows of data
        last_rows = df.tail(50)
        close_values = last_rows['close'].values
        cache_key = f"pattern_detection:{hash(tuple(close_values))}"
        
        # Check cache
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['data']
        
        # Prepare results
        patterns = []
        
        try:
            # For demonstration, we'll implement simplified detection algorithms
            # In practice, more sophisticated algorithms would be used
            
            # Check for double top pattern
            double_top = self._detect_double_top(df)
            if double_top['detected']:
                patterns.append({
                    'name': 'double_top',
                    'confidence': double_top['confidence'],
                    'start_index': double_top['start_index'],
                    'end_index': double_top['end_index'],
                    'signal': self.pattern_characteristics['double_top']['signal'],
                    'description': self.pattern_characteristics['double_top']['description']
                })
            
            # Check for double bottom pattern
            double_bottom = self._detect_double_bottom(df)
            if double_bottom['detected']:
                patterns.append({
                    'name': 'double_bottom',
                    'confidence': double_bottom['confidence'],
                    'start_index': double_bottom['start_index'],
                    'end_index': double_bottom['end_index'],
                    'signal': self.pattern_characteristics['double_bottom']['signal'],
                    'description': self.pattern_characteristics['double_bottom']['description']
                })
            
            # Check for head and shoulders pattern
            head_shoulders = self._detect_head_and_shoulders(df)
            if head_shoulders['detected']:
                patterns.append({
                    'name': 'head_and_shoulders',
                    'confidence': head_shoulders['confidence'],
                    'start_index': head_shoulders['start_index'],
                    'end_index': head_shoulders['end_index'],
                    'signal': self.pattern_characteristics['head_and_shoulders']['signal'],
                    'description': self.pattern_characteristics['head_and_shoulders']['description']
                })
            
            # Check for inverse head and shoulders pattern
            inv_head_shoulders = self._detect_inverse_head_and_shoulders(df)
            if inv_head_shoulders['detected']:
                patterns.append({
                    'name': 'inverse_head_and_shoulders',
                    'confidence': inv_head_shoulders['confidence'],
                    'start_index': inv_head_shoulders['start_index'],
                    'end_index': inv_head_shoulders['end_index'],
                    'signal': self.pattern_characteristics['inverse_head_and_shoulders']['signal'],
                    'description': self.pattern_characteristics['inverse_head_and_shoulders']['description']
                })
            
            # Check for triangle pattern
            triangle = self._detect_triangle(df)
            if triangle['detected']:
                patterns.append({
                    'name': triangle['type'],
                    'confidence': triangle['confidence'],
                    'start_index': triangle['start_index'],
                    'end_index': triangle['end_index'],
                    'signal': 'bullish' if triangle['type'] == 'ascending_triangle' else 
                              'bearish' if triangle['type'] == 'descending_triangle' else 'neutral',
                    'description': self.pattern_characteristics['triangle']['description']
                })
            
            # Check for channel pattern
            channel = self._detect_channel(df)
            if channel['detected']:
                patterns.append({
                    'name': channel['type'],
                    'confidence': channel['confidence'],
                    'start_index': channel['start_index'],
                    'end_index': channel['end_index'],
                    'signal': 'bullish' if channel['type'] == 'ascending_channel' else 'bearish',
                    'description': self.pattern_characteristics['channel']['description']
                })
            
            # Sort patterns by confidence
            patterns = sorted(patterns, key=lambda p: p['confidence'], reverse=True)
        
        except Exception as e:
            print(f"Error detecting patterns: {e}")
        
        # Prepare result
        result = {
            'patterns': patterns,
            'count': len(patterns),
            'timestamp': datetime.now().isoformat()
        }
        
        # Update cache
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'data': result
        }
        
        return result
    
    def _detect_double_top(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect double top pattern.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with detection result
        """
        # This is a simplified implementation for demonstration
        # In practice, more sophisticated algorithms would be used
        
        # Default result
        result = {
            'detected': False,
            'confidence': 0.0,
            'start_index': 0,
            'end_index': 0
        }
        
        try:
            # Check if there's enough data
            if len(df) < 20:
                return result
            
            # Get high prices
            highs = df['high'].values
            
            # Identify potential peaks (local maxima)
            peaks = []
            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
                   highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    peaks.append((i, highs[i]))
            
            # Need at least 2 peaks for a double top
            if len(peaks) < 2:
                return result
            
            # Look at the last 3 peaks
            recent_peaks = peaks[-3:]
            
            # Need at least 2 recent peaks
            if len(recent_peaks) < 2:
                return result
            
            # Check the last two peaks
            peak1_idx, peak1_val = recent_peaks[-2]
            peak2_idx, peak2_val = recent_peaks[-1]
            
            # Check if peaks are close in value but separated in time
            value_diff_pct = abs(peak1_val - peak2_val) / peak1_val
            time_diff = peak2_idx - peak1_idx
            
            # For a valid double top:
            # 1. Peaks should be close in value (within 3%)
            # 2. Peaks should be separated by some time (at least 5 bars)
            # 3. There should be a trough between the peaks
            
            if value_diff_pct <= 0.03 and time_diff >= 5:
                # Check for trough between peaks
                trough_idx = peak1_idx
                trough_val = float('inf')
                
                for i in range(peak1_idx + 1, peak2_idx):
                    if df['low'].iloc[i] < trough_val:
                        trough_val = df['low'].iloc[i]
                        trough_idx = i
                
                # The trough should be significantly lower than the peaks
                trough_diff_pct = (peak1_val - trough_val) / peak1_val
                
                if trough_diff_pct >= 0.05:
                    # Calculate confidence based on how well it fits pattern criteria
                    value_confidence = max(0, 1 - value_diff_pct/0.03)
                    time_confidence = min(1, time_diff/10)
                    trough_confidence = min(1, trough_diff_pct/0.1)
                    
                    confidence = (value_confidence + time_confidence + trough_confidence) / 3
                    
                    # Current price should be below the neckline (support level at the trough)
                    current_price = df['close'].iloc[-1]
                    if current_price < trough_val:
                        confidence += 0.2
                        confidence = min(confidence, 0.95)  # Cap at 0.95
                    
                    result = {
                        'detected': True,
                        'confidence': confidence,
                        'start_index': peak1_idx,
                        'end_index': len(df) - 1
                    }
        
        except Exception as e:
            print(f"Error detecting double top pattern: {e}")
        
        return result
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect double bottom pattern.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with detection result
        """
        # This is a simplified implementation for demonstration
        # In practice, more sophisticated algorithms would be used
        
        # Default result
        result = {
            'detected': False,
            'confidence': 0.0,
            'start_index': 0,
            'end_index': 0
        }
        
        try:
            # Check if there's enough data
            if len(df) < 20:
                return result
            
            # Get low prices
            lows = df['low'].values
            
            # Identify potential troughs (local minima)
            troughs = []
            for i in range(2, len(lows) - 2):
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
                   lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    troughs.append((i, lows[i]))
            
            # Need at least 2 troughs for a double bottom
            if len(troughs) < 2:
                return result
            
            # Look at the last 3 troughs
            recent_troughs = troughs[-3:]
            
            # Need at least 2 recent troughs
            if len(recent_troughs) < 2:
                return result
            
            # Check the last two troughs
            trough1_idx, trough1_val = recent_troughs[-2]
            trough2_idx, trough2_val = recent_troughs[-1]
            
            # Check if troughs are close in value but separated in time
            value_diff_pct = abs(trough1_val - trough2_val) / trough1_val
            time_diff = trough2_idx - trough1_idx
            
            # For a valid double bottom:
            # 1. Troughs should be close in value (within 3%)
            # 2. Troughs should be separated by some time (at least 5 bars)
            # 3. There should be a peak between the troughs
            
            if value_diff_pct <= 0.03 and time_diff >= 5:
                # Check for peak between troughs
                peak_idx = trough1_idx
                peak_val = float('-inf')
                
                for i in range(trough1_idx + 1, trough2_idx):
                    if df['high'].iloc[i] > peak_val:
                        peak_val = df['high'].iloc[i]
                        peak_idx = i
                
                # The peak should be significantly higher than the troughs
                peak_diff_pct = (peak_val - trough1_val) / trough1_val
                
                if peak_diff_pct >= 0.05:
                    # Calculate confidence based on how well it fits pattern criteria
                    value_confidence = max(0, 1 - value_diff_pct/0.03)
                    time_confidence = min(1, time_diff/10)
                    peak_confidence = min(1, peak_diff_pct/0.1)
                    
                    confidence = (value_confidence + time_confidence + peak_confidence) / 3
                    
                    # Current price should be above the neckline (resistance level at the peak)
                    current_price = df['close'].iloc[-1]
                    if current_price > peak_val:
                        confidence += 0.2
                        confidence = min(confidence, 0.95)  # Cap at 0.95
                    
                    result = {
                        'detected': True,
                        'confidence': confidence,
                        'start_index': trough1_idx,
                        'end_index': len(df) - 1
                    }
        
        except Exception as e:
            print(f"Error detecting double bottom pattern: {e}")
        
        return result
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect head and shoulders pattern.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with detection result
        """
        # This is a simplified implementation for demonstration
        # In practice, more sophisticated algorithms would be used
        
        # Default result
        result = {
            'detected': False,
            'confidence': 0.0,
            'start_index': 0,
            'end_index': 0
        }
        
        try:
            # For simplicity, we'll use a probabilistic approach
            # In a real system, this would be replaced with actual pattern detection
            
            # Check if there's enough data
            if len(df) < 30:
                return result
            
            # Detect a strong uptrend followed by a potential reversal
            returns = df['close'].pct_change(5).iloc[-10:]
            
            # Check for an uptrend followed by a potential topping pattern
            if returns.iloc[0:5].mean() > 0.02 and returns.iloc[5:].mean() < 0:
                # Simulate a pattern detection with some randomness
                confidence = random.uniform(0.4, 0.7)
                
                # The more negative the recent returns, the higher the confidence
                confidence -= returns.iloc[5:].mean() * 2
                confidence = min(max(confidence, 0.0), 0.9)
                
                if confidence > 0.5:  # Only return if confidence is reasonable
                    result = {
                        'detected': True,
                        'confidence': confidence,
                        'start_index': max(0, len(df) - 30),
                        'end_index': len(df) - 1
                    }
        
        except Exception as e:
            print(f"Error detecting head and shoulders pattern: {e}")
        
        return result
    
    def _detect_inverse_head_and_shoulders(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect inverse head and shoulders pattern.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with detection result
        """
        # This is a simplified implementation for demonstration
        # In practice, more sophisticated algorithms would be used
        
        # Default result
        result = {
            'detected': False,
            'confidence': 0.0,
            'start_index': 0,
            'end_index': 0
        }
        
        try:
            # For simplicity, we'll use a probabilistic approach
            # In a real system, this would be replaced with actual pattern detection
            
            # Check if there's enough data
            if len(df) < 30:
                return result
            
            # Detect a strong downtrend followed by a potential reversal
            returns = df['close'].pct_change(5).iloc[-10:]
            
            # Check for a downtrend followed by a potential bottoming pattern
            if returns.iloc[0:5].mean() < -0.02 and returns.iloc[5:].mean() > 0:
                # Simulate a pattern detection with some randomness
                confidence = random.uniform(0.4, 0.7)
                
                # The more positive the recent returns, the higher the confidence
                confidence += returns.iloc[5:].mean() * 2
                confidence = min(max(confidence, 0.0), 0.9)
                
                if confidence > 0.5:  # Only return if confidence is reasonable
                    result = {
                        'detected': True,
                        'confidence': confidence,
                        'start_index': max(0, len(df) - 30),
                        'end_index': len(df) - 1
                    }
        
        except Exception as e:
            print(f"Error detecting inverse head and shoulders pattern: {e}")
        
        return result
    
    def _detect_triangle(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect triangle patterns (ascending, descending, or symmetrical).
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with detection result
        """
        # This is a simplified implementation for demonstration
        # In practice, more sophisticated algorithms would be used
        
        # Default result
        result = {
            'detected': False,
            'type': 'triangle',
            'confidence': 0.0,
            'start_index': 0,
            'end_index': 0
        }
        
        try:
            # For simplicity, we'll use a probabilistic approach
            # In a real system, this would be replaced with actual pattern detection
            
            # Check if there's enough data
            if len(df) < 20:
                return result
            
            # Look at price volatility decreasing - a characteristic of triangles
            recent_data = df.tail(20)
            volatility = (recent_data['high'] - recent_data['low']) / recent_data['close']
            
            early_volatility = volatility.iloc[:10].mean()
            late_volatility = volatility.iloc[10:].mean()
            
            # Check for narrowing volatility
            if late_volatility < early_volatility * 0.7:
                # Determine type of triangle
                recent_highs = recent_data['high']
                recent_lows = recent_data['low']
                
                high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
                low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
                
                triangle_type = 'symmetrical_triangle'
                
                if high_slope < -0.001 and low_slope > 0.001:
                    triangle_type = 'symmetrical_triangle'
                elif high_slope > -0.001 and low_slope > 0.001:
                    triangle_type = 'ascending_triangle'
                elif high_slope < -0.001 and low_slope < 0.001:
                    triangle_type = 'descending_triangle'
                
                # Calculate confidence based on volatility reduction
                volatility_ratio = late_volatility / early_volatility
                confidence = max(0, min(1, 1 - volatility_ratio))
                
                # Adjust confidence based on how clear the pattern is
                if triangle_type == 'symmetrical_triangle':
                    if abs(high_slope) > 0.01 and abs(low_slope) > 0.01:
                        confidence *= 0.8
                elif triangle_type == 'ascending_triangle':
                    if abs(high_slope) > 0.005 or low_slope < 0.005:
                        confidence *= 0.8
                elif triangle_type == 'descending_triangle':
                    if abs(low_slope) > 0.005 or high_slope > -0.005:
                        confidence *= 0.8
                
                if confidence > 0.4:  # Only return if confidence is reasonable
                    result = {
                        'detected': True,
                        'type': triangle_type,
                        'confidence': confidence,
                        'start_index': len(df) - 20,
                        'end_index': len(df) - 1
                    }
        
        except Exception as e:
            print(f"Error detecting triangle pattern: {e}")
        
        return result
    
    def _detect_channel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect channel patterns (ascending or descending).
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with detection result
        """
        # This is a simplified implementation for demonstration
        # In practice, more sophisticated algorithms would be used
        
        # Default result
        result = {
            'detected': False,
            'type': 'channel',
            'confidence': 0.0,
            'start_index': 0,
            'end_index': 0
        }
        
        try:
            # For simplicity, we'll use a probabilistic approach
            # In a real system, this would be replaced with actual pattern detection
            
            # Check if there's enough data
            if len(df) < 20:
                return result
            
            # Look at the trend of highs and lows
            recent_data = df.tail(20)
            recent_highs = recent_data['high']
            recent_lows = recent_data['low']
            
            high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            # Check if both high and low trends are moving in the same direction
            if (high_slope > 0.001 and low_slope > 0.001) or (high_slope < -0.001 and low_slope < -0.001):
                # Determine channel type
                channel_type = 'ascending_channel' if high_slope > 0 else 'descending_channel'
                
                # Check if the slopes are roughly parallel
                slope_ratio = min(abs(high_slope), abs(low_slope)) / max(abs(high_slope), abs(low_slope))
                
                if slope_ratio > 0.7:  # Fairly parallel slopes
                    # Calculate confidence based on how well prices respect the channel
                    confidence = slope_ratio
                    
                    # Check for at least two touches of each boundary
                    high_line = np.polyval([high_slope, np.polyfit(range(len(recent_highs)), recent_highs, 1)[1]], range(len(recent_highs)))
                    low_line = np.polyval([low_slope, np.polyfit(range(len(recent_lows)), recent_lows, 1)[1]], range(len(recent_lows)))
                    
                    high_touches = sum(1 for i in range(len(recent_highs)) if abs(recent_highs.iloc[i] - high_line[i]) / recent_highs.iloc[i] < 0.01)
                    low_touches = sum(1 for i in range(len(recent_lows)) if abs(recent_lows.iloc[i] - low_line[i]) / recent_lows.iloc[i] < 0.01)
                    
                    if high_touches >= 2 and low_touches >= 2:
                        confidence += 0.2
                    
                    confidence = min(confidence, 0.9)
                    
                    if confidence > 0.5:  # Only return if confidence is reasonable
                        result = {
                            'detected': True,
                            'type': channel_type,
                            'confidence': confidence,
                            'start_index': len(df) - 20,
                            'end_index': len(df) - 1
                        }
        
        except Exception as e:
            print(f"Error detecting channel pattern: {e}")
        
        return result
    
    def get_pattern_interpretation(self, pattern_name: str) -> Dict[str, Any]:
        """
        Get detailed interpretation of a chart pattern.
        
        Args:
            pattern_name: Name of the pattern
            
        Returns:
            Dictionary with pattern interpretation
        """
        if pattern_name in self.pattern_characteristics:
            return self.pattern_characteristics[pattern_name]
        
        # Handle specific triangle and channel types
        if pattern_name.endswith('_triangle'):
            base_info = self.pattern_characteristics['triangle'].copy()
            
            if pattern_name == 'ascending_triangle':
                base_info['signal'] = 'bullish'
                base_info['description'] = 'A bullish continuation pattern with a flat top resistance line and rising bottom support line'
            elif pattern_name == 'descending_triangle':
                base_info['signal'] = 'bearish'
                base_info['description'] = 'A bearish continuation pattern with a flat bottom support line and falling top resistance line'
            elif pattern_name == 'symmetrical_triangle':
                base_info['signal'] = 'neutral'
                base_info['description'] = 'A continuation pattern with converging trend lines, often continuing the previous trend'
            
            return base_info
        
        if pattern_name.endswith('_channel'):
            base_info = self.pattern_characteristics['channel'].copy()
            
            if pattern_name == 'ascending_channel':
                base_info['signal'] = 'bullish'
                base_info['description'] = 'A bullish continuation pattern with parallel rising support and resistance lines'
            elif pattern_name == 'descending_channel':
                base_info['signal'] = 'bearish'
                base_info['description'] = 'A bearish continuation pattern with parallel falling support and resistance lines'
            
            return base_info
        
        # Default interpretation
        return {
            'type': 'unknown',
            'trend_before': 'any',
            'signal': 'neutral',
            'reliability': 0.5,
            'description': 'Unknown pattern type'
        } 