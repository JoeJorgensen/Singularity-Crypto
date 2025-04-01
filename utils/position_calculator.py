"""
Position sizing calculator for cryptocurrency trading.
"""
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import math

import pandas as pd
import numpy as np

from utils.risk_manager import RiskManager

class PositionCalculator:
    """
    Class to calculate optimal position sizes for cryptocurrency trading.
    """
    
    def __init__(self, risk_manager: Optional[RiskManager] = None, config: Optional[Dict] = None):
        """
        Initialize PositionCalculator with risk manager and configuration.
        
        Args:
            risk_manager: RiskManager instance
            config: Configuration dictionary with position sizing parameters
        """
        self.config = config or {}
        self.risk_manager = risk_manager or RiskManager(config)
    
    def calculate_basic_position_size(
        self,
        account_balance: float,
        risk_percentage: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Calculate basic position size based on risk percentage.
        
        Args:
            account_balance: Current account balance
            risk_percentage: Percentage of account to risk (0-1)
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            
        Returns:
            Position size in units
        """
        # Calculate dollar risk
        dollar_risk = account_balance * risk_percentage
        
        # Calculate price risk per unit
        price_risk = abs(entry_price - stop_loss_price)
        if price_risk == 0:
            return 0
        
        # Calculate position size
        position_size = dollar_risk / price_risk
        
        return position_size
    
    def calculate_atr_based_position_size(
        self,
        account_balance: float,
        risk_percentage: float,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 2.0
    ) -> float:
        """
        Calculate position size based on ATR for stop loss placement.
        
        Args:
            account_balance: Current account balance
            risk_percentage: Percentage of account to risk (0-1)
            entry_price: Entry price for the trade
            atr: Average True Range
            atr_multiplier: Multiplier for ATR to set stop loss distance
            
        Returns:
            Position size in units
        """
        # Calculate dollar risk
        dollar_risk = account_balance * risk_percentage
        
        # Calculate stop loss distance based on ATR
        stop_loss_distance = atr * atr_multiplier
        
        # Calculate position size
        position_size = dollar_risk / stop_loss_distance
        
        return position_size
    
    def calculate_volatility_adjusted_position_size(
        self,
        account_balance: float,
        risk_percentage: float,
        entry_price: float,
        price_data: pd.DataFrame,
        lookback_days: int = 20
    ) -> float:
        """
        Calculate position size adjusted for recent volatility.
        
        Args:
            account_balance: Current account balance
            risk_percentage: Percentage of account to risk (0-1)
            entry_price: Entry price for the trade
            price_data: DataFrame with price data
            lookback_days: Number of days to look back for volatility calculation
            
        Returns:
            Position size in units
        """
        if len(price_data) < 2:
            return 0
        
        # Calculate daily returns volatility
        returns = price_data['close'].pct_change().dropna()
        if len(returns) < lookback_days:
            lookback_days = len(returns)
        
        volatility = returns.iloc[-lookback_days:].std()
        
        # Adjust risk percentage based on volatility
        # Higher volatility -> Lower risk
        volatility_factor = 0.15  # Baseline volatility for crypto
        risk_adjustment = volatility_factor / max(volatility, 0.01)
        adjusted_risk = risk_percentage * min(risk_adjustment, 2.0)  # Cap at 2x increase
        
        # Calculate dollar risk
        dollar_risk = account_balance * adjusted_risk
        
        # Calculate position size based on 2 x daily volatility for stop loss
        stop_loss_distance = entry_price * volatility * 2
        
        # Calculate position size
        position_size = dollar_risk / stop_loss_distance
        
        return position_size
    
    def calculate_kelly_position_size(
        self,
        account_balance: float,
        win_rate: float,
        profit_loss_ratio: float,
        max_kelly_percentage: float = 0.2
    ) -> float:
        """
        Calculate position size using the Kelly Criterion.
        
        Args:
            account_balance: Current account balance
            win_rate: Probability of winning (0-1)
            profit_loss_ratio: Ratio of average profit to average loss
            max_kelly_percentage: Maximum allowed Kelly percentage
            
        Returns:
            Position size in account value terms
        """
        if win_rate <= 0 or win_rate >= 1 or profit_loss_ratio <= 0:
            return 0
        
        # Calculate Kelly percentage
        kelly = win_rate - ((1 - win_rate) / profit_loss_ratio)
        
        # Kelly can be negative, in which case don't trade
        if kelly <= 0:
            return 0
        
        # Most practitioners use a fraction of Kelly
        half_kelly = kelly * 0.5
        
        # Cap at max_kelly_percentage
        kelly_percentage = min(half_kelly, max_kelly_percentage)
        
        # Calculate position value
        position_value = account_balance * kelly_percentage
        
        return position_value
    
    def calculate_optimal_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        signal_strength: float,
        recent_volatility: float,
        max_position_percentage: Optional[float] = None
    ) -> Dict:
        """
        Calculate optimal position size considering multiple factors.
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            take_profit_price: Take profit price for the trade
            signal_strength: Strength of the trading signal (0-1)
            recent_volatility: Recent price volatility
            max_position_percentage: Maximum position size as percentage of account
            
        Returns:
            Dictionary with position size and calculation details
        """
        if max_position_percentage is None:
            max_position_percentage = self.config.get('risk_management', {}).get('max_position_size', 0.9)
        
        # Get default risk per trade
        risk_per_trade = self.config.get('risk_management', {}).get('risk_per_trade', 0.02)
        
        # Adjust risk based on signal strength
        adjusted_risk = risk_per_trade * (0.5 + signal_strength * 0.5)
        
        # Adjust risk based on volatility
        volatility_factor = 0.15 / max(recent_volatility, 0.01)
        volatility_adjusted_risk = adjusted_risk * min(volatility_factor, 2.0)
        
        # Calculate risk amount in dollars
        risk_amount = account_balance * volatility_adjusted_risk
        
        # Calculate standard position size based on risk per trade
        standard_position_size = self.calculate_basic_position_size(
            account_balance, volatility_adjusted_risk, entry_price, stop_loss_price
        )
        
        # Calculate profit/loss ratio
        price_risk = abs(entry_price - stop_loss_price)
        price_reward = abs(entry_price - take_profit_price)
        profit_loss_ratio = price_reward / price_risk if price_risk > 0 else 0
        
        # Estimate win rate from signal strength
        win_rate = 0.5 + (signal_strength * 0.2)  # Range 0.5-0.7 based on signal
        
        # Calculate Kelly position size
        kelly_percentage = self.risk_manager.calculate_kelly_criterion(win_rate, profit_loss_ratio)
        kelly_position_value = account_balance * kelly_percentage
        kelly_position_size = kelly_position_value / entry_price
        
        # Use the more conservative position size
        position_size = min(standard_position_size, kelly_position_size)
        
        # Calculate position value
        position_value = position_size * entry_price
        
        # Cap at max position percentage
        max_position_value = account_balance * max_position_percentage
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            position_value = max_position_value
        
        # Calculate final position percentage
        position_percentage = position_value / account_balance
        
        return {
            "position_size": position_size,
            "position_value": position_value,
            "position_percentage": position_percentage,
            "risk_per_trade": volatility_adjusted_risk,
            "risk_amount": risk_amount,
            "standard_position_size": standard_position_size,
            "kelly_position_size": kelly_position_size,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "price_risk": price_risk,
            "price_reward": price_reward,
            "max_position_percentage": max_position_percentage,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_pyramid_positions(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        signal_strength: float,
        num_positions: int = 3,
        pyramiding_factor: float = 0.7
    ) -> List[Dict]:
        """
        Calculate multiple position sizes for pyramiding into a trade.
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            take_profit_price: Take profit price for the trade
            signal_strength: Strength of the trading signal (0-1)
            num_positions: Number of positions to pyramid
            pyramiding_factor: Factor to reduce each subsequent position by
            
        Returns:
            List of dictionaries with position sizes for pyramiding
        """
        # Calculate base position size
        base_position_result = self.calculate_optimal_position_size(
            account_balance, entry_price, stop_loss_price, take_profit_price, signal_strength, 0.15
        )
        
        base_position_size = base_position_result['position_size']
        
        # Create pyramid positions
        pyramid_positions = []
        
        for i in range(num_positions):
            # Calculate position size for this level
            position_size = base_position_size * (pyramiding_factor ** i)
            position_value = position_size * entry_price
            
            # Calculate entry price for this level
            if i == 0:
                level_entry_price = entry_price
            else:
                # Only support long positions since crypto accounts are non-marginable and don't support short selling
                # Move entry price toward take profit for subsequent entries in a long position
                distance = (take_profit_price - entry_price) / (num_positions + 1)
                level_entry_price = entry_price + (distance * i)
            
            pyramid_positions.append({
                "level": i + 1,
                "position_size": position_size,
                "position_value": position_value,
                "entry_price": level_entry_price,
                "stop_loss_price": stop_loss_price,
                "percentage_of_base": (pyramiding_factor ** i) * 100
            })
        
        return pyramid_positions
    
    def calculate_portfolio_allocation(
        self,
        account_balance: float,
        signals: List[Dict],
        max_portfolio_risk: float = 0.05,
        max_position_percentage: float = 0.2
    ) -> List[Dict]:
        """
        Calculate optimal position sizes for multiple trading signals within a portfolio.
        
        Args:
            account_balance: Current account balance
            signals: List of dictionaries with signal information
            max_portfolio_risk: Maximum portfolio risk
            max_position_percentage: Maximum position size as percentage of account
            
        Returns:
            List of dictionaries with position sizes for each signal
        """
        # Sort signals by expected value
        sorted_signals = sorted(signals, key=lambda x: x.get('expected_value', 0), reverse=True)
        
        # Calculate total signal strength
        total_strength = sum(max(0.01, abs(s.get('signal', 0))) for s in sorted_signals)
        
        # Calculate total portfolio risk
        portfolio_risk = min(max_portfolio_risk, sum(s.get('risk_per_trade', 0.01) for s in sorted_signals))
        
        # Calculate position sizes
        positions = []
        remaining_balance = account_balance
        
        for signal in sorted_signals:
            if remaining_balance <= 0:
                break
                
            # Extract signal details
            entry_price = signal.get('entry_point', 0)
            stop_loss_price = signal.get('stop_loss', 0)
            take_profit_price = signal.get('take_profit', 0)
            signal_strength = abs(signal.get('signal', 0)) / max(0.01, total_strength)
            
            # Skip invalid signals
            if entry_price <= 0 or stop_loss_price <= 0 or take_profit_price <= 0:
                continue
            
            # Calculate position weighting based on signal strength and portfolio risk
            position_risk = signal.get('risk_per_trade', 0.01) * signal_strength
            weighted_risk = min(position_risk, 0.05)  # Cap individual position risk
            
            # Calculate position size
            position_result = self.calculate_optimal_position_size(
                remaining_balance,
                entry_price,
                stop_loss_price,
                take_profit_price,
                signal_strength,
                signal.get('recent_volatility', 0.15),
                max_position_percentage
            )
            
            # Cap position value
            max_position_value = account_balance * max_position_percentage
            position_value = min(position_result['position_value'], max_position_value)
            position_size = position_value / entry_price
            
            # Add to positions list
            positions.append({
                "symbol": signal.get('symbol', 'unknown'),
                "direction": signal.get('direction', 'none'),
                "position_size": position_size,
                "position_value": position_value,
                "percentage_of_account": position_value / account_balance,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "signal_strength": signal_strength,
                "expected_value": signal.get('expected_value', 0),
                "risk_reward_ratio": signal.get('risk_reward_ratio', 0)
            })
            
            # Update remaining balance
            remaining_balance -= position_value
        
        return positions
    
    def format_position_size(
        self,
        position_size: float,
        min_qty: float = 0.0001,
        round_to: Optional[int] = None
    ) -> float:
        """
        Format position size to meet exchange requirements.
        
        Args:
            position_size: Calculated position size
            min_qty: Minimum quantity allowed
            round_to: Decimal places to round to (e.g., 4 for 0.0001)
            
        Returns:
            Formatted position size
        """
        # Ensure position size meets minimum quantity
        if position_size < min_qty:
            return 0
        
        # Round to specific decimal places if specified
        if round_to is not None:
            multiplier = 10 ** round_to
            position_size = math.floor(position_size * multiplier) / multiplier
        
        return position_size 