"""
Risk management utilities for cryptocurrency trading.
"""
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import math

import pandas as pd
import numpy as np

class RiskManager:
    """
    Class to handle risk management for cryptocurrency trading.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize RiskManager with configuration.
        
        Args:
            config: Configuration dictionary with risk management parameters
        """
        self.config = config or {}
        
        # Extract risk management parameters with defaults
        self.risk_per_trade = self.config.get('risk_management', {}).get('risk_per_trade', 0.02)
        self.max_position_size = self.config.get('risk_management', {}).get('max_position_size', 0.9)
        self.max_drawdown = self.config.get('risk_management', {}).get('max_drawdown', 0.15)
        self.var_limit = self.config.get('risk_management', {}).get('var_limit', 0.05)
        self.min_risk_reward_ratio = self.config.get('risk_management', {}).get('min_risk_reward_ratio', 1.5)
        self.max_trades_per_day = self.config.get('trading', {}).get('max_trades_per_day', 5)
        self.stop_loss_percentage = self.config.get('risk_management', {}).get('stop_loss_percentage', 0.03)
        self.take_profit_percentage = self.config.get('risk_management', {}).get('take_profit_percentage', 0.045)
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on risk per trade.
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            risk_per_trade: Override the default risk per trade percentage
            
        Returns:
            Quantity to buy/sell
        """
        if risk_per_trade is None:
            risk_per_trade = self.risk_per_trade
        
        # Calculate risk amount
        risk_amount = account_balance * risk_per_trade
        
        # Calculate risk per unit
        price_risk = abs(entry_price - stop_loss_price)
        if price_risk == 0:
            price_risk = entry_price * self.stop_loss_percentage
        
        # Calculate position size
        position_size = risk_amount / price_risk
        
        # Calculate position value
        position_value = position_size * entry_price
        
        # Check if position value exceeds max position size
        max_position_value = account_balance * self.max_position_size
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
        
        return position_size
    
    def calculate_risk_reward_ratio(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float
    ) -> float:
        """
        Calculate risk-reward ratio for a trade.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            take_profit_price: Take profit price for the trade
            
        Returns:
            Risk-reward ratio (reward/risk)
        """
        risk = abs(entry_price - stop_loss_price)
        reward = abs(entry_price - take_profit_price)
        
        if risk == 0:
            return float('inf')
        
        return reward / risk
    
    def check_risk_reward_filter(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        min_risk_reward_ratio: Optional[float] = None
    ) -> bool:
        """
        Check if a trade passes the risk-reward filter.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            take_profit_price: Take profit price for the trade
            min_risk_reward_ratio: Override the default minimum risk-reward ratio
            
        Returns:
            True if trade passes the filter, False otherwise
        """
        if min_risk_reward_ratio is None:
            min_risk_reward_ratio = self.min_risk_reward_ratio
        
        risk_reward_ratio = self.calculate_risk_reward_ratio(entry_price, stop_loss_price, take_profit_price)
        return risk_reward_ratio >= min_risk_reward_ratio
    
    def calculate_expected_value(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        win_probability: float
    ) -> float:
        """
        Calculate expected value of a trade.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            take_profit_price: Take profit price for the trade
            win_probability: Probability of winning the trade (0-1)
            
        Returns:
            Expected value as a percentage of entry price
        """
        risk = abs(entry_price - stop_loss_price) / entry_price
        reward = abs(entry_price - take_profit_price) / entry_price
        
        expected_value = (win_probability * reward) - ((1 - win_probability) * risk)
        return expected_value
    
    def check_max_drawdown(
        self,
        current_drawdown: float,
        potential_loss_percentage: float,
        max_drawdown: Optional[float] = None
    ) -> bool:
        """
        Check if a trade would exceed maximum drawdown.
        
        Args:
            current_drawdown: Current drawdown as a percentage
            potential_loss_percentage: Potential loss from the trade as a percentage
            max_drawdown: Override the default maximum drawdown percentage
            
        Returns:
            True if trade would not exceed maximum drawdown, False otherwise
        """
        if max_drawdown is None:
            max_drawdown = self.max_drawdown
        
        # Calculate potential new drawdown
        potential_drawdown = current_drawdown + potential_loss_percentage
        
        return potential_drawdown <= max_drawdown
    
    def calculate_var(
        self,
        price_data: pd.DataFrame,
        position_size: float,
        confidence_level: float = 0.95,
        lookback_days: int = 20
    ) -> float:
        """
        Calculate Value at Risk (VaR) for a position.
        
        Args:
            price_data: DataFrame with price data
            position_size: Position size in units
            confidence_level: Confidence level for VaR calculation
            lookback_days: Number of days to look back for historical data
            
        Returns:
            Value at Risk as a percentage of position value
        """
        if len(price_data) < lookback_days:
            lookback_days = len(price_data)
        
        # Calculate daily returns
        returns = price_data['close'].pct_change().dropna().iloc[-lookback_days:]
        
        # Calculate VaR
        var = np.percentile(returns, 100 * (1 - confidence_level))
        
        # Scale VaR to position value
        position_value = position_size * price_data['close'].iloc[-1]
        var_value = position_value * abs(var)
        
        return var_value / position_value
    
    def check_var_limit(
        self,
        price_data: pd.DataFrame,
        position_size: float,
        var_limit: Optional[float] = None,
        confidence_level: float = 0.95
    ) -> bool:
        """
        Check if a position passes the Value at Risk (VaR) limit.
        
        Args:
            price_data: DataFrame with price data
            position_size: Position size in units
            var_limit: Override the default VaR limit
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            True if position passes the filter, False otherwise
        """
        if var_limit is None:
            var_limit = self.var_limit
        
        var = self.calculate_var(price_data, position_size, confidence_level)
        return var <= var_limit
    
    def check_max_trades_per_day(
        self,
        recent_trades: List[Dict],
        max_trades_per_day: Optional[int] = None
    ) -> bool:
        """
        Check if maximum trades per day limit has been reached.
        
        Args:
            recent_trades: List of recent trades with timestamp
            max_trades_per_day: Override the default maximum trades per day
            
        Returns:
            True if limit has not been reached, False otherwise
        """
        if max_trades_per_day is None:
            max_trades_per_day = self.max_trades_per_day
        
        # Get current date
        today = datetime.now().date()
        
        # Count trades for today
        today_trades = sum(
            1 for trade in recent_trades
            if datetime.fromisoformat(trade['timestamp']).date() == today
        )
        
        return today_trades < max_trades_per_day
    
    def adjust_stop_loss_take_profit(
        self,
        entry_price: float,
        atr: float,
        direction: str = 'long',
        risk_reward_ratio: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels based on ATR.
        
        Args:
            entry_price: Entry price for the trade
            atr: Average True Range
            direction: Trade direction ('long' or 'short')
            risk_reward_ratio: Override the default risk-reward ratio
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if risk_reward_ratio is None:
            risk_reward_ratio = self.min_risk_reward_ratio
        
        # Default ATR multiplier for stop loss
        atr_multiplier = 2.0
        
        if direction.lower() == 'long':
            stop_loss_price = entry_price - (atr * atr_multiplier)
            take_profit_price = entry_price + (atr * atr_multiplier * risk_reward_ratio)
        else:  # short
            stop_loss_price = entry_price + (atr * atr_multiplier)
            take_profit_price = entry_price - (atr * atr_multiplier * risk_reward_ratio)
        
        return stop_loss_price, take_profit_price
    
    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win_loss_ratio: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win_loss_ratio: Average win/loss ratio
            
        Returns:
            Kelly percentage (0-1)
        """
        if win_rate <= 0 or win_rate >= 1 or avg_win_loss_ratio <= 0:
            return 0
        
        kelly = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
        
        # Limit Kelly to a reasonable range to avoid excessive leverage
        return max(0, min(kelly, self.max_position_size))
    
    def optimize_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        win_probability: float,
        current_drawdown: float,
        price_data: pd.DataFrame,
        recent_trades: List[Dict]
    ) -> Dict:
        """
        Optimize position size based on all risk management criteria.
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            take_profit_price: Take profit price for the trade
            win_probability: Probability of winning the trade (0-1)
            current_drawdown: Current drawdown as a percentage
            price_data: DataFrame with price data
            recent_trades: List of recent trades with timestamp
            
        Returns:
            Dictionary with optimal position size and risk metrics
        """
        # Calculate standard position size based on risk per trade
        standard_position_size = self.calculate_position_size(
            account_balance, entry_price, stop_loss_price
        )
        
        # Calculate Kelly position size
        risk = abs(entry_price - stop_loss_price) / entry_price
        reward = abs(entry_price - take_profit_price) / entry_price
        avg_win_loss_ratio = reward / risk
        kelly_position = self.calculate_kelly_criterion(win_probability, avg_win_loss_ratio)
        kelly_position_size = account_balance * kelly_position / entry_price
        
        # Use the more conservative position size
        position_size = min(standard_position_size, kelly_position_size)
        
        # Check if position would exceed max drawdown
        potential_loss = position_size * abs(entry_price - stop_loss_price)
        potential_loss_percentage = potential_loss / account_balance
        max_drawdown_ok = self.check_max_drawdown(current_drawdown, potential_loss_percentage)
        
        # Check VaR limit
        var_ok = self.check_var_limit(price_data, position_size)
        
        # Check max trades per day
        trades_ok = self.check_max_trades_per_day(recent_trades)
        
        # Check risk-reward ratio
        risk_reward_ok = self.check_risk_reward_filter(entry_price, stop_loss_price, take_profit_price)
        
        # Calculate expected value
        expected_value = self.calculate_expected_value(
            entry_price, stop_loss_price, take_profit_price, win_probability
        )
        
        # If any checks fail, return 0 position size
        if not all([max_drawdown_ok, var_ok, trades_ok, risk_reward_ok]):
            position_size = 0
        
        # Return comprehensive results
        return {
            "position_size": position_size,
            "position_value": position_size * entry_price,
            "percentage_of_balance": (position_size * entry_price) / account_balance,
            "expected_value": expected_value,
            "risk_reward_ratio": avg_win_loss_ratio,
            "potential_loss": potential_loss_percentage,
            "max_drawdown_ok": max_drawdown_ok,
            "var_ok": var_ok,
            "trades_ok": trades_ok,
            "risk_reward_ok": risk_reward_ok,
            "kelly_percentage": kelly_position,
            "standard_risk_percentage": self.risk_per_trade,
            "max_position_size": self.max_position_size,
            "timestamp": datetime.now().isoformat()
        } 