"""
RiskAnalyzer - Analyze trading risk and calculate risk metrics.
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import time
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import math
from scipy import stats
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RiskAnalyzer:
    """
    Analyze trading risk and calculate various risk metrics.
    This implementation includes Value at Risk (VaR), Expected Shortfall (ES),
    maximum drawdown, volatility analysis, and other risk management tools.
    """
    
    def __init__(self):
        """Initialize risk analyzer."""
        self.cache = {}
        self.cache_expiry = 1800  # 30 minutes
        
        # Default values for VaR calculation
        self.default_confidence_level = 0.95  # 95% confidence level
        self.default_time_horizon = 1  # 1 day
        
        # Maximum acceptable values (configurable via risk management settings)
        self.max_acceptable_var = 0.05  # 5% maximum acceptable VaR
        self.max_acceptable_drawdown = 0.15  # 15% maximum acceptable drawdown
        self.max_acceptable_volatility = 0.03  # 3% daily volatility
    
    def calculate_var(self, returns: Optional[pd.Series] = None, 
                     confidence_level: float = 0.95,
                     time_horizon: int = 1,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of historical returns
            confidence_level: Confidence level for VaR calculation (default: 0.95)
            time_horizon: Time horizon in days (default: 1)
            method: VaR calculation method ('historical', 'parametric', or 'monte_carlo')
            
        Returns:
            Value at Risk (VaR) as a decimal (e.g., 0.05 for 5% VaR)
        """
        # If no returns provided, use simulated returns
        if returns is None:
            # Generate random returns with negative skew and excess kurtosis
            mean_return = 0.001  # 0.1% daily mean return
            std_dev = 0.02  # 2% daily standard deviation
            skew = -0.5  # Negative skew (more extreme negative returns)
            kurtosis = 3  # Excess kurtosis (fatter tails than normal distribution)
            
            # Generate random returns using a skewed t-distribution
            returns = pd.Series(
                stats.skewnorm.rvs(a=skew, loc=mean_return, scale=std_dev, size=252)
            )
        
        # Check if returns is empty
        if len(returns) == 0:
            return self.max_acceptable_var
        
        # Calculate VaR based on specified method
        if method == 'historical':
            # Historical simulation method
            var = -np.percentile(returns, 100 * (1 - confidence_level))
        elif method == 'parametric':
            # Parametric method (assuming normal distribution)
            mean = returns.mean()
            std_dev = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean + z_score * std_dev)
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            mean = returns.mean()
            std_dev = returns.std()
            
            # Generate 10,000 random returns
            simulated_returns = np.random.normal(mean, std_dev, 10000)
            var = -np.percentile(simulated_returns, 100 * (1 - confidence_level))
        else:
            raise ValueError(f"Unknown VaR calculation method: {method}")
        
        # Scale VaR for the specified time horizon
        var = var * math.sqrt(time_horizon)
        
        return max(0, var)
    
    def calculate_expected_shortfall(self, returns: Optional[pd.Series] = None,
                                    confidence_level: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (ES), also known as Conditional Value at Risk (CVaR).
        
        Args:
            returns: Series of historical returns
            confidence_level: Confidence level for ES calculation (default: 0.95)
            
        Returns:
            Expected Shortfall (ES) as a decimal
        """
        # If no returns provided, use simulated returns
        if returns is None:
            # Generate random returns with negative skew and excess kurtosis
            mean_return = 0.001  # 0.1% daily mean return
            std_dev = 0.02  # 2% daily standard deviation
            skew = -0.5  # Negative skew (more extreme negative returns)
            kurtosis = 3  # Excess kurtosis (fatter tails than normal distribution)
            
            # Generate random returns using a skewed t-distribution
            returns = pd.Series(
                stats.skewnorm.rvs(a=skew, loc=mean_return, scale=std_dev, size=252)
            )
        
        # Check if returns is empty
        if len(returns) == 0:
            return self.max_acceptable_var * 1.5  # ES is typically higher than VaR
        
        # Calculate VaR
        var = self.calculate_var(returns, confidence_level)
        
        # Calculate ES as the average of returns beyond VaR
        tail_returns = returns[returns < -var]
        
        if len(tail_returns) == 0:
            return var  # Fallback if no tail returns
        
        es = -tail_returns.mean()
        
        return max(0, es)
    
    def calculate_max_drawdown(self, prices: Optional[pd.Series] = None) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            prices: Series of historical prices
            
        Returns:
            Maximum drawdown as a decimal
        """
        # If no prices provided, generate simulated prices
        if prices is None:
            # Generate random price series with some trend and volatility
            base_price = 100
            n_days = 252
            
            # Generate returns with a slight positive drift
            returns = np.random.normal(0.0005, 0.015, n_days)
            
            # Convert returns to prices
            prices = pd.Series(base_price * np.cumprod(1 + returns))
        
        # Check if prices is empty
        if len(prices) == 0:
            return self.max_acceptable_drawdown
        
        # Calculate running maximum
        running_max = prices.cummax()
        
        # Calculate drawdowns
        drawdowns = (prices - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = abs(drawdowns.min())
        
        return max_drawdown
    
    def calculate_sharpe_ratio(self, returns: Optional[pd.Series] = None,
                              risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of historical returns
            risk_free_rate: Annual risk-free rate (default: 0.02 for 2%)
            
        Returns:
            Sharpe ratio
        """
        # If no returns provided, use simulated returns
        if returns is None:
            # Generate random returns with a slight positive drift
            returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        
        # Check if returns is empty
        if len(returns) == 0:
            return 0
        
        # Convert annual risk-free rate to match the returns frequency (assuming daily returns)
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate excess returns
        excess_returns = returns - daily_risk_free
        
        # Calculate annualized Sharpe ratio
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns: Optional[pd.Series] = None,
                               risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Series of historical returns
            risk_free_rate: Annual risk-free rate (default: 0.02 for 2%)
            
        Returns:
            Sortino ratio
        """
        # If no returns provided, use simulated returns
        if returns is None:
            # Generate random returns with a slight positive drift
            returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        
        # Check if returns is empty
        if len(returns) == 0:
            return 0
        
        # Convert annual risk-free rate to match the returns frequency (assuming daily returns)
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate excess returns
        excess_returns = returns - daily_risk_free
        
        # Calculate downside deviation (only negative excess returns)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
        
        # Calculate annualized Sortino ratio
        sortino_ratio = excess_returns.mean() * 252 / downside_deviation
        
        return sortino_ratio
    
    def analyze_risk(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trading risk for a cryptocurrency.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with risk analysis
        """
        # Check if the dataframe has expected columns
        required_columns = ['close']
        if not all(col in df.columns for col in required_columns):
            return {
                'var': self.max_acceptable_var,
                'expected_shortfall': self.max_acceptable_var * 1.5,
                'max_drawdown': self.max_acceptable_drawdown,
                'volatility': self.max_acceptable_volatility,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'risk_level': 'unknown',
                'timestamp': datetime.now().isoformat()
            }
        
        # Create a cache key based on the last row of data
        if len(df) == 0:
            return {
                'var': self.max_acceptable_var,
                'expected_shortfall': self.max_acceptable_var * 1.5,
                'max_drawdown': self.max_acceptable_drawdown,
                'volatility': self.max_acceptable_volatility,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'risk_level': 'unknown',
                'timestamp': datetime.now().isoformat()
            }
        
        last_row = df.iloc[-1]
        cache_key = f"risk_analysis:{last_row['close']}:{last_row.name}"
        
        # Check cache
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['data']
        
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        
        try:
            # Calculate risk metrics
            var = self.calculate_var(returns)
            es = self.calculate_expected_shortfall(returns)
            max_drawdown = self.calculate_max_drawdown(df['close'])
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            sortino_ratio = self.calculate_sortino_ratio(returns)
            
            # Determine risk level
            risk_score = (
                var / self.max_acceptable_var * 0.3 +
                max_drawdown / self.max_acceptable_drawdown * 0.3 +
                volatility / self.max_acceptable_volatility * 0.4
            )
            
            risk_level = 'medium'
            if risk_score < 0.5:
                risk_level = 'low'
            elif risk_score > 1.0:
                risk_level = 'high'
            
            # Prepare result
            result = {
                'var': var,
                'expected_shortfall': es,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            print(f"Error analyzing risk: {e}")
            result = {
                'var': self.max_acceptable_var,
                'expected_shortfall': self.max_acceptable_var * 1.5,
                'max_drawdown': self.max_acceptable_drawdown,
                'volatility': self.max_acceptable_volatility,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'risk_level': 'unknown',
                'timestamp': datetime.now().isoformat()
            }
        
        # Update cache
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'data': result
        }
        
        return result
    
    def calculate_position_risk(self, position_size: float, entry_price: float,
                              stop_loss_price: float, account_balance: float) -> Dict[str, Any]:
        """
        Calculate risk for a specific position.
        
        Args:
            position_size: Size of the position in units
            entry_price: Entry price per unit
            stop_loss_price: Stop loss price per unit
            account_balance: Total account balance
            
        Returns:
            Dictionary with position risk metrics
        """
        try:
            # Calculate position value
            position_value = position_size * entry_price
            
            # Calculate potential loss
            potential_loss = abs(position_size * (entry_price - stop_loss_price))
            
            # Calculate risk percentage (potential loss as percentage of account balance)
            risk_percentage = potential_loss / account_balance
            
            # Calculate risk/reward ratio (assuming 2:1 reward/risk)
            reward_to_risk = 2.0
            take_profit_price = entry_price + (entry_price - stop_loss_price) * reward_to_risk
            
            # Determine if risk is acceptable
            is_acceptable = risk_percentage <= 0.02  # 2% max risk per trade
            
            return {
                'position_value': position_value,
                'potential_loss': potential_loss,
                'risk_percentage': risk_percentage,
                'is_acceptable': is_acceptable,
                'take_profit_price': take_profit_price,
                'reward_to_risk': reward_to_risk
            }
        
        except Exception as e:
            print(f"Error calculating position risk: {e}")
            return {
                'position_value': 0,
                'potential_loss': 0,
                'risk_percentage': 0,
                'is_acceptable': False,
                'take_profit_price': 0,
                'reward_to_risk': 0
            }
    
    def optimize_position_size(self, account_balance: float, entry_price: float,
                             stop_loss_price: float, max_risk_percentage: float = 0.02) -> float:
        """
        Optimize position size based on risk parameters.
        
        Args:
            account_balance: Total account balance
            entry_price: Entry price per unit
            stop_loss_price: Stop loss price per unit
            max_risk_percentage: Maximum risk percentage per trade (default: 0.02 for 2%)
            
        Returns:
            Optimal position size in units
        """
        try:
            # Calculate price difference
            price_diff = abs(entry_price - stop_loss_price)
            
            if price_diff == 0:
                return 0  # Cannot calculate position size
            
            # Calculate maximum acceptable loss
            max_loss = account_balance * max_risk_percentage
            
            # Calculate position size
            position_size = max_loss / price_diff
            
            return position_size
        
        except Exception as e:
            print(f"Error optimizing position size: {e}")
            return 0
    
    def calculate_portfolio_var(self, positions: List[Dict[str, Any]],
                              correlation_matrix: Optional[pd.DataFrame] = None,
                              confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) for a portfolio of positions.
        
        Args:
            positions: List of position dictionaries with 'value' and 'volatility' keys
            correlation_matrix: Correlation matrix for position returns
            confidence_level: Confidence level for VaR calculation (default: 0.95)
            
        Returns:
            Portfolio VaR as a decimal
        """
        try:
            # Extract position values and volatilities
            values = np.array([p['value'] for p in positions])
            volatilities = np.array([p['volatility'] for p in positions])
            
            # Calculate weights
            total_value = sum(values)
            weights = values / total_value if total_value > 0 else np.zeros_like(values)
            
            # If correlation matrix is not provided, assume all correlations are 0.5
            if correlation_matrix is None:
                n = len(positions)
                correlation_matrix = np.ones((n, n)) * 0.5
                np.fill_diagonal(correlation_matrix, 1)
            
            # Convert to covariance matrix
            covariance_matrix = np.zeros_like(correlation_matrix)
            for i in range(len(correlation_matrix)):
                for j in range(len(correlation_matrix)):
                    covariance_matrix[i, j] = correlation_matrix[i, j] * volatilities[i] * volatilities[j]
            
            # Calculate portfolio variance
            portfolio_variance = weights.dot(covariance_matrix).dot(weights)
            
            # Calculate portfolio volatility
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate z-score for the confidence level
            z_score = stats.norm.ppf(confidence_level)
            
            # Calculate portfolio VaR
            portfolio_var = total_value * portfolio_volatility * z_score
            
            return portfolio_var
        
        except Exception as e:
            print(f"Error calculating portfolio VaR: {e}")
            return 0 