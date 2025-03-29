"""
Portfolio Optimizer - Optimize crypto portfolio allocation using various techniques
including Modern Portfolio Theory, risk-based allocation, and optimization algorithms.
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import time
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PortfolioOptimizer:
    """
    Optimize crypto portfolio allocation using various techniques including
    Modern Portfolio Theory, risk-based allocation, and optimization algorithms.
    """
    
    def __init__(self):
        """Initialize portfolio optimizer."""
        self.cache = {}
        self.cache_expiry = 1800  # 30 minutes
        
        # Default risk-free rate (adjustable)
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Default constraints
        self.min_allocation = 0.05  # Minimum 5% allocation to any asset
        self.max_allocation = 0.40  # Maximum 40% allocation to any asset
    
    def _prepare_returns(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare returns data from price dataframes.
        
        Args:
            price_data: Dictionary of symbol to price DataFrame
            
        Returns:
            DataFrame with daily returns for each symbol
        """
        # Extract returns
        returns_dict = {}
        for symbol, df in price_data.items():
            if 'close' in df.columns and len(df) > 0:
                returns = df['close'].pct_change().dropna()
                returns_dict[symbol] = returns
        
        # Create DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Fill NAs with 0s (for assets with different trading days)
        returns_df = returns_df.fillna(0)
        
        return returns_df
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio metrics for given weights and returns.
        
        Args:
            weights: Array of asset weights
            returns: DataFrame of asset returns
            
        Returns:
            Dictionary with portfolio metrics
        """
        # Calculate portfolio expected return
        expected_returns = returns.mean()
        portfolio_return = np.sum(expected_returns * weights) * 252  # Annualized
        
        # Calculate portfolio volatility
        cov_matrix = returns.cov() * 252  # Annualized
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Calculate Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Calculate downside risk (using only negative returns)
        negative_returns = returns.copy()
        negative_returns[negative_returns > 0] = 0
        downside_volatility = np.sqrt(np.dot(weights.T, np.dot(negative_returns.cov() * 252, weights)))
        
        # Calculate Sortino ratio
        sortino_ratio = (portfolio_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else float('inf')
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio
        }
    
    def _portfolio_negative_sharpe(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """
        Calculate negative Sharpe ratio for optimization.
        
        Args:
            weights: Array of asset weights
            returns: DataFrame of asset returns
            
        Returns:
            Negative Sharpe ratio
        """
        metrics = self._calculate_portfolio_metrics(weights, returns)
        return -metrics['sharpe_ratio']
    
    def _portfolio_negative_sortino(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """
        Calculate negative Sortino ratio for optimization.
        
        Args:
            weights: Array of asset weights
            returns: DataFrame of asset returns
            
        Returns:
            Negative Sortino ratio
        """
        metrics = self._calculate_portfolio_metrics(weights, returns)
        return -metrics['sortino_ratio']
    
    def _portfolio_variance(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """
        Calculate portfolio variance for optimization.
        
        Args:
            weights: Array of asset weights
            returns: DataFrame of asset returns
            
        Returns:
            Portfolio variance
        """
        cov_matrix = returns.cov() * 252  # Annualized
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    def optimize_sharpe_ratio(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Optimize portfolio allocation to maximize Sharpe ratio.
        
        Args:
            price_data: Dictionary of symbol to price DataFrame
            
        Returns:
            Dictionary with optimized weights and metrics
        """
        # Check cache
        cache_key = f"optimize_sharpe:{hash(frozenset(price_data.keys()))}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['data']
        
        try:
            # Prepare returns
            returns = self._prepare_returns(price_data)
            
            if returns.empty or returns.shape[1] == 0:
                return {
                    'weights': {},
                    'metrics': {
                        'expected_return': 0,
                        'volatility': 0,
                        'sharpe_ratio': 0,
                        'sortino_ratio': 0
                    },
                    'timestamp': datetime.now().isoformat()
                }
            
            # Initial weights (equal)
            num_assets = returns.shape[1]
            initial_weights = np.array([1.0 / num_assets] * num_assets)
            
            # Constraints
            bounds = [(self.min_allocation, self.max_allocation) for _ in range(num_assets)]
            
            # Sum of weights = 1
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Optimize
            result = minimize(
                self._portfolio_negative_sharpe,
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Get optimized weights
            optimized_weights = result['x']
            
            # Calculate metrics
            metrics = self._calculate_portfolio_metrics(optimized_weights, returns)
            
            # Create result
            weights_dict = {symbol: round(weight, 4) for symbol, weight in zip(returns.columns, optimized_weights)}
            result = {
                'weights': weights_dict,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update cache
            self.cache[cache_key] = {
                'timestamp': time.time(),
                'data': result
            }
            
            return result
            
        except Exception as e:
            print(f"Error optimizing Sharpe ratio: {e}")
            return {
                'weights': {symbol: 1.0 / len(price_data) for symbol in price_data.keys()},
                'metrics': {
                    'expected_return': 0,
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def optimize_minimum_variance(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Optimize portfolio allocation to minimize variance.
        
        Args:
            price_data: Dictionary of symbol to price DataFrame
            
        Returns:
            Dictionary with optimized weights and metrics
        """
        # Check cache
        cache_key = f"optimize_min_var:{hash(frozenset(price_data.keys()))}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['data']
        
        try:
            # Prepare returns
            returns = self._prepare_returns(price_data)
            
            if returns.empty or returns.shape[1] == 0:
                return {
                    'weights': {},
                    'metrics': {
                        'expected_return': 0,
                        'volatility': 0,
                        'sharpe_ratio': 0,
                        'sortino_ratio': 0
                    },
                    'timestamp': datetime.now().isoformat()
                }
            
            # Initial weights (equal)
            num_assets = returns.shape[1]
            initial_weights = np.array([1.0 / num_assets] * num_assets)
            
            # Constraints
            bounds = [(self.min_allocation, self.max_allocation) for _ in range(num_assets)]
            
            # Sum of weights = 1
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Optimize
            result = minimize(
                self._portfolio_variance,
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Get optimized weights
            optimized_weights = result['x']
            
            # Calculate metrics
            metrics = self._calculate_portfolio_metrics(optimized_weights, returns)
            
            # Create result
            weights_dict = {symbol: round(weight, 4) for symbol, weight in zip(returns.columns, optimized_weights)}
            result = {
                'weights': weights_dict,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update cache
            self.cache[cache_key] = {
                'timestamp': time.time(),
                'data': result
            }
            
            return result
            
        except Exception as e:
            print(f"Error optimizing minimum variance: {e}")
            return {
                'weights': {symbol: 1.0 / len(price_data) for symbol in price_data.keys()},
                'metrics': {
                    'expected_return': 0,
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def optimize_sortino_ratio(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Optimize portfolio allocation to maximize Sortino ratio.
        
        Args:
            price_data: Dictionary of symbol to price DataFrame
            
        Returns:
            Dictionary with optimized weights and metrics
        """
        # Check cache
        cache_key = f"optimize_sortino:{hash(frozenset(price_data.keys()))}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['data']
        
        try:
            # Prepare returns
            returns = self._prepare_returns(price_data)
            
            if returns.empty or returns.shape[1] == 0:
                return {
                    'weights': {},
                    'metrics': {
                        'expected_return': 0,
                        'volatility': 0,
                        'sharpe_ratio': 0,
                        'sortino_ratio': 0
                    },
                    'timestamp': datetime.now().isoformat()
                }
            
            # Initial weights (equal)
            num_assets = returns.shape[1]
            initial_weights = np.array([1.0 / num_assets] * num_assets)
            
            # Constraints
            bounds = [(self.min_allocation, self.max_allocation) for _ in range(num_assets)]
            
            # Sum of weights = 1
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Optimize
            result = minimize(
                self._portfolio_negative_sortino,
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Get optimized weights
            optimized_weights = result['x']
            
            # Calculate metrics
            metrics = self._calculate_portfolio_metrics(optimized_weights, returns)
            
            # Create result
            weights_dict = {symbol: round(weight, 4) for symbol, weight in zip(returns.columns, optimized_weights)}
            result = {
                'weights': weights_dict,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update cache
            self.cache[cache_key] = {
                'timestamp': time.time(),
                'data': result
            }
            
            return result
            
        except Exception as e:
            print(f"Error optimizing Sortino ratio: {e}")
            return {
                'weights': {symbol: 1.0 / len(price_data) for symbol in price_data.keys()},
                'metrics': {
                    'expected_return': 0,
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_efficient_frontier(self, price_data: Dict[str, pd.DataFrame], 
                                  points: int = 20) -> List[Dict[str, Any]]:
        """
        Generate points on the efficient frontier.
        
        Args:
            price_data: Dictionary of symbol to price DataFrame
            points: Number of points to generate
            
        Returns:
            List of dictionaries with weights and metrics for each point
        """
        # Check cache
        cache_key = f"efficient_frontier:{hash(frozenset(price_data.keys()))}:{points}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['data']
        
        try:
            # Prepare returns
            returns = self._prepare_returns(price_data)
            
            if returns.empty or returns.shape[1] == 0:
                return []
            
            # Get minimum variance and maximum return portfolios
            min_var_weights = self.optimize_minimum_variance(price_data)['weights']
            min_var_weights_array = np.array([min_var_weights.get(symbol, 0) for symbol in returns.columns])
            min_var_metrics = self._calculate_portfolio_metrics(min_var_weights_array, returns)
            
            # Find maximum return asset
            expected_returns = returns.mean() * 252  # Annualized
            max_return_symbol = expected_returns.idxmax()
            max_return_weights = {symbol: self.min_allocation for symbol in returns.columns}
            remaining_weight = 1 - self.min_allocation * (len(returns.columns) - 1)
            max_return_weights[max_return_symbol] = remaining_weight
            max_return_weights_array = np.array([max_return_weights.get(symbol, 0) for symbol in returns.columns])
            max_return_metrics = self._calculate_portfolio_metrics(max_return_weights_array, returns)
            
            # Define return range
            min_return = min_var_metrics['expected_return']
            max_return = max_return_metrics['expected_return']
            return_targets = np.linspace(min_return, max_return, points)
            
            frontier_points = []
            
            # For each return target, find the minimum variance portfolio
            for target_return in return_targets:
                # Initial weights (equal)
                num_assets = returns.shape[1]
                initial_weights = np.array([1.0 / num_assets] * num_assets)
                
                # Constraints
                bounds = [(self.min_allocation, self.max_allocation) for _ in range(num_assets)]
                
                # Sum of weights = 1
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.sum(expected_returns * x) * 252 - target_return}
                ]
                
                # Optimize
                try:
                    result = minimize(
                        self._portfolio_variance,
                        initial_weights,
                        args=(returns,),
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints
                    )
                    
                    # Get optimized weights
                    optimized_weights = result['x']
                    
                    # Calculate metrics
                    metrics = self._calculate_portfolio_metrics(optimized_weights, returns)
                    
                    # Create result
                    weights_dict = {symbol: round(weight, 4) for symbol, weight in zip(returns.columns, optimized_weights)}
                    frontier_points.append({
                        'weights': weights_dict,
                        'metrics': metrics
                    })
                except:
                    # Skip if optimization fails
                    continue
            
            # Update cache
            self.cache[cache_key] = {
                'timestamp': time.time(),
                'data': frontier_points
            }
            
            return frontier_points
            
        except Exception as e:
            print(f"Error generating efficient frontier: {e}")
            return []
    
    def risk_parity_allocation(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate risk parity allocation where each asset contributes equally to portfolio risk.
        
        Args:
            price_data: Dictionary of symbol to price DataFrame
            
        Returns:
            Dictionary with weights and metrics
        """
        # Check cache
        cache_key = f"risk_parity:{hash(frozenset(price_data.keys()))}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['data']
        
        try:
            # Prepare returns
            returns = self._prepare_returns(price_data)
            
            if returns.empty or returns.shape[1] == 0:
                return {
                    'weights': {},
                    'metrics': {
                        'expected_return': 0,
                        'volatility': 0,
                        'sharpe_ratio': 0,
                        'sortino_ratio': 0
                    },
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate covariance matrix
            cov_matrix = returns.cov() * 252  # Annualized
            
            # Calculate volatilities
            vols = np.sqrt(np.diag(cov_matrix))
            
            # Calculate risk parity weights (inverse of volatility)
            weights = 1 / vols
            weights = weights / np.sum(weights)  # Normalize
            
            # Apply min and max constraints
            num_assets = len(weights)
            
            def _adjust_weights(w, n):
                """Adjust weights to satisfy min and max constraints."""
                adjusted = np.copy(w)
                
                # Identify assets below min or above max
                below_min = adjusted < self.min_allocation
                above_max = adjusted > self.max_allocation
                
                # Set assets below min to min
                adjusted[below_min] = self.min_allocation
                
                # Set assets above max to max
                adjusted[above_max] = self.max_allocation
                
                # Normalize remaining weights
                normal_assets = ~(below_min | above_max)
                if np.any(normal_assets):
                    remaining_weight = 1 - np.sum(adjusted[below_min]) - np.sum(adjusted[above_max])
                    adjusted[normal_assets] = adjusted[normal_assets] / np.sum(adjusted[normal_assets]) * remaining_weight
                
                return adjusted
            
            # Apply adjustments
            weights = _adjust_weights(weights, num_assets)
            
            # Calculate metrics
            metrics = self._calculate_portfolio_metrics(weights, returns)
            
            # Create result
            weights_dict = {symbol: round(weight, 4) for symbol, weight in zip(returns.columns, weights)}
            result = {
                'weights': weights_dict,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update cache
            self.cache[cache_key] = {
                'timestamp': time.time(),
                'data': result
            }
            
            return result
            
        except Exception as e:
            print(f"Error calculating risk parity allocation: {e}")
            return {
                'weights': {symbol: 1.0 / len(price_data) for symbol in price_data.keys()},
                'metrics': {
                    'expected_return': 0,
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def correlations_analysis(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze correlations between assets.
        
        Args:
            price_data: Dictionary of symbol to price DataFrame
            
        Returns:
            Dictionary with correlation analysis
        """
        try:
            # Prepare returns
            returns = self._prepare_returns(price_data)
            
            if returns.empty or returns.shape[1] < 2:
                return {
                    'correlation_matrix': pd.DataFrame().to_dict(),
                    'avg_correlation': 0,
                    'most_correlated_pair': [],
                    'least_correlated_pair': [],
                    'diversification_score': 0
                }
            
            # Calculate correlation matrix
            corr_matrix = returns.corr()
            
            # Find average correlation
            corr_values = corr_matrix.values
            n = corr_matrix.shape[0]
            avg_corr = (corr_values.sum() - n) / (n * n - n)  # Excluding diagonal
            
            # Find most and least correlated pairs
            np.fill_diagonal(corr_values, np.nan)
            most_corr_idx = np.nanargmax(corr_values)
            least_corr_idx = np.nanargmin(corr_values)
            
            most_corr_i, most_corr_j = most_corr_idx // n, most_corr_idx % n
            least_corr_i, least_corr_j = least_corr_idx // n, least_corr_idx % n
            
            most_corr_pair = [corr_matrix.index[most_corr_i], corr_matrix.columns[most_corr_j]]
            least_corr_pair = [corr_matrix.index[least_corr_i], corr_matrix.columns[least_corr_j]]
            
            # Calculate diversification score (1 - average correlation)
            # Higher score means better diversification
            diversification_score = 1 - abs(avg_corr)
            
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'avg_correlation': avg_corr,
                'most_correlated_pair': most_corr_pair,
                'least_correlated_pair': least_corr_pair,
                'diversification_score': diversification_score
            }
            
        except Exception as e:
            print(f"Error analyzing correlations: {e}")
            return {
                'correlation_matrix': {},
                'avg_correlation': 0,
                'most_correlated_pair': [],
                'least_correlated_pair': [],
                'diversification_score': 0
            }
    
    def optimize_custom(self, price_data: Dict[str, pd.DataFrame], 
                      target_return: Optional[float] = None,
                      target_volatility: Optional[float] = None,
                      custom_constraints: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Optimize portfolio with custom constraints.
        
        Args:
            price_data: Dictionary of symbol to price DataFrame
            target_return: Target annual return (if None, maximize Sharpe ratio)
            target_volatility: Target annual volatility (if None, no constraint)
            custom_constraints: Dictionary with custom asset allocation constraints
            
        Returns:
            Dictionary with optimized weights and metrics
        """
        try:
            # Prepare returns
            returns = self._prepare_returns(price_data)
            
            if returns.empty or returns.shape[1] == 0:
                return {
                    'weights': {},
                    'metrics': {
                        'expected_return': 0,
                        'volatility': 0,
                        'sharpe_ratio': 0,
                        'sortino_ratio': 0
                    },
                    'timestamp': datetime.now().isoformat()
                }
            
            # Initial weights (equal)
            num_assets = returns.shape[1]
            initial_weights = np.array([1.0 / num_assets] * num_assets)
            
            # Prepare constraints
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Add target return constraint
            if target_return is not None:
                expected_returns = returns.mean() * 252  # Annualized
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x: np.sum(expected_returns * x) * 252 - target_return
                })
            
            # Add target volatility constraint
            if target_volatility is not None:
                cov_matrix = returns.cov() * 252  # Annualized
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))) - target_volatility
                })
            
            # Set bounds
            bounds = []
            for i, symbol in enumerate(returns.columns):
                if custom_constraints and symbol in custom_constraints:
                    min_val = max(self.min_allocation, custom_constraints[symbol].get('min', self.min_allocation))
                    max_val = min(self.max_allocation, custom_constraints[symbol].get('max', self.max_allocation))
                    bounds.append((min_val, max_val))
                else:
                    bounds.append((self.min_allocation, self.max_allocation))
            
            # Choose objective function
            objective_function = self._portfolio_negative_sharpe
            
            # Optimize
            result = minimize(
                objective_function,
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Get optimized weights
            optimized_weights = result['x']
            
            # Calculate metrics
            metrics = self._calculate_portfolio_metrics(optimized_weights, returns)
            
            # Create result
            weights_dict = {symbol: round(weight, 4) for symbol, weight in zip(returns.columns, optimized_weights)}
            return {
                'weights': weights_dict,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error optimizing custom portfolio: {e}")
            return {
                'weights': {symbol: 1.0 / len(price_data) for symbol in price_data.keys()},
                'metrics': {
                    'expected_return': 0,
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0
                },
                'timestamp': datetime.now().isoformat()
            } 