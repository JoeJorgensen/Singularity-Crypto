"""
PricePredictor - Machine learning model for cryptocurrency price prediction.
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import time
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PricePredictor:
    """
    Machine learning model for cryptocurrency price prediction.
    This implementation uses a simplified prediction logic with simulation.
    In a production environment, this would use more sophisticated ML models.
    """
    
    def __init__(self):
        """Initialize price predictor."""
        self.models = {}
        self.scalers = {}
        self.cache = {}
        self.cache_expiry = 1800  # 30 minutes
        self.lookback_period = 20  # Number of days to look back for training
        self.prediction_horizon = 7  # Number of days to predict forward
        
        # Initialize random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Prepare data for prediction by scaling and creating features.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Tuple of (X, y, scaler)
        """
        # Create a copy to avoid modifying the original dataframe
        price_data = df.copy()
        
        # Make sure the dataframe has expected columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in price_data.columns:
                raise ValueError(f"Required column '{col}' not found in input data")
        
        # Feature engineering
        price_data['prev_close'] = price_data['close'].shift(1)
        price_data['price_change'] = price_data['close'] - price_data['prev_close']
        price_data['pct_change'] = price_data['price_change'] / price_data['prev_close']
        price_data['volatility'] = price_data['high'] - price_data['low']
        price_data['range_ratio'] = price_data['volatility'] / price_data['prev_close']
        
        # Add lag features
        for lag in [1, 2, 3, 5, 8, 13]:
            price_data[f'close_lag_{lag}'] = price_data['close'].shift(lag)
            price_data[f'volume_lag_{lag}'] = price_data['volume'].shift(lag)
        
        # Add moving averages
        for window in [5, 10, 20]:
            price_data[f'close_ma_{window}'] = price_data['close'].rolling(window=window).mean()
            price_data[f'volume_ma_{window}'] = price_data['volume'].rolling(window=window).mean()
        
        # Drop NaN values
        price_data.dropna(inplace=True)
        
        # Separate features and target
        features = price_data.drop(['open', 'high', 'low', 'close'], axis=1)
        target = price_data['close']
        
        # Scale features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        
        return features_scaled, target.values, scaler
    
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train prediction models.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Dictionary with trained models
        """
        # Determine training and validation split
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        rf_val_pred = rf_model.predict(X_val)
        rf_val_mse = mean_squared_error(y_val, rf_val_pred)
        
        # Train Linear Regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_val_pred = lr_model.predict(X_val)
        lr_val_mse = mean_squared_error(y_val, lr_val_pred)
        
        # Select best model
        if rf_val_mse <= lr_val_mse:
            best_model = rf_model
            best_model_name = 'random_forest'
            val_mse = rf_val_mse
        else:
            best_model = lr_model
            best_model_name = 'linear_regression'
            val_mse = lr_val_mse
        
        return {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'val_mse': val_mse,
            'random_forest': rf_model,
            'linear_regression': lr_model
        }
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict future price movement for a cryptocurrency.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with prediction results
        """
        # Create a cache key based on the last row of data
        if len(df) == 0:
            return {
                'direction': 'neutral',
                'confidence': 0,
                'price_target': None,
                'horizon': '24h',
                'timestamp': datetime.now().isoformat()
            }
        
        last_row = df.iloc[-1]
        cache_key = f"price_prediction:{last_row['close']}:{last_row.name}"
        
        # Check cache
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['data']
        
        # For this implementation, we'll use a simplified approach with some randomness
        # In a production environment, you would use actual ML predictions here
        current_price = last_row['close']
        
        try:
            # Extract recent price trend
            recent_prices = df['close'].tail(10).values
            short_term_trend = (recent_prices[-1] / recent_prices[0]) - 1  # Percentage change
            
            # Add some randomness while keeping alignment with the recent trend
            random_factor = random.uniform(-0.5, 0.5)
            trend_weight = 0.7
            random_weight = 0.3
            predicted_direction_value = trend_weight * short_term_trend + random_weight * random_factor
            
            # Determine direction and confidence
            confidence = min(abs(predicted_direction_value) * 2, 0.95)  # Scale to reasonable confidence
            
            if predicted_direction_value > 0.01:
                direction = 'bullish'
                price_change = confidence * random.uniform(0.01, 0.05)  # 1% to 5% increase
            elif predicted_direction_value < -0.01:
                direction = 'bearish'
                price_change = -confidence * random.uniform(0.01, 0.05)  # 1% to 5% decrease
            else:
                direction = 'neutral'
                price_change = random.uniform(-0.01, 0.01)  # -1% to 1% change
                confidence = random.uniform(0.2, 0.4)  # Lower confidence for neutral
            
            # Calculate predicted price
            price_target = current_price * (1 + price_change)
            
            # Prepare result
            result = {
                'direction': direction,
                'confidence': confidence,
                'price_target': price_target,
                'horizon': '24h',
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            print(f"Error making price prediction: {e}")
            result = {
                'direction': 'neutral',
                'confidence': 0,
                'price_target': None,
                'horizon': '24h',
                'timestamp': datetime.now().isoformat()
            }
        
        # Update cache
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'data': result
        }
        
        return result
    
    def predict_price_range(self, df: pd.DataFrame, days: int = 7) -> Dict[str, Any]:
        """
        Predict price range for the next n days.
        
        Args:
            df: DataFrame with price data
            days: Number of days to predict
            
        Returns:
            Dictionary with price range prediction
        """
        if len(df) < 30:
            return {
                'status': 'error',
                'message': 'Insufficient data for prediction',
                'timestamp': datetime.now().isoformat()
            }
        
        # Extract recent price data
        current_price = df['close'].iloc[-1]
        
        try:
            # Calculate historical volatility
            daily_returns = df['close'].pct_change().dropna()
            volatility = daily_returns.std()
            
            # Project forward with increasing uncertainty
            predictions = []
            lower_bounds = []
            upper_bounds = []
            
            # Add current price as baseline
            date = datetime.now()
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': current_price,
                'lower_bound': current_price,
                'upper_bound': current_price
            })
            
            # Calculate drift based on recent trend
            recent_trend = df['close'].pct_change().tail(30).mean()
            drift = recent_trend * 0.7  # Dampen the drift for more conservative estimates
            
            # Generate future predictions
            price = current_price
            for i in range(1, days + 1):
                date = date + timedelta(days=1)
                
                # Add increasing uncertainty over time
                uncertainty_multiplier = (1 + (i / 10))
                day_volatility = volatility * uncertainty_multiplier
                
                # Calculate price with drift and random component
                random_component = np.random.normal(0, day_volatility)
                price = price * (1 + drift + random_component)
                
                # Calculate confidence interval (roughly 95% confidence)
                confidence_interval = price * day_volatility * 1.96
                lower_bound = price - confidence_interval
                upper_bound = price + confidence_interval
                
                predictions.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'price': price,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })
            
            return {
                'status': 'success',
                'current_price': current_price,
                'predicted_prices': predictions,
                'volatility': volatility,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            print(f"Error in price range prediction: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def evaluate_accuracy(self, historic_predictions: List[Dict[str, Any]], 
                         actual_prices: pd.Series) -> Dict[str, Any]:
        """
        Evaluate prediction accuracy based on historical predictions.
        
        Args:
            historic_predictions: List of historical predictions
            actual_prices: Series of actual prices
            
        Returns:
            Dictionary with accuracy metrics
        """
        if not historic_predictions or len(actual_prices) == 0:
            return {
                'accuracy': 0,
                'direction_accuracy': 0,
                'mae': 0,
                'rmse': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            correct_directions = 0
            errors = []
            
            for pred in historic_predictions:
                # Skip if we don't have actual data for this prediction
                pred_date = datetime.fromisoformat(pred['timestamp'])
                pred_date_str = pred_date.strftime('%Y-%m-%d')
                
                if pred_date_str not in actual_prices.index:
                    continue
                
                # Get actual price
                actual_price = actual_prices[pred_date_str]
                
                # Check direction prediction
                if (pred['direction'] == 'bullish' and actual_price > pred['price_target']) or \
                   (pred['direction'] == 'bearish' and actual_price < pred['price_target']) or \
                   (pred['direction'] == 'neutral' and abs(actual_price - pred['price_target']) / pred['price_target'] < 0.01):
                    correct_directions += 1
                
                # Calculate error
                error = abs(actual_price - pred['price_target'])
                errors.append(error)
            
            # Calculate metrics
            if len(errors) > 0:
                direction_accuracy = correct_directions / len(errors)
                mae = sum(errors) / len(errors)
                rmse = (sum([e ** 2 for e in errors]) / len(errors)) ** 0.5
                
                return {
                    'accuracy': 1 - (mae / actual_prices.mean()),
                    'direction_accuracy': direction_accuracy,
                    'mae': mae,
                    'rmse': rmse,
                    'prediction_count': len(errors),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'accuracy': 0,
                    'direction_accuracy': 0,
                    'mae': 0,
                    'rmse': 0,
                    'prediction_count': 0,
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            print(f"Error evaluating prediction accuracy: {e}")
            return {
                'accuracy': 0,
                'direction_accuracy': 0,
                'mae': 0,
                'rmse': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            } 