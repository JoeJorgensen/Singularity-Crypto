"""
Sentiment Predictor - Predict cryptocurrency market sentiment using machine learning.
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import time
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SentimentPredictor:
    """
    Predict cryptocurrency market sentiment using machine learning and various signals.
    This model combines sentiment data from social media, news, and market data
    to predict future market sentiment.
    """
    
    def __init__(self):
        """Initialize sentiment predictor."""
        self.cache = {}
        self.cache_expiry = 1800  # 30 minutes
        
        # Models for different timeframes
        self.models = {
            'short_term': None,  # 1-3 days
            'medium_term': None,  # 1-2 weeks
            'long_term': None    # 1 month+
        }
        
        # Sentiment thresholds
        self.extremely_bearish = -0.75
        self.bearish = -0.25
        self.neutral_low = -0.25
        self.neutral_high = 0.25
        self.bullish = 0.25
        self.extremely_bullish = 0.75
        
        # Prediction horizons in days
        self.horizons = {
            'short_term': 3,
            'medium_term': 10,
            'long_term': 30
        }
        
        # Scaler for features
        self.scalers = {
            'short_term': StandardScaler(),
            'medium_term': StandardScaler(),
            'long_term': StandardScaler()
        }
    
    def _prepare_features(self, market_data: pd.DataFrame,
                         social_sentiment: pd.DataFrame,
                         news_sentiment: pd.DataFrame,
                         on_chain_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare features for sentiment prediction.
        
        Args:
            market_data: DataFrame with market data (prices, volumes, etc.)
            social_sentiment: DataFrame with social media sentiment
            news_sentiment: DataFrame with news sentiment
            on_chain_data: Optional DataFrame with on-chain metrics
            
        Returns:
            DataFrame with prepared features
        """
        features = pd.DataFrame(index=market_data.index)
        
        # Market features
        if 'close' in market_data.columns:
            # Price changes
            features['price_change_1d'] = market_data['close'].pct_change(1)
            features['price_change_3d'] = market_data['close'].pct_change(3)
            features['price_change_7d'] = market_data['close'].pct_change(7)
            
            # Moving averages
            features['ma_7d'] = market_data['close'].rolling(7).mean()
            features['ma_14d'] = market_data['close'].rolling(14).mean()
            features['ma_30d'] = market_data['close'].rolling(30).mean()
            
            # Moving average crossovers
            features['ma_crossover_7_30'] = features['ma_7d'] - features['ma_30d']
            features['ma_crossover_7_14'] = features['ma_7d'] - features['ma_14d']
            
            # Volatility
            features['volatility_7d'] = market_data['close'].pct_change().rolling(7).std()
            features['volatility_14d'] = market_data['close'].pct_change().rolling(14).std()
        
        if 'volume' in market_data.columns:
            # Volume changes
            features['volume_change_1d'] = market_data['volume'].pct_change(1)
            features['volume_change_3d'] = market_data['volume'].pct_change(3)
            features['volume_change_7d'] = market_data['volume'].pct_change(7)
            
            # Relative volume
            features['relative_volume_7d'] = market_data['volume'] / market_data['volume'].rolling(7).mean()
            features['relative_volume_14d'] = market_data['volume'] / market_data['volume'].rolling(14).mean()
        
        # Social sentiment features
        if 'sentiment_score' in social_sentiment.columns:
            # Align dates
            social_sentiment = social_sentiment.reindex(features.index, method='ffill')
            
            features['social_sentiment'] = social_sentiment['sentiment_score']
            features['social_sentiment_ma_3d'] = social_sentiment['sentiment_score'].rolling(3).mean()
            features['social_sentiment_ma_7d'] = social_sentiment['sentiment_score'].rolling(7).mean()
            
            # Sentiment momentum
            features['social_sentiment_momentum'] = features['social_sentiment_ma_3d'] - features['social_sentiment_ma_7d']
        
        # News sentiment features
        if 'sentiment_score' in news_sentiment.columns:
            # Align dates
            news_sentiment = news_sentiment.reindex(features.index, method='ffill')
            
            features['news_sentiment'] = news_sentiment['sentiment_score']
            features['news_sentiment_ma_3d'] = news_sentiment['sentiment_score'].rolling(3).mean()
            features['news_sentiment_ma_7d'] = news_sentiment['sentiment_score'].rolling(7).mean()
            
            # Sentiment momentum
            features['news_sentiment_momentum'] = features['news_sentiment_ma_3d'] - features['news_sentiment_ma_7d']
        
        # On-chain features
        if on_chain_data is not None:
            # Align dates
            on_chain_data = on_chain_data.reindex(features.index, method='ffill')
            
            # Add on-chain features
            for col in on_chain_data.columns:
                features[f'onchain_{col}'] = on_chain_data[col]
        
        # Combined sentiment
        if 'social_sentiment' in features.columns and 'news_sentiment' in features.columns:
            features['combined_sentiment'] = (features['social_sentiment'] + features['news_sentiment']) / 2
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def _create_target(self, market_data: pd.DataFrame, horizon: int) -> pd.Series:
        """
        Create target variable for sentiment prediction.
        
        Args:
            market_data: DataFrame with market data
            horizon: Prediction horizon in days
            
        Returns:
            Series with target sentiment labels
        """
        # Calculate future returns
        future_returns = market_data['close'].pct_change(horizon).shift(-horizon)
        
        # Create sentiment labels based on future returns
        sentiment = pd.Series(index=future_returns.index, dtype='object')
        
        # Define thresholds for sentiment labels
        sentiment[future_returns <= self.extremely_bearish] = 'extremely_bearish'
        sentiment[(future_returns > self.extremely_bearish) & (future_returns <= self.bearish)] = 'bearish'
        sentiment[(future_returns > self.neutral_low) & (future_returns < self.neutral_high)] = 'neutral'
        sentiment[(future_returns >= self.bullish) & (future_returns < self.extremely_bullish)] = 'bullish'
        sentiment[future_returns >= self.extremely_bullish] = 'extremely_bullish'
        
        return sentiment
    
    def train(self, market_data: pd.DataFrame,
             social_sentiment: pd.DataFrame,
             news_sentiment: pd.DataFrame,
             on_chain_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train sentiment prediction models.
        
        Args:
            market_data: DataFrame with market data (prices, volumes, etc.)
            social_sentiment: DataFrame with social media sentiment
            news_sentiment: DataFrame with news sentiment
            on_chain_data: Optional DataFrame with on-chain metrics
            
        Returns:
            Dictionary with training results
        """
        # Prepare features
        features = self._prepare_features(market_data, social_sentiment, news_sentiment, on_chain_data)
        
        # Train models for different timeframes
        results = {}
        
        for timeframe, horizon in self.horizons.items():
            # Create target
            target = self._create_target(market_data, horizon)
            
            # Align target with features
            aligned_data = pd.concat([features, target], axis=1).dropna()
            
            if len(aligned_data) < 30:  # Not enough data
                results[timeframe] = {
                    'success': False,
                    'message': f"Not enough data for {timeframe} model"
                }
                continue
            
            X = aligned_data.iloc[:, :-1]  # All columns except the last one (target)
            y = aligned_data.iloc[:, -1]   # Last column (target)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.scalers[timeframe].fit(X_train)
            X_train_scaled = self.scalers[timeframe].transform(X_train)
            X_test_scaled = self.scalers[timeframe].transform(X_test)
            
            # Try both classifiers
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
            rf_accuracy = rf_model.score(X_test_scaled, y_test)
            
            lr_model = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
            lr_model.fit(X_train_scaled, y_train)
            lr_accuracy = lr_model.score(X_test_scaled, y_test)
            
            # Choose the best model
            if rf_accuracy >= lr_accuracy:
                self.models[timeframe] = rf_model
                accuracy = rf_accuracy
                model_type = 'RandomForest'
            else:
                self.models[timeframe] = lr_model
                accuracy = lr_accuracy
                model_type = 'LogisticRegression'
            
            # Store results
            results[timeframe] = {
                'success': True,
                'accuracy': accuracy,
                'model_type': model_type,
                'feature_importance': self._get_feature_importance(timeframe)
            }
        
        return results
    
    def _get_feature_importance(self, timeframe: str) -> Dict[str, float]:
        """
        Get feature importance for a trained model.
        
        Args:
            timeframe: Timeframe of the model
            
        Returns:
            Dictionary with feature importance
        """
        model = self.models[timeframe]
        if model is None:
            return {}
        
        # Get feature names
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        else:
            return {}
        
        # Get importance
        if hasattr(model, 'feature_importances_'):  # RandomForest
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):  # LogisticRegression
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            return {}
        
        # Create dictionary
        importance_dict = dict(zip(feature_names, importance))
        
        # Sort by importance
        return {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
    
    def predict(self, market_data: pd.DataFrame,
               social_sentiment: pd.DataFrame,
               news_sentiment: pd.DataFrame,
               on_chain_data: Optional[pd.DataFrame] = None,
               timeframe: str = 'short_term') -> Dict[str, Any]:
        """
        Predict sentiment for a specific timeframe.
        
        Args:
            market_data: DataFrame with market data
            social_sentiment: DataFrame with social media sentiment
            news_sentiment: DataFrame with news sentiment
            on_chain_data: Optional DataFrame with on-chain metrics
            timeframe: Timeframe for prediction ('short_term', 'medium_term', or 'long_term')
            
        Returns:
            Dictionary with prediction results
        """
        # Check if model is trained
        if self.models[timeframe] is None:
            return {
                'success': False,
                'message': f"Model for {timeframe} is not trained"
            }
        
        # Check cache
        cache_key = f"sentiment_prediction:{timeframe}:{hash(str(market_data.iloc[-1:].to_dict()))}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['data']
        
        try:
            # Prepare features
            features = self._prepare_features(market_data, social_sentiment, news_sentiment, on_chain_data)
            
            if len(features) == 0:
                return {
                    'success': False,
                    'message': "No features could be prepared from the provided data"
                }
            
            # Get latest data point
            latest_features = features.iloc[-1:].values
            
            # Scale features
            latest_scaled = self.scalers[timeframe].transform(latest_features)
            
            # Make prediction
            prediction = self.models[timeframe].predict(latest_scaled)[0]
            
            # Get prediction probabilities
            probabilities = self.models[timeframe].predict_proba(latest_scaled)[0]
            probability_dict = dict(zip(self.models[timeframe].classes_, probabilities))
            
            # Calculate sentiment score (-1 to 1)
            sentiment_scores = {
                'extremely_bearish': -1.0,
                'bearish': -0.5,
                'neutral': 0,
                'bullish': 0.5,
                'extremely_bullish': 1.0
            }
            
            sentiment_score = sum(probability_dict.get(label, 0) * score 
                                  for label, score in sentiment_scores.items())
            
            # Prepare result
            result = {
                'success': True,
                'prediction': prediction,
                'probabilities': probability_dict,
                'sentiment_score': sentiment_score,
                'timeframe': timeframe,
                'horizon_days': self.horizons[timeframe],
                'timestamp': datetime.now().isoformat()
            }
            
            # Update cache
            self.cache[cache_key] = {
                'timestamp': time.time(),
                'data': result
            }
            
            return result
            
        except Exception as e:
            print(f"Error predicting sentiment: {e}")
            return {
                'success': False,
                'message': f"Error predicting sentiment: {str(e)}"
            }
    
    def get_sentiment_trend(self, market_data: pd.DataFrame,
                           social_sentiment: pd.DataFrame,
                           news_sentiment: pd.DataFrame,
                           on_chain_data: Optional[pd.DataFrame] = None,
                           days: int = 30) -> Dict[str, Any]:
        """
        Get sentiment trend for the past N days.
        
        Args:
            market_data: DataFrame with market data
            social_sentiment: DataFrame with social media sentiment
            news_sentiment: DataFrame with news sentiment
            on_chain_data: Optional DataFrame with on-chain metrics
            days: Number of days to analyze
            
        Returns:
            Dictionary with sentiment trend analysis
        """
        try:
            # Prepare features
            features = self._prepare_features(market_data, social_sentiment, news_sentiment, on_chain_data)
            
            if len(features) == 0:
                return {
                    'success': False,
                    'message': "No features could be prepared from the provided data"
                }
            
            # Get data for past N days
            recent_features = features.iloc[-days:]
            
            # Analyze for each timeframe
            results = {}
            
            for timeframe in self.models:
                if self.models[timeframe] is None:
                    continue
                
                # Scale features
                scaled_features = self.scalers[timeframe].transform(recent_features.values)
                
                # Make predictions
                predictions = self.models[timeframe].predict(scaled_features)
                
                # Get prediction probabilities
                probabilities = self.models[timeframe].predict_proba(scaled_features)
                
                # Calculate sentiment scores
                sentiment_scores = {
                    'extremely_bearish': -1.0,
                    'bearish': -0.5,
                    'neutral': 0,
                    'bullish': 0.5,
                    'extremely_bullish': 1.0
                }
                
                sentiment_trend = []
                
                for i, date in enumerate(recent_features.index):
                    probs = probabilities[i]
                    prob_dict = dict(zip(self.models[timeframe].classes_, probs))
                    
                    score = sum(prob_dict.get(label, 0) * score 
                                for label, score in sentiment_scores.items())
                    
                    sentiment_trend.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'prediction': predictions[i],
                        'sentiment_score': score
                    })
                
                # Calculate trend metrics
                scores = [item['sentiment_score'] for item in sentiment_trend]
                
                trend_metrics = {
                    'start_score': scores[0] if scores else 0,
                    'end_score': scores[-1] if scores else 0,
                    'avg_score': sum(scores) / len(scores) if scores else 0,
                    'min_score': min(scores) if scores else 0,
                    'max_score': max(scores) if scores else 0,
                    'trend_direction': 'improving' if scores[-1] > scores[0] else 'deteriorating' if scores[-1] < scores[0] else 'stable',
                    'volatility': np.std(scores) if len(scores) > 1 else 0
                }
                
                results[timeframe] = {
                    'sentiment_trend': sentiment_trend,
                    'trend_metrics': trend_metrics
                }
            
            return {
                'success': True,
                'timeframes': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment trend: {e}")
            return {
                'success': False,
                'message': f"Error analyzing sentiment trend: {str(e)}"
            }
    
    def simulate_sentiment_data(self, days: int = 60) -> Dict[str, pd.DataFrame]:
        """
        Generate simulated sentiment data for testing.
        
        Args:
            days: Number of days to simulate
            
        Returns:
            Dictionary with simulated DataFrames
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create simulated market data
        base_price = 10000
        daily_returns = np.random.normal(0.001, 0.025, len(date_range))
        prices = base_price * np.cumprod(1 + daily_returns)
        
        volumes = np.random.lognormal(mean=np.log(1e9), sigma=0.3, size=len(date_range))
        
        market_data = pd.DataFrame({
            'close': prices,
            'volume': volumes
        }, index=date_range)
        
        # Create simulated social sentiment
        social_sentiment = pd.DataFrame({
            'sentiment_score': np.random.normal(0.1, 0.3, len(date_range)),
            'mentions': np.random.lognormal(mean=np.log(1000), sigma=0.5, size=len(date_range))
        }, index=date_range)
        
        # Create simulated news sentiment
        news_sentiment = pd.DataFrame({
            'sentiment_score': np.random.normal(0, 0.2, len(date_range)),
            'article_count': np.random.poisson(20, len(date_range))
        }, index=date_range)
        
        # Create simulated on-chain data
        on_chain_data = pd.DataFrame({
            'active_addresses': np.random.lognormal(mean=np.log(100000), sigma=0.2, size=len(date_range)),
            'transaction_count': np.random.lognormal(mean=np.log(500000), sigma=0.3, size=len(date_range)),
            'avg_transaction_value': np.random.lognormal(mean=np.log(500), sigma=0.4, size=len(date_range))
        }, index=date_range)
        
        return {
            'market_data': market_data,
            'social_sentiment': social_sentiment,
            'news_sentiment': news_sentiment,
            'on_chain_data': on_chain_data
        } 