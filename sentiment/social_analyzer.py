"""
SocialAnalyzer - Analyze sentiment from social media for cryptocurrencies.
"""
from typing import Dict, List, Any, Optional
import os
import time
import json
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
from api.openai_api import OpenAIAPI

# Load environment variables
load_dotenv()

class SocialAnalyzer:
    """
    Analyze sentiment from social media for cryptocurrencies.
    This is a simulated implementation, as real social media APIs would require additional setup.
    """
    
    def __init__(self):
        """Initialize social analyzer."""
        self.openai = OpenAIAPI()
        self.cache = {}
        self.cache_expiry = 1800  # 30 minutes
        self.simulated_data = self._initialize_simulated_data()
    
    def _initialize_simulated_data(self) -> Dict[str, Any]:
        """
        Initialize simulated social media data for common cryptocurrencies.
        
        Returns:
            Dictionary with simulated data
        """
        return {
            'BTC': {
                'sentiment_history': [
                    {'timestamp': (datetime.now() - timedelta(hours=24)).isoformat(), 'score': 0.3},
                    {'timestamp': (datetime.now() - timedelta(hours=18)).isoformat(), 'score': 0.4},
                    {'timestamp': (datetime.now() - timedelta(hours=12)).isoformat(), 'score': 0.2},
                    {'timestamp': (datetime.now() - timedelta(hours=6)).isoformat(), 'score': 0.1},
                    {'timestamp': datetime.now().isoformat(), 'score': 0.25}
                ],
                'mentions': 3500,
                'trending_topics': ['Bitcoin halving', 'BTC ETF', 'Institutional adoption']
            },
            'ETH': {
                'sentiment_history': [
                    {'timestamp': (datetime.now() - timedelta(hours=24)).isoformat(), 'score': 0.1},
                    {'timestamp': (datetime.now() - timedelta(hours=18)).isoformat(), 'score': 0.2},
                    {'timestamp': (datetime.now() - timedelta(hours=12)).isoformat(), 'score': 0.4},
                    {'timestamp': (datetime.now() - timedelta(hours=6)).isoformat(), 'score': 0.3},
                    {'timestamp': datetime.now().isoformat(), 'score': 0.35}
                ],
                'mentions': 2200,
                'trending_topics': ['Ethereum upgrades', 'ETH staking', 'L2 scaling']
            },
            'XRP': {
                'sentiment_history': [
                    {'timestamp': (datetime.now() - timedelta(hours=24)).isoformat(), 'score': -0.2},
                    {'timestamp': (datetime.now() - timedelta(hours=18)).isoformat(), 'score': -0.1},
                    {'timestamp': (datetime.now() - timedelta(hours=12)).isoformat(), 'score': 0.1},
                    {'timestamp': (datetime.now() - timedelta(hours=6)).isoformat(), 'score': 0.2},
                    {'timestamp': datetime.now().isoformat(), 'score': 0.15}
                ],
                'mentions': 1200,
                'trending_topics': ['XRP SEC case', 'Ripple partnerships', 'Cross-border payments']
            },
            'SOL': {
                'sentiment_history': [
                    {'timestamp': (datetime.now() - timedelta(hours=24)).isoformat(), 'score': 0.4},
                    {'timestamp': (datetime.now() - timedelta(hours=18)).isoformat(), 'score': 0.3},
                    {'timestamp': (datetime.now() - timedelta(hours=12)).isoformat(), 'score': 0.25},
                    {'timestamp': (datetime.now() - timedelta(hours=6)).isoformat(), 'score': 0.5},
                    {'timestamp': datetime.now().isoformat(), 'score': 0.45}
                ],
                'mentions': 1800,
                'trending_topics': ['Solana DeFi', 'Network stability', 'NFT marketplace']
            }
        }
    
    def analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze sentiment from social media for a specific cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            
        Returns:
            Dictionary with social media sentiment analysis
        """
        # Clean symbol
        symbol = symbol.upper()
        if '/' in symbol:
            symbol = symbol.split('/')[0]
        
        # Check cache
        cache_key = f"social_sentiment:{symbol}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['data']
        
        # Prepare result
        sentiment_data = {
            'symbol': symbol,
            'score': 0,
            'sentiment': 'neutral',
            'mentions': 0,
            'trending_topics': [],
            'sentiment_change_24h': 0,
            'trend': 'neutral',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Use simulated data for common cryptos, or generate random data for others
            if symbol in self.simulated_data:
                data = self.simulated_data[symbol]
                
                # Get latest sentiment score
                latest_sentiment = data['sentiment_history'][-1]['score']
                
                # Calculate sentiment change over 24 hours
                sentiment_24h_ago = data['sentiment_history'][0]['score']
                sentiment_change = latest_sentiment - sentiment_24h_ago
                
                # Determine trend
                trend = 'neutral'
                if sentiment_change > 0.1:
                    trend = 'bullish'
                elif sentiment_change < -0.1:
                    trend = 'bearish'
                
                # Update sentiment data
                sentiment_data['score'] = latest_sentiment
                sentiment_data['mentions'] = data['mentions']
                sentiment_data['trending_topics'] = data['trending_topics']
                sentiment_data['sentiment_change_24h'] = sentiment_change
                sentiment_data['trend'] = trend
                
                # Determine sentiment label
                if latest_sentiment > 0.2:
                    sentiment_data['sentiment'] = 'positive'
                elif latest_sentiment < -0.2:
                    sentiment_data['sentiment'] = 'negative'
                else:
                    sentiment_data['sentiment'] = 'neutral'
            else:
                # For other cryptos, use OpenAI to generate a basic sentiment analysis
                # This would typically be replaced with actual social media API analysis
                try:
                    simulated_prompt = f"Generate a brief sentiment analysis for {symbol} cryptocurrency based on current trends. Include a sentiment score between -1 and 1, where 1 is very positive and -1 is very negative."
                    
                    response = self.openai.get_sentiment_analysis(simulated_prompt)
                    
                    # Extract sentiment score - assuming a reasonable response format
                    sentiment_score = 0
                    
                    for line in response.split('\n'):
                        if 'score' in line.lower():
                            try:
                                # Extract number from this line
                                sentiment_score = float([s for s in line.split() if s.replace('-', '').replace('.', '').isdigit()][0])
                                # Ensure score is in range [-1, 1]
                                sentiment_score = max(min(sentiment_score, 1), -1)
                            except:
                                sentiment_score = 0
                    
                    # Update sentiment data
                    sentiment_data['score'] = sentiment_score
                    sentiment_data['mentions'] = 500  # Default value
                    
                    # Determine sentiment label
                    if sentiment_score > 0.2:
                        sentiment_data['sentiment'] = 'positive'
                        sentiment_data['trend'] = 'bullish'
                    elif sentiment_score < -0.2:
                        sentiment_data['sentiment'] = 'negative'
                        sentiment_data['trend'] = 'bearish'
                    else:
                        sentiment_data['sentiment'] = 'neutral'
                        sentiment_data['trend'] = 'neutral'
                
                except Exception as e:
                    print(f"Error using OpenAI for sentiment generation: {e}")
        
        except Exception as e:
            print(f"Error analyzing social media sentiment: {e}")
        
        # Update cache
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'data': sentiment_data
        }
        
        return sentiment_data
    
    def get_trend_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get trending topics and sentiment trend for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            
        Returns:
            Dictionary with trend analysis
        """
        # Clean symbol
        symbol = symbol.upper()
        if '/' in symbol:
            symbol = symbol.split('/')[0]
        
        # Get base sentiment data
        sentiment_data = self.analyze_sentiment(symbol)
        
        # Additional trend analysis
        trend_data = {
            'symbol': symbol,
            'trending_topics': sentiment_data.get('trending_topics', []),
            'sentiment_trend': sentiment_data.get('trend', 'neutral'),
            'sentiment_history': []
        }
        
        # Get simulated sentiment history if available
        if symbol in self.simulated_data:
            trend_data['sentiment_history'] = self.simulated_data[symbol]['sentiment_history']
        
        return trend_data 