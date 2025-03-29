"""
NewsAnalyzer - Analyze sentiment from cryptocurrency news.
"""
from typing import Dict, List, Any, Optional
import os
import time
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from api.finnhub_api import FinnhubAPI

# Load environment variables
load_dotenv()

class NewsAnalyzer:
    """
    Analyze sentiment from cryptocurrency news using Finnhub or other news APIs.
    """
    
    def __init__(self):
        """Initialize news analyzer."""
        self.finnhub = FinnhubAPI()
        self.cache = {}
        self.cache_expiry = 3600  # 1 hour
    
    def analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze sentiment from news for a specific cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            
        Returns:
            Dictionary with news sentiment analysis
        """
        # Check cache
        cache_key = f"news_sentiment:{symbol}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['data']
        
        # Prepare result
        sentiment_data = {
            'symbol': symbol,
            'score': 0,
            'sentiment': 'neutral',
            'news_count': 0,
            'positive_news': 0,
            'negative_news': 0,
            'neutral_news': 0,
            'important_events': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Get crypto news from Finnhub
            news = self.finnhub.get_crypto_news()
            
            # Filter news for the specified cryptocurrency
            symbol_news = []
            for article in news:
                # Check if the symbol is mentioned in the headline or summary
                headline = article.get('headline', '').lower()
                summary = article.get('summary', '').lower()
                
                if (symbol.lower() in headline or 
                    symbol.lower() in summary or 
                    'crypto' in headline or 
                    'cryptocurrency' in headline):
                    symbol_news.append(article)
            
            sentiment_data['news_count'] = len(symbol_news)
            
            if symbol_news:
                # Analyze sentiment for each article
                positive_count = 0
                negative_count = 0
                neutral_count = 0
                important_events = []
                
                for article in symbol_news:
                    # Use Finnhub sentiment if available, otherwise simple keyword analysis
                    sentiment_score = article.get('sentiment', 0)
                    
                    if sentiment_score == 0:
                        # Simple keyword-based sentiment analysis
                        headline = article.get('headline', '').lower()
                        summary = article.get('summary', '').lower()
                        
                        positive_keywords = ['bullish', 'surge', 'rally', 'gain', 'up', 'rise', 'positive', 'growth']
                        negative_keywords = ['bearish', 'crash', 'fall', 'drop', 'down', 'negative', 'loss', 'concern']
                        
                        positive_matches = sum(1 for keyword in positive_keywords if keyword in headline or keyword in summary)
                        negative_matches = sum(1 for keyword in negative_keywords if keyword in headline or keyword in summary)
                        
                        if positive_matches > negative_matches:
                            sentiment_score = 0.5
                        elif negative_matches > positive_matches:
                            sentiment_score = -0.5
                        else:
                            sentiment_score = 0
                    
                    # Categorize article sentiment
                    if sentiment_score > 0.2:
                        positive_count += 1
                    elif sentiment_score < -0.2:
                        negative_count += 1
                    else:
                        neutral_count += 1
                    
                    # Check for important events
                    importance_keywords = ['major', 'breaking', 'announcement', 'launch', 'partnership', 'regulation']
                    if any(keyword in article.get('headline', '').lower() for keyword in importance_keywords):
                        important_events.append({
                            'headline': article.get('headline'),
                            'url': article.get('url'),
                            'datetime': article.get('datetime')
                        })
                
                # Update sentiment data
                sentiment_data['positive_news'] = positive_count
                sentiment_data['negative_news'] = negative_count
                sentiment_data['neutral_news'] = neutral_count
                sentiment_data['important_events'] = important_events[:5]  # Limit to top 5 important events
                
                # Calculate overall sentiment score (-1 to 1)
                if sentiment_data['news_count'] > 0:
                    sentiment_data['score'] = (positive_count - negative_count) / sentiment_data['news_count']
                
                # Determine sentiment label
                if sentiment_data['score'] > 0.2:
                    sentiment_data['sentiment'] = 'positive'
                elif sentiment_data['score'] < -0.2:
                    sentiment_data['sentiment'] = 'negative'
                else:
                    sentiment_data['sentiment'] = 'neutral'
        
        except Exception as e:
            print(f"Error analyzing news sentiment: {e}")
        
        # Update cache
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'data': sentiment_data
        }
        
        return sentiment_data
    
    def get_recent_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get recent news for a specific cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            limit: Maximum number of news items to return
            
        Returns:
            List of news items
        """
        try:
            # Get crypto news from Finnhub
            news = self.finnhub.get_crypto_news()
            
            # Filter news for the specified cryptocurrency
            symbol_news = []
            for article in news:
                # Check if the symbol is mentioned in the headline or summary
                headline = article.get('headline', '').lower()
                summary = article.get('summary', '').lower()
                
                if (symbol.lower() in headline or 
                    symbol.lower() in summary or 
                    'crypto' in headline or 
                    'cryptocurrency' in headline):
                    symbol_news.append({
                        'headline': article.get('headline'),
                        'summary': article.get('summary'),
                        'url': article.get('url'),
                        'source': article.get('source'),
                        'datetime': article.get('datetime'),
                        'image': article.get('image')
                    })
                    
                    if len(symbol_news) >= limit:
                        break
            
            return symbol_news
        
        except Exception as e:
            print(f"Error getting recent news: {e}")
            return [] 