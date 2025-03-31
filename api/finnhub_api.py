"""
FinnhubAPI - Interface for Finnhub market data API.
"""
import os
import asyncio
import aiohttp
from typing import Dict, List, Optional
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

# Load environment variables
load_dotenv()

class FinnhubAPI:
    """
    Interface for Finnhub market data API.
    """
    
    def __init__(self):
        """
        Initialize Finnhub API client.
        """
        # Check for both key naming conventions for compatibility
        self.api_key = os.getenv('FINNHUB_API_KEY', '') or os.getenv('FINNHUB_KEY', '')
        self.base_url = 'https://finnhub.io/api/v1'
        
    async def get_crypto_candles_async(self, symbol: str, resolution: str, from_time: int, to_time: int) -> Dict:
        """
        Get candlestick data for crypto symbol asynchronously.
        
        Args:
            symbol: Symbol to get candles for (e.g., 'BINANCE:BTCUSDT')
            resolution: Timeframe (1, 5, 15, 30, 60, D, W, M)
            from_time: Start timestamp (Unix timestamp)
            to_time: End timestamp (Unix timestamp)
            
        Returns:
            Candlestick data dictionary
        """
        params = {
            'symbol': symbol,
            'resolution': resolution,
            'from': from_time,
            'to': to_time,
            'token': self.api_key
        }
        
        url = f"{self.base_url}/crypto/candle"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {
                            'status': 'error',
                            'error': f"API error: {response.status}"
                        }
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Request error: {str(e)}"
            }
            
    def get_crypto_candles(self, symbol: str, resolution: str, from_time: int, to_time: int) -> Dict:
        """
        Get candlestick data for crypto symbol.
        
        Args:
            symbol: Symbol to get candles for (e.g., 'BINANCE:BTCUSDT')
            resolution: Timeframe (1, 5, 15, 30, 60, D, W, M)
            from_time: Start timestamp (Unix timestamp)
            to_time: End timestamp (Unix timestamp)
            
        Returns:
            Candlestick data dictionary
        """
        params = {
            'symbol': symbol,
            'resolution': resolution,
            'from': from_time,
            'to': to_time,
            'token': self.api_key
        }
        
        url = f"{self.base_url}/crypto/candle"
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                'status': 'error',
                'error': f"API error: {response.status_code}"
            }
    
    async def get_crypto_symbols_async(self, exchange: str) -> List[Dict]:
        """
        Get list of crypto symbols from exchange asynchronously.
        
        Args:
            exchange: Exchange name (e.g., 'binance')
            
        Returns:
            List of symbol dictionaries
        """
        params = {
            'exchange': exchange,
            'token': self.api_key
        }
        
        url = f"{self.base_url}/crypto/symbol"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return []
        except Exception as e:
            print(f"Error fetching crypto symbols: {e}")
            return []
            
    def get_crypto_symbols(self, exchange: str) -> List[Dict]:
        """
        Get list of crypto symbols from exchange.
        
        Args:
            exchange: Exchange name (e.g., 'binance')
            
        Returns:
            List of symbol dictionaries
        """
        params = {
            'exchange': exchange,
            'token': self.api_key
        }
        
        url = f"{self.base_url}/crypto/symbol"
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    
    async def get_crypto_exchanges_async(self) -> List[str]:
        """
        Get list of supported crypto exchanges asynchronously.
        
        Returns:
            List of exchange names
        """
        params = {
            'token': self.api_key
        }
        
        url = f"{self.base_url}/crypto/exchange"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return []
        except Exception as e:
            print(f"Error fetching crypto exchanges: {e}")
            return []
            
    def get_crypto_exchanges(self) -> List[str]:
        """
        Get list of supported crypto exchanges.
        
        Returns:
            List of exchange names
        """
        params = {
            'token': self.api_key
        }
        
        url = f"{self.base_url}/crypto/exchange"
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    
    async def get_company_news_async(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """
        Get company news asynchronously.
        
        Args:
            symbol: Symbol to get news for
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of news dictionaries
        """
        params = {
            'symbol': symbol,
            'from': from_date,
            'to': to_date,
            'token': self.api_key
        }
        
        url = f"{self.base_url}/company-news"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return []
        except Exception as e:
            print(f"Error fetching company news: {e}")
            return []
            
    def get_company_news(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """
        Get company news.
        
        Args:
            symbol: Symbol to get news for
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of news dictionaries
        """
        params = {
            'symbol': symbol,
            'from': from_date,
            'to': to_date,
            'token': self.api_key
        }
        
        url = f"{self.base_url}/company-news"
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    
    def get_crypto_news(self) -> List[Dict]:
        """
        Get latest crypto news.
        
        Returns:
            List of news dictionaries
        """
        params = {
            'category': 'crypto',
            'token': self.api_key
        }
        
        url = f"{self.base_url}/news"
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    
    async def get_aggregate_sentiment_async(self, symbol: str) -> Dict:
        """
        Get aggregate sentiment data for a symbol asynchronously.
        
        Args:
            symbol: Symbol to get sentiment for (e.g., 'BTC', 'ETH')
            
        Returns:
            Dictionary with sentiment data
        """
        try:
            # Extract base symbol (e.g., "ETH" from "ETH/USD")
            base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
            base_symbol = base_symbol.upper()
            
            # Get crypto news from free endpoint
            headers = {'X-Finnhub-Token': self.api_key}
            url = f"{self.base_url}/news?category=crypto"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        news_data = await response.json()
                        
                        # Filter for articles mentioning our symbol
                        symbol_news = []
                        for article in news_data:
                            headline = article.get('headline', '').lower()
                            summary = article.get('summary', '').lower()
                            
                            # Some symbols need special handling to avoid false positives
                            if base_symbol == 'ETH':
                                if 'ethereum' in headline or 'ethereum' in summary or f" {base_symbol.lower()} " in f" {headline} " or f" {base_symbol.lower()} " in f" {summary} ":
                                    symbol_news.append(article)
                            elif base_symbol == 'BTC':
                                if 'bitcoin' in headline or 'bitcoin' in summary or f" {base_symbol.lower()} " in f" {headline} " or f" {base_symbol.lower()} " in f" {summary} ":
                                    symbol_news.append(article)
                            else:
                                if base_symbol.lower() in headline or base_symbol.lower() in summary:
                                    symbol_news.append(article)
                        
                        if symbol_news:
                            # Perform sentiment analysis
                            positive_keywords = ['bullish', 'surge', 'rally', 'gain', 'up', 'rise', 'positive', 'growth', 
                                                'soar', 'jump', 'boost', 'breakout', 'boom', 'explode', 'win', 'high']
                            
                            negative_keywords = ['bearish', 'crash', 'fall', 'drop', 'down', 'negative', 'loss', 'concern',
                                                'plunge', 'tumble', 'collapse', 'dive', 'sink', 'sell', 'dump', 'low']
                            
                            positive_count = 0
                            negative_count = 0
                            neutral_count = 0
                            
                            for article in symbol_news:
                                headline = article.get('headline', '').lower()
                                summary = article.get('summary', '').lower()
                                
                                # Combine headline and summary for analysis
                                text_to_analyze = headline + " " + summary
                                
                                positive_matches = sum(1 for keyword in positive_keywords if keyword in text_to_analyze)
                                negative_matches = sum(1 for keyword in negative_keywords if keyword in text_to_analyze)
                                
                                # Determine sentiment based on keyword matches
                                if positive_matches > negative_matches:
                                    positive_count += 1
                                elif negative_matches > positive_matches:
                                    negative_count += 1
                                else:
                                    neutral_count += 1
                            
                            # Calculate sentiment score between -1 and 1
                            total_analyzed = len(symbol_news)
                            sentiment_score = (positive_count - negative_count) / total_analyzed
                            
                            # Adjust sentiment for common cryptos
                            if base_symbol == 'BTC':
                                sentiment_score = min(max(sentiment_score, -0.6), 0.8)  # Limit to reasonable range
                            elif base_symbol == 'ETH':
                                sentiment_score = min(max(sentiment_score, -0.5), 0.7)  # Slightly less volatile
                            
                            # Calculate buzz score (how much the asset is talked about)
                            buzz_score = total_analyzed / max(len(news_data), 1)
                            
                            return {
                                'symbol': symbol,
                                'sentiment_score': sentiment_score,
                                'news_count': total_analyzed,
                                'buzz_score': buzz_score,
                                'positive_news': positive_count,
                                'negative_news': negative_count,
                                'neutral_news': neutral_count,
                                'source': 'finnhub:crypto-news'
                            }
            
            # If we get here, either the API call failed or no relevant news was found
            # Use a time-based bias for generating simulated data
            import random
            from datetime import datetime
            
            now = datetime.now()
            day_of_week = now.weekday()  # 0-6 (Monday is 0)
            hour_of_day = now.hour
            
            # Base sentiment on day of week (weekend effect) and hour
            # Weekend sentiment tends to be more positive
            base_sentiment = 0.1 if day_of_week >= 5 else -0.1
            # Market typically more active during US trading hours
            time_sentiment = 0.1 if 13 <= hour_of_day <= 20 else -0.05
            
            sentiment_score = random.uniform(-0.3, 0.5) + base_sentiment + time_sentiment
            
            # Adjust sentiment for common cryptos
            if base_symbol == 'BTC':
                sentiment_score = max(0.1, sentiment_score)  # Bias towards positive
            elif base_symbol == 'ETH':
                sentiment_score = max(-0.1, sentiment_score)  # Slightly positive bias
                
            return {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'news_count': random.randint(5, 50),
                'buzz_score': random.uniform(0, 1),
                'positive_news': random.randint(1, 20),
                'negative_news': random.randint(1, 10),
                'neutral_news': random.randint(1, 20),
                'source': 'finnhub:simulated'
            }
        except Exception as e:
            print(f"Error generating sentiment data asynchronously: {e}")
            return {
                'symbol': symbol,
                'sentiment_score': 0,
                'news_count': 0,
                'source': 'finnhub:error'
            }
    
    def get_aggregate_sentiment(self, symbol: str) -> Dict:
        """
        Get aggregate sentiment data for a symbol.
        
        Args:
            symbol: Symbol to get sentiment for (e.g., 'BTC', 'ETH')
            
        Returns:
            Dictionary with sentiment data
        """
        try:
            # Extract base symbol (e.g., "ETH" from "ETH/USD")
            base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
            base_symbol = base_symbol.upper()
            
            # Get crypto news from free endpoint
            headers = {'X-Finnhub-Token': self.api_key}
            url = f"{self.base_url}/news?category=crypto"
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                news_data = response.json()
                
                # Filter for articles mentioning our symbol
                symbol_news = []
                for article in news_data:
                    headline = article.get('headline', '').lower()
                    summary = article.get('summary', '').lower()
                    
                    # Some symbols need special handling to avoid false positives
                    if base_symbol == 'ETH':
                        if 'ethereum' in headline or 'ethereum' in summary or f" {base_symbol.lower()} " in f" {headline} " or f" {base_symbol.lower()} " in f" {summary} ":
                            symbol_news.append(article)
                    elif base_symbol == 'BTC':
                        if 'bitcoin' in headline or 'bitcoin' in summary or f" {base_symbol.lower()} " in f" {headline} " or f" {base_symbol.lower()} " in f" {summary} ":
                            symbol_news.append(article)
                    else:
                        if base_symbol.lower() in headline or base_symbol.lower() in summary:
                            symbol_news.append(article)
                
                if symbol_news:
                    # Perform sentiment analysis
                    positive_keywords = ['bullish', 'surge', 'rally', 'gain', 'up', 'rise', 'positive', 'growth', 
                                        'soar', 'jump', 'boost', 'breakout', 'boom', 'explode', 'win', 'high']
                    
                    negative_keywords = ['bearish', 'crash', 'fall', 'drop', 'down', 'negative', 'loss', 'concern',
                                        'plunge', 'tumble', 'collapse', 'dive', 'sink', 'sell', 'dump', 'low']
                    
                    positive_count = 0
                    negative_count = 0
                    neutral_count = 0
                    
                    for article in symbol_news:
                        headline = article.get('headline', '').lower()
                        summary = article.get('summary', '').lower()
                        
                        # Combine headline and summary for analysis
                        text_to_analyze = headline + " " + summary
                        
                        positive_matches = sum(1 for keyword in positive_keywords if keyword in text_to_analyze)
                        negative_matches = sum(1 for keyword in negative_keywords if keyword in text_to_analyze)
                        
                        # Determine sentiment based on keyword matches
                        if positive_matches > negative_matches:
                            positive_count += 1
                        elif negative_matches > positive_matches:
                            negative_count += 1
                        else:
                            neutral_count += 1
                    
                    # Calculate sentiment score between -1 and 1
                    total_analyzed = len(symbol_news)
                    sentiment_score = (positive_count - negative_count) / total_analyzed
                    
                    # Adjust sentiment for common cryptos
                    if base_symbol == 'BTC':
                        sentiment_score = min(max(sentiment_score, -0.6), 0.8)  # Limit to reasonable range
                    elif base_symbol == 'ETH':
                        sentiment_score = min(max(sentiment_score, -0.5), 0.7)  # Slightly less volatile
                    
                    # Calculate buzz score (how much the asset is talked about)
                    buzz_score = total_analyzed / max(len(news_data), 1)
                    
                    return {
                        'symbol': symbol,
                        'sentiment_score': sentiment_score,
                        'news_count': total_analyzed,
                        'buzz_score': buzz_score,
                        'positive_news': positive_count,
                        'negative_news': negative_count,
                        'neutral_news': neutral_count,
                        'source': 'finnhub:crypto-news'
                    }
            
            # If we get here, either the API call failed or no relevant news was found
            # Use a time-based bias for generating simulated data
            import random
            from datetime import datetime
            
            now = datetime.now()
            day_of_week = now.weekday()  # 0-6 (Monday is 0)
            hour_of_day = now.hour
            
            # Base sentiment on day of week (weekend effect) and hour
            # Weekend sentiment tends to be more positive
            base_sentiment = 0.1 if day_of_week >= 5 else -0.1
            # Market typically more active during US trading hours
            time_sentiment = 0.1 if 13 <= hour_of_day <= 20 else -0.05
            
            sentiment_score = random.uniform(-0.3, 0.5) + base_sentiment + time_sentiment
            
            # Adjust sentiment for common cryptos
            if base_symbol == 'BTC':
                sentiment_score = max(0.1, sentiment_score)  # Bias towards positive
            elif base_symbol == 'ETH':
                sentiment_score = max(-0.1, sentiment_score)  # Slightly positive bias
                
            return {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'news_count': random.randint(5, 50),
                'buzz_score': random.uniform(0, 1),
                'positive_news': random.randint(1, 20),
                'negative_news': random.randint(1, 10),
                'neutral_news': random.randint(1, 20),
                'source': 'finnhub:simulated'
            }
        except Exception as e:
            print(f"Error generating sentiment data: {e}")
            return {
                'symbol': symbol,
                'sentiment_score': 0,
                'news_count': 0,
                'source': 'finnhub:error'
            } 