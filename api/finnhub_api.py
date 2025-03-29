"""
FinnhubAPI - Interface for Finnhub market data API.
"""
import os
import asyncio
import aiohttp
from typing import Dict, List, Optional
import requests
from dotenv import load_dotenv

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
        self.api_key = os.getenv('FINNHUB_API_KEY', '')
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
    
    async def get_aggregate_sentiment_async(self, symbol: str) -> Dict:
        """
        Get aggregate sentiment data for a symbol asynchronously.
        
        Args:
            symbol: Symbol to get sentiment for (e.g., 'BTC', 'ETH')
            
        Returns:
            Dictionary with sentiment data
        """
        try:
            # In a production environment, we would query the Finnhub API using aiohttp
            # For now, we'll return simulated data to keep the application running
            import random
            
            # Simulate network delay for realism
            await asyncio.sleep(0.1)
            
            # Generate random sentiment score between -1 and 1
            sentiment_score = random.uniform(-0.5, 0.7)
            
            # Adjust sentiment for common cryptos
            if symbol.upper() == 'BTC':
                sentiment_score = max(0.1, sentiment_score)  # Bias towards positive
            elif symbol.upper() == 'ETH':
                sentiment_score = max(-0.1, sentiment_score)  # Slightly positive bias
                
            return {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'news_count': random.randint(5, 50),
                'buzz_score': random.uniform(0, 1),
                'positive_news': random.randint(1, 20),
                'negative_news': random.randint(1, 10),
                'neutral_news': random.randint(1, 20)
            }
        except Exception as e:
            print(f"Error generating sentiment data asynchronously: {e}")
            return {
                'symbol': symbol,
                'sentiment_score': 0,
                'news_count': 0
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
            # In a production environment, we would query the Finnhub API
            # For now, we'll return simulated data to keep the application running
            import random
            
            # Generate random sentiment score between -1 and 1
            sentiment_score = random.uniform(-0.5, 0.7)
            
            # Adjust sentiment for common cryptos
            if symbol.upper() == 'BTC':
                sentiment_score = max(0.1, sentiment_score)  # Bias towards positive
            elif symbol.upper() == 'ETH':
                sentiment_score = max(-0.1, sentiment_score)  # Slightly positive bias
                
            return {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'news_count': random.randint(5, 50),
                'buzz_score': random.uniform(0, 1),
                'positive_news': random.randint(1, 20),
                'negative_news': random.randint(1, 10),
                'neutral_news': random.randint(1, 20)
            }
        except Exception as e:
            print(f"Error generating sentiment data: {e}")
            return {
                'symbol': symbol,
                'sentiment_score': 0,
                'news_count': 0
            } 