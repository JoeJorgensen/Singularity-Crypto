#!/usr/bin/env python3
"""
Test script for Finnhub API integration, focusing on sentiment data.
This script tests both the simulated data and attempts to make real API calls.
"""
import os
import asyncio
import json
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import sys

# Add the project directory to the Python path
sys.path.append('.')

# Import the Finnhub API from the project
from api.finnhub_api import FinnhubAPI

# Load environment variables
load_dotenv()

def test_current_implementation():
    """Test the current implementation which uses simulated data."""
    print("\n----- Testing Current Implementation (Simulated Data) -----")
    
    # Initialize the API client
    finnhub_api = FinnhubAPI()
    
    # Print the API key length for debugging (without revealing the key)
    key_length = len(finnhub_api.api_key) if finnhub_api.api_key else 0
    print(f"API Key loaded: {'Yes' if key_length > 0 else 'No'} (length: {key_length})")
    
    # Test for different symbols
    for symbol in ['ETH', 'BTC', 'XRP']:
        # Get sentiment data
        sentiment = finnhub_api.get_aggregate_sentiment(symbol)
        
        print(f"\nSentiment for {symbol}:")
        print(f"  Score: {sentiment.get('sentiment_score', 0)}")
        print(f"  News Count: {sentiment.get('news_count', 0)}")
        print(f"  Method: Simulated data (current implementation)")

async def test_real_api_call():
    """Test making real API calls to Finnhub for sentiment data."""
    print("\n----- Testing Direct Finnhub API Calls -----")
    
    # Initialize the API client
    finnhub_api = FinnhubAPI()
    api_key = finnhub_api.api_key
    
    if not api_key:
        print("ERROR: No Finnhub API key found. Please set FINNHUB_API_KEY in your .env file.")
        return
    
    print(f"API Key loaded: Yes (length: {len(api_key)})")
    
    # Test for different symbols
    for symbol in ['ETH', 'BTC', 'XRP']:
        # Implementation 1: Buzz API (sentiment)
        try:
            url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}"
            headers = {'X-Finnhub-Token': api_key}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                sentiment_data = response.json()
                print(f"\nSentiment for {symbol} (News Sentiment API):")
                print(f"  Buzz Score: {sentiment_data.get('buzz', {}).get('buzz', 'N/A')}")
                print(f"  Sentiment Score: {sentiment_data.get('sentiment', {}).get('bullishPercent', 'N/A')}")
                print(f"  Data source: Finnhub News Sentiment API")
            else:
                print(f"\nERROR for {symbol} (News Sentiment API): Status code {response.status_code}")
                print(f"  Response: {response.text}")
        except Exception as e:
            print(f"\nException for {symbol} (News Sentiment API): {str(e)}")
        
        # Implementation 2: News API with sentiment analysis
        try:
            # Calculate last 7 days for news
            today = datetime.now()
            week_ago = today - timedelta(days=7)
            today_str = today.strftime('%Y-%m-%d')
            week_ago_str = week_ago.strftime('%Y-%m-%d')
            
            # For crypto, we need to modify the symbol format
            crypto_symbol = f"CRYPTO:{symbol}"
            
            url = f"https://finnhub.io/api/v1/company-news?symbol={crypto_symbol}&from={week_ago_str}&to={today_str}"
            headers = {'X-Finnhub-Token': api_key}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                news_data = response.json()
                
                print(f"\nNews for {symbol} (Company News API):")
                print(f"  News Count: {len(news_data)}")
                
                # Simple analysis to calculate sentiment from headlines
                if news_data:
                    positive_keywords = ['bullish', 'surge', 'rally', 'gain', 'up', 'rise', 'positive', 'growth']
                    negative_keywords = ['bearish', 'crash', 'fall', 'drop', 'down', 'negative', 'loss', 'concern']
                    
                    positive_count = 0
                    negative_count = 0
                    
                    for article in news_data[:10]:  # Analyze first 10 articles
                        headline = article.get('headline', '').lower()
                        
                        positive_matches = sum(1 for keyword in positive_keywords if keyword in headline)
                        negative_matches = sum(1 for keyword in negative_keywords if keyword in headline)
                        
                        if positive_matches > negative_matches:
                            positive_count += 1
                        elif negative_matches > positive_matches:
                            negative_count += 1
                    
                    total_analyzed = min(10, len(news_data))
                    sentiment_score = (positive_count - negative_count) / total_analyzed if total_analyzed > 0 else 0
                    
                    print(f"  Calculated Sentiment Score: {sentiment_score:.2f} (from {total_analyzed} articles)")
                    print(f"  Positive articles: {positive_count}, Negative articles: {negative_count}")
                    print(f"  Data source: Finnhub Company News API + custom analysis")
                else:
                    print("  No news articles found for analysis")
            else:
                print(f"\nERROR for {symbol} (Company News API): Status code {response.status_code}")
                print(f"  Response: {response.text}")
        except Exception as e:
            print(f"\nException for {symbol} (Company News API): {str(e)}")
        
        # Add a small delay between API calls to avoid rate limiting
        await asyncio.sleep(1)

def test_updated_implementation():
    """Test an updated implementation for the FinnhubAPI class."""
    print("\n----- Testing Updated Implementation (Real API) -----")
    
    # Define an updated implementation with proper API calls
    class UpdatedFinnhubAPI:
        def __init__(self):
            self.api_key = os.getenv('FINNHUB_API_KEY', '') or os.getenv('FINNHUB_KEY', '')
            self.base_url = 'https://finnhub.io/api/v1'
            
        def get_aggregate_sentiment(self, symbol: str) -> dict:
            """Get aggregate sentiment data using a real API call."""
            headers = {'X-Finnhub-Token': self.api_key}
            
            # First try news sentiment API
            url = f"{self.base_url}/news-sentiment?symbol={symbol}"
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if data and 'sentiment' in data:
                        sentiment_score = data['sentiment'].get('bullishPercent', 0.5) - 0.5  # Convert to -0.5 to 0.5 range
                        return {
                            'symbol': symbol,
                            'sentiment_score': sentiment_score,
                            'news_count': data.get('buzz', {}).get('articlesInLastWeek', 0),
                            'buzz_score': data.get('buzz', {}).get('buzz', 0),
                            'positive_news': data.get('sentiment', {}).get('bullishPercent', 0) * 100,
                            'negative_news': (1 - data.get('sentiment', {}).get('bullishPercent', 0)) * 100,
                            'neutral_news': 0,  # Not provided by API
                            'source': 'finnhub:news-sentiment'
                        }
            except Exception as e:
                print(f"Error with news-sentiment API: {e}")
            
            # Fallback to news API with custom sentiment analysis
            try:
                # Calculate last 7 days for news
                today = datetime.now()
                week_ago = today - timedelta(days=7)
                today_str = today.strftime('%Y-%m-%d')
                week_ago_str = week_ago.strftime('%Y-%m-%d')
                
                # For crypto, use CRYPTO: prefix
                crypto_symbol = f"CRYPTO:{symbol}"
                
                url = f"{self.base_url}/company-news?symbol={crypto_symbol}&from={week_ago_str}&to={today_str}"
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    news_data = response.json()
                    
                    if news_data:
                        positive_keywords = ['bullish', 'surge', 'rally', 'gain', 'up', 'rise', 'positive', 'growth']
                        negative_keywords = ['bearish', 'crash', 'fall', 'drop', 'down', 'negative', 'loss', 'concern']
                        
                        positive_count = 0
                        negative_count = 0
                        neutral_count = 0
                        
                        for article in news_data:
                            headline = article.get('headline', '').lower()
                            
                            positive_matches = sum(1 for keyword in positive_keywords if keyword in headline)
                            negative_matches = sum(1 for keyword in negative_keywords if keyword in headline)
                            
                            if positive_matches > negative_matches:
                                positive_count += 1
                            elif negative_matches > positive_matches:
                                negative_count += 1
                            else:
                                neutral_count += 1
                        
                        total_analyzed = len(news_data)
                        sentiment_score = (positive_count - negative_count) / total_analyzed if total_analyzed > 0 else 0
                        
                        return {
                            'symbol': symbol,
                            'sentiment_score': sentiment_score,
                            'news_count': total_analyzed,
                            'buzz_score': total_analyzed / 100,  # Arbitrary scale
                            'positive_news': positive_count,
                            'negative_news': negative_count,
                            'neutral_news': neutral_count,
                            'source': 'finnhub:company-news'
                        }
                    
                    # Return empty data if no news found
                    return {
                        'symbol': symbol,
                        'sentiment_score': 0,
                        'news_count': 0,
                        'buzz_score': 0,
                        'source': 'finnhub:no-news'
                    }
            except Exception as e:
                print(f"Error with company-news API: {e}")
            
            # If all else fails, return default values
            return {
                'symbol': symbol,
                'sentiment_score': 0,
                'news_count': 0,
                'source': 'fallback'
            }
    
    # Test the updated implementation
    updated_api = UpdatedFinnhubAPI()
    
    # Print the API key length for debugging (without revealing the key)
    key_length = len(updated_api.api_key) if updated_api.api_key else 0
    print(f"API Key loaded: {'Yes' if key_length > 0 else 'No'} (length: {key_length})")
    
    # Test for different symbols
    for symbol in ['ETH', 'BTC', 'XRP']:
        # Get sentiment data
        sentiment = updated_api.get_aggregate_sentiment(symbol)
        
        print(f"\nSentiment for {symbol}:")
        print(f"  Score: {sentiment.get('sentiment_score', 0)}")
        print(f"  News Count: {sentiment.get('news_count', 0)}")
        print(f"  Buzz Score: {sentiment.get('buzz_score', 0)}")
        print(f"  Source: {sentiment.get('source', 'unknown')}")
        print(f"  Positive News: {sentiment.get('positive_news', 0)}")
        print(f"  Negative News: {sentiment.get('negative_news', 0)}")
        
        # Add a small delay between API calls to avoid rate limiting
        time.sleep(1)

async def main():
    # Test the current implementation (simulated data)
    test_current_implementation()
    
    # Test real API call
    await test_real_api_call()
    
    # Test updated implementation
    test_updated_implementation()
    
    print("\n----- Recommended Implementation -----")
    print("Based on the test results, the updated implementation should be integrated into")
    print("the FinnhubAPI class to replace the simulated data with real API calls.")
    print("This will provide actual sentiment data instead of random values.")

if __name__ == "__main__":
    import time
    asyncio.run(main()) 