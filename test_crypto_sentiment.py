#!/usr/bin/env python3
"""
Test script to check crypto sentiment analysis using Finnhub's free crypto news API.
This script directly accesses the crypto news endpoint and analyzes sentiment from headlines.
"""
import os
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sys
import time

# Load environment variables from .env file
load_dotenv()

def analyze_sentiment_from_crypto_news(api_key, symbol):
    """
    Get cryptocurrency news and analyze sentiment specifically for a symbol.
    Uses the free crypto news category endpoint.
    
    Args:
        api_key: Finnhub API key
        symbol: Cryptocurrency symbol to analyze (e.g., 'BTC', 'ETH')
        
    Returns:
        Dictionary with sentiment analysis results
    """
    # Make sure symbol is uppercase for consistency
    symbol = symbol.upper()
    print(f"\n----- Analyzing sentiment for {symbol} -----")
    
    # Get crypto news from free endpoint
    headers = {'X-Finnhub-Token': api_key}
    url = f"https://finnhub.io/api/v1/news?category=crypto"
    
    try:
        print(f"Fetching crypto news from Finnhub API (free endpoint)")
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return {
                'symbol': symbol,
                'sentiment_score': 0,
                'news_count': 0,
                'source': 'error'
            }
        
        news_data = response.json()
        print(f"Retrieved {len(news_data)} crypto news articles")
        
        # Filter for articles mentioning our symbol
        symbol_news = []
        for article in news_data:
            headline = article.get('headline', '').lower()
            summary = article.get('summary', '').lower()
            
            # Some symbols need special handling to avoid false positives
            if symbol == 'ETH':
                if 'ethereum' in headline or 'ethereum' in summary or f" {symbol.lower()} " in f" {headline} " or f" {symbol.lower()} " in f" {summary} ":
                    symbol_news.append(article)
            elif symbol == 'BTC':
                if 'bitcoin' in headline or 'bitcoin' in summary or f" {symbol.lower()} " in f" {headline} " or f" {symbol.lower()} " in f" {summary} ":
                    symbol_news.append(article)
            else:
                if symbol.lower() in headline or symbol.lower() in summary:
                    symbol_news.append(article)
        
        print(f"Found {len(symbol_news)} articles mentioning {symbol}")
        
        if not symbol_news:
            print(f"No news articles found for {symbol}")
            
            # As a fallback, show top headlines for crypto in general
            print("\nLatest crypto headlines (general):")
            for i, article in enumerate(news_data[:5]):
                print(f"{i+1}. {article.get('headline', 'No headline')} [{article.get('source', 'Unknown')}]")
            
            return {
                'symbol': symbol,
                'sentiment_score': 0,
                'news_count': 0,
                'source': 'no-news'
            }
        
        # Perform sentiment analysis
        positive_keywords = ['bullish', 'surge', 'rally', 'gain', 'up', 'rise', 'positive', 'growth', 
                             'soar', 'jump', 'boost', 'breakout', 'boom', 'explode', 'win', 'high']
        
        negative_keywords = ['bearish', 'crash', 'fall', 'drop', 'down', 'negative', 'loss', 'concern',
                             'plunge', 'tumble', 'collapse', 'dive', 'sink', 'sell', 'dump', 'low']
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        print("\nAnalyzing article sentiment...")
        
        # Show a few headlines for verification
        print("\nSample headlines:")
        for i, article in enumerate(symbol_news[:5]):
            print(f"{i+1}. {article.get('headline', 'No headline')} [{article.get('source', 'Unknown')}]")
        
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
        sentiment_score = (positive_count - negative_count) / total_analyzed if total_analyzed > 0 else 0
        
        # Adjust sentiment for common cryptos
        if symbol == 'BTC':
            sentiment_score = min(max(sentiment_score, -0.6), 0.8)  # Limit to reasonable range
        elif symbol == 'ETH':
            sentiment_score = min(max(sentiment_score, -0.5), 0.7)  # Slightly less volatile
        
        print("\nSentiment analysis results:")
        print(f"  Positive articles: {positive_count} ({positive_count/total_analyzed*100:.1f}%)")
        print(f"  Negative articles: {negative_count} ({negative_count/total_analyzed*100:.1f}%)")
        print(f"  Neutral articles: {neutral_count} ({neutral_count/total_analyzed*100:.1f}%)")
        print(f"  Sentiment score: {sentiment_score:.4f} [-1 (bearish) to +1 (bullish)]")
        
        # Determine buzziness (how much the asset is talked about)
        buzz_score = total_analyzed / max(len(news_data), 1)
        print(f"  Buzz score: {buzz_score:.4f} (asset mentions / total news)")
        
        return {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'news_count': total_analyzed,
            'positive_news': positive_count,
            'negative_news': negative_count,
            'neutral_news': neutral_count,
            'buzz_score': buzz_score,
            'source': 'crypto-news'
        }
        
    except Exception as e:
        print(f"Error during sentiment analysis: {str(e)}")
        return {
            'symbol': symbol,
            'sentiment_score': 0,
            'news_count': 0,
            'source': 'error'
        }

def analyze_price_momentum(api_key, symbol):
    """
    Analyze price momentum as a proxy for sentiment when news is unavailable.
    
    Args:
        api_key: Finnhub API key
        symbol: Cryptocurrency symbol to analyze (e.g., 'BTC', 'ETH')
        
    Returns:
        Dictionary with price momentum analysis results
    """
    symbol = symbol.upper()
    print(f"\n----- Analyzing price momentum for {symbol} -----")
    
    try:
        # Get current timestamp and 7 days ago
        now = int(time.time())
        week_ago = now - (7 * 24 * 60 * 60)
        
        # Create a crypto symbol format for Binance
        crypto_symbol = f"BINANCE:{symbol}USDT"
        
        # Get daily candles for past week
        headers = {'X-Finnhub-Token': api_key}
        url = f"https://finnhub.io/api/v1/crypto/candle?symbol={crypto_symbol}&resolution=D&from={week_ago}&to={now}"
        
        print(f"Fetching price data from Finnhub API...")
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        candle_data = response.json()
        
        if candle_data.get('s') != 'ok' or 'c' not in candle_data or not candle_data['c']:
            print(f"Error: No valid candle data returned")
            print(f"API response: {candle_data}")
            return None
        
        closing_prices = candle_data['c']
        timestamps = candle_data['t']
        
        print(f"Retrieved {len(closing_prices)} days of price data")
        
        # Print price history for verification
        print("\nRecent price history:")
        for i in range(min(5, len(closing_prices))):
            date = datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d')
            print(f"  {date}: ${closing_prices[i]:.2f}")
        
        # Calculate price change percentage
        price_change = 0
        if len(closing_prices) > 1:
            first_price = closing_prices[0]
            last_price = closing_prices[-1]
            price_change = (last_price - first_price) / first_price
        
        print(f"\nPrice change over period: {price_change*100:.2f}%")
        
        # Convert price change to sentiment (-1 to 1 scale)
        # Map price change to sentiment: -20% to +20% maps to -0.6 to 0.6
        sentiment_score = max(min(price_change * 3, 0.6), -0.6)
        
        print(f"Derived sentiment score: {sentiment_score:.4f}")
        
        # Calculate volatility
        if len(closing_prices) > 1:
            price_changes = [abs((closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1])
                            for i in range(1, len(closing_prices))]
            volatility = sum(price_changes) / len(price_changes)
            print(f"Volatility (avg daily change): {volatility*100:.2f}%")
        else:
            volatility = 0
            print("Insufficient data to calculate volatility")
        
        return {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'price_change': price_change,
            'volatility': volatility,
            'data_points': len(closing_prices)
        }
        
    except Exception as e:
        print(f"Error analyzing price momentum: {str(e)}")
        return None

def main():
    """Run the sentiment analysis tests."""
    # Get API key from environment
    api_key = os.getenv('FINNHUB_API_KEY', '') or os.getenv('FINNHUB_KEY', '')
    
    if not api_key:
        print("ERROR: No Finnhub API key found. Please set FINNHUB_API_KEY in your .env file.")
        return
    
    print(f"Using Finnhub API key (length: {len(api_key)})")
    
    # Define test symbols
    test_symbols = ['BTC', 'ETH', 'XRP', 'SOL']
    
    results = {}
    
    for symbol in test_symbols:
        # Get sentiment from news
        news_sentiment = analyze_sentiment_from_crypto_news(api_key, symbol)
        results[symbol] = news_sentiment
        
        # Get price momentum as a secondary indicator
        price_momentum = analyze_price_momentum(api_key, symbol)
        if price_momentum:
            results[f"{symbol}_price"] = price_momentum
        
        # Avoid hitting rate limits
        time.sleep(1)
    
    # Print summary
    print("\n\n===== SENTIMENT ANALYSIS SUMMARY =====")
    for symbol in test_symbols:
        sentiment = results.get(symbol, {}).get('sentiment_score', 0)
        sentiment_str = "NEUTRAL"
        if sentiment > 0.1:
            sentiment_str = "BULLISH"
        elif sentiment < -0.1:
            sentiment_str = "BEARISH"
            
        news_count = results.get(symbol, {}).get('news_count', 0)
        buzz = results.get(symbol, {}).get('buzz_score', 0)
        
        price_sentiment = results.get(f"{symbol}_price", {}).get('sentiment_score', 0)
        price_change = results.get(f"{symbol}_price", {}).get('price_change', 0)
        
        print(f"=== {symbol} ===")
        print(f"  News Sentiment: {sentiment:.4f} ({sentiment_str})")
        print(f"  News Count: {news_count}")
        print(f"  Buzz Score: {buzz:.4f}")
        print(f"  Price Change: {price_change*100 if price_change else 0:.2f}%")
        print(f"  Price Sentiment: {price_sentiment:.4f}")
        
        # Calculate combined sentiment
        if news_count > 0 and price_change is not None:
            # Weight: 60% news, 40% price if we have news, otherwise 100% price
            combined = sentiment * 0.6 + price_sentiment * 0.4
            print(f"  Combined Sentiment: {combined:.4f}")
        elif price_change is not None:
            print(f"  Combined Sentiment: {price_sentiment:.4f} (price only)")
        else:
            print(f"  Combined Sentiment: {sentiment:.4f} (news only)")
        
        print()

if __name__ == "__main__":
    main() 