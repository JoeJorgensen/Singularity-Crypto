"""
WhaleTracker - Track large cryptocurrency transactions and whale activity.
"""
from typing import Dict, List, Any, Optional
import os
import time
import json
import random
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WhaleTracker:
    """
    Track large cryptocurrency transactions and whale activity.
    This is a simulated implementation, as real on-chain data would require
    integration with blockchain APIs and data providers.
    """
    
    def __init__(self):
        """Initialize whale activity tracker."""
        self.cache = {}
        self.cache_expiry = 1800  # 30 minutes
        self.whale_thresholds = {
            'BTC': 50,       # 50+ BTC is considered a whale transaction
            'ETH': 500,      # 500+ ETH is considered a whale transaction
            'SOL': 10000,    # 10,000+ SOL is considered a whale transaction
            'XRP': 100000,   # 100,000+ XRP is considered a whale transaction
            'default': 100000  # Default value in USD
        }
        self.simulated_data = self._initialize_simulated_data()
    
    def _initialize_simulated_data(self) -> Dict[str, Any]:
        """
        Initialize simulated whale activity data for common cryptocurrencies.
        
        Returns:
            Dictionary with simulated data
        """
        now = datetime.now()
        
        # Create mock data for top coins
        return {
            'BTC': {
                'whale_transactions': [
                    {
                        'timestamp': (now - timedelta(hours=22)).isoformat(),
                        'amount': 125,
                        'value_usd': 7500000,
                        'from_exchange': False,
                        'to_exchange': False,
                        'transaction_type': 'transfer'
                    },
                    {
                        'timestamp': (now - timedelta(hours=16)).isoformat(),
                        'amount': 85,
                        'value_usd': 5100000,
                        'from_exchange': True,
                        'to_exchange': False,
                        'transaction_type': 'withdrawal'
                    },
                    {
                        'timestamp': (now - timedelta(hours=8)).isoformat(),
                        'amount': 200,
                        'value_usd': 12000000,
                        'from_exchange': False,
                        'to_exchange': False,
                        'transaction_type': 'transfer'
                    },
                    {
                        'timestamp': (now - timedelta(hours=3)).isoformat(),
                        'amount': 75,
                        'value_usd': 4500000,
                        'from_exchange': False,
                        'to_exchange': True,
                        'transaction_type': 'deposit'
                    }
                ],
                'total_whale_outflows_24h': 210,
                'total_whale_inflows_24h': 75,
                'unique_active_whales': 18,
                'whale_sentiment': 'accumulation'  # More outflows than inflows
            },
            'ETH': {
                'whale_transactions': [
                    {
                        'timestamp': (now - timedelta(hours=23)).isoformat(),
                        'amount': 1200,
                        'value_usd': 3600000,
                        'from_exchange': True,
                        'to_exchange': False,
                        'transaction_type': 'withdrawal'
                    },
                    {
                        'timestamp': (now - timedelta(hours=18)).isoformat(),
                        'amount': 850,
                        'value_usd': 2550000,
                        'from_exchange': False,
                        'to_exchange': False,
                        'transaction_type': 'transfer'
                    },
                    {
                        'timestamp': (now - timedelta(hours=10)).isoformat(),
                        'amount': 1500,
                        'value_usd': 4500000,
                        'from_exchange': False,
                        'to_exchange': True,
                        'transaction_type': 'deposit'
                    },
                    {
                        'timestamp': (now - timedelta(hours=4)).isoformat(),
                        'amount': 2000,
                        'value_usd': 6000000,
                        'from_exchange': False,
                        'to_exchange': True,
                        'transaction_type': 'deposit'
                    }
                ],
                'total_whale_outflows_24h': 2050,
                'total_whale_inflows_24h': 3500,
                'unique_active_whales': 12,
                'whale_sentiment': 'distribution'  # More inflows than outflows (to exchanges)
            },
            'SOL': {
                'whale_transactions': [
                    {
                        'timestamp': (now - timedelta(hours=21)).isoformat(),
                        'amount': 35000,
                        'value_usd': 3150000,
                        'from_exchange': True,
                        'to_exchange': False,
                        'transaction_type': 'withdrawal'
                    },
                    {
                        'timestamp': (now - timedelta(hours=15)).isoformat(),
                        'amount': 22000,
                        'value_usd': 1980000,
                        'from_exchange': False,
                        'to_exchange': False,
                        'transaction_type': 'transfer'
                    },
                    {
                        'timestamp': (now - timedelta(hours=12)).isoformat(),
                        'amount': 18000,
                        'value_usd': 1620000,
                        'from_exchange': False,
                        'to_exchange': True,
                        'transaction_type': 'deposit'
                    },
                    {
                        'timestamp': (now - timedelta(hours=5)).isoformat(),
                        'amount': 40000,
                        'value_usd': 3600000,
                        'from_exchange': True,
                        'to_exchange': False,
                        'transaction_type': 'withdrawal'
                    }
                ],
                'total_whale_outflows_24h': 75000,
                'total_whale_inflows_24h': 18000,
                'unique_active_whales': 9,
                'whale_sentiment': 'accumulation'  # More outflows than inflows
            }
        }
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze whale activity for a specific cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            
        Returns:
            Dictionary with whale activity analysis
        """
        # Clean symbol
        symbol = symbol.upper()
        if '/' in symbol:
            symbol = symbol.split('/')[0]
        
        # Check cache
        cache_key = f"whale_activity:{symbol}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['data']
        
        # Prepare result
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'whale_transactions_24h': [],
            'total_whale_outflows_24h': 0,
            'total_whale_inflows_24h': 0,
            'net_whale_flow': 0,
            'whale_activity_level': 'low',
            'whale_sentiment': 'neutral',
            'activity_score': 0  # -1 to 1 scale
        }
        
        try:
            # Use simulated data for common cryptocurrencies
            if symbol in self.simulated_data:
                data = self.simulated_data[symbol]
                
                # Calculate net whale flow (negative = more outflows = accumulation)
                outflows = data['total_whale_outflows_24h']
                inflows = data['total_whale_inflows_24h']
                net_flow = inflows - outflows
                
                # Normalize to -1 to 1 scale
                max_flow = max(inflows, outflows)
                if max_flow > 0:
                    activity_score = -1 * net_flow / max_flow  # Negative to maintain: negative = accumulation = bullish
                else:
                    activity_score = 0
                
                # Determine activity level
                activity_level = 'medium'
                if len(data['whale_transactions']) > 5:
                    activity_level = 'high'
                elif len(data['whale_transactions']) < 3:
                    activity_level = 'low'
                
                # Update result
                result['whale_transactions_24h'] = data['whale_transactions']
                result['total_whale_outflows_24h'] = outflows
                result['total_whale_inflows_24h'] = inflows
                result['net_whale_flow'] = net_flow
                result['whale_activity_level'] = activity_level
                result['whale_sentiment'] = data['whale_sentiment']
                result['activity_score'] = activity_score
            else:
                # Generate random data for other cryptocurrencies
                transactions = []
                outflows = 0
                inflows = 0
                
                # Generate random transactions
                for _ in range(random.randint(1, 5)):
                    amount = random.randint(50, 500)
                    is_inflow = random.choice([True, False])
                    
                    transaction = {
                        'timestamp': (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                        'amount': amount,
                        'value_usd': amount * 1000,  # Simplified calculation
                        'from_exchange': is_inflow,
                        'to_exchange': not is_inflow,
                        'transaction_type': 'deposit' if is_inflow else 'withdrawal'
                    }
                    
                    transactions.append(transaction)
                    
                    if is_inflow:
                        inflows += amount
                    else:
                        outflows += amount
                
                # Calculate net flow and activity score
                net_flow = inflows - outflows
                max_flow = max(inflows, outflows)
                if max_flow > 0:
                    activity_score = -1 * net_flow / max_flow
                else:
                    activity_score = 0
                
                # Determine whale sentiment
                sentiment = 'neutral'
                if activity_score > 0.3:
                    sentiment = 'accumulation'
                elif activity_score < -0.3:
                    sentiment = 'distribution'
                
                # Determine activity level
                activity_level = 'medium'
                if len(transactions) > 3:
                    activity_level = 'high'
                elif len(transactions) < 2:
                    activity_level = 'low'
                
                # Update result
                result['whale_transactions_24h'] = transactions
                result['total_whale_outflows_24h'] = outflows
                result['total_whale_inflows_24h'] = inflows
                result['net_whale_flow'] = net_flow
                result['whale_activity_level'] = activity_level
                result['whale_sentiment'] = sentiment
                result['activity_score'] = activity_score
        
        except Exception as e:
            print(f"Error analyzing whale activity: {e}")
        
        # Update cache
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'data': result
        }
        
        return result
    
    def get_recent_transactions(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get recent whale transactions for a specific cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            limit: Maximum number of transactions to return
            
        Returns:
            List of recent whale transactions
        """
        # Clean symbol
        symbol = symbol.upper()
        if '/' in symbol:
            symbol = symbol.split('/')[0]
        
        transactions = []
        
        try:
            # Use simulated data for common cryptocurrencies
            if symbol in self.simulated_data:
                data = self.simulated_data[symbol]
                transactions = data['whale_transactions'][:limit]
            else:
                # Generate random transactions
                for _ in range(min(limit, random.randint(1, 5))):
                    amount = random.randint(50, 500)
                    is_inflow = random.choice([True, False])
                    
                    transaction = {
                        'timestamp': (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                        'amount': amount,
                        'value_usd': amount * 1000,  # Simplified calculation
                        'from_exchange': is_inflow,
                        'to_exchange': not is_inflow,
                        'transaction_type': 'deposit' if is_inflow else 'withdrawal'
                    }
                    
                    transactions.append(transaction)
        
        except Exception as e:
            print(f"Error getting recent whale transactions: {e}")
        
        return transactions
    
    def get_whale_trend(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get whale activity trend for a specific cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            days: Number of days for trend analysis
            
        Returns:
            Dictionary with whale activity trend analysis
        """
        # Clean symbol
        symbol = symbol.upper()
        if '/' in symbol:
            symbol = symbol.split('/')[0]
        
        result = {
            'symbol': symbol,
            'days_analyzed': days,
            'overall_trend': 'neutral',
            'daily_stats': [],
            'whale_count_trend': 'stable',
            'accumulation_days': 0,
            'distribution_days': 0,
            'neutral_days': 0
        }
        
        try:
            now = datetime.now()
            
            # Generate daily stats
            accumulation_days = 0
            distribution_days = 0
            neutral_days = 0
            
            for i in range(days):
                date = now - timedelta(days=i)
                
                # Randomly determine daily sentiment
                random_value = random.uniform(-1, 1)
                
                # For common cryptos, bias the sentiment based on their overall simulated trend
                if symbol in self.simulated_data:
                    if self.simulated_data[symbol]['whale_sentiment'] == 'accumulation':
                        random_value = random.uniform(0, 1)  # Bias towards accumulation
                    elif self.simulated_data[symbol]['whale_sentiment'] == 'distribution':
                        random_value = random.uniform(-1, 0)  # Bias towards distribution
                
                sentiment = 'neutral'
                if random_value > 0.3:
                    sentiment = 'accumulation'
                    accumulation_days += 1
                elif random_value < -0.3:
                    sentiment = 'distribution'
                    distribution_days += 1
                else:
                    neutral_days += 1
                
                # Create daily stats
                daily_stat = {
                    'date': date.strftime('%Y-%m-%d'),
                    'whale_transactions': random.randint(1, 10),
                    'unique_whales': random.randint(5, 20),
                    'net_flow': random_value,
                    'sentiment': sentiment
                }
                
                result['daily_stats'].append(daily_stat)
            
            # Determine overall trend
            if accumulation_days > distribution_days and accumulation_days > neutral_days:
                result['overall_trend'] = 'accumulation'
            elif distribution_days > accumulation_days and distribution_days > neutral_days:
                result['overall_trend'] = 'distribution'
            else:
                result['overall_trend'] = 'neutral'
            
            # Update result
            result['accumulation_days'] = accumulation_days
            result['distribution_days'] = distribution_days
            result['neutral_days'] = neutral_days
            
            # Determine whale count trend
            first_day_whales = result['daily_stats'][-1]['unique_whales']
            last_day_whales = result['daily_stats'][0]['unique_whales']
            
            if last_day_whales > first_day_whales * 1.2:
                result['whale_count_trend'] = 'increasing'
            elif last_day_whales < first_day_whales * 0.8:
                result['whale_count_trend'] = 'decreasing'
            else:
                result['whale_count_trend'] = 'stable'
        
        except Exception as e:
            print(f"Error analyzing whale activity trend: {e}")
        
        return result 