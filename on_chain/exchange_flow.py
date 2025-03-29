"""
ExchangeFlow - Analyze cryptocurrency exchange inflows and outflows.
"""
from typing import Dict, List, Any, Optional
import os
import time
import json
from datetime import datetime, timedelta
import random
import pandas as pd

class ExchangeFlow:
    """
    Analyze cryptocurrency exchange inflows and outflows.
    This is a simulated implementation, as real on-chain data would require
    integration with services like Glassnode, Nansen, etc.
    """
    
    def __init__(self):
        """Initialize exchange flow analyzer."""
        self.cache = {}
        self.cache_expiry = 3600  # 1 hour
        self.simulated_data = self._initialize_simulated_data()
    
    def _initialize_simulated_data(self) -> Dict[str, Any]:
        """
        Initialize simulated exchange flow data for common cryptocurrencies.
        
        Returns:
            Dictionary with simulated data
        """
        now = datetime.now()
        
        # Create mock data for BTC, ETH, and a few other top coins
        return {
            'BTC': {
                'exchange_inflow': [
                    {'timestamp': (now - timedelta(hours=24)).isoformat(), 'value': 1200},
                    {'timestamp': (now - timedelta(hours=18)).isoformat(), 'value': 980},
                    {'timestamp': (now - timedelta(hours=12)).isoformat(), 'value': 1100},
                    {'timestamp': (now - timedelta(hours=6)).isoformat(), 'value': 1300},
                    {'timestamp': now.isoformat(), 'value': 1150}
                ],
                'exchange_outflow': [
                    {'timestamp': (now - timedelta(hours=24)).isoformat(), 'value': 1100},
                    {'timestamp': (now - timedelta(hours=18)).isoformat(), 'value': 1050},
                    {'timestamp': (now - timedelta(hours=12)).isoformat(), 'value': 1180},
                    {'timestamp': (now - timedelta(hours=6)).isoformat(), 'value': 1250},
                    {'timestamp': now.isoformat(), 'value': 1350}
                ],
                'net_flow_trend': 'accumulation',  # more outflow than inflow = accumulation
                'large_transactions': [
                    {'timestamp': (now - timedelta(hours=10)).isoformat(), 'value': 120, 'type': 'outflow'},
                    {'timestamp': (now - timedelta(hours=8)).isoformat(), 'value': 85, 'type': 'inflow'},
                    {'timestamp': (now - timedelta(hours=3)).isoformat(), 'value': 150, 'type': 'outflow'}
                ]
            },
            'ETH': {
                'exchange_inflow': [
                    {'timestamp': (now - timedelta(hours=24)).isoformat(), 'value': 12500},
                    {'timestamp': (now - timedelta(hours=18)).isoformat(), 'value': 11800},
                    {'timestamp': (now - timedelta(hours=12)).isoformat(), 'value': 13000},
                    {'timestamp': (now - timedelta(hours=6)).isoformat(), 'value': 12200},
                    {'timestamp': now.isoformat(), 'value': 11500}
                ],
                'exchange_outflow': [
                    {'timestamp': (now - timedelta(hours=24)).isoformat(), 'value': 11800},
                    {'timestamp': (now - timedelta(hours=18)).isoformat(), 'value': 12200},
                    {'timestamp': (now - timedelta(hours=12)).isoformat(), 'value': 12700},
                    {'timestamp': (now - timedelta(hours=6)).isoformat(), 'value': 12000},
                    {'timestamp': now.isoformat(), 'value': 11800}
                ],
                'net_flow_trend': 'neutral',  # roughly equal inflow and outflow
                'large_transactions': [
                    {'timestamp': (now - timedelta(hours=16)).isoformat(), 'value': 850, 'type': 'inflow'},
                    {'timestamp': (now - timedelta(hours=12)).isoformat(), 'value': 920, 'type': 'outflow'},
                    {'timestamp': (now - timedelta(hours=4)).isoformat(), 'value': 780, 'type': 'inflow'}
                ]
            },
            'SOL': {
                'exchange_inflow': [
                    {'timestamp': (now - timedelta(hours=24)).isoformat(), 'value': 62000},
                    {'timestamp': (now - timedelta(hours=18)).isoformat(), 'value': 58000},
                    {'timestamp': (now - timedelta(hours=12)).isoformat(), 'value': 65000},
                    {'timestamp': (now - timedelta(hours=6)).isoformat(), 'value': 72000},
                    {'timestamp': now.isoformat(), 'value': 68000}
                ],
                'exchange_outflow': [
                    {'timestamp': (now - timedelta(hours=24)).isoformat(), 'value': 55000},
                    {'timestamp': (now - timedelta(hours=18)).isoformat(), 'value': 52000},
                    {'timestamp': (now - timedelta(hours=12)).isoformat(), 'value': 48000},
                    {'timestamp': (now - timedelta(hours=6)).isoformat(), 'value': 51000},
                    {'timestamp': now.isoformat(), 'value': 49000}
                ],
                'net_flow_trend': 'distribution',  # more inflow than outflow = distribution
                'large_transactions': [
                    {'timestamp': (now - timedelta(hours=18)).isoformat(), 'value': 15000, 'type': 'inflow'},
                    {'timestamp': (now - timedelta(hours=8)).isoformat(), 'value': 12000, 'type': 'inflow'},
                    {'timestamp': (now - timedelta(hours=2)).isoformat(), 'value': 8000, 'type': 'outflow'}
                ]
            }
        }
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze exchange flows for a specific cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            
        Returns:
            Dictionary with exchange flow analysis
        """
        # Clean symbol
        symbol = symbol.upper()
        if '/' in symbol:
            symbol = symbol.split('/')[0]
        
        # Check cache
        cache_key = f"exchange_flow:{symbol}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['data']
        
        # Prepare result
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'inflow_24h': 0,
            'outflow_24h': 0,
            'net_flow': 0,
            'net_flow_trend': 'neutral',
            'large_transactions': [],
            'exchange_balance_change_24h': 0
        }
        
        try:
            # Use simulated data for common cryptocurrencies
            if symbol in self.simulated_data:
                data = self.simulated_data[symbol]
                
                # Calculate 24h inflow and outflow
                inflow_24h = data['exchange_inflow'][-1]['value']
                outflow_24h = data['exchange_outflow'][-1]['value']
                
                # Calculate net flow (negative = more outflow than inflow)
                net_flow = (inflow_24h - outflow_24h) / max(inflow_24h, outflow_24h)  # Normalized between -1 and 1
                
                # Get exchange balance change
                exchange_balance_24h = (
                    data['exchange_inflow'][-1]['value'] - data['exchange_inflow'][0]['value']
                ) - (
                    data['exchange_outflow'][-1]['value'] - data['exchange_outflow'][0]['value']
                )
                
                # Update result
                result['inflow_24h'] = inflow_24h
                result['outflow_24h'] = outflow_24h
                result['net_flow'] = net_flow
                result['net_flow_trend'] = data['net_flow_trend']
                result['large_transactions'] = data['large_transactions']
                result['exchange_balance_change_24h'] = exchange_balance_24h
            else:
                # Generate random data for other cryptocurrencies
                inflow = random.randint(1000, 10000)
                outflow = random.randint(1000, 10000)
                net_flow = (inflow - outflow) / max(inflow, outflow)  # Normalized between -1 and 1
                
                # Determine trend
                trend = 'neutral'
                if net_flow > 0.1:
                    trend = 'distribution'  # More coins moving to exchanges (potentially bearish)
                elif net_flow < -0.1:
                    trend = 'accumulation'  # More coins moving out of exchanges (potentially bullish)
                
                # Generate random large transactions
                large_transactions = []
                for _ in range(random.randint(1, 3)):
                    large_transactions.append({
                        'timestamp': (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                        'value': random.randint(100, 1000),
                        'type': random.choice(['inflow', 'outflow'])
                    })
                
                # Update result
                result['inflow_24h'] = inflow
                result['outflow_24h'] = outflow
                result['net_flow'] = net_flow
                result['net_flow_trend'] = trend
                result['large_transactions'] = large_transactions
                result['exchange_balance_change_24h'] = inflow - outflow
        
        except Exception as e:
            print(f"Error analyzing exchange flows: {e}")
        
        # Update cache
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'data': result
        }
        
        return result
    
    def get_historical_flows(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """
        Get historical exchange flows for a specific cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical flow data
        """
        # Clean symbol
        symbol = symbol.upper()
        if '/' in symbol:
            symbol = symbol.split('/')[0]
        
        # Prepare data
        data = {
            'timestamp': [],
            'inflow': [],
            'outflow': [],
            'net_flow': []
        }
        
        try:
            # Generate historical data
            now = datetime.now()
            
            if symbol in self.simulated_data:
                # Use available simulated data and extend it
                sim_data = self.simulated_data[symbol]
                inflow_base = sim_data['exchange_inflow'][-1]['value']
                outflow_base = sim_data['exchange_outflow'][-1]['value']
                
                # Add real data points
                for i, inflow_point in enumerate(sim_data['exchange_inflow']):
                    outflow_point = sim_data['exchange_outflow'][i]
                    
                    data['timestamp'].append(datetime.fromisoformat(inflow_point['timestamp']))
                    data['inflow'].append(inflow_point['value'])
                    data['outflow'].append(outflow_point['value'])
                    data['net_flow'].append(outflow_point['value'] - inflow_point['value'])
                
                # Add simulated historical data
                for i in range(1, days * 24 - len(sim_data['exchange_inflow']) + 1):
                    timestamp = now - timedelta(hours=i + 24)
                    
                    # Add random variation to inflow/outflow
                    inflow = inflow_base * (1 + random.uniform(-0.15, 0.15))
                    outflow = outflow_base * (1 + random.uniform(-0.15, 0.15))
                    
                    data['timestamp'].append(timestamp)
                    data['inflow'].append(inflow)
                    data['outflow'].append(outflow)
                    data['net_flow'].append(outflow - inflow)
            else:
                # Generate completely random data
                inflow_base = random.randint(1000, 10000)
                outflow_base = random.randint(1000, 10000)
                
                for i in range(days * 24):
                    timestamp = now - timedelta(hours=i)
                    
                    # Add random variation to inflow/outflow
                    inflow = inflow_base * (1 + random.uniform(-0.2, 0.2))
                    outflow = outflow_base * (1 + random.uniform(-0.2, 0.2))
                    
                    data['timestamp'].append(timestamp)
                    data['inflow'].append(inflow)
                    data['outflow'].append(outflow)
                    data['net_flow'].append(outflow - inflow)
        
        except Exception as e:
            print(f"Error getting historical exchange flows: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df.sort_values('timestamp', inplace=True)
        
        return df 