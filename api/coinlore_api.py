"""
CoinloreAPI - Wrapper for Coinlore API for basic on-chain cryptocurrency metrics.
"""
import os
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CoinloreAPI:
    """
    Wrapper for Coinlore API, providing on-chain metrics for cryptocurrencies.
    """
    
    def __init__(self):
        """
        Initialize CoinloreAPI.
        """
        self.base_url = 'https://api.coinlore.net/api'
        
        # Rate limiting settings
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        # Cache for coin data
        self.coin_list_cache = None
        self.coin_list_cache_time = 0
        self.cache_expiry = 3600  # 1 hour cache expiry
        
        # Async rate limiting
        self.request_lock = asyncio.Lock()
    
    async def _make_request_async(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make an asynchronous request to the Coinlore API with rate limiting.
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters
            
        Returns:
            Response as a dictionary or list
        """
        # Apply rate limiting using asyncio lock
        async with self.request_lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last_request)
            
            # Update last request time
            self.last_request_time = time.time()
        
        # Make request
        try:
            url = f"{self.base_url}/{endpoint}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"Coinlore API error: {response.status}")
                        return {}
        except Exception as e:
            print(f"Coinlore API async request error: {e}")
            return {}
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make a request to the Coinlore API with rate limiting.
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters
            
        Returns:
            Response as a dictionary or list
        """
        # Apply rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        
        # Update last request time
        self.last_request_time = time.time()
        
        # Make request
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Coinlore API request error: {e}")
            return {}
    
    async def get_coin_list_async(self, refresh: bool = False) -> List[Dict]:
        """
        Get list of all coins with their IDs asynchronously.
        
        Args:
            refresh: Force refresh the cache
            
        Returns:
            List of coin dictionaries
        """
        current_time = time.time()
        
        # Check if we can use the cache
        if (not refresh and 
            self.coin_list_cache is not None and 
            current_time - self.coin_list_cache_time <= self.cache_expiry):
            return self.coin_list_cache
            
        # Get all coins (paginated)
        all_coins = []
        start = 0
        limit = 100
        
        while True:
            coins = await self._make_request_async('tickers', {'start': start, 'limit': limit})
            
            if not coins or not isinstance(coins, list) or len(coins) == 0:
                break
            
            all_coins.extend(coins)
            
            if len(coins) < limit:
                break
            
            start += limit
        
        # Update cache
        self.coin_list_cache = all_coins
        self.coin_list_cache_time = current_time
        
        return all_coins
    
    def get_coin_list(self, refresh: bool = False) -> List[Dict]:
        """
        Get list of all coins with their IDs.
        
        Args:
            refresh: Force refresh the cache
            
        Returns:
            List of coin dictionaries
        """
        current_time = time.time()
        
        # Check if we need to refresh the cache
        if (self.coin_list_cache is None or 
            refresh or 
            current_time - self.coin_list_cache_time > self.cache_expiry):
            
            # Get all coins (paginated)
            all_coins = []
            start = 0
            limit = 100
            
            while True:
                coins = self._make_request('tickers', {'start': start, 'limit': limit})
                
                if not coins or not isinstance(coins, list) or len(coins) == 0:
                    break
                
                all_coins.extend(coins)
                
                if len(coins) < limit:
                    break
                
                start += limit
            
            # Update cache
            self.coin_list_cache = all_coins
            self.coin_list_cache_time = current_time
        
        return self.coin_list_cache
    
    async def get_coin_id_async(self, symbol: str) -> Optional[str]:
        """
        Get coin ID for a symbol asynchronously.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Coin ID or None if not found
        """
        symbol = symbol.upper()
        coins = await self.get_coin_list_async()
        
        for coin in coins:
            if coin.get('symbol') == symbol:
                return coin.get('id')
        
        return None
    
    def get_coin_id(self, symbol: str) -> Optional[str]:
        """
        Get coin ID for a symbol.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Coin ID or None if not found
        """
        symbol = symbol.upper()
        coins = self.get_coin_list()
        
        for coin in coins:
            if coin.get('symbol') == symbol:
                return coin.get('id')
        
        return None
    
    async def get_coin_info_async(self, symbol: str) -> Dict:
        """
        Get detailed information for a coin asynchronously.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Dictionary with coin information
        """
        coin_id = await self.get_coin_id_async(symbol)
        
        if not coin_id:
            return {
                'symbol': symbol,
                'error': 'Coin not found',
                'timestamp': datetime.now().isoformat()
            }
        
        # Get coin info
        coins = await self._make_request_async('ticker', {'id': coin_id})
        
        if isinstance(coins, list) and len(coins) > 0:
            coin_info = coins[0]
            return {
                'symbol': symbol,
                'name': coin_info.get('name', ''),
                'rank': int(coin_info.get('rank', 0)),
                'price_usd': float(coin_info.get('price_usd', 0)),
                'market_cap_usd': float(coin_info.get('market_cap_usd', 0)),
                'volume_24h_usd': float(coin_info.get('volume24', 0)),
                'percent_change_24h': float(coin_info.get('percent_change_24h', 0)),
                'percent_change_7d': float(coin_info.get('percent_change_7d', 0)),
                'percent_change_1h': float(coin_info.get('percent_change_1h', 0)),
                'total_supply': float(coin_info.get('tsupply', 0)),
                'circulating_supply': float(coin_info.get('csupply', 0)),
                'timestamp': datetime.now().isoformat()
            }
        
        return {
            'symbol': symbol,
            'error': 'Could not retrieve coin info',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_coin_info(self, symbol: str) -> Dict:
        """
        Get detailed information for a coin.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Dictionary with coin information
        """
        coin_id = self.get_coin_id(symbol)
        
        if not coin_id:
            return {
                'symbol': symbol,
                'error': 'Coin not found',
                'timestamp': datetime.now().isoformat()
            }
        
        # Get coin info
        coins = self._make_request(f'ticker', {'id': coin_id})
        
        if isinstance(coins, list) and len(coins) > 0:
            coin_info = coins[0]
            return {
                'symbol': symbol,
                'name': coin_info.get('name', ''),
                'rank': int(coin_info.get('rank', 0)),
                'price_usd': float(coin_info.get('price_usd', 0)),
                'market_cap_usd': float(coin_info.get('market_cap_usd', 0)),
                'volume_24h_usd': float(coin_info.get('volume24', 0)),
                'percent_change_24h': float(coin_info.get('percent_change_24h', 0)),
                'percent_change_7d': float(coin_info.get('percent_change_7d', 0)),
                'percent_change_1h': float(coin_info.get('percent_change_1h', 0)),
                'total_supply': float(coin_info.get('tsupply', 0)),
                'circulating_supply': float(coin_info.get('csupply', 0)),
                'timestamp': datetime.now().isoformat()
            }
        
        return {
            'symbol': symbol,
            'error': 'Could not retrieve coin info',
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_global_metrics_async(self) -> Dict:
        """
        Get global cryptocurrency market metrics asynchronously.
        
        Returns:
            Dictionary with global market metrics
        """
        global_data = await self._make_request_async('global')
        
        if global_data and isinstance(global_data, list) and len(global_data) > 0:
            data = global_data[0]
            return {
                'total_market_cap_usd': float(data.get('total_mcap', 0)),
                'total_volume_24h_usd': float(data.get('total_volume', 0)),
                'btc_dominance': float(data.get('btc_d', 0)),
                'eth_dominance': float(data.get('eth_d', 0)),
                'active_markets': int(data.get('active_markets', 0)),
                'active_cryptocurrencies': int(data.get('active_currencies', 0)),
                'timestamp': datetime.now().isoformat()
            }
        
        return {
            'error': 'Could not retrieve global metrics',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_global_metrics(self) -> Dict:
        """
        Get global cryptocurrency market metrics.
        
        Returns:
            Dictionary with global market metrics
        """
        global_data = self._make_request('global')
        
        if global_data and isinstance(global_data, list) and len(global_data) > 0:
            data = global_data[0]
            return {
                'total_market_cap_usd': float(data.get('total_mcap', 0)),
                'total_volume_24h_usd': float(data.get('total_volume', 0)),
                'btc_dominance': float(data.get('btc_d', 0)),
                'eth_dominance': float(data.get('eth_d', 0)),
                'active_markets': int(data.get('active_markets', 0)),
                'active_cryptocurrencies': int(data.get('active_currencies', 0)),
                'timestamp': datetime.now().isoformat()
            }
        
        return {
            'error': 'Could not retrieve global metrics',
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_on_chain_metrics_async(self, symbol: str) -> Dict:
        """
        Get on-chain metrics for a cryptocurrency asynchronously.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Dictionary with on-chain metrics
        """
        # Get basic coin info first
        coin_info = await self.get_coin_info_async(symbol)
        
        # Extract relevant metrics
        return {
            'symbol': symbol,
            'price_usd': coin_info.get('price_usd', 0),
            'market_cap_usd': coin_info.get('market_cap_usd', 0),
            'volume_24h_usd': coin_info.get('volume_24h_usd', 0),
            'percent_change_1h': coin_info.get('percent_change_1h', 0),
            'percent_change_24h': coin_info.get('percent_change_24h', 0),
            'percent_change_7d': coin_info.get('percent_change_7d', 0),
            'circulating_supply': coin_info.get('circulating_supply', 0),
            'total_supply': coin_info.get('total_supply', 0),
            'supply_ratio': (
                coin_info.get('circulating_supply', 0) / coin_info.get('total_supply', 1)
                if coin_info.get('total_supply', 0) > 0 else 0
            ),
            'rank': coin_info.get('rank', 0),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_on_chain_metrics(self, symbol: str) -> Dict:
        """
        Get on-chain metrics for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Dictionary with on-chain metrics
        """
        # Get basic coin info first
        coin_info = self.get_coin_info(symbol)
        
        # Extract relevant metrics
        return {
            'symbol': symbol,
            'price_usd': coin_info.get('price_usd', 0),
            'market_cap_usd': coin_info.get('market_cap_usd', 0),
            'volume_24h_usd': coin_info.get('volume_24h_usd', 0),
            'percent_change_1h': coin_info.get('percent_change_1h', 0),
            'percent_change_24h': coin_info.get('percent_change_24h', 0),
            'percent_change_7d': coin_info.get('percent_change_7d', 0),
            'circulating_supply': coin_info.get('circulating_supply', 0),
            'total_supply': coin_info.get('total_supply', 0),
            'supply_ratio': (
                coin_info.get('circulating_supply', 0) / coin_info.get('total_supply', 1)
                if coin_info.get('total_supply', 0) > 0 else 0
            ),
            'rank': coin_info.get('rank', 0),
            'timestamp': datetime.now().isoformat()
        }