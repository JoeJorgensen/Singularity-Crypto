"""
Whale tracker for monitoring large cryptocurrency transactions
"""
import logging
import time
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from .alchemy.client import AlchemyClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Whale thresholds (in USD equivalent)
WHALE_THRESHOLDS = {
    "eth": 100000,  # $100k+ for ETH
    "btc": 250000,  # $250k+ for BTC
    "usdt": 500000,  # $500k+ for stablecoins
    "usdc": 500000,
    "link": 50000,  # $50k+ for alts
    "uni": 50000,
    "aave": 50000,
}

class WhaleTracker:
    """
    Tracks and analyzes large cryptocurrency transactions
    """
    
    def __init__(self):
        """Initialize the whale tracker"""
        self.alchemy_client = AlchemyClient()
        self.whale_txs_cache = {}
        self.cache_timestamp = 0
        self.cache_duration = 600  # 10 minutes cache
        self.price_cache = {}
        
    def get_token_price(self, token_symbol: str) -> float:
        """
        Get the current price of a token in USD
        
        This is a simplified implementation. In production, you'd use a proper price feed.
        
        Args:
            token_symbol: Symbol of the token
            
        Returns:
            float: Token price in USD
        """
        # Check cache first
        if token_symbol in self.price_cache and (time.time() - self.price_cache[token_symbol]["timestamp"] < 3600):
            return self.price_cache[token_symbol]["price"]
        
        # Simple placeholder prices - in production, use a real price API
        current_prices = {
            "eth": 2000.0,
            "btc": 35000.0,
            "usdt": 1.0,
            "usdc": 1.0,
            "link": 7.0,
            "uni": 5.0,
            "aave": 80.0,
        }
        
        price = current_prices.get(token_symbol.lower(), 0)
        
        # Update cache
        self.price_cache[token_symbol] = {
            "price": price,
            "timestamp": time.time()
        }
        
        return price
    
    def track_whale_transactions(self, token_symbol: str, hours: int = 24) -> Dict[str, Any]:
        """
        Track large transactions for a specific token
        
        Args:
            token_symbol: Symbol of the token to track (e.g., 'ETH', 'BTC')
            hours: Number of hours to look back
            
        Returns:
            Dict containing whale transaction metrics
        """
        # Use cache if available and not expired
        cache_key = f"{token_symbol}_{hours}"
        current_time = time.time()
        
        if cache_key in self.whale_txs_cache and (current_time - self.cache_timestamp < self.cache_duration):
            logger.debug(f"Using cached whale transaction data for {token_symbol}")
            return self.whale_txs_cache[cache_key]
        
        logger.info(f"Tracking whale transactions for {token_symbol} over past {hours} hours")
        
        # Get token price
        token_price = self.get_token_price(token_symbol)
        
        # Get whale threshold in token amount
        usd_threshold = WHALE_THRESHOLDS.get(token_symbol.lower(), 100000)
        token_threshold = usd_threshold / token_price if token_price > 0 else 0
        
        # Check if using test mode
        is_mock_data = os.getenv("ALCHEMY_TEST_MODE", "false").lower() == "true"
        
        # Calculate block range (approximate)
        # Assuming ~12 second block time for Ethereum
        blocks_per_hour = 3600 // 12
        current_block = self.alchemy_client.get_block_number()
        from_block = current_block - (blocks_per_hour * hours)
        
        # Configure the parameters based on token type
        if token_symbol.lower() == "eth":
            # For native ETH
            params = {
                "fromBlock": hex(from_block),
                "toBlock": "latest",
                "category": ["external"]
            }
        else:
            # For ERC-20 tokens
            token_contracts = self._get_token_contract(token_symbol)
            if not token_contracts:
                logger.warning(f"No contract address found for {token_symbol}")
                return {
                    "token": token_symbol,
                    "whale_threshold_usd": usd_threshold,
                    "whale_threshold_tokens": round(token_threshold, 6),
                    "whale_count": 0,
                    "whale_volume": 0,
                    "total_volume": 0,
                    "whale_volume_percent": 0,
                    "buy_volume": 0,
                    "sell_volume": 0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "transactions": [],
                    "timestamp": datetime.now().isoformat(),
                    "is_mock_data": is_mock_data
                }
                
            params = {
                "fromBlock": hex(from_block),
                "toBlock": "latest",
                "category": ["erc20"],
                "contractAddresses": token_contracts
            }
        
        # Get transfers data
        transfers = self.alchemy_client.get_asset_transfers(params)
        
        # Process the transfers to identify whale transactions
        whale_txs = []
        total_volume = 0
        whale_volume = 0
        
        for transfer in transfers.get("transfers", []):
            # Extract addresses with proper checks
            from_address = transfer.get("from", "")
            to_address = transfer.get("to", "")
            
            # Safely convert value to float
            try:
                value = float(transfer.get("value", 0))
            except (ValueError, TypeError):
                logger.debug(f"Skipping transfer with invalid value: {transfer}")
                continue
            
            # Skip if address is missing or value is zero/negative
            if not from_address or not to_address or value <= 0:
                logger.debug(f"Skipping transfer with missing address or zero/negative value")
                continue
                
            total_volume += value
            
            # Check if this is a whale transaction
            if value >= token_threshold:
                whale_volume += value
                
                # Format the transaction data
                whale_txs.append({
                    "hash": transfer.get("hash", ""),
                    "from": from_address,
                    "to": to_address,
                    "value": value,
                    "usd_value": round(value * token_price, 2),
                    "timestamp": transfer.get("metadata", {}).get("blockTimestamp", "")
                })
        
        # Sort whale transactions by value (descending)
        whale_txs.sort(key=lambda x: x["value"], reverse=True)
        
        # Calculate metrics
        whale_count = len(whale_txs)
        
        # Initialize direction metrics
        buy_volume = 0
        sell_volume = 0
        buy_count = 0
        sell_count = 0
        
        # Simplified heuristic: transfers to exchanges are considered sells
        from .exchange_flow import EXCHANGE_WALLETS
        exchange_addresses = {addr.lower() for addr in EXCHANGE_WALLETS.keys()}
        
        for tx in whale_txs:
            to_address = tx["to"].lower() if tx["to"] else ""
            from_address = tx["from"].lower() if tx["from"] else ""
            
            if to_address in exchange_addresses:
                # Treating as a sell (retail selling to exchanges)
                sell_volume += tx["value"]
                sell_count += 1
            elif from_address in exchange_addresses:
                # Treating as a buy (retail buying from exchanges)
                buy_volume += tx["value"]
                buy_count += 1
            else:
                # Wallet-to-wallet transfer
                # Could be categorized further with more sophisticated analysis
                pass
        
        # Compute whale metrics
        result = {
            "token": token_symbol,
            "whale_threshold_usd": usd_threshold,
            "whale_threshold_tokens": round(token_threshold, 6),
            "whale_count": whale_count,
            "whale_volume": round(whale_volume, 4),
            "total_volume": round(total_volume, 4),
            "whale_volume_percent": round(whale_volume / total_volume * 100 if total_volume > 0 else 0, 2),
            "buy_volume": round(buy_volume, 4),
            "sell_volume": round(sell_volume, 4),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "transactions": whale_txs[:10],  # Limit to top 10 to avoid excessive data
            "timestamp": datetime.now().isoformat(),
            "is_mock_data": is_mock_data
        }
        
        # Update cache
        self.whale_txs_cache[cache_key] = result
        self.cache_timestamp = current_time
        
        logger.debug(f"Completed whale transaction analysis for {token_symbol}")
        return result
    
    def get_whale_signals(self, token_symbol: str, hours: int = 24) -> Dict[str, float]:
        """
        Generate trading signals based on whale activity
        
        Args:
            token_symbol: Symbol of the token to analyze
            hours: Number of hours to look back
            
        Returns:
            Dict of signal values between -1.0 and 1.0
        """
        whale_data = self.track_whale_transactions(token_symbol, hours)
        
        # No data case
        if whale_data["whale_count"] == 0:
            return {
                "whale_activity_signal": 0,
                "whale_confidence": 0,
                "whale_accumulation": 0
            }
        
        # Calculate signals
        
        # 1. Whale activity signal: positive when buying exceeds selling
        if whale_data["buy_volume"] + whale_data["sell_volume"] > 0:
            if whale_data["buy_volume"] > whale_data["sell_volume"]:
                # More buying than selling - bullish
                whale_activity_signal = min(1.0, (whale_data["buy_volume"] - whale_data["sell_volume"]) / 
                                         (whale_data["buy_volume"] + whale_data["sell_volume"]) * 2)
            else:
                # More selling than buying - bearish
                whale_activity_signal = max(-1.0, (whale_data["buy_volume"] - whale_data["sell_volume"]) / 
                                          (whale_data["buy_volume"] + whale_data["sell_volume"]) * 2)
        else:
            whale_activity_signal = 0
        
        # 2. Whale confidence: higher when whale volume dominates total volume
        whale_confidence = min(1.0, whale_data["whale_volume_percent"] / 50)
        
        # 3. Whale accumulation: are whales cumulatively buying or selling?
        if whale_data["buy_count"] + whale_data["sell_count"] > 0:
            whale_accumulation = (whale_data["buy_count"] - whale_data["sell_count"]) / (whale_data["buy_count"] + whale_data["sell_count"])
        else:
            whale_accumulation = 0
        
        return {
            "whale_activity_signal": round(whale_activity_signal, 2),
            "whale_confidence": round(whale_confidence, 2),
            "whale_accumulation": round(whale_accumulation, 2)
        }
    
    def _get_token_contract(self, token_symbol: str) -> List[str]:
        """
        Get the contract address for a token symbol
        
        Args:
            token_symbol: Token symbol (e.g., 'ETH', 'BTC')
            
        Returns:
            List of contract addresses
        """
        # Common token contracts - same as in ExchangeFlowAnalyzer
        token_contracts = {
            "btc": ["0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"],  # WBTC
            "eth": [], # Native ETH
            "usdt": ["0xdAC17F958D2ee523a2206206994597C13D831ec7"],
            "usdc": ["0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"],
            "link": ["0x514910771AF9Ca656af840dff83E8264EcF986CA"],
            "uni": ["0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"],
            "aave": ["0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9"],
        }
        
        return token_contracts.get(token_symbol.lower(), []) 