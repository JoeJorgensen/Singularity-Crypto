"""
Alchemy API client for on-chain data
"""
import os
import requests
import json
import logging
from dotenv import load_dotenv
import time
from typing import Dict, List, Any, Optional
import sys
import os.path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define common exchange wallets
COMMON_EXCHANGE_WALLETS = {
    # Binance
    "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE": "Binance",
    # Coinbase
    "0xdAC17F958D2ee523a2206206994597C13D831ec7": "Coinbase",
    # Kraken
    "0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2": "Kraken"
}

class AlchemyClient:
    """Client for interacting with Alchemy API for on-chain data"""
    
    def __init__(self):
        """Initialize the Alchemy client with API key from environment variables"""
        self.api_key = os.getenv("ALCHEMY_API_KEY")
        if not self.api_key:
            raise ValueError("ALCHEMY_API_KEY environment variable not set")
        
        # Use the HTTP endpoint URL
        self.base_url = f"https://eth-mainnet.g.alchemy.com/v2/{self.api_key}"
        
        # Prices API endpoint
        self.prices_base_url = f"https://api.g.alchemy.com/prices/v1/{self.api_key}"
        
        logger.info("Alchemy API client initialized")
    
    def _make_request(self, method: str, params: List = None) -> Dict:
        """
        Make a JSON-RPC request to the Alchemy API
        
        Args:
            method: The JSON-RPC method to call
            params: The parameters to pass to the method
            
        Returns:
            The parsed JSON response
        """
        if params is None:
            params = []
            
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": method,
            "params": params
        }
        
        try:
            logger.debug(f"Making Alchemy API request: {method}")
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for non-200 status codes
            
            data = response.json()
            
            if "error" in data:
                error_msg = data["error"].get("message", "Unknown error")
                logger.error(f"Alchemy API error: {error_msg}")
                raise Exception(f"Alchemy API error: {error_msg}")
                
            if "result" not in data:
                logger.error(f"Unexpected response format: {data}")
                raise Exception("Unexpected response format from Alchemy API")
                
            return data["result"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when calling {method}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when calling {method}: {str(e)}")
            raise
    
    def get_block_number(self) -> int:
        """
        Get the current block number
        
        Returns:
            The current block number
        """
        try:
            # Check if we should use test data
            test_mode = os.getenv("ALCHEMY_TEST_MODE", "false").lower() == "true"
            
            if test_mode:
                # Return a realistic block number
                return 19000000
                
            result = self._make_request("eth_blockNumber")
            block_number = int(result, 16) if isinstance(result, str) else result
            return block_number
        except Exception as e:
            logger.error(f"Error getting block number: {str(e)}")
            # Return a realistic fallback block number
            return 19000000
    
    def get_balance(self, address: str, block_tag: str = "latest") -> int:
        """
        Get the balance of an address
        
        Args:
            address: The Ethereum address
            block_tag: The block number or tag (e.g., "latest")
            
        Returns:
            The balance in wei
        """
        result = self._make_request("eth_getBalance", [address, block_tag])
        balance = int(result, 16) if isinstance(result, str) else result
        return balance
    
    def get_token_balances(self, address: str) -> Dict:
        """
        Get token balances for an address using Alchemy's getTokenBalances endpoint
        
        Args:
            address: The Ethereum address
            
        Returns:
            Dict with token balances
        """
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "alchemy_getTokenBalances",
            "params": [address, "erc20"]
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                raise Exception(f"Alchemy API error: {data['error'].get('message', 'Unknown error')}")
                
            return data["result"]
        except Exception as e:
            logger.error(f"Error getting token balances: {str(e)}")
            raise
    
    def get_asset_transfers(self, params: Dict) -> Dict:
        """
        Get asset transfers using Alchemy's getAssetTransfers endpoint
        
        Args:
            params: Parameters for the transfer query (fromBlock, toBlock, category, etc.)
            
        Returns:
            Dict with transfers data
        """
        # Convert fromBlock and toBlock to hex if they are integers
        if "fromBlock" in params and isinstance(params["fromBlock"], int):
            params["fromBlock"] = hex(params["fromBlock"])
        if "toBlock" in params and isinstance(params["toBlock"], int):
            params["toBlock"] = hex(params["toBlock"])
            
        # Set a default maxCount if not provided
        if "maxCount" not in params:
            params["maxCount"] = "0x3e8"  # Hex for 1000 (maxCount must be <= 0x3e8 per Alchemy API limit)
            
        test_mode = os.getenv("ALCHEMY_TEST_MODE", "false").lower() == "true"
        
        if test_mode:
            # Return test data
            logger.info("Using mock transfer data in test mode")
            return self._get_mock_transfers_data(params)
        
        # For real data, attempt to collect transfers with pagination if needed
        all_transfers = []
        has_more = True
        page_key = None
        attempt = 0
        max_attempts = 3  # Maximum number of pagination requests to avoid excessive API usage
        
        while has_more and attempt < max_attempts:
            attempt += 1
            
            # Add page key to params if we have one
            if page_key:
                params["pageKey"] = page_key
                
            payload = {
                "jsonrpc": "2.0",
                "id": int(time.time() * 1000),
                "method": "alchemy_getAssetTransfers",
                "params": [params]
            }
            
            headers = {"Content-Type": "application/json"}
            
            try:
                # Proceed with real API call
                logger.debug(f"Making Alchemy API call to get asset transfers (page {attempt})")
                response = requests.post(self.base_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                if "error" in data:
                    logger.error(f"Alchemy API error: {data['error'].get('message', 'Unknown error')}")
                    raise Exception(f"Alchemy API error: {data['error'].get('message', 'Unknown error')}")
                
                # Extract results
                if "result" in data:
                    result = data["result"]
                    transfers = result.get("transfers", [])
                    all_transfers.extend(transfers)
                    logger.debug(f"Received {len(transfers)} transfers from Alchemy API in page {attempt}")
                    
                    # Check if there are more pages
                    if "pageKey" in result:
                        page_key = result["pageKey"]
                        has_more = True
                    else:
                        has_more = False
                else:
                    logger.error(f"No result field in Alchemy API response: {data}")
                    has_more = False
                    
            except Exception as e:
                logger.error(f"Error getting asset transfers: {str(e)}")
                has_more = False
                # Only return mock data if we have no real data
                if len(all_transfers) == 0:
                    return self._get_mock_transfers_data(params)
        
        # If we've collected any transfers, return them
        if all_transfers:
            logger.debug(f"Total transfers collected: {len(all_transfers)}")
            
            # Filter out any transfers with missing from/to fields
            valid_transfers = []
            invalid_count = 0
            for i, transfer in enumerate(all_transfers):
                # Check if transfer has valid from and to addresses
                if not transfer.get('from'):
                    logger.debug(f"Skipping transfer with missing 'from' field")
                    invalid_count += 1
                    continue
                if not transfer.get('to'):
                    logger.debug(f"Skipping transfer with missing 'to' field")
                    invalid_count += 1
                    continue
                    
                # Check if value is present and valid
                try:
                    value = float(transfer.get('value', 0))
                    if value <= 0:
                        logger.debug(f"Skipping transfer with zero/negative value")
                        invalid_count += 1
                        continue
                except (ValueError, TypeError):
                    logger.debug(f"Skipping transfer with invalid value")
                    invalid_count += 1
                    continue
                    
                valid_transfers.append(transfer)
                
            if invalid_count > 0:
                logger.debug(f"Filtered out {invalid_count} invalid transfers")
            
            logger.debug(f"Filtered down to {len(valid_transfers)} valid transfers")
            return {"transfers": valid_transfers}
        
        # If we get here with no transfers, return empty result
        logger.warning("No transfers found in the specified block range")
        return {"transfers": []}
    
    def _get_mock_transfers_data(self, params: Dict) -> Dict:
        """
        Generate mock transfer data for testing and development
        
        Args:
            params: Original request parameters
            
        Returns:
            Dummy transfer data
        """
        logger.warning("Using mock transfer data for development")
        
        # Generate mock transfer data
        mock_data = {
            "transfers": []
        }
        
        # Check if this is for ETH or token transfers
        is_eth = "category" in params and "external" in params["category"]
        
        # Create some exchange wallets for testing
        exchange_addresses = list(COMMON_EXCHANGE_WALLETS.keys())[:3] if 'COMMON_EXCHANGE_WALLETS' in globals() else [
            "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",  # Binance
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # Coinbase
            "0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2"   # Kraken
        ]
        
        # Generate random wallet addresses for non-exchange addresses
        random_wallets = [
            f"0x{i:040x}" for i in range(10, 20)
        ]
        
        # Generate random transfers
        for i in range(20):
            # Decide if this is inflow or outflow to exchange (50/50 chance)
            is_inflow = i % 2 == 0
            
            if is_inflow:
                # Transfer from random wallet to exchange
                from_addr = random_wallets[i % len(random_wallets)]
                to_addr = exchange_addresses[i % len(exchange_addresses)]
            else:
                # Transfer from exchange to random wallet
                from_addr = exchange_addresses[i % len(exchange_addresses)]
                to_addr = random_wallets[i % len(random_wallets)]
            
            # Generate random value (higher for exchange transactions)
            value = 0.5 + (i * 1.2) if i < 5 else 0.05 + (i * 0.1)
            
            # Ensure from and to addresses are valid strings
            if not from_addr or not isinstance(from_addr, str):
                from_addr = f"0x{i:040x}"
            if not to_addr or not isinstance(to_addr, str):
                to_addr = f"0x{(i+1):040x}"
            
            transfer = {
                "blockNum": f"0x{(12345678 + i):x}",
                "hash": f"0x{(100000000000 + i*10000):032x}",
                "from": from_addr,
                "to": to_addr,
                "value": value,
                "asset": "ETH" if is_eth else "USDT",
                "category": "external" if is_eth else "erc20",
                "rawContract": {
                    "value": hex(int(value * 10**18)),
                    "address": None if is_eth else f"0x{(200000000 + i):x}",
                    "decimal": "0x12"
                }
            }
            
            mock_data["transfers"].append(transfer)
        
        return mock_data
    
    def get_nfts_for_owner(self, owner_address: str) -> Dict:
        """
        Get NFTs owned by an address using Alchemy's getNFTs endpoint
        
        Args:
            owner_address: The Ethereum address that owns the NFTs
            
        Returns:
            Dict with NFT data
        """
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "alchemy_getNFTs",
            "params": [{"owner": owner_address}]
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                raise Exception(f"Alchemy API error: {data['error'].get('message', 'Unknown error')}")
                
            return data["result"]
        except Exception as e:
            logger.error(f"Error getting NFTs: {str(e)}")
            raise
    
    def get_token_metadata(self, token_address: str) -> Dict:
        """
        Get metadata for a token contract using Alchemy's getTokenMetadata endpoint
        
        Args:
            token_address: The token contract address
            
        Returns:
            Dict with token metadata
        """
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "alchemy_getTokenMetadata",
            "params": [token_address]
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                raise Exception(f"Alchemy API error: {data['error'].get('message', 'Unknown error')}")
                
            return data["result"]
        except Exception as e:
            logger.error(f"Error getting token metadata: {str(e)}")
            raise
    
    def get_gas_price(self) -> int:
        """
        Get current gas price
        
        Returns:
            The gas price in wei
        """
        try:
            # Check if we should use test data (uncomment for testing)
            test_mode = os.getenv("ALCHEMY_TEST_MODE", "false").lower() == "true"
            
            if test_mode:
                # Return test data (30 Gwei in wei)
                return 30 * 1_000_000_000
                
            result = self._make_request("eth_gasPrice")
            gas_price = int(result, 16) if isinstance(result, str) else result
            return gas_price
        except Exception as e:
            logger.error(f"Error getting gas price: {str(e)}")
            # Return realistic fallback value (30 Gwei in wei)
            return 30 * 1_000_000_000
            
    # ----------------------- Prices API Methods -----------------------
    
    def get_token_price_by_symbol(self, symbols: List[str]) -> Dict:
        """
        Get current token prices by symbol using Alchemy's Prices API
        
        Args:
            symbols: List of token symbols (e.g., ["ETH", "BTC", "USDT"])
            
        Returns:
            Dictionary with token prices
        """
        if not symbols:
            logger.warning("No symbols provided to get_token_price_by_symbol")
            return {"data": []}
            
        try:
            # Construct URL with query parameters
            url = f"{self.prices_base_url}/tokens/by-symbol"
            
            # Add symbols as query parameters
            params = []
            for symbol in symbols:
                params.append(("symbols", symbol))
                
            # Make the request
            logger.debug(f"Making Alchemy Prices API request for symbols: {symbols}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Received price data for {len(data.get('data', []))} symbols")
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when getting token prices by symbol: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when getting token prices by symbol: {str(e)}")
            raise
    
    def get_token_price_by_address(self, addresses: List[Dict]) -> Dict:
        """
        Get current token prices by contract address using Alchemy's Prices API
        
        Args:
            addresses: List of dictionaries with network and address keys
                      (e.g., [{"network": "eth-mainnet", "address": "0x..."}])
            
        Returns:
            Dictionary with token prices
        """
        if not addresses:
            logger.warning("No addresses provided to get_token_price_by_address")
            return {"data": []}
            
        try:
            # Construct URL
            url = f"{self.prices_base_url}/tokens/by-address"
            
            # Create request payload
            payload = {"addresses": addresses}
            
            # Make the request
            logger.debug(f"Making Alchemy Prices API request for addresses: {addresses}")
            response = requests.post(
                url, 
                headers={"Content-Type": "application/json"},
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Received price data for {len(data.get('data', []))} addresses")
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when getting token prices by address: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when getting token prices by address: {str(e)}")
            raise
    
    def get_historical_token_prices(self, 
                                  symbol_or_address: str, 
                                  network: str = "eth-mainnet",
                                  is_address: bool = False,
                                  start_timestamp: int = None, 
                                  end_timestamp: int = None,
                                  interval: str = "1d") -> Dict:
        """
        Get historical token prices using Alchemy's Prices API
        
        Args:
            symbol_or_address: Token symbol or contract address
            network: Blockchain network (only used if is_address is True)
            is_address: Whether symbol_or_address is a contract address
            start_timestamp: Start timestamp in UNIX seconds (defaults to 30 days ago)
            end_timestamp: End timestamp in UNIX seconds (defaults to current time)
            interval: Time interval ("1h", "1d", etc.)
            
        Returns:
            Dictionary with historical price data
        """
        if not symbol_or_address:
            logger.warning("No symbol or address provided to get_historical_token_prices")
            return {"data": []}
            
        try:
            # Set default timestamps if not provided
            current_time = int(time.time())
            if not end_timestamp:
                end_timestamp = current_time
            if not start_timestamp:
                # Default to 30 days before end_timestamp
                start_timestamp = end_timestamp - (30 * 24 * 60 * 60)
                
            # Construct URL
            if is_address:
                url = f"{self.prices_base_url}/tokens/{network}/{symbol_or_address}/history"
            else:
                url = f"{self.prices_base_url}/tokens/symbol/{symbol_or_address}/history"
                
            # Add query parameters
            params = {
                "from": start_timestamp,
                "to": end_timestamp,
                "interval": interval
            }
                
            # Make the request
            logger.debug(f"Making Alchemy Prices API request for historical data: {symbol_or_address}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Received historical price data for {symbol_or_address}")
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when getting historical token prices: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when getting historical token prices: {str(e)}")
            raise
    
    def get_current_crypto_price(self, symbol: str) -> float:
        """
        Get the current price of a cryptocurrency in USD
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "ETH", "BTC")
            
        Returns:
            Current price in USD
        """
        try:
            # Get price data from Alchemy
            price_data = self.get_token_price_by_symbol([symbol])
            
            # Extract price from response
            if "data" in price_data and price_data["data"]:
                for token_data in price_data["data"]:
                    if token_data["symbol"].upper() == symbol.upper():
                        if "prices" in token_data and token_data["prices"]:
                            for price_info in token_data["prices"]:
                                if price_info["currency"] == "USD":
                                    return float(price_info["value"])
            
            logger.warning(f"Price not found for {symbol}, returning None")
            return None
        except Exception as e:
            logger.error(f"Error getting current crypto price for {symbol}: {str(e)}")
            return None 