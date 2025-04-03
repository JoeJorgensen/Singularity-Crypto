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
import logging
from .alchemy.client import AlchemyClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Known exchange wallets - this would be expanded in production
EXCHANGE_WALLETS = {
    # Binance
    "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE": "Binance",
    "0xD551234AE421e3BCBA99A0Da6d736074f22192FF": "Binance",
    "0x564286362092D8e7936f0549571a803B203aAceD": "Binance",
    # Coinbase
    "0xdAC17F958D2ee523a2206206994597C13D831ec7": "Coinbase",
    "0x71660c4005BA85c37ccec55d0C4493E66Fe775d3": "Coinbase",
    # Kraken
    "0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2": "Kraken",
    # FTX (defunct but still useful for historical analysis)
    "0xC098B2a3Aa256D2140208C3de6543aAEf5cd3A94": "FTX",
    # Gemini
    "0x5f65f7b609678448494De4C87521CdF6cEf1e932": "Gemini",
}

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

class ExchangeFlowAnalyzer:
    """
    Analyzes cryptocurrency flows in and out of exchanges
    """
    
    def __init__(self):
        """Initialize the exchange flow analyzer with Alchemy client and exchange list"""
        # Initialize Alchemy client
        from on_chain.alchemy.client import AlchemyClient
        self.alchemy_client = AlchemyClient()
        
        # Initialize caches
        self.flows_cache = {}
        self.cache_timestamp = 0
        self.cache_duration = 600  # Cache for 10 minutes
        
        # Define exchange wallets list - adding more well-known exchanges
        self.exchange_wallets = {
            # Binance - Main Hot Wallets
            "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE": "Binance",
            "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": "Binance",
            "0x28c6c06298d514db089934071355e5743bf21d60": "Binance",
            "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "Binance",
            "0xbe0eb53f46cd790cd13851d5eff43d12404d33e8": "Binance",
            "0xf977814e90da44bfa03b6295a0616a897441acec": "Binance",
            "0x8315177ab297ba92a06054ce80a67ed4dbd7ed3a": "Binance",
            
            # Coinbase - Main Wallets
            "0xdAC17F958D2ee523a2206206994597C13D831ec7": "Coinbase",
            "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "Coinbase",
            "0x503828976d22510aad0201ac7ec88293211d23da": "Coinbase",
            "0xa090e606e30bd747d4e6245a1517ebe430f0057e": "Coinbase",
            "0xeb2629a2734e272bcc07bda959863f316f4bd4cf": "Coinbase",
            
            # Kraken
            "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "Kraken",
            "0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13": "Kraken",
            "0xa24787320ade7fdc6faa9362ef22c1b350e9f4a4": "Kraken",
            "0xda9dfa130df4de4673b89022ee50ff26f6ea73cf": "Kraken",
            
            # FTX (historical)
            "0x2faf487a4414fe77e2327f0bf4ae2a264a776ad2": "FTX",
            "0xc098b2a3aa256d2140208c3de6543aaef5cd3a94": "FTX",
            
            # Huobi
            "0xab5c66752a9e8167967685f1450532fb96d5d24f": "Huobi",
            "0x6748f50f686bfbca6fe8ad62b22228b87f31ff2b": "Huobi",
            "0xfdb16996831753d5331ff813c29a93c76834a0ad": "Huobi",
            "0xeee28d484628d41a82d01e21d12e2e78d69920da": "Huobi",
            
            # OKX
            "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": "OKX",
            "0x236f9f97e0e62388479bf9e5ba4889e46b0273c3": "OKX",
            "0x5041ed759dd4afc3a72b8192c143f72f4724081a": "OKX",
            
            # Kucoin
            "0x0861fca546225fbf8806986d211c8398f7457734": "Kucoin",
            "0xd6216fc19db775df9774a6e33526131da7d19a2c": "Kucoin",
            
            # Bitfinex
            "0x77134cbc06cb00b66f4c7e623d5fdbf6777635ec": "Bitfinex",
            "0x742d35cc6634c0532925a3b844bc454e4438f44e": "Bitfinex",
            "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa": "Bitfinex",
            
            # Gemini
            "0x61edcdf5bb737adffe5043706e7c5bb1f1a56eea": "Gemini",
            "0x5f65f7b609678448494de4c87521cdf6cef1e932": "Gemini",
            
            # Gate.io
            "0x0d0707963952f2fba59dd06f2b425ace40b492fe": "Gate.io",
            "0x7793cd85c11a924478d358d49b05b37e91b5810f": "Gate.io",
            
            # Bitstamp
            "0x00bdb5699745f5b860228c8f939abf1b9ae374ed": "Bitstamp",
            "0x1522900b6dafac587d499a862861c0869be6e428": "Bitstamp",
            
            # Bittrex
            "0xfbb1b73c4f0bda4f67dca266ce6ef42f520fbb98": "Bittrex",
            "0xe94b04a0fed112f3664e45adb2b8915693dd5ff3": "Bittrex",
            
            # Bybit
            "0x7ee71053e1b6a1b40e233a458839ecaeb1d0f97e": "Bybit",
            "0xbe0eb53f46cd790cd13851d5eff43d12404d33e8": "Bybit",
        }
        
        logger.info(f"Exchange flow analyzer initialized with {len(self.exchange_wallets)} exchange wallet addresses")
        
    def is_exchange_wallet(self, address: str) -> bool:
        """
        Check if an address belongs to a known exchange
        
        Args:
            address: Ethereum address to check
            
        Returns:
            True if the address belongs to a known exchange, False otherwise
        """
        if not address or len(address) < 10:
            logger.warning(f"Invalid address: {address}")
            return False
        
        # Convert address to lowercase for case-insensitive comparison
        address_lower = address.lower()
        
        # Check if address is in our exchange wallets dictionary
        return any(wallet_addr.lower() == address_lower for wallet_addr in self.exchange_wallets.keys())
    
    def get_exchange_name(self, address: str) -> Optional[str]:
        """
        Get the name of the exchange for an address
        
        Args:
            address: Ethereum address to check
            
        Returns:
            Name of the exchange or None if not found
        """
        if not address or len(address) < 10:
            return None
        
        # Convert address to lowercase for case-insensitive comparison
        address_lower = address.lower()
        
        # Find the exchange name
        for wallet_addr, name in self.exchange_wallets.items():
            if wallet_addr.lower() == address_lower:
                return name
                
        return None
    
    def get_exchange_flows(self, token_symbol: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get exchange flow data for a specific token
        
        Args:
            token_symbol: Symbol of the token to analyze (e.g., 'ETH', 'BTC')
            hours: Number of hours to look back
            
        Returns:
            Dict containing exchange flow metrics and visualization data
        """
        try:
            # Get the raw flow data first
            raw_flow_data = self.analyze_token_flows(token_symbol, hours)
            flow_signals = self.get_flow_signals(token_symbol, hours)
            
            # Check if using test mode
            is_mock_data = os.getenv("ALCHEMY_TEST_MODE", "false").lower() == "true"
            
            # Format data for UI display
            result = {
                "token": token_symbol.upper(),
                "timeframe": f"{hours}h",
                "timestamp": datetime.now().isoformat(),
                "inflow": {
                    "volume": raw_flow_data["inflow_volume"],
                    "count": raw_flow_data["inflow_count"]
                },
                "outflow": {
                    "volume": raw_flow_data["outflow_volume"],
                    "count": raw_flow_data["outflow_count"]
                },
                "net_flow": raw_flow_data["net_flow"],
                "signals": flow_signals,
                "exchanges": raw_flow_data["exchange_data"],
                "chart_data": self._prepare_chart_data(raw_flow_data),
                "trend": self._determine_flow_trend(raw_flow_data["net_flow"]),
                "is_mock_data": is_mock_data
            }
            
            return result
        except Exception as e:
            logger.error(f"Error getting exchange flows: {str(e)}")
            # Return minimal data to prevent UI errors
            return {
                "token": token_symbol.upper(),
                "timeframe": f"{hours}h",
                "timestamp": datetime.now().isoformat(),
                "inflow": {"volume": 0, "count": 0},
                "outflow": {"volume": 0, "count": 0},
                "net_flow": 0,
                "signals": {"exchange_flow_signal": 0, "flow_strength": 0, "exchange_accumulation": 0},
                "exchanges": [],
                "chart_data": {"inflows": [], "outflows": []},
                "trend": "neutral",
                "is_mock_data": True
            }
    
    def _prepare_chart_data(self, flow_data: Dict[str, Any]) -> Dict[str, List]:
        """
        Prepare chart data from flow data
        
        Args:
            flow_data: Flow data from analyze_token_flows
            
        Returns:
            Dict with chart-ready data
        """
        # In a full implementation, this would format time-series data
        # For this simplified version, we'll create placeholder data for exchanges
        
        inflows = []
        outflows = []
        
        for exchange_data in flow_data.get("exchange_data", []):
            exchange_name = exchange_data.get("exchange", "Unknown")
            inflows.append({
                "exchange": exchange_name,
                "value": exchange_data.get("inflow", 0)
            })
            outflows.append({
                "exchange": exchange_name,
                "value": exchange_data.get("outflow", 0)
            })
        
        return {
            "inflows": inflows,
            "outflows": outflows
        }
    
    def _determine_flow_trend(self, net_flow: float) -> str:
        """
        Determine the flow trend based on net flow
        
        Args:
            net_flow: Net flow value (positive = inflow to exchanges)
            
        Returns:
            Trend description
        """
        if abs(net_flow) < 0.5:
            return "neutral"
        elif net_flow > 0:
            return "exchange_inflow"  # Potentially bearish
        else:
            return "exchange_outflow"  # Potentially bullish
        
    def analyze_token_flows(self, token_symbol: str, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze the exchange flows for a specific token over a given time period
        
        Args:
            token_symbol: Symbol of the token (e.g., 'ETH')
            hours: Number of hours to look back
            
        Returns:
            Dictionary with exchange flow analysis
        """
        # Clean token symbol
        token_symbol = token_symbol.lower()
        
        # Use cache if available and not expired
        cache_key = f"flow_{token_symbol}_{hours}"
        current_time = time.time()
        
        if cache_key in self.flows_cache and (current_time - self.cache_timestamp < self.cache_duration):
            logger.debug(f"Using cached exchange flow data for {token_symbol}")
            return self.flows_cache[cache_key]
            
        # Log start of analysis
        logger.info(f"Analyzing exchange flows for {token_symbol} over past {hours} hours")
        
        # Process logic based on test mode
        test_mode = os.getenv("ALCHEMY_TEST_MODE", "false").lower() == "true"
        logger.debug(f"ALCHEMY_TEST_MODE is set to: {test_mode}")
        
        # Calculate block range - expand range to ensure we catch more transactions
        # Ethereum blocks are ~12-13 seconds apart, but we'll use a more conservative value
        # to ensure we don't miss transactions due to varying block times
        blocks_per_hour = int(3600 / 12.5) # More accurate blocks per hour calculation
        current_block = self.alchemy_client.get_block_number()
        
        # Add a 10% buffer to the block range to ensure we don't miss transactions
        # due to variations in block times
        from_block = current_block - int(blocks_per_hour * hours * 1.1)
        
        logger.info(f"Querying transfers from block {from_block} to current block {current_block}")
        
        # Configure the parameters based on token type
        if token_symbol == "eth":
            # For native ETH
            params = {
                "fromBlock": hex(from_block),
                "toBlock": "latest",
                "category": ["external"],
                # Set a higher limit to catch more transactions
                "maxCount": "0x3e8"  # Hex for 1000
            }
        else:
            # For ERC-20 tokens
            token_contracts = self._get_token_contract(token_symbol)
            if not token_contracts:
                logger.warning(f"No contract address found for {token_symbol}")
                return {
                    "token": token_symbol,
                    "inflow_volume": 0,
                    "outflow_volume": 0,
                    "net_flow": 0,
                    "inflow_count": 0,
                    "outflow_count": 0,
                    "exchange_data": []
                }
                
            params = {
                "fromBlock": hex(from_block),
                "toBlock": "latest",
                "category": ["erc20"],
                "contractAddresses": token_contracts,
                "maxCount": "0x3e8"  # Hex for 1000
            }
            
        # Get transfers data
        transfers = self.alchemy_client.get_asset_transfers(params)
        
        # Process the transfers
        inflow_volume = 0
        outflow_volume = 0
        inflow_count = 0
        outflow_count = 0
        exchange_data = {}
        
        # Log the number of transfers received
        transfer_list = transfers.get("transfers", [])
        logger.info(f"Received {len(transfer_list)} transfers from Alchemy API")
        
        # Check if we have any malformed transfers
        for i, transfer in enumerate(transfer_list):
            if "from" not in transfer or not transfer["from"]:
                logger.warning(f"Transfer #{i} missing 'from' field: {transfer}")
            if "to" not in transfer or not transfer["to"]:
                logger.warning(f"Transfer #{i} missing 'to' field: {transfer}")
            if "value" not in transfer:
                logger.warning(f"Transfer #{i} missing 'value' field: {transfer}")
        
        # Group exchanges for easier identification
        exchange_addresses = set(addr.lower() for addr in self.exchange_wallets.keys())
        
        # First pass - identify and store exchanges with actual transactions
        active_exchanges = {}
        
        for transfer in transfer_list:
            # Extract addresses with proper checks
            from_address = transfer.get("from", "").lower() if transfer.get("from") else ""
            to_address = transfer.get("to", "").lower() if transfer.get("to") else ""
            
            # Safely convert value to float
            try:
                value = float(transfer.get("value", 0))
            except (ValueError, TypeError):
                logger.debug(f"Skipping transfer with invalid value: {transfer}")
                continue
            
            # Skip if address is missing or value is zero
            if not from_address or not to_address or value <= 0:
                logger.debug(f"Skipping transfer with missing address or zero value: {transfer}")
                continue
                
            # Check if transfer involves an exchange
            from_is_exchange = from_address in exchange_addresses
            to_is_exchange = to_address in exchange_addresses
            
            # Record active exchanges
            if from_is_exchange:
                exchange_name = self.get_exchange_name(from_address)
                if exchange_name:
                    active_exchanges[from_address] = exchange_name
            
            if to_is_exchange:
                exchange_name = self.get_exchange_name(to_address)
                if exchange_name:
                    active_exchanges[to_address] = exchange_name
        
        # Log active exchanges
        logger.info(f"Found {len(active_exchanges)} active exchanges in the transfers")
        
        # Second pass - calculate flows
        for transfer in transfer_list:
            # Extract addresses with proper checks
            from_address = transfer.get("from", "").lower() if transfer.get("from") else ""
            to_address = transfer.get("to", "").lower() if transfer.get("to") else ""
            
            # Safely convert value to float
            try:
                value = float(transfer.get("value", 0))
            except (ValueError, TypeError):
                continue
                
            hash_val = transfer.get("hash", "")
            
            # Skip if address is missing or value is zero/negative
            if not from_address or not to_address or value <= 0:
                continue
                
            # Determine if addresses are exchanges
            from_is_exchange = from_address in exchange_addresses
            to_is_exchange = to_address in exchange_addresses
            
            # Skip transfers between exchanges (internal movements)
            if from_is_exchange and to_is_exchange:
                logger.debug(f"Skipping internal exchange transfer: {hash_val}")
                continue
                
            if to_is_exchange and not from_is_exchange:
                # Inflow to exchange
                inflow_volume += value
                inflow_count += 1
                
                exchange_name = active_exchanges.get(to_address, "Unknown Exchange")
                if exchange_name not in exchange_data:
                    exchange_data[exchange_name] = {"inflow": 0, "outflow": 0}
                exchange_data[exchange_name]["inflow"] += value
                
                logger.debug(f"Inflow: {value} ETH to {exchange_name} in tx {hash_val}")
                
            elif from_is_exchange and not to_is_exchange:
                # Outflow from exchange
                outflow_volume += value
                outflow_count += 1
                
                exchange_name = active_exchanges.get(from_address, "Unknown Exchange")
                if exchange_name not in exchange_data:
                    exchange_data[exchange_name] = {"inflow": 0, "outflow": 0}
                exchange_data[exchange_name]["outflow"] += value
                
                logger.debug(f"Outflow: {value} ETH from {exchange_name} in tx {hash_val}")
        
        # Calculate net flow (positive means more inflow to exchanges)
        net_flow = inflow_volume - outflow_volume
        
        # Format results
        result = {
            "token": token_symbol,
            "inflow_volume": round(inflow_volume, 4),
            "outflow_volume": round(outflow_volume, 4),
            "net_flow": round(net_flow, 4),
            "inflow_count": inflow_count,
            "outflow_count": outflow_count,
            "exchange_data": [
                {
                    "exchange": exchange,
                    "inflow": round(data["inflow"], 4),
                    "outflow": round(data["outflow"], 4),
                    "net_flow": round(data["inflow"] - data["outflow"], 4)
                }
                for exchange, data in exchange_data.items()
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        # Update cache
        self.flows_cache[cache_key] = result
        self.cache_timestamp = current_time
        
        logger.debug(f"Completed exchange flow analysis for {token_symbol}: inflow={inflow_volume}, outflow={outflow_volume}, net_flow={net_flow}")
        
        return result
    
    def get_flow_signals(self, token_symbol: str, hours: int = 24) -> Dict[str, float]:
        """
        Generate trading signals based on exchange flows
        
        Args:
            token_symbol: Symbol of the token to analyze
            hours: Number of hours to look back
            
        Returns:
            Dict of signal values between -1.0 and 1.0
        """
        flow_data = self.analyze_token_flows(token_symbol, hours)
        
        # No data case
        if flow_data["inflow_volume"] == 0 and flow_data["outflow_volume"] == 0:
            return {
                "exchange_flow_signal": 0,
                "flow_strength": 0,
                "exchange_accumulation": 0
            }
        
        # Calculate signals
        
        # 1. Net flow signal: positive when outflows exceed inflows (bullish)
        if flow_data["inflow_volume"] + flow_data["outflow_volume"] > 0:
            net_flow_ratio = flow_data["net_flow"] / (flow_data["inflow_volume"] + flow_data["outflow_volume"])
            # Inverse the signal since outflows from exchanges are bullish
            exchange_flow_signal = -1 * max(min(net_flow_ratio * 3, 1.0), -1.0)
        else:
            exchange_flow_signal = 0
        
        # 2. Flow strength: higher when total volume is significant
        # This is a simple placeholder implementation
        flow_strength = min(1.0, (flow_data["inflow_volume"] + flow_data["outflow_volume"]) / 1000)
        
        # 3. Exchange accumulation: are exchanges accumulating or distributing?
        exchange_count = len(flow_data["exchange_data"])
        if exchange_count > 0:
            accumulating_exchanges = sum(1 for exchange in flow_data["exchange_data"] if exchange["net_flow"] > 0)
            distributing_exchanges = sum(1 for exchange in flow_data["exchange_data"] if exchange["net_flow"] < 0)
            
            if accumulating_exchanges + distributing_exchanges > 0:
                exchange_accumulation = (accumulating_exchanges - distributing_exchanges) / (accumulating_exchanges + distributing_exchanges)
            else:
                exchange_accumulation = 0
        else:
            exchange_accumulation = 0
        
        return {
            "exchange_flow_signal": round(exchange_flow_signal, 2),
            "flow_strength": round(flow_strength, 2),
            "exchange_accumulation": round(exchange_accumulation, 2)
        }
    
    def _get_token_contract(self, token_symbol: str) -> List[str]:
        """
        Get the contract address for a token symbol
        
        Args:
            token_symbol: Token symbol (e.g., 'ETH', 'BTC')
            
        Returns:
            List of contract addresses
        """
        # Common token contracts
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