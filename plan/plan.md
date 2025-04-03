# Implementation Plan: Integrating Alchemy for On-Chain Data

## Introduction

This document outlines our plan to replace simulated on-chain data with real blockchain data using Alchemy's API. Alchemy is a blockchain developer platform that provides access to on-chain data across multiple networks with significant advantages:

- 300M free compute units per month
- Access to archive data at no additional cost
- Simple HTTP API (no blockchain wallet needed)
- Full support for major networks including Ethereum, Polygon, and Arbitrum
- Enhanced APIs for token data, transfers, and transaction information
- Well-documented SDKs for JavaScript/TypeScript, Python and more

## Current Implementation Overview

Currently, our on-chain data implementation:

1. Uses mostly simulated data in `exchange_flow.py`, `whale_tracker.py`, and `network_metrics.py`
2. Relies on `CoinloreAPI` for minimal real on-chain metrics
3. Provides this data to the sentiment predictor and other analysis tools

## Alchemy Configuration

Our Alchemy setup includes the following configuration:

1. **App Creation**:
   - App Name: "CryptoTrader On-Chain Data"
   - Description: "On-chain data analytics for crypto trading signals"
   - Environment: "Development" (with plan to create production version later)

2. **Networks**:
   - Primary: Ethereum Mainnet (via HTTPS) for comprehensive on-chain data
   - Secondary: Polygon for lower cost transaction data
   - Additional: Arbitrum for L2 scaling solutions

3. **API Services Used**:
   - Supernode API for core blockchain data
   - Token API for token balances and metadata
   - Transfers API for exchange flow and whale tracking
   - Archive data for historical blockchain state information

4. **Connection Method**:
   - HTTPS endpoints for all on-chain data requests
   - WebSockets not required for our current implementation

## Implementation Phases

### Phase 1: Set Up Alchemy API Integration (Week 1)

1. **Create a new AlchemyAPI class**
   - Create `api/alchemy_api.py` for handling Alchemy API requests
   - Implement request handling with rate limiting
   - Set up HTTP request mechanism
   - Implement API key authentication
   - Implement caching similar to other APIs in the system

2. **Basic API Structure**
   ```python
   class AlchemyAPI:
       def __init__(self):
           self.api_key = os.getenv('ALCHEMY_API_KEY')
           self.eth_url = os.getenv('ALCHEMY_ETH_URL')
           self.polygon_url = os.getenv('ALCHEMY_POLYGON_URL')
           self.arbitrum_url = os.getenv('ALCHEMY_ARBITRUM_URL')
           self.cache = {}
           self.cache_expiry = 3600  # 1 hour
           self.last_request_time = 0
           self.min_request_interval = 0.2  # seconds
           
       def _make_request(self, method, params=None, network='ethereum'):
           """Make a JSON-RPC request to Alchemy API"""
           # Select appropriate network URL
           if network == 'ethereum':
               url = self.eth_url
           elif network == 'polygon':
               url = self.polygon_url
           elif network == 'arbitrum':
               url = self.arbitrum_url
           else:
               raise ValueError(f"Unsupported network: {network}")
               
           # Apply rate limiting
           current_time = time.time()
           time_since_last_request = current_time - self.last_request_time
           if time_since_last_request < self.min_request_interval:
               time.sleep(self.min_request_interval - time_since_last_request)
           
           # Update last request time
           self.last_request_time = time.time()
           
           # Prepare payload
           payload = {
               "jsonrpc": "2.0",
               "id": 1,
               "method": method,
               "params": params or []
           }
           
           # Make request with error handling and retries
           response = requests.post(url, json=payload)
           response.raise_for_status()
           return response.json()
           
       async def _make_request_async(self, method, params=None, network='ethereum'):
           """Async version of the request method"""
           # Similar implementation with aiohttp
           
       def get_token_data(self, token_address, network='ethereum'):
           """Get token data for a specific token address"""
           # Implementation for token data retrieval
   ```

3. **Environment Configuration**
   - Update `.env` file with required Alchemy credentials:
     ```
     ALCHEMY_API_KEY=your_api_key
     ALCHEMY_ETH_URL=https://eth-mainnet.g.alchemy.com/v2/your_api_key
     ALCHEMY_POLYGON_URL=https://polygon-mainnet.g.alchemy.com/v2/your_api_key
     ALCHEMY_ARBITRUM_URL=https://arb-mainnet.g.alchemy.com/v2/your_api_key
     ```

### Phase 2: Create Real On-Chain Data Providers (Week 2)

1. **Implement RealExchangeFlow**
   - Create a new class `RealExchangeFlow` in `on_chain/exchange_flow.py`
   - Use Alchemy's Token and Transfers APIs to track exchange deposits/withdrawals
   - Connect to known exchange wallet addresses
   - Standardize output to match the current interface
   
2. **Implement RealWhaleTracker**
   - Create a new class `RealWhaleTracker` in `on_chain/whale_tracker.py`
   - Use Alchemy API to query large transactions based on our whale thresholds
   - Track transactions to/from known exchange addresses
   - Maintain the same interface as the current `WhaleTracker`
   
3. **Implement RealNetworkMetrics**
   - Create a new class `RealNetworkMetrics` in `on_chain/network_metrics.py`
   - Use Alchemy's Supernode API to query network health metrics
   - Retrieve active addresses, transaction counts, and gas metrics
   - Match the interface of the current `NetworkMetrics`

### Phase 3: Integration and Data Transformation (Week 3)

1. **Create Data Transformers**
   - Implement methods to convert Alchemy JSON responses to pandas DataFrames
   - Ensure compatibility with the sentiment predictor's expected format
   - Build utility functions for common data transformations
   
2. **Update On-Chain Interfaces**
   - Add factory methods to create either simulated or real implementations
   - Add configuration option to choose data source
   - Enable graceful fallback to simulated data when needed

3. **Update __init__.py Files**
   - Update module exports to include new implementations
   - Maintain backward compatibility

### Phase 4: Testing and Optimization (Week 4)

1. **Create Test Cases**
   - Write tests for Alchemy API requests
   - Test data transformation functions
   - Compare real vs simulated data
   
2. **Optimize Performance**
   - Implement request batching where possible
   - Optimize caching for frequently accessed data
   - Monitor compute unit usage to stay within free tier limits

3. **Implement Fallback Mechanisms**
   - Create graceful degradation for API rate limits or errors
   - Set up fallbacks to simulated data when real data isn't available
   - Handle network-specific issues with automatic failover

## Required New Dependencies

```
requests==2.31.0       # HTTP library (may already be installed)
aiohttp==3.8.4        # Async HTTP client for async methods
backoff==2.2.1        # Retry mechanism with exponential backoff
python-dotenv==1.0.0  # Environment variable management
web3==6.0.0           # Optional - for additional Ethereum utilities
```

## On-Chain Data Points to Collect

1. **Exchange Flow Data**
   - Exchange deposits/withdrawals
   - Token transfers between known exchanges
   - DEX liquidity changes
   - Net flow metrics (inflow vs outflow)

2. **Whale Activity Data**
   - Large transactions based on token-specific thresholds
   - Smart money wallet activity
   - Token concentration metrics
   - Whale sentiment indicators

3. **Network Health Metrics**
   - Daily active addresses
   - Transaction counts and fees
   - Gas prices and network congestion
   - New address growth
   - Block production rates

## Implementation Details

### Example Alchemy API Request for Active Addresses

```python
def get_active_addresses(self, days=1, network='ethereum'):
    """Get count of active addresses for the past N days"""
    # Calculate block numbers approximately
    current_block = self._make_request('eth_blockNumber', network=network)
    current_block_num = int(current_block['result'], 16)
    
    # Approximate blocks per day (15s block time for Ethereum)
    blocks_per_day = 24 * 60 * 60 // 15
    
    # Calculate starting block
    start_block = current_block_num - (blocks_per_day * days)
    
    # Get unique addresses that made transactions
    # This requires tracking blocks and analyzing transactions
    # within those blocks, and is a simplified example
    unique_addresses = set()
    
    for block_num in range(start_block, current_block_num, 1000):
        # Get block by number
        block = self._make_request('eth_getBlockByNumber', 
                               [hex(block_num), True],
                               network=network)
        
        # Process transactions in the block
        if 'result' in block and block['result'] and 'transactions' in block['result']:
            for tx in block['result']['transactions']:
                if 'from' in tx:
                    unique_addresses.add(tx['from'])
                if 'to' in tx and tx['to']:
                    unique_addresses.add(tx['to'])
    
    return {
        'active_addresses': len(unique_addresses),
        'time_period_days': days,
        'network': network,
        'timestamp': datetime.now().isoformat()
    }
```

### Data Transformation Example

```python
def transform_blockchain_data_to_dataframe(data, days=30):
    """Transform blockchain metrics to a DataFrame for the sentiment predictor"""
    # Create date range for the past N days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create empty DataFrame with dates as index
    df = pd.DataFrame(index=date_range)
    
    # Process active addresses data
    if 'active_addresses' in data:
        active_addresses = data['active_addresses']
        # If we have daily data
        if isinstance(active_addresses, dict) and all(k.isdigit() for k in active_addresses.keys()):
            for day_str, value in active_addresses.items():
                day = int(day_str)
                date = end_date - timedelta(days=day)
                if date in df.index:
                    df.at[date, 'active_addresses'] = value
        # If we have a single value for the most recent day
        elif isinstance(active_addresses, (int, float)):
            # Use the value for the most recent day and interpolate/approximate for previous days
            df.at[end_date, 'active_addresses'] = active_addresses
            # Simple linear decay for older dates (this is a placeholder approach)
            decay_factor = 0.98
            current_value = active_addresses
            for date in date_range[:-1][::-1]:  # Iterate in reverse order excluding the last day
                current_value *= decay_factor
                df.at[date, 'active_addresses'] = current_value
    
    # Process other metrics similarly...
    
    # Forward fill any missing values
    df = df.ffill().bfill()
    
    return df
```

## Benefits of Alchemy Integration

1. **Improved Prediction Accuracy**
   - Real on-chain data provides actual market signals vs. simulated patterns
   - Blockchain data has proven predictive power for price movements
   - Enhanced sentiment signals from real market activity

2. **Simplified Integration**
   - HTTP endpoints only require API keys, no blockchain wallet needed
   - Consistent interfaces across multiple networks
   - Well-documented APIs with SDKs

3. **Cost Efficiency**
   - Free tier of 300M compute units per month is generous for our use case
   - No additional cost for archive data access
   - Predictable scaling as usage grows

4. **Enhanced Features**
   - Access to data across multiple networks (Ethereum, Polygon, Arbitrum)
   - Historical data for backtesting and model training
   - Access to specialized APIs like Token API for additional insights

## Future Enhancements

Once the basic Alchemy integration is complete, we can consider these enhancements:

1. **Real-time Alert System**
   - Add WebSocket connection for immediate notification of whale movements
   - Set up custom webhooks for specific on-chain events
   - Create real-time alert dashboard in the Streamlit app

2. **Advanced On-Chain Analytics**
   - Implement token concentration analysis
   - Track smart contract interactions for DeFi protocols
   - Monitor NFT market activity as a sentiment indicator

3. **Cross-Chain Integration**
   - Expand to additional networks like Solana, Avalanche, etc.
   - Implement cross-chain flow analysis
   - Correlate multi-chain metrics for more robust signals
