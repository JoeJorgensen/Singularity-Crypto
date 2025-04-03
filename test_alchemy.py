"""
Test script for Alchemy integration
"""
import logging
import time
from on_chain.alchemy import AlchemyClient
from on_chain import ExchangeFlowAnalyzer, WhaleTracker, NetworkMetricsAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_alchemy_client():
    """Test direct Alchemy client functionality"""
    logger.info("Testing Alchemy client...")
    
    client = AlchemyClient()
    
    # Test getting the current block number
    logger.info("Getting current block number...")
    block_number = client.get_block_number()
    logger.info(f"Current block number: {block_number}")
    
    # Test getting gas price
    logger.info("Getting current gas price...")
    gas_price = client.get_gas_price()
    gas_price_gwei = int(gas_price) / 1_000_000_000
    logger.info(f"Current gas price: {gas_price_gwei} gwei")
    
    # Test getting token metadata
    logger.info("Getting token metadata for USDC...")
    token_metadata = client.get_token_metadata("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
    logger.info(f"USDC token metadata: {token_metadata}")
    
    return True

def test_exchange_flow_analyzer():
    """Test exchange flow analyzer"""
    logger.info("Testing Exchange Flow Analyzer...")
    
    analyzer = ExchangeFlowAnalyzer()
    
    # Test analyzing token flows for ETH
    logger.info("Analyzing ETH exchange flows...")
    eth_flows = analyzer.analyze_token_flows("eth", hours=24)
    logger.info(f"ETH exchange flow analysis: {eth_flows}")
    
    # Test getting signals
    logger.info("Getting ETH exchange flow signals...")
    eth_signals = analyzer.get_flow_signals("eth", hours=24)
    logger.info(f"ETH exchange flow signals: {eth_signals}")
    
    return True

def test_whale_tracker():
    """Test whale tracker"""
    logger.info("Testing Whale Tracker...")
    
    tracker = WhaleTracker()
    
    # Test tracking whale transactions for ETH
    logger.info("Tracking ETH whale transactions...")
    eth_whales = tracker.track_whale_transactions("eth", hours=24)
    logger.info(f"ETH whale transactions: {eth_whales}")
    
    # Test getting signals
    logger.info("Getting ETH whale activity signals...")
    eth_signals = tracker.get_whale_signals("eth", hours=24)
    logger.info(f"ETH whale activity signals: {eth_signals}")
    
    return True

def test_network_metrics_analyzer():
    """Test network metrics analyzer"""
    logger.info("Testing Network Metrics Analyzer...")
    
    analyzer = NetworkMetricsAnalyzer()
    
    # Test getting network metrics
    logger.info("Getting Ethereum network metrics...")
    metrics = analyzer.get_network_metrics()
    logger.info(f"Ethereum network metrics: {metrics}")
    
    # Test getting signals
    logger.info("Getting network metrics signals...")
    signals = analyzer.get_network_signals()
    logger.info(f"Network metrics signals: {signals}")
    
    return True

if __name__ == "__main__":
    logger.info("Starting Alchemy integration tests...")
    
    try:
        # Test Alchemy client
        test_alchemy_client()
        
        # Test exchange flow analyzer
        test_exchange_flow_analyzer()
        
        # Test whale tracker
        test_whale_tracker()
        
        # Test network metrics analyzer
        test_network_metrics_analyzer()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        raise 