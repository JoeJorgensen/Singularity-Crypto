#!/usr/bin/env python3
"""
Test script for the Alchemy client to verify the fix for handling missing from/to fields in transfers.
"""
import os
import logging
import sys
from pprint import pprint
from on_chain.alchemy.client import AlchemyClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_alchemy_transfers():
    """Test the Alchemy client's handling of transfers with missing data."""
    # Enable test mode to use mock data
    os.environ["ALCHEMY_TEST_MODE"] = "true"
    
    try:
        # Initialize the Alchemy client
        api_key = os.getenv("ALCHEMY_API_KEY")
        if not api_key:
            # If no API key is set, use a placeholder for testing
            os.environ["ALCHEMY_API_KEY"] = "test_api_key"
            
        client = AlchemyClient()
        
        # Get ETH transfers
        params = {
            "fromBlock": "0x0",
            "toBlock": "latest",
            "category": ["external"]
        }
        
        logger.info("Requesting ETH transfers from Alchemy API (mock data)")
        transfers = client.get_asset_transfers(params)
        
        # Verify the results
        logger.info(f"Got {len(transfers.get('transfers', []))} transfers")
        
        # Check if any transfers have missing from/to fields (there should be none)
        invalid_transfers = [
            t for t in transfers.get('transfers', []) 
            if not t.get('from') or not t.get('to')
        ]
        
        logger.info(f"Number of transfers with missing from/to fields: {len(invalid_transfers)}")
        assert len(invalid_transfers) == 0, "Found transfers with missing from/to fields"
        
        logger.info("All transfers have valid from/to fields - test passed!")
        
        # Print a sample transfer for inspection
        if transfers.get('transfers'):
            logger.info("Sample transfer:")
            pprint(transfers['transfers'][0])
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_alchemy_transfers()
    sys.exit(0 if success else 1) 