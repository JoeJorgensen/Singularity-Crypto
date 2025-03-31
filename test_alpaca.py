#!/usr/bin/env python
"""
Diagnostic script to test Alpaca API and websocket connections
"""
from api.alpaca_api import AlpacaAPI
import json
import time
import inspect
import threading

def main():
    # Create simple config
    config = {'data_optimization': {'cache_ttl': 60}}
    
    # Initialize API
    print("Initializing Alpaca API...")
    api = AlpacaAPI(config)
    
    # Test direct API call
    print("\nTesting direct API call for ETH/USD:")
    try:
        # Get data - using a direct approach to see the raw response
        symbol = 'ETH/USD'
        timeframe = '1Min'
        limit = 5
        
        # Use the _parse_timeframe method to get timeframe object
        tf = api._parse_timeframe(timeframe)
        
        # Import CryptoBarsRequest for direct usage
        from alpaca.data.requests import CryptoBarsRequest
        
        # Create request parameters directly
        request_params = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            limit=limit
        )
        
        # Get the raw response
        print("Making direct API request...")
        bars_response = api.data_client.get_crypto_bars(request_params)
        
        # Inspect the type and structure of the response
        print(f"\nResponse type: {type(bars_response)}")
        print(f"Response attributes: {dir(bars_response)}")
        
        # Look for key properties
        if hasattr(bars_response, 'data'):
            print("\nData attribute exists:")
            print(f"Data type: {type(bars_response.data)}")
            if symbol in bars_response.data:
                print(f"Symbol '{symbol}' exists in data")
                print(f"Number of bars: {len(bars_response.data[symbol])}")
                print(f"First bar: {bars_response.data[symbol][0]}")
        elif hasattr(bars_response, 'bars'):
            print("\nBars attribute exists:")
            print(f"Bars type: {type(bars_response.bars)}")
            if symbol in bars_response.bars:
                print(f"Symbol '{symbol}' exists in bars")
                print(f"Number of bars: {len(bars_response.bars[symbol])}")
                print(f"First bar: {bars_response.bars[symbol][0]}")
        
        # Convert to dict for easier debugging
        try:
            response_dict = bars_response.__dict__
            print("\nResponse as dictionary:")
            print(json.dumps(response_dict, default=str, indent=2))
        except:
            print("Could not convert response to dictionary")
        
        # Try our normal get_crypto_bars method
        print("\nNow trying the standard get_crypto_bars method:")
        data = api.get_crypto_bars(symbol, timeframe, limit)
        print(f"Received data shape: {data.shape}")
        print(f"Data preview:\n{data.head(2)}")
        
    except Exception as e:
        print(f"Error in get_crypto_bars: {str(e)}")
    
    # Separate test for websocket - keep it brief
    print("\nTesting websocket connection (brief connection only):")
    try:
        # Use a separate thread to avoid blocking
        def test_websocket():
            try:
                print("Starting websocket for ETH/USD...")
                api.start_websocket(['ETH/USD'])
                print("Websocket started, waiting 3 seconds for data...")
                
                # Wait a few seconds for any data
                time.sleep(3)
                
                # Get whatever data we have
                ws_data = api.get_latest_websocket_data('ETH/USD')
                print("Websocket data received:")
                print(json.dumps(ws_data, default=str, indent=2))
                
                # Always stop the websocket when done
                print("Stopping websocket connection...")
                if hasattr(api, '_ws_client') and api._ws_client:
                    api._ws_client.stop()
                
                print("Websocket test completed")
            except Exception as e:
                print(f"Error in websocket thread: {str(e)}")
        
        # Start the websocket test in a separate thread
        ws_thread = threading.Thread(target=test_websocket)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait for the thread to complete, but limit to 5 seconds max
        ws_thread.join(timeout=5)
        
        if ws_thread.is_alive():
            print("Websocket test is taking too long - continuing with test")
        
    except Exception as e:
        print(f"Error in websocket test: {str(e)}")
        
    print("\nDiagnostic test completed")

if __name__ == "__main__":
    main() 