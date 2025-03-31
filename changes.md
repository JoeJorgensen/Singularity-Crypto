# Changes Made

1. Fixed 'position does not exist' error handling in AlpacaAPI's get_position method
2. Updated position access in trading_app.py to handle None values safely
3. Added proper error filtering in log display
4. Updated position handling in OrderManager to log position-not-found as debug instead of error
5. Added UI refresh on position close to update the interface immediately

## Changes

2024-03-28: Added dedicated models for signal generation and position calculation
2024-03-29: Added support for multi-pair trading
2024-03-30: Added custom Alpaca API wrapper with improved error handling
2024-03-30: Added TA-Lib alternatives for Python 3.12 compatibility
2024-03-30: Improved sentiment analysis module with caching
2024-03-30: Added technical analysis visualization component
2024-03-30: Updated .gitignore to exclude token files and logs

## Fixes

2024-03-31: Fixed Alpaca API websocket and crypto feed issues with Streamlit Cloud:
1. Updated the CryptoDataStream initialization to use feed="us" instead of CryptoFeed.US
2. Removed max_reconnect_attempts and other parameters that aren't supported in alpaca-py==0.10.0
3. Removed feed parameter from CryptoHistoricalDataClient initialization
4. Added test script to verify connection to Alpaca API works correctly

