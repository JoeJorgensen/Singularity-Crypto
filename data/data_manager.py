"""
Data Manager for handling ETH/USD market data retrieval and processing.
"""
import pandas as pd
import numpy as np
from technical.indicators import TechnicalIndicators

class DataManager:
    """Class for managing ETH/USD market data retrieval and processing."""
    
    def __init__(self, alpaca, config):
        """
        Initialize the DataManager.
        
        Args:
            alpaca: AlpacaAPI instance for data retrieval
            config: Application configuration
        """
        self.alpaca = alpaca
        self.config = config
        self.indicators = TechnicalIndicators()
    
    def get_bars(self, timeframe, limit=100):
        """
        Get historical bar data for ETH/USD.
        
        Args:
            timeframe: Time interval (e.g., '1Min', '5Min', '1D')
            limit: Maximum number of bars to retrieve
            
        Returns:
            DataFrame containing historical price data
        """
        try:
            # Use Alpaca API to get bars data for ETH/USD only
            symbol = 'ETH/USD'
            bars = self.alpaca.get_bars(symbol, timeframe, limit)
            
            if bars is None or len(bars) == 0:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            
            # Set index to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            print(f"Error fetching ETH/USD bars data: {e}")
            return pd.DataFrame()
    
    def get_bars_with_indicators(self, timeframe, limit=100):
        """
        Get historical ETH/USD bar data with technical indicators added.
        
        Args:
            timeframe: Time interval (e.g., '1Min', '5Min', '1D')
            limit: Maximum number of bars to retrieve
            
        Returns:
            DataFrame containing historical price data with indicators
        """
        df = self.get_bars(timeframe, limit)
        
        if df.empty:
            return df
        
        # Add technical indicators
        df = self.indicators.add_all_indicators(df)
        
        return df 