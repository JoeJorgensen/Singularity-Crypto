import pandas as pd
import numpy as np
import logging
import sys
import warnings
from technical.indicators import TechnicalIndicators

# Configure logging to see warnings
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
warnings.filterwarnings('always', category=FutureWarning)  # Show future warnings

def test_int64_column_assignment():
    """Test directly assigning float values to int64 columns to see the warning."""
    print("\nTesting direct assignment of float values to int64 columns...")
    
    # Create a simple dataframe with int64 columns
    df = pd.DataFrame({
        'int_col': np.arange(10, dtype=np.int64)
    })
    
    print(f"Column dtypes before: {df.dtypes}")
    
    # Try to assign float values to the int64 column - this should show the warning
    try:
        print("Attempting to assign float array to int64 column:")
        float_values = np.array([4.900566, 0.0, 0.0, 4.900794, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        df['int_col'] = float_values
        print("Assignment complete - check for warnings in output")
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"Column dtypes after: {df.dtypes}")

def create_test_dataframe(size=100):
    """Create a sample dataframe with OHLCV data."""
    date_range = pd.date_range(start='2023-01-01', periods=size, freq='h')
    
    # Create random data for testing
    np.random.seed(42)  # For reproducibility
    base_price = 1800.0
    volatility = 50.0
    
    # Generate random price movements
    price_changes = np.random.normal(0, 1, size) * volatility
    prices = base_price + np.cumsum(price_changes)
    
    # Generate OHLCV data
    high = prices + np.random.uniform(5, 20, size)
    low = prices - np.random.uniform(5, 20, size)
    close = prices + np.random.normal(0, 5, size)
    volume = np.random.uniform(10, 1000, size) * 1000
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=date_range)
    
    return df

def create_mock_dataframe_with_int64_columns():
    """Create a dataframe specifically designed to trigger the FutureWarning."""
    date_range = pd.date_range(start='2023-01-01', periods=30, freq='h')
    
    # Create a dataframe with basic data
    df = pd.DataFrame({
        'close': np.random.normal(1800, 50, 30),
        'high': np.random.normal(1850, 30, 30),
        'low': np.random.normal(1750, 30, 30),
    }, index=date_range)
    
    # Force integer dtypes with explicit conversion
    # First create as numpy arrays with int64 type
    obv_values = np.zeros(30, dtype=np.int64)
    volume_sma_values = np.zeros(30, dtype=np.int64)
    volume_osc_values = np.zeros(30, dtype=np.int64)
    mfi_values = np.ones(30, dtype=np.int64) * 50
    
    # Then add to dataframe, ensuring dtype is preserved
    df['obv'] = obv_values
    df['volume_sma20'] = volume_sma_values
    df['volume_osc'] = volume_osc_values
    df['mfi'] = mfi_values
    
    # Verify the dtypes
    print(f"Column dtypes in mock dataframe: {df.dtypes}")
    
    # Return the dataframe
    return df

def test_volume_indicators_with_int64():
    """Test with int64 columns that should trigger the warning."""
    print("\nTesting specifically to trigger FutureWarning...")
    df = create_mock_dataframe_with_int64_columns()
    
    # Create a challenging scenario: delete volume column to force error
    if 'volume' in df.columns:
        del df['volume']
    
    # Now try to add volume indicators, which should hit the exception handler
    print("Calling add_volume_indicators with missing volume column...")
    result_df = TechnicalIndicators.add_volume_indicators(df)
    
    print(f"Column dtypes after: {result_df.dtypes}")
    
    # Print values of the added/modified columns
    for col in ['obv', 'vwap', 'mfi', 'volume_sma20', 'volume_osc']:
        if col in result_df.columns:
            print(f"{col} values: {result_df[col].iloc[:5]}")

def test_reproduce_future_warning():
    """Test case designed to reproduce the exact FutureWarning from the error."""
    print("\nTesting reproduction of FutureWarning...")
    df = create_problematic_dataframe()
    
    print(f"Column dtypes before: {df.dtypes}")
    
    # Add a small amount of volume data that will cause obv to be computed with float values
    # but try to assign to an int64 column
    df['volume'] = np.random.normal(100, 10, 30)
    
    # This should now trigger the FutureWarning when calculating the indicators
    result_df = TechnicalIndicators.add_volume_indicators(df)
    
    print(f"Column dtypes after: {result_df.dtypes}")
    print(f"OBV values: {result_df['obv'].iloc[:5]}")

def create_problematic_dataframe():
    """Create a dataframe that will trigger the FutureWarning."""
    date_range = pd.date_range(start='2023-01-01', periods=30, freq='h')
    
    # Create a dataframe with some columns as int64 that will cause issues
    df = pd.DataFrame({
        'close': np.random.normal(1800, 50, 30),
        'high': np.random.normal(1850, 30, 30),
        'low': np.random.normal(1750, 30, 30),
    }, index=date_range)
    
    # Pre-create the indicator columns as int64 (this is key to trigger the warning)
    # Use numpy arrays to ensure int64 dtype is preserved
    df['obv'] = np.zeros(30, dtype=np.int64)
    df['volume_sma20'] = np.zeros(30, dtype=np.int64)
    df['volume_osc'] = np.zeros(30, dtype=np.int64)
    
    return df

def test_add_volume_indicators_error_case():
    """Test the error case in add_volume_indicators that leads to the FutureWarning."""
    print("\nTesting error case scenario...")
    df = create_incomplete_dataframe()
    
    print(f"Column dtypes before: {df.dtypes}")
    
    # This should trigger the exception and fallback to setting 0 values
    result_df = TechnicalIndicators.add_volume_indicators(df)
    
    print(f"Column dtypes after: {result_df.dtypes}")
    print(f"Columns that should have been added: {set(['obv', 'vwap', 'mfi', 'volume_sma20', 'volume_osc'])}")
    print(f"Actual columns: {set(result_df.columns) - set(df.columns)}")
    
    # Print values of the added columns
    for col in ['obv', 'vwap', 'mfi', 'volume_sma20', 'volume_osc']:
        if col in result_df.columns:
            print(f"{col} values: {result_df[col].values}")

def create_incomplete_dataframe():
    """Create a dataframe that will trigger the exception case in add_volume_indicators."""
    date_range = pd.date_range(start='2023-01-01', periods=30, freq='h')
    
    # Create a dataframe with some missing columns to force an exception
    df = pd.DataFrame({
        'close': np.random.normal(1800, 50, 30),
        # Missing 'high', 'low', 'volume' needed for indicators
    }, index=date_range)
    
    # Set dtype explicitly to int64 for demonstration
    df['col_int64'] = np.arange(30, dtype=np.int64)
    
    return df

def test_add_volume_indicators():
    """Test the add_volume_indicators function to reproduce the warning."""
    print("Creating test dataframe...")
    df = create_test_dataframe(size=30)  # Create smaller dataset to ensure error occurs
    
    # Set dtypes explicitly to match what might be happening in production
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    
    # Try with different volume types to see which one triggers the warning
    tests = [
        ("float64", lambda x: x.astype(float)),
        ("int64", lambda x: x.astype(np.int64)),
        ("small int values", lambda x: (x / 1000).astype(np.int64))
    ]
    
    for test_name, converter in tests:
        print(f"\nTesting with volume as {test_name}")
        test_df = df.copy()
        test_df['volume'] = converter(df['volume'])
        print(f"Volume dtype before: {test_df['volume'].dtype}")
        
        # Apply volume indicators
        try:
            result_df = TechnicalIndicators.add_volume_indicators(test_df)
            print(f"Success! Volume indicators added without warnings")
            print(f"OBV dtype: {result_df['obv'].dtype}")
            print(f"VWAP dtype: {result_df['vwap'].dtype}")
            print(f"MFI dtype: {result_df['mfi'].dtype}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing technical indicators...")
    # Run the most direct test first
    test_int64_column_assignment()
    # Then test our specific error case
    test_volume_indicators_with_int64()
    # Run other tests
    test_add_volume_indicators()
    test_add_volume_indicators_error_case()
    test_reproduce_future_warning()
    print("Done!") 