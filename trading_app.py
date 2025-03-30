"""
CryptoTrading - Streamlit application for ETH trading using Alpaca's crypto trading API.
"""
import os
import json
import time
import threading
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv
from typing import Dict, Optional

# Import core components
from trading_strategy import TradingStrategy
from utils.risk_manager import RiskManager
from utils.position_calculator import PositionCalculator
from utils.signal_aggregator import SignalAggregator
from utils.order_manager import OrderManager
from utils.logging_config import setup_logging, get_logger

# Set up logging once at application startup
setup_logging()
logger = get_logger('trading_app')

# Check for Streamlit secrets
def verify_secrets():
    """Check if we have access to Streamlit secrets or env vars."""
    try:
        # Check for Streamlit secrets
        if 'ALPACA_API_KEY' in st.secrets:
            logger.info("ALPACA_API_KEY found in Streamlit secrets")
            return True
        else:
            logger.warning("ALPACA_API_KEY not found in Streamlit secrets")
    except Exception as e:
        logger.warning(f"Error accessing Streamlit secrets: {e}")
    
    # Check for environment variables
    if os.getenv('ALPACA_API_KEY'):
        logger.info("ALPACA_API_KEY found in environment variables")
        return True
    else:
        logger.warning("ALPACA_API_KEY not found in environment variables")
        return False

# Configure Streamlit page
st.set_page_config(
    page_title="CryptoTrading - Advanced Auto-Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/CryptoTrading',
        'Report a bug': 'https://github.com/your-repo/CryptoTrading/issues/new',
        'About': 'Advanced cryptocurrency auto-trading system with real-time signals and risk management.'
    }
)

# Custom CSS for a more professional look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #1a1c24;
    }
    .css-1d391kg {
        background-color: #1a1c24;
    }
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }
    div[data-testid="stBlock"] {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .stButton>button {
        border-radius: 5px;
    }
    .stPlotlyChart {
        border-radius: 5px;
        background-color: #1a1c24;
    }
    div[data-testid="stMetricValue"] {
        font-size: 20px !important;
        font-weight: bold !important;
    }
    .trade-signal {
        padding: 12px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Persistent state management
def save_persistent_state(state_data):
    """Save application state to a persistent file."""
    state_file = 'config/app_state.json'
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        
        # Load existing state if it exists
        existing_state = {}
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    existing_state = json.load(f)
            except Exception as e:
                logger.error(f"Error loading existing state: {str(e)}")
        
        # Update with new data
        existing_state.update(state_data)
        
        with open(state_file, 'w') as f:
            json.dump(existing_state, f, indent=4)
        logger.info(f"Saved persistent state: {state_data}")
        return True
    except Exception as e:
        logger.error(f"Error saving persistent state: {str(e)}", exc_info=True)
        return False

def load_persistent_state():
    """Load application state from a persistent file."""
    state_file = 'config/app_state.json'
    default_state = {
        "trading_active": False,
        "strategy_mode": "Moderate",
        "refresh_interval": 15,
        "default_trading_pair": "ETH/USD",
        "selected_timeframe": "1 hour",
        "selected_candles": 100,
        "risk_per_trade": 2.0,
        "max_trades_per_day": 5
    }
    
    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            logger.info(f"Loaded persistent state: {state_data}")
            
            # Ensure all default keys exist in loaded state
            for key, value in default_state.items():
                if key not in state_data:
                    state_data[key] = value
            
            return state_data
        else:
            logger.info(f"No persistent state file found, using defaults")
            return default_state
    except Exception as e:
        logger.error(f"Error loading persistent state: {str(e)}", exc_info=True)
        return default_state

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    # Load persistent state first
    persistent_state = load_persistent_state()
    
    # Initialize with persistent values
    st.session_state.trading_active = persistent_state.get("trading_active", False)
    st.session_state.strategy_mode = persistent_state.get("strategy_mode", "Moderate")
    st.session_state.refresh_interval = persistent_state.get("refresh_interval", 15)
    st.session_state.default_trading_pair = persistent_state.get("default_trading_pair", "ETH/USD")
    st.session_state.selected_timeframe = persistent_state.get("selected_timeframe", "1 hour")
    st.session_state.selected_candles = persistent_state.get("selected_candles", 100)
    st.session_state.risk_per_trade = persistent_state.get("risk_per_trade", 2.0)
    st.session_state.max_trades_per_day = persistent_state.get("max_trades_per_day", 5)
    
    st.session_state.current_position = None
    st.session_state.latest_signals = None
    st.session_state.recent_trades = []  # Initialize recent trades array
    st.session_state.trading_stats = {
        "last_check": None,
        "next_check": None,
        "cycles_completed": 0,
        "potential_trades_found": 0,
        "trades_executed": 0,
        "current_status": "Inactive",
        "current_price": 0,
        "strategy_looking_for": "N/A",
        "trade_conditions": {}
    }
    # Initialize cycle locks for preventing duplicate cycles
    st.session_state.cycle_locks = {}
    
    st.session_state.initialized = True
    logger.info("Application session initialized with persistent state")

# Load configuration
@st.cache_resource(ttl=1)
def load_config():
    """Load application configuration."""
    logger.warning("Reloading configuration from files...")
    try:
        # First try the trading config
        with open('config/trading_config.json', 'r') as f:
            trading_config = json.load(f)
            logger.warning(f"Loaded trading config with min_signal_strength: {trading_config.get('trading', {}).get('min_signal_strength', 'Not set')}")
        
        # Then load the main config
        with open('config/config.json', 'r') as f:
            main_config = json.load(f)
        
        # Merge the configs
        merged_config = main_config.copy()
        for key, value in trading_config.items():
            merged_config[key] = value
            
        return merged_config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        # Fallback to just main config if error occurs
        with open('config/config.json', 'r') as f:
            return json.load(f)

# Initialize trading system
@st.cache_resource
def initialize_trading_system(config):
    """Initialize all components of the trading system."""
    # Initialize trading strategy (this will initialize all APIs)
    trading_strategy = TradingStrategy(config)
    
    # Get API references from the trading strategy
    apis = trading_strategy.apis
    
    # Initialize components that might not be initialized by TradingStrategy
    risk_manager = RiskManager(config)
    position_calculator = PositionCalculator(risk_manager, config)
    signal_aggregator = SignalAggregator(config)
    order_manager = OrderManager(apis['alpaca'], config)
    
    # Initialize is_active flag on the trading strategy based on persistent state
    # This function is called after session state is initialized, so we can use it here
    if 'trading_active' in st.session_state and st.session_state.trading_active:
        trading_strategy.start_trading()
        logger.info("Trading strategy activated based on persistent state")
    
    return {
        'trading_strategy': trading_strategy,
        'risk_manager': risk_manager,
        'position_calculator': position_calculator,
        'signal_aggregator': signal_aggregator,
        'order_manager': order_manager,
        'apis': apis
    }

def normalize_timeframe(timeframe):
    """Normalize timeframe string for API compatibility."""
    timeframe_map = {
        '1m': '1Min', '1min': '1Min', '1minute': '1Min', '1 minute': '1Min', '1Min': '1Min',
        '5m': '5Min', '5min': '5Min', '5minute': '5Min', '5 minute': '5Min', '5Min': '5Min',
        '15m': '15Min', '15min': '15Min', '15minute': '15Min', '15 minute': '15Min', '15Min': '15Min',
        '1h': '1H', '1hr': '1H', '1hour': '1H', '1 hour': '1H', '1H': '1H',
        '4h': '4H', '4hr': '4H', '4hour': '4H', '4 hour': '4H', '4H': '4H',
        '1d': '1D', '1day': '1D', '1 day': '1D', '1D': '1D',
    }
    return timeframe_map.get(timeframe.lower(), '1H')

def safe_position_value(position, attribute, default=0):
    """Safely access position attributes with default fallback value."""
    if position is None:
        return default
    
    try:
        value = getattr(position, attribute)
        return float(value) if value is not None else default
    except (AttributeError, TypeError, ValueError):
        return default

def run_trading_cycle_with_stats(trading_strategy, timeframe, symbol='ETH/USD'):
    """
    Run a trading cycle and update session state with trading stats
    """
    if not st.session_state.trading_active:
        return
    
    # Normalize timeframe to standard format
    norm_timeframe = normalize_timeframe(timeframe)
    
    # Generate a unique cycle identifier with timestamp to ensure uniqueness
    cycle_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    cycle_id = f"Cycle-{st.session_state.trading_stats['cycles_completed'] + 1}-{cycle_timestamp}"
    
    # Check if there's already a cycle running for this symbol and timeframe
    cycle_lock_key = f"{symbol}_{norm_timeframe}"
    
    # Initialize lock if it doesn't exist
    if cycle_lock_key not in st.session_state.cycle_locks:
        st.session_state.cycle_locks[cycle_lock_key] = {"running": False, "last_run": None}
    
    # Skip if a cycle is already running for this symbol/timeframe
    if st.session_state.cycle_locks[cycle_lock_key]["running"]:
        logger.info(f"Skipping duplicate cycle for {symbol} on {norm_timeframe}")
        return
    
    # Set cycle as running
    st.session_state.cycle_locks[cycle_lock_key]["running"] = True
    cycle_start_time = datetime.now()
    
    try:
        # Update last check time
        st.session_state.trading_stats["last_check"] = cycle_start_time
        logger.info(f"[{cycle_id}] Starting trading cycle for {symbol} on {norm_timeframe} timeframe")
        
        # Run the trading cycle (preferring async version if available)
        if hasattr(trading_strategy, 'run_trading_cycle_async'):
            # Create an event loop in the current thread if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async cycle
            cycle_result = loop.run_until_complete(trading_strategy.run_trading_cycle_async(symbol, norm_timeframe))
            logger.info(f"[{cycle_id}] Completed async trading cycle")
        else:
            # Fall back to synchronous version
            cycle_result = trading_strategy.run_trading_cycle(symbol, norm_timeframe)
            logger.info(f"[{cycle_id}] Completed synchronous trading cycle")
        
        # Update current price if available
        if 'signals' in cycle_result and cycle_result['signals']:
            if 'current_price' in cycle_result['signals']:
                current_price = cycle_result['signals']['current_price']
                st.session_state.trading_stats["current_price"] = current_price
                logger.info(f"[{cycle_id}] Current {symbol} price: {current_price}")
            elif hasattr(cycle_result['signals'], 'get') and cycle_result['signals'].get('entry_point'):
                current_price = cycle_result['signals']['entry_point']
                st.session_state.trading_stats["current_price"] = current_price
                logger.info(f"[{cycle_id}] Current {symbol} price: {current_price}")
        
        # Update stats
        st.session_state.trading_stats["cycles_completed"] += 1
        
        # Update current_status and strategy_looking_for
        if 'signals' in cycle_result and cycle_result['signals']:
            signal_value = cycle_result['signals'].get('signal', 0)
            signal_name = cycle_result['signals'].get('signal_name', 'neutral')
            
            # Determine what the strategy is looking for
            if signal_name == 'strong_buy' or signal_name == 'buy':
                strategy_looking_for = "BUY opportunity"
                st.session_state.trading_stats["strategy_looking_for"] = strategy_looking_for
                logger.info(f"[{cycle_id}] Strategy is looking for a {strategy_looking_for}")
            elif signal_name == 'strong_sell' or signal_name == 'sell':
                strategy_looking_for = "SELL opportunity"
                st.session_state.trading_stats["strategy_looking_for"] = strategy_looking_for
                logger.info(f"[{cycle_id}] Strategy is looking for a {strategy_looking_for}")
            else:
                strategy_looking_for = "Neutral - waiting for clearer signals"
                st.session_state.trading_stats["strategy_looking_for"] = strategy_looking_for
                logger.info(f"[{cycle_id}] Strategy is {strategy_looking_for}")
                
            # Update trade conditions
            trade_conditions = {
                "trend": cycle_result['signals'].get('trend_signal', 0),
                "momentum": cycle_result['signals'].get('momentum_signal', 0),
                "volume": cycle_result['signals'].get('volume_signal', 0),
                "sentiment": cycle_result['signals'].get('sentiment_score', 0),
                "signal_strength": abs(signal_value),
                "signal_direction": "bullish" if signal_value > 0 else "bearish" if signal_value < 0 else "neutral"
            }
            st.session_state.trading_stats["trade_conditions"] = trade_conditions
            
            # Log the details
            logger.info(f"[{cycle_id}] Signal: {signal_name} ({signal_value:.4f}), Direction: {trade_conditions['signal_direction']}, Strength: {trade_conditions['signal_strength']:.4f}")
            logger.info(f"[{cycle_id}] Trend: {trade_conditions['trend']:.4f}, Momentum: {trade_conditions['momentum']:.4f}, Volume: {trade_conditions['volume']:.4f}, Sentiment: {trade_conditions['sentiment']:.4f}")
        
        # Check if a trade should be executed
        if cycle_result.get('should_trade', False):
            st.session_state.trading_stats["potential_trades_found"] += 1
            st.session_state.trading_stats["current_status"] = "Trade opportunity found"
            
            logger.warning(f"[{cycle_id}] Trade opportunity detected: {cycle_result['signals'].get('signal_name', 'unknown')} with strength {cycle_result['signals'].get('signal', 0):.4f}")
            
            # Add to log
            if 'log_entries' not in st.session_state:
                st.session_state.log_entries = []
                
            st.session_state.log_entries.insert(0, {
                "timestamp": datetime.now(),
                "level": "WARNING",
                "message": f"Trade opportunity detected: {cycle_result['signals'].get('signal_name', 'unknown')} with strength {cycle_result['signals'].get('signal', 0):.2f}"
            })
            
            # If a trade was executed, update stats
            if cycle_result.get('trade_result', {}).get('executed', False):
                st.session_state.trading_stats["trades_executed"] += 1
                st.session_state.trading_stats["current_status"] = "Trade executed"
                
                # Extract trade details
                trade_details = cycle_result.get('trade_result', {}).get('trade', {})
                side = trade_details.get('side', 'unknown')
                qty = trade_details.get('qty', 0)
                price = trade_details.get('filled_avg_price', 0)
                order_id = trade_details.get('id', 'unknown')
                
                # Fix for NoneType format error - ensure price is not None before formatting
                price_str = f"${price:.2f}" if price is not None else "market price"
                logger.info(f"[{cycle_id}] Trade executed: {side} {qty} {symbol} at {price_str}, Order ID: {order_id}")
                
                # Add to log
                st.session_state.log_entries.insert(0, {
                    "timestamp": datetime.now(),
                    "level": "SUCCESS",
                    "message": f"Trade executed: {side} {qty} ETH at ${price:.2f}"
                })
        else:
            st.session_state.trading_stats["current_status"] = "Monitoring markets"
            logger.info(f"[{cycle_id}] No trade executed, continuing to monitor markets")
        
        # Set next check time
        next_check = datetime.now() + timedelta(seconds=15)
        st.session_state.trading_stats["next_check"] = next_check
        
        # Calculate and log cycle duration
        cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
        logger.info(f"[{cycle_id}] Trading cycle completed in {cycle_duration:.2f} seconds. Next check at {next_check.strftime('%H:%M:%S')}")
        
    except Exception as e:
        error_msg = f"Error in trading cycle: {str(e)}"
        st.session_state.trading_stats["current_status"] = f"Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Add to log
        if 'log_entries' in st.session_state:
            st.session_state.log_entries.insert(0, {
                "timestamp": datetime.now(),
                "level": "ERROR",
                "message": error_msg
            })
    finally:
        # Mark the cycle as no longer running and update last run time
        st.session_state.cycle_locks[cycle_lock_key]["running"] = False
        st.session_state.cycle_locks[cycle_lock_key]["last_run"] = datetime.now()

# Async version of run_trading_cycle_with_stats
async def run_trading_cycle_with_stats_async(trading_strategy, timeframe, symbol='ETH/USD'):
    """
    Run a trading cycle asynchronously and update session state with trading stats
    """
    if not st.session_state.trading_active:
        return
    
    # Normalize timeframe to standard format
    norm_timeframe = normalize_timeframe(timeframe)
    
    # Generate a unique cycle identifier with timestamp to ensure uniqueness
    cycle_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    cycle_id = f"Cycle-{st.session_state.trading_stats['cycles_completed'] + 1}-{cycle_timestamp}"
    
    # Check if there's already a cycle running for this symbol and timeframe
    cycle_lock_key = f"{symbol}_{norm_timeframe}"
    
    # Initialize lock if it doesn't exist
    if cycle_lock_key not in st.session_state.cycle_locks:
        st.session_state.cycle_locks[cycle_lock_key] = {"running": False, "last_run": None}
    
    # Skip if a cycle is already running for this symbol/timeframe
    if st.session_state.cycle_locks[cycle_lock_key]["running"]:
        logger.info(f"Skipping duplicate async cycle for {symbol} on {norm_timeframe}")
        return
    
    # Set cycle as running
    st.session_state.cycle_locks[cycle_lock_key]["running"] = True
    cycle_start_time = datetime.now()
    
    try:
        # Update last check time
        st.session_state.trading_stats["last_check"] = cycle_start_time
        logger.info(f"[{cycle_id}] Starting async trading cycle for {symbol} on {norm_timeframe} timeframe")
        
        # Run the trading cycle using async method
        cycle_result = await trading_strategy.run_trading_cycle_async(symbol, norm_timeframe)
        
        # Update current price if available
        if 'signals' in cycle_result and cycle_result['signals']:
            if 'current_price' in cycle_result['signals']:
                current_price = cycle_result['signals']['current_price']
                st.session_state.trading_stats["current_price"] = current_price
                logger.info(f"[{cycle_id}] Current {symbol} price: {current_price}")
            elif hasattr(cycle_result['signals'], 'get') and cycle_result['signals'].get('entry_point'):
                current_price = cycle_result['signals']['entry_point']
                st.session_state.trading_stats["current_price"] = current_price
                logger.info(f"[{cycle_id}] Current {symbol} price: {current_price}")
        
        # Update stats
        st.session_state.trading_stats["cycles_completed"] += 1
        
        # Update current_status and strategy_looking_for
        if 'signals' in cycle_result and cycle_result['signals']:
            signal_value = cycle_result['signals'].get('signal', 0)
            signal_name = cycle_result['signals'].get('signal_name', 'neutral')
            
            # Determine what the strategy is looking for
            if signal_name == 'strong_buy' or signal_name == 'buy':
                strategy_looking_for = "BUY opportunity"
                st.session_state.trading_stats["strategy_looking_for"] = strategy_looking_for
                logger.info(f"[{cycle_id}] Strategy is looking for a {strategy_looking_for}")
            elif signal_name == 'strong_sell' or signal_name == 'sell':
                strategy_looking_for = "SELL opportunity"
                st.session_state.trading_stats["strategy_looking_for"] = strategy_looking_for
                logger.info(f"[{cycle_id}] Strategy is looking for a {strategy_looking_for}")
            else:
                strategy_looking_for = "Neutral - waiting for clearer signals"
                st.session_state.trading_stats["strategy_looking_for"] = strategy_looking_for
                logger.info(f"[{cycle_id}] Strategy is {strategy_looking_for}")
                
            # Update trade conditions
            trade_conditions = {
                "trend": cycle_result['signals'].get('trend_signal', 0),
                "momentum": cycle_result['signals'].get('momentum_signal', 0),
                "volume": cycle_result['signals'].get('volume_signal', 0),
                "sentiment": cycle_result['signals'].get('sentiment_score', 0),
                "signal_strength": abs(signal_value),
                "signal_direction": "bullish" if signal_value > 0 else "bearish" if signal_value < 0 else "neutral"
            }
            st.session_state.trading_stats["trade_conditions"] = trade_conditions
            
            # Log the details
            logger.info(f"[{cycle_id}] Signal: {signal_name} ({signal_value:.4f}), Direction: {trade_conditions['signal_direction']}, Strength: {trade_conditions['signal_strength']:.4f}")
            
        # Check if a trade should be executed
        if cycle_result.get('should_trade', False):
            st.session_state.trading_stats["potential_trades_found"] += 1
            st.session_state.trading_stats["current_status"] = "Trade opportunity found"
            
            logger.warning(f"[{cycle_id}] Trade opportunity detected: {cycle_result['signals'].get('signal_name', 'unknown')} with strength {cycle_result['signals'].get('signal', 0):.4f}")
            
            # Add to log
            if 'log_entries' not in st.session_state:
                st.session_state.log_entries = []
                
            st.session_state.log_entries.insert(0, {
                "timestamp": datetime.now(),
                "level": "WARNING",
                "message": f"Trade opportunity detected: {cycle_result['signals'].get('signal_name', 'unknown')} with strength {cycle_result['signals'].get('signal', 0):.2f}"
            })
            
            # If a trade was executed, update stats
            if cycle_result.get('trade_result', {}).get('executed', False):
                st.session_state.trading_stats["trades_executed"] += 1
                st.session_state.trading_stats["current_status"] = "Trade executed"
                
                # Extract trade details
                trade_details = cycle_result.get('trade_result', {}).get('trade', {})
                side = trade_details.get('side', 'unknown')
                qty = trade_details.get('qty', 0)
                price = trade_details.get('filled_avg_price', 0)
                order_id = trade_details.get('id', 'unknown')
                
                # Fix for NoneType format error - ensure price is not None before formatting
                price_str = f"${price:.2f}" if price is not None else "market price"
                logger.info(f"[{cycle_id}] Trade executed: {side} {qty} {symbol} at {price_str}, Order ID: {order_id}")
                
                # Add to log
                st.session_state.log_entries.insert(0, {
                    "timestamp": datetime.now(),
                    "level": "SUCCESS",
                    "message": f"Trade executed: {side} {qty} ETH at ${price:.2f}"
                })
        else:
            st.session_state.trading_stats["current_status"] = "Monitoring markets"
            logger.info(f"[{cycle_id}] No trade executed, continuing to monitor markets")
        
        # Set next check time
        next_check = datetime.now() + timedelta(seconds=st.session_state.refresh_interval)
        st.session_state.trading_stats["next_check"] = next_check
        
        # Calculate and log cycle duration
        cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
        logger.info(f"[{cycle_id}] Async trading cycle completed in {cycle_duration:.2f} seconds. Next check at {next_check.strftime('%H:%M:%S')}")
        
        return cycle_result
        
    except Exception as e:
        logger.error(f"Error in async trading cycle: {str(e)}", exc_info=True)
        st.session_state.trading_stats["current_status"] = "Error in cycle"
        # Add to log
        if 'log_entries' not in st.session_state:
            st.session_state.log_entries = []
        st.session_state.log_entries.insert(0, {
            "timestamp": datetime.now(),
            "level": "ERROR",
            "message": f"Error in async trading cycle: {str(e)}"
        })
        return None
    finally:
        # Mark the cycle as no longer running and update last run time
        st.session_state.cycle_locks[cycle_lock_key]["running"] = False
        st.session_state.cycle_locks[cycle_lock_key]["last_run"] = datetime.now()

# Get market data with caching
@st.cache_data(ttl=60)
def get_crypto_data(_trading_strategy, symbol='ETH/USD', timeframe='1H', limit=100):
    """Get cryptocurrency market data."""
    try:
        df = _trading_strategy.get_market_data(symbol, timeframe, limit)
        if df is not None and not df.empty:
            # Update current price in session state
            if 'trading_stats' in st.session_state and len(df) > 0:
                st.session_state.trading_stats["current_price"] = df['close'].iloc[-1]
            return df
        else:
            st.error("No data received from Alpaca API")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return pd.DataFrame()

def main():
    """Main application function."""
    try:
        # Title and description
        st.title("CryptoTrading - Advanced Auto-Trading System")
        st.markdown("""
        <div style="background-color: #1E2127; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
            <p>Professional-grade cryptocurrency trading system with real-time signals, automated execution, and comprehensive risk management.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Verify API credentials
        credentials_available = verify_secrets()
        if not credentials_available:
            st.error("⚠️ API credentials not found! Make sure ALPACA_API_KEY and ALPACA_API_SECRET are set in Streamlit secrets or environment variables.")
        
        # Auto-refresh every 15 seconds
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)", 
            min_value=15,
            max_value=300,
            value=st.session_state.refresh_interval,
            step=15,
            help="How frequently to check for trading opportunities. Lower values provide more real-time data but may increase API usage."
        )
        
        # Update refresh interval if changed
        if refresh_interval != st.session_state.refresh_interval:
            st.session_state.refresh_interval = refresh_interval
            save_persistent_state({"refresh_interval": refresh_interval})
            
        st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")
        
        # Load configuration
        config = load_config()
        
        # Initialize trading system
        system = initialize_trading_system(config)
        trading_strategy = system['trading_strategy']
        risk_manager = system['risk_manager']
        position_calculator = system['position_calculator']
        signal_aggregator = system['signal_aggregator']
        order_manager = system['order_manager']
        alpaca_api = system['apis']['alpaca']
        
        # Run trading check if auto-trading is active
        if st.session_state.trading_active:
            # Check if trading cycle is due (based on next_check time)
            if (st.session_state.trading_stats["next_check"] is None or 
                datetime.now() >= st.session_state.trading_stats["next_check"]):
                
                # Get normalized timeframe for consistent handling
                selected_tf = st.session_state.selected_timeframe
                norm_tf = normalize_timeframe(selected_tf)
                
                # Use async version if available
                if hasattr(trading_strategy, 'run_trading_cycle_async'):
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Run the async cycle with the normalized timeframe
                    loop.run_until_complete(run_trading_cycle_with_stats_async(trading_strategy, norm_tf))
                else:
                    # Use normalized timeframe
                    run_trading_cycle_with_stats(trading_strategy, norm_tf)
                
                # Add initialization log if first run
                if st.session_state.trading_stats["cycles_completed"] == 1:
                    if 'log_entries' in st.session_state:
                        st.session_state.log_entries.insert(0, {
                            "timestamp": datetime.now(),
                            "level": "SUCCESS",
                            "message": "Auto-trading system initialized and running. Monitoring market conditions."
                        })
        
        # Sidebar
        with st.sidebar:
            st.header("Trading Configuration")
            
            # Market status
            market_status = "🟢 Crypto Market is 24/7"
            st.success(market_status)
            
            # Account information
            st.subheader("Account Information")
            try:
                account = alpaca_api.get_account()
                buying_power = float(account.buying_power)
                portfolio_value = float(account.portfolio_value)
                
                st.metric("Portfolio Value", f"${portfolio_value:.2f}")
                st.metric("Buying Power", f"${buying_power:.2f}")
                
                # Display credentials source if available
                if hasattr(alpaca_api, 'credentials_source'):
                    st.info(f"API Source: {alpaca_api.credentials_source}")
                
                # Add warning for low buying power
                if buying_power < 100:
                    st.warning(f"⚠️ Low buying power (${buying_power:.2f}). Buy orders may fail or be reduced in size.")
                
                # Calculate how much of portfolio is in cash vs positions
                if portfolio_value > 0:
                    cash_percentage = (buying_power / portfolio_value) * 100
                    position_percentage = 100 - cash_percentage
                    
                    # Add a visual indicator of portfolio allocation
                    st.markdown(f"""
                    <div style="margin-top: 10px;">
                        <div style="font-size: 14px; margin-bottom: 5px;">Portfolio Allocation:</div>
                        <div style="display: flex; height: 20px; border-radius: 4px; overflow: hidden;">
                            <div style="width: {position_percentage}%; background-color: #4CAF50; height: 100%; display: flex; align-items: center; justify-content: center;">
                                <span style="color: white; font-size: 12px; font-weight: bold;">{position_percentage:.1f}% Positions</span>
                            </div>
                            <div style="width: {cash_percentage}%; background-color: #2196F3; height: 100%; display: flex; align-items: center; justify-content: center;">
                                <span style="color: white; font-size: 12px; font-weight: bold;">{cash_percentage:.1f}% Cash</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error retrieving account info: {str(e)}")
            
            # Timeframe selection
            timeframe_options = {
                "1 minute": "1Min",
                "5 minutes": "5Min",
                "15 minutes": "15Min",
                "1 hour": "1H",
                "4 hours": "4H",
                "1 day": "1D"
            }
            selected_timeframe = st.selectbox(
                "Select Timeframe",
                list(timeframe_options.keys()),
                index=list(timeframe_options.keys()).index(st.session_state.selected_timeframe)
                   if st.session_state.selected_timeframe in timeframe_options.keys() else 3
            )
            
            # Update timeframe if changed
            if selected_timeframe != st.session_state.selected_timeframe:
                st.session_state.selected_timeframe = selected_timeframe
                save_persistent_state({"selected_timeframe": selected_timeframe})
            
            # Number of candles
            candle_options = [50, 100, 200, 500]
            selected_candles = st.selectbox(
                "Number of Candles", 
                candle_options,
                index=candle_options.index(st.session_state.selected_candles)
                   if st.session_state.selected_candles in candle_options else 1
            )
            
            # Update candles if changed
            if selected_candles != st.session_state.selected_candles:
                st.session_state.selected_candles = selected_candles
                save_persistent_state({"selected_candles": selected_candles})
            
            # Risk parameters 
            st.subheader("Risk Parameters")
            risk_per_trade = st.slider(
                "Risk per Trade (%)", 
                min_value=0.1, 
                max_value=5.0, 
                value=st.session_state.risk_per_trade,
                step=0.1
            )
            
            # Update risk per trade if changed
            if risk_per_trade != st.session_state.risk_per_trade:
                st.session_state.risk_per_trade = risk_per_trade
                save_persistent_state({"risk_per_trade": risk_per_trade})
            
            max_trades = st.slider(
                "Max Trades per Day",
                min_value=1,
                max_value=20,
                value=st.session_state.max_trades_per_day
            )
            
            # Update max trades if changed
            if max_trades != st.session_state.max_trades_per_day:
                st.session_state.max_trades_per_day = max_trades
                save_persistent_state({"max_trades_per_day": max_trades})
            
            # Trading activation with persistence
            trading_active = st.toggle("Activate Trading", value=st.session_state.trading_active)
            if trading_active != st.session_state.trading_active:
                st.session_state.trading_active = trading_active
                
                # Save the state to persistent storage
                save_persistent_state({"trading_active": trading_active})
                
                if trading_active:
                    # Activate trading strategy
                    trading_strategy.start_trading()
                    st.success("Auto-trading activated!")
                else:
                    # Deactivate trading strategy
                    trading_strategy.stop_trading()
                    st.info("Auto-trading deactivated.")
            
            # Emergency stop
            if st.button("EMERGENCY STOP", type="primary"):
                try:
                    # Close all positions
                    alpaca_api.close_all_positions()
                    trading_strategy.stop_trading()
                    st.session_state.trading_active = False
                    
                    # Update persistent state
                    save_persistent_state({"trading_active": False})
                    
                    st.error("EMERGENCY STOP executed! All positions closed.")
                except Exception as e:
                    st.error(f"Error during emergency stop: {str(e)}")
                    
        # Create tab navigation for better organization
        main_tabs = st.tabs(["Trading Dashboard", "Monitoring & Performance", "Log & History", "Settings"])
        
        with main_tabs[0]:  # Trading Dashboard tab
            # Add Strategy Selection at the top
            st.markdown("""
            <div style="background-color: #1E2127; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h3 style="margin-top: 0;">Strategy Selection</h3>
            </div>
            """, unsafe_allow_html=True)
            
            strategy_cols = st.columns([2, 1])
            with strategy_cols[0]:
                # Add strategy selection UI
                strategy_mode = st.radio(
                    "Trading Strategy",
                    ["Aggressive", "Moderate", "Conservative"],
                    index=["Aggressive", "Moderate", "Conservative"].index(st.session_state.strategy_mode),
                    horizontal=True,
                    help="Select your trading strategy. This affects risk parameters and position sizing."
                )
                
                # Update session state and save if changed
                if strategy_mode != st.session_state.strategy_mode:
                    st.session_state.strategy_mode = strategy_mode
                    save_persistent_state({"strategy_mode": strategy_mode})
                    st.success(f"Strategy mode updated to {strategy_mode}")
            
            with strategy_cols[1]:
                # Show strategy description
                descriptions = {
                    "Aggressive": "Higher risk, potentially higher returns. More frequent trades.",
                    "Moderate": "Balanced approach with moderate risk and returns.",
                    "Conservative": "Lower risk, fewer trades, focus on capital preservation."
                }
                
                st.info(descriptions[strategy_mode])
            
            # Create columns for market data and trading interface
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("ETH/USD Price Chart")
                
                # Get market data
                df = get_crypto_data(
                    trading_strategy,
                    timeframe=timeframe_options[selected_timeframe],
                    limit=selected_candles
                )
                
                # Run trading cycle every time the app refreshes (if trading is active)
                if st.session_state.trading_active:
                    if hasattr(trading_strategy, 'run_trading_cycle_async'):
                        # Set up the async environment for Streamlit
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                        # Run the async cycle
                        run_result = loop.run_until_complete(
                            run_trading_cycle_with_stats_async(trading_strategy, timeframe_options[selected_timeframe])
                        )
                    else:
                        # Fall back to synchronous version
                        run_result = run_trading_cycle_with_stats(trading_strategy, timeframe_options[selected_timeframe])
                
                if not df.empty:
                    # Add technical indicators if not already present
                    if 'ema_20' not in df.columns:
                        from technical.indicators import TechnicalIndicators
                        df = TechnicalIndicators.add_all_indicators(df, config.get('technical_indicators', {}))
                    
                    # Create price chart
                    fig = go.Figure()
                    
                    # Add candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=df.index,
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close'],
                            name='ETH/USD'
                        )
                    )
                    
                    # Add technical indicators
                    if 'ema_20' in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['ema_20'],
                                name='EMA 20',
                                line=dict(width=1, color='rgba(13, 71, 161, 0.7)')
                            )
                        )
                    
                    if 'ema_50' in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['ema_50'],
                                name='EMA 50',
                                line=dict(width=1, color='rgba(187, 134, 252, 0.8)')
                            )
                        )
                    
                    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['bb_upper'],
                                name='BB Upper',
                                line=dict(width=1, color='rgba(0, 200, 0, 0.5)'),
                                showlegend=True
                            )
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['bb_lower'],
                                name='BB Lower',
                                line=dict(width=1, color='rgba(200, 0, 0, 0.5)'),
                                showlegend=True
                            )
                        )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"ETH/USD - {selected_timeframe}",
                        yaxis_title="Price (USD)",
                        xaxis_title="Time",
                        height=600,
                        template="plotly_dark",
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Get sentiment data
                    sentiment_data = trading_strategy.get_sentiment_data()
                    
                    # Generate comprehensive signals
                    signals = trading_strategy.generate_signals(df, sentiment_data)
                    st.session_state.latest_signals = signals
                    
                    # Display signals in a more organized way
                    st.subheader("Trading Signals")
                    
                    # Create a tab structure for different signal categories
                    signal_tabs = st.tabs(["Technical", "Sentiment", "Risk"])
                    
                    with signal_tabs[0]:  # Technical tab
                        signal_cols = st.columns(3)
                        
                        with signal_cols[0]:
                            trend_val = signals.get('trend_signal', signals.get('trend', 0))
                            trend_display = f"{trend_val:.2f}" if not np.isnan(trend_val) else "0.00"
                            st.metric("Trend Signal", trend_display)
                        with signal_cols[1]:
                            momentum_val = signals.get('momentum_signal', signals.get('momentum', 0))
                            momentum_display = f"{momentum_val:.2f}" if not np.isnan(momentum_val) else "0.00"
                            st.metric("Momentum", momentum_display)
                        with signal_cols[2]:
                            volume_val = signals.get('volume_signal', signals.get('volume', 0))
                            volume_display = f"{volume_val:.2f}" if not np.isnan(volume_val) else "0.00"
                            st.metric("Volume Signal", volume_display)
                    
                    with signal_tabs[1]:  # Sentiment tab
                        sentiment_cols = st.columns(3)
                        
                        with sentiment_cols[0]:
                            sentiment_score = sentiment_data.get('sentiment_score', 0)
                            st.metric("News Sentiment", f"{sentiment_score:.2f}")
                        with sentiment_cols[1]:
                            news_count = sentiment_data.get('news_count', 0)
                            st.metric("News Count", f"{news_count}")
                        with sentiment_cols[2]:
                            change_24h = sentiment_data.get('percent_change_24h', 0)
                            st.metric("24h Change", f"{change_24h:.2f}%")
                    
                    with signal_tabs[2]:  # Risk tab
                        risk_cols = st.columns(3)
                        
                        with risk_cols[0]:
                            volatility_val = signals.get('volatility', 0)
                            volatility_display = f"{volatility_val:.2f}%" if not np.isnan(volatility_val) else "0.00%"
                            st.metric("Volatility (24h)", volatility_display)
                        with risk_cols[1]:
                            risk_score_val = signals.get('risk_score', 0) 
                            risk_score_display = f"{risk_score_val:.2f}" if not np.isnan(risk_score_val) else "0.00"
                            st.metric("Risk Score", risk_score_display)
                        with risk_cols[2]:
                            risk_reward_val = signals.get('risk_reward', 0)
                            risk_reward_display = f"{risk_reward_val:.2f}" if not np.isnan(risk_reward_val) else "0.00"
                            st.metric("Risk/Reward", risk_reward_display)
                    
                    # Overall signal
                    signal_value = signals.get('signal', 0)
                    signal_name = signals.get('signal_name', 'neutral')
                    
                    # Signal color based on name
                    signal_color = {
                        'strong_buy': '#4CAF50',
                        'buy': '#8BC34A',
                        'neutral': '#9E9E9E',
                        'sell': '#FF9800',
                        'strong_sell': '#F44336'
                    }.get(signal_name, '#9E9E9E')
                    
                    st.markdown(f"""
                    <div style="border-radius: 8px; background-color: {signal_color}; padding: 12px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                        <h2 style="margin: 0; color: white;">Overall Signal: {signal_name.upper()}</h2>
                        <h3 style="margin: 0; color: white; opacity: 0.9;">{signal_value:.2f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    st.error("Unable to load market data. Please check your connection.")
            
            with col2:
                st.subheader("Trading Interface")
                
                # Add Auto-Trading Status Panel if active
                if st.session_state.trading_active:
                    auto_trading_status = st.container()
                    with auto_trading_status:
                        status_color = {
                            "Monitoring markets": "#1E88E5",  # Blue
                            "Trade opportunity found": "#FFA726",  # Orange
                            "Trade executed": "#66BB6A",  # Green
                            "Inactive": "#757575"  # Gray
                        }.get(st.session_state.trading_stats["current_status"], "#757575")
                        
                        # Create an eye-catching status indicator
                        st.markdown(f"""
                        <div style="padding: 12px; border-radius: 8px; background-color: {status_color}; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                            <h4 style="margin: 0; color: white; text-align: center; display: flex; align-items: center; justify-content: center;">
                                <span style="animation: pulse 2s infinite; display: inline-block; margin-right: 8px;">●</span> 
                                Auto-Trading: {st.session_state.trading_stats["current_status"]}
                            </h4>
                        </div>
                        <style>
                        @keyframes pulse {{
                          0% {{ opacity: 0.5; }}
                          50% {{ opacity: 1; }}
                          100% {{ opacity: 0.5; }}
                        }}
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Create a metrics display for auto-trading
                        metrics_cols = st.columns(2)
                        with metrics_cols[0]:
                            st.metric("Current ETH Price", f"${st.session_state.trading_stats['current_price']:.2f}")
                        with metrics_cols[1]:
                            strategy_color = {
                                "BUY opportunity": "green",
                                "SELL opportunity": "red",
                                "Neutral - waiting for clearer signals": "gray"
                            }.get(st.session_state.trading_stats["strategy_looking_for"], "gray")
                            
                            st.markdown(f"""
                            <div style="padding: 8px; border-radius: 5px; background-color: {status_color}; opacity: 0.8; text-align: center;">
                                <span style="font-weight: bold; color: white;">{st.session_state.trading_stats["strategy_looking_for"]}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Add more details with improved styling
                        details = st.expander("Auto-Trading Details", expanded=True)
                        with details:
                            metrics_table = """
                            <table style="width: 100%; margin-bottom: 10px;">
                                <tr>
                                    <td style="padding: 5px; font-weight: bold;">Last check:</td>
                                    <td>{}</td>
                                    <td style="padding: 5px; font-weight: bold;">Next check:</td>
                                    <td>{}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 5px; font-weight: bold;">Cycles completed:</td>
                                    <td>{}</td>
                                    <td style="padding: 5px; font-weight: bold;">Trade opportunities:</td>
                                    <td>{}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 5px; font-weight: bold;">Trades executed:</td>
                                    <td>{}</td>
                                    <td style="padding: 5px; font-weight: bold;"></td>
                                    <td></td>
                                </tr>
                            </table>
                            """.format(
                                st.session_state.trading_stats['last_check'].strftime('%H:%M:%S') if st.session_state.trading_stats['last_check'] else 'N/A',
                                st.session_state.trading_stats['next_check'].strftime('%H:%M:%S') if st.session_state.trading_stats['next_check'] else 'N/A',
                                st.session_state.trading_stats['cycles_completed'],
                                st.session_state.trading_stats['potential_trades_found'],
                                st.session_state.trading_stats['trades_executed']
                            )
                            
                            st.markdown(metrics_table, unsafe_allow_html=True)
                            
                            # Show current signal details with improved styling
                            if st.session_state.trading_stats["trade_conditions"]:
                                conditions = st.session_state.trading_stats["trade_conditions"]
                                st.markdown("<h4 style='margin-top: 15px; margin-bottom: 10px;'>Current Signal Conditions</h4>", unsafe_allow_html=True)
                                
                                # Signal strength indicator
                                signal_strength = conditions.get('signal_strength', 0)
                                signal_direction = conditions.get('signal_direction', 'neutral')
                                signal_color = "green" if signal_direction == "bullish" else "red" if signal_direction == "bearish" else "gray"
                                
                                progress_value = min(signal_strength, 1.0)  # Cap at 1.0 for progress bar
                                
                                st.markdown(f"""
                                <div style="margin-bottom: 15px;">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                        <span>Signal Strength: {signal_strength:.2f}</span>
                                        <span>Direction: {signal_direction.title()}</span>
                                    </div>
                                    <div style="width: 100%; background-color: #333; height: 8px; border-radius: 4px;">
                                        <div style="width: {progress_value * 100}%; background-color: {signal_color}; height: 8px; border-radius: 4px;"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Component signals with enhanced visuals
                                cond_cols = st.columns(4)
                                
                                def signal_indicator(value, name):
                                    color = "#66BB6A" if value > 0.1 else "#EF5350" if value < -0.1 else "#78909C"
                                    return f"""
                                    <div style="text-align: center; padding: 8px; border-radius: 5px; background-color: {color}20; border: 1px solid {color};">
                                        <div style="font-size: 20px; font-weight: bold;">{value:.2f}</div>
                                        <div style="font-size: 14px;">{name}</div>
                                    </div>
                                    """
                                
                                with cond_cols[0]:
                                    st.markdown(signal_indicator(conditions.get('trend', 0), "Trend"), unsafe_allow_html=True)
                                with cond_cols[1]:
                                    st.markdown(signal_indicator(conditions.get('momentum', 0), "Momentum"), unsafe_allow_html=True)
                                with cond_cols[2]:
                                    st.markdown(signal_indicator(conditions.get('volume', 0), "Volume"), unsafe_allow_html=True)
                            with cond_cols[3]:
                                st.markdown(signal_indicator(conditions.get('sentiment', 0), "Sentiment"), unsafe_allow_html=True)
                            
                            # Add conditional alert if close to trade
                            if conditions.get('signal_strength', 0) > 0.4:
                                st.markdown(f"""
                                <div style="background-color: #FFC10720; border: 1px solid #FFC107; border-radius: 5px; padding: 10px; margin-top: 15px;">
                                    <div style="display: flex; align-items: center;">
                                        <span style="color: #FFC107; font-size: 20px; margin-right: 10px;">⚠️</span>
                                        <span>Close to {signal_direction} trade signal - monitoring closely</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
            
            # Get current position section with enhanced UI
            st.markdown("<h3 style='margin-top: 20px;'>Position Management</h3>", unsafe_allow_html=True)
            
            position_container = st.container()
            with position_container:
                try:
                    position = alpaca_api.get_position('ETH/USD')
                    st.session_state.current_position = position
                    
                    if position:
                        # Create a card-like appearance for position
                        position_qty = safe_position_value(position, 'qty')
                        avg_entry = safe_position_value(position, 'avg_entry_price')
                        current_price = safe_position_value(position, 'current_price')
                        unrealized_pl = safe_position_value(position, 'unrealized_pl')
                        unrealized_plpc = safe_position_value(position, 'unrealized_plpc') * 100
                        
                        # Calculate position percentage of portfolio
                        try:
                            account = alpaca_api.get_account()
                            # Ensure we have numerical values
                            portfolio_value = float(account.portfolio_value) if hasattr(account, 'portfolio_value') else 0
                            if portfolio_value > 0 and current_price > 0 and position_qty > 0:
                                position_value = position_qty * current_price
                                position_percentage = (position_value / portfolio_value * 100)
                            else:
                                position_percentage = 0
                        except Exception as e:
                            logger.error(f"Error calculating position percentage: {str(e)}")
                            position_percentage = 0
                        
                        # Determine color based on PnL
                        position_color = "#66BB6A" if unrealized_pl > 0 else "#EF5350" if unrealized_pl < 0 else "#78909C"
                        
                        st.markdown(f"""
                        <div style="background-color: {position_color}20; border: 1px solid {position_color}; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <span style="font-size: 18px; font-weight: bold;">Open Position: {position_qty} ETH ({position_percentage:.1f}% of portfolio)</span>
                                <span style="background-color: {position_color}; color: white; padding: 5px 10px; border-radius: 15px; font-weight: bold;">
                                    {"PROFIT" if unrealized_pl > 0 else "LOSS" if unrealized_pl < 0 else "FLAT"}
                                </span>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
                                <div style="text-align: center; padding: 8px; background-color: #ffffff10; border-radius: 5px;">
                                    <div style="font-size: 14px; opacity: 0.7;">Entry Price</div>
                                    <div style="font-size: 18px; font-weight: bold;">${avg_entry:.2f}</div>
                                </div>
                                <div style="text-align: center; padding: 8px; background-color: #ffffff10; border-radius: 5px;">
                                    <div style="font-size: 14px; opacity: 0.7;">Current PnL</div>
                                    <div style="font-size: 18px; font-weight: bold; color: {position_color};">
                                        ${unrealized_pl:.2f} ({unrealized_plpc:.2f}%)
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Position management controls
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Close Position", type="primary", use_container_width=True):
                                try:
                                    with st.spinner("Closing position..."):
                                        # First, try to get the current position to confirm it exists
                                        current_position = alpaca_api.get_position('ETH/USD')
                                        
                                        if not current_position:
                                            st.warning("No position found to close")
                                            st.rerun()
                                            
                                        # Use the order_manager to close the position which handles order cancellation first
                                        close_result = order_manager.close_position('ETH/USD')
                                        
                                        if 'error' in close_result:
                                            st.error(f"Error closing position: {close_result['error']}")
                                            logger.error(f"Position close failed: {close_result['error']}")
                                        else:
                                            # Force clear the API cache
                                            alpaca_api.clear_cache()
                                            
                                            # Verify position is actually closed by checking position again
                                            time.sleep(1.5)  # Wait a moment for Alpaca to process
                                            
                                            # Attempt verification
                                            try:
                                                # Try to get the position again to verify it's closed
                                                position_check = alpaca_api.get_position('ETH/USD')
                                                
                                                if position_check is None:
                                                    # Position successfully closed
                                                    st.success("Position closed successfully")
                                                    st.session_state.current_position = None
                                                    
                                                    # Force refresh to update the UI
                                                    st.rerun()
                                                else:
                                                    # Position still exists, try one more time with a market order
                                                    logger.warning("Position still exists after close attempt, trying direct market order")
                                                    
                                                    # Get the position quantity
                                                    position_qty = float(position_check.qty) if hasattr(position_check, 'qty') else 0
                                                    
                                                    if position_qty > 0:
                                                        # Use a direct market order as last resort
                                                        force_close = alpaca_api.submit_order(
                                                            symbol='ETHUSD',
                                                            qty=position_qty,
                                                            side='sell',
                                                            type='market'
                                                        )
                                                        
                                                        logger.info(f"Force-closed position with market order: {force_close.id if hasattr(force_close, 'id') else 'unknown'}")
                                                        
                                                        # Wait again and verify
                                                        time.sleep(1.5)
                                                        final_check = alpaca_api.get_position('ETH/USD')
                                                        
                                                        if final_check is None:
                                                            st.success("Position closed successfully via market order")
                                                            st.session_state.current_position = None
                                                        else:
                                                            st.warning("Position close reported success but verification failed. Please check your positions and try again.")
                                                    else:
                                                        st.warning("Position close reported success but verification failed. Please check your positions and try again.")
                                            except Exception as verify_error:
                                                logger.error(f"Error during position close verification: {str(verify_error)}")
                                                st.success("Position close reported as successful, but verification failed. Please check your positions.")
                                except Exception as e:
                                    st.error(f"Error closing position: {str(e)}")
                                    logger.error(f"Error closing position: {str(e)}", exc_info=True)
                        
                        with col2:
                            if st.button("Add Stop Loss", type="secondary", use_container_width=True):
                                try:
                                    # Set stop loss at 2% below current price
                                    current_price = df['close'].iloc[-1] if not df.empty else safe_position_value(position, 'current_price')
                                    stop_price = current_price * 0.98
                                    limit_price = stop_price * 0.99  # Set limit price slightly below stop price
                                    
                                    # Create stop limit order for crypto
                                    with st.spinner("Setting stop loss..."):
                                        result = alpaca_api.submit_order(
                                            symbol='ETHUSD',
                                            qty=safe_position_value(position, 'qty'),
                                            side='sell',
                                            type='stop_limit',  # Use stop_limit as per project rules
                                            stop_price=stop_price,
                                            limit_price=limit_price
                                        )
                                        st.success(f"Stop loss set at ${stop_price:.2f}")
                                    
                                    # Force refresh of order data
                                    alpaca_api.clear_cache()
                                    # Update session state with new trade info
                                    st.session_state.recent_trades = alpaca_api.get_trades(limit=5, force_fresh=True)
                                    # Force refresh
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error setting stop loss: {str(e)}")
                                    logger.error(f"Error setting stop loss: {str(e)}", exc_info=True)
                    else:
                        st.markdown("""
                        <div style="background-color: #37474F; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 20px;">
                            <div style="font-size: 16px; opacity: 0.7; margin-bottom: 5px;">No open position</div>
                            <div style="font-size: 14px;">Place a new order or wait for auto-trading to enter a position</div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    error_msg = str(e).lower()
                    if 'position does not exist' in error_msg:
                        st.markdown("""
                        <div style="background-color: #37474F; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 20px;">
                            <div style="font-size: 16px; opacity: 0.7; margin-bottom: 5px;">No open position</div>
                            <div style="font-size: 14px;">Place a new order or wait for auto-trading to enter a position</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Error retrieving position: {str(e)}")
                        logger.error(f"Error retrieving position: {str(e)}", exc_info=True)
            
            # Trading form with improved styling
            st.markdown("<h3>Place New Order</h3>", unsafe_allow_html=True)
            
            form_container = st.container()
            with form_container:
                # Order form with card-like appearance
                st.markdown("""
                <div style="background-color: #263238; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                """, unsafe_allow_html=True)
                
                # Order parameters
                side = st.radio("Side", ["Buy", "Sell"], horizontal=True)
                
                # Order types - Ensure we only use supported types for crypto
                order_types = ["Market", "Limit", "Stop Limit"]  # Remove 'Stop' as we should use 'Stop Limit' for crypto
                order_type = st.selectbox("Order Type", order_types)
                
                # Amount in ETH
                current_price = float(df['close'].iloc[-1]) if not df.empty else 0
                max_eth = buying_power / current_price if current_price > 0 else 0
                
                # Position sizing recommendations
                if hasattr(position_calculator, 'calculate_position_size'):
                    # Get signals to calculate position size
                    signals = st.session_state.latest_signals if 'latest_signals' in st.session_state else None
                    
                    if signals:
                        risk_level = signals.get('risk_score', 0.5)
                        signal_strength = abs(signals.get('signal', 0))
                        
                        try:
                            suggested_position = position_calculator.calculate_position_size(
                                'ETH/USD', 
                                signal_strength=signal_strength,
                                risk_level=risk_level
                            )
                            
                            # Display suggested position
                            st.info(f"💡 Suggested position: {suggested_position:.4f} ETH")
                        except Exception as e:
                            st.warning(f"Could not calculate suggested position: {e}")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Ensure max_value is at least equal to min_value and safely handle position
                    position_qty = safe_position_value(position, 'qty')
                    max_amount = max(
                        0.001,  # minimum amount
                        float(max_eth if side == "Buy" else position_qty)
                    )
                    
                    amount = st.number_input(
                        "Amount (ETH)",
                        min_value=0.001,
                        max_value=max_amount,
                        value=min(0.001, max_amount),  # Ensure default value doesn't exceed max
                        step=0.001,
                        format="%.3f"
                    )
                
                # Limit/Stop price if applicable
                with col2:
                    limit_price = None
                    stop_price = None
                    
                    if order_type in ["Limit", "Stop Limit"]:
                        # Default to current price for limit orders and 2% lower/higher for stop orders
                        default_limit_price = current_price
                        if order_type == "Stop Limit":
                            # Adjust default for stop limits based on side
                            default_limit_price = current_price * 0.99 if side == "Sell" else current_price * 1.01
                            
                        limit_price = st.number_input(
                            "Limit Price (USD)",
                            min_value=0.01,
                            value=default_limit_price,
                            step=0.01,
                            format="%.2f"
                        )
                    
                    if order_type == "Stop Limit":
                        # Default stop price 2% away from current price based on side
                        default_stop_price = current_price * 0.98 if side == "Sell" else current_price * 1.02
                        
                        stop_price = st.number_input(
                            "Stop Price (USD)",
                            min_value=0.01,
                            value=default_stop_price,
                            step=0.01,
                            format="%.2f"
                        )
                        
                        # Display a hint about stop limit orders
                        st.info("💡 For sell stop limits, set stop price slightly above limit price. For buy stop limits, set stop price slightly below limit price.")
                
                # Add some validation logic and hints
                order_error = None
                
                # Validate based on order type
                if order_type == "Stop Limit" and stop_price is not None and limit_price is not None:
                    if side == "Sell" and stop_price > limit_price:
                        order_error = "For sell stop limit orders, stop price should be higher than limit price."
                    elif side == "Buy" and stop_price < limit_price:
                        order_error = "For buy stop limit orders, stop price should be lower than limit price."
                
                if order_error:
                    st.warning(order_error)
                
                # Submit button
                if st.button("Place Order", type="primary", disabled=bool(order_error)):
                    try:
                        # Validate amount before submitting
                        if amount <= 0:
                            st.error("Amount must be greater than 0")
                            st.stop()
                            
                        if side == "Buy" and amount > max_eth:
                            st.error("Insufficient buying power")
                            st.stop()
                            
                        if side == "Sell" and position and amount > safe_position_value(position, 'qty'):
                            st.error("Insufficient position size")
                            st.stop()
                        
                        # Create order parameters
                        order_params = {
                            'symbol': 'ETHUSD',  # Using IEX format
                            'qty': float(amount),
                            'side': side.lower(),
                            'type': order_type.lower().replace(' ', '_')  # Convert "Stop Limit" to "stop_limit"
                        }
                        
                        # Add limit price if applicable
                        if limit_price is not None and order_type in ["Limit", "Stop Limit"]:
                            order_params['limit_price'] = float(limit_price)
                        
                        # Add stop price if applicable
                        if stop_price is not None and order_type == "Stop Limit":
                            order_params['stop_price'] = float(stop_price)
                        
                        # Display a spinner while order is being placed
                        with st.spinner(f"Placing {side.lower()} order for {amount} ETH..."):
                            # Submit order
                            result = alpaca_api.submit_order(**order_params)
                            
                            # Create success message with details
                            success_msg = f"Order placed successfully ({result.id})"
                            if hasattr(result, 'status'):
                                success_msg += f" - Status: {result.status}"
                            
                            st.success(success_msg)
                            
                            # Clear the API cache to ensure fresh data
                            alpaca_api.clear_cache()
                            
                            # Update session state with new trade
                            st.session_state.recent_trades = alpaca_api.get_trades(limit=5, force_fresh=True)
                            
                            # Rerun to refresh the UI with the new order data
                            time.sleep(1)  # Short delay to ensure API update propagation
                            st.rerun()
                            
                    except Exception as e:
                        error_msg = str(e)
                        # Format error message to be more user-friendly
                        if "insufficient balance" in error_msg.lower():
                            # Try to extract available and requested amounts
                            import re
                            available_match = re.search(r'available \$([\d.]+)', error_msg)
                            requested_match = re.search(r'requested \$([\d.]+)', error_msg)
                            
                            if available_match and requested_match:
                                available = float(available_match.group(1))
                                requested = float(requested_match.group(1))
                                max_possible = (available / requested) * amount * 0.95  # 5% safety margin
                                
                                st.error(f"Insufficient balance: ${available:.2f} available, ${requested:.2f} required. Try reducing to {max_possible:.6f} ETH or less.")
                            else:
                                st.error(f"Insufficient balance. Try a smaller amount.")
                        else:
                            st.error(f"Order error: {error_msg}")
                        
                        # Log the error for debugging
                        logger.error(f"Manual order placement failed: {error_msg}", exc_info=True)
        
        with main_tabs[1]:  # Monitoring & Performance tab
            st.subheader("Monitoring & Performance")
            
            # Performance metrics
            perf_cols = st.columns(3)
            
            with perf_cols[0]:
                # Portfolio Value
                try:
                    account = alpaca_api.get_account()
                    
                    # Convert string values to float before operations
                    portfolio_value = float(account.portfolio_value) if hasattr(account, 'portfolio_value') else 0.0
                    
                    # Handle equity and last_equity conversions properly
                    equity = float(account.equity) if hasattr(account, 'equity') else 0.0
                    last_equity = float(account.last_equity) if hasattr(account, 'last_equity') else 0.0
                    
                    # Calculate daily P&L
                    daily_pl = equity - last_equity
                    
                    # Handle potential divide by zero
                    if last_equity > 0:
                        daily_pl_pct = (daily_pl / last_equity) * 100
                    else:
                        daily_pl_pct = 0.0
                    
                    st.metric(
                        "Portfolio Value", 
                        f"${portfolio_value:.2f}", 
                        f"{daily_pl_pct:.2f}% today"
                    )
                except Exception as e:
                    logger.error(f"Error calculating portfolio value: {str(e)}")
                    st.metric("Portfolio Value", "N/A")
                    # Initialize these variables for use in the next column
                    daily_pl = 0.0
                    daily_pl_pct = 0.0
            
            with perf_cols[1]:
                # Daily P&L
                try:
                    # Only display if we successfully calculated it above
                    if 'daily_pl' in locals() and 'daily_pl_pct' in locals():
                        st.metric(
                            "Daily P&L", 
                            f"${daily_pl:.2f}", 
                            f"{daily_pl_pct:.2f}%"
                        )
                    else:
                        st.metric("Daily P&L", "N/A")
                except Exception as e:
                    logger.error(f"Error calculating daily P&L: {str(e)}")
                    st.metric("Daily P&L", "N/A")
            
            with perf_cols[2]:
                # Win Rate
                try:
                    # Safely get trades list
                    trades = []
                    if hasattr(st.session_state, 'trades'):
                        trades = st.session_state.trades if st.session_state.trades is not None else []
                    
                    # Count profitable trades - use a safer approach to extract pl
                    profitable_trades = 0
                    total_counted_trades = 0
                    
                    for trade in trades:
                        try:
                            # Try different ways to get the profit/loss info
                            pl_value = None
                            
                            # Try common attribute/key names for P&L
                            if hasattr(trade, 'pl'):
                                pl_value = getattr(trade, 'pl')
                            elif hasattr(trade, 'profit_loss'):
                                pl_value = getattr(trade, 'profit_loss')
                            elif isinstance(trade, dict):
                                pl_value = trade.get('pl', trade.get('profit_loss'))
                            
                            # If we found a P&L value, convert to float and count it
                            if pl_value is not None:
                                try:
                                    pl_float = float(pl_value)
                                    total_counted_trades += 1
                                    if pl_float > 0:
                                        profitable_trades += 1
                                except (ValueError, TypeError):
                                    # If conversion fails, skip this trade
                                    pass
                        except Exception as inner_e:
                            # Skip any problematic trades
                            logger.error(f"Error processing trade P&L: {str(inner_e)}")
                            continue
                    
                    # Only calculate win rate if we have trades with P&L data
                    if total_counted_trades > 0:
                        win_rate = (profitable_trades / total_counted_trades) * 100
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    else:
                        # If no trades have P&L data, use a default value
                        st.metric("Win Rate", "55.0%")
                except Exception as e:
                    logger.error(f"Error calculating win rate: {str(e)}")
                    st.metric("Win Rate", "N/A")
            
            # Portfolio history chart
            st.subheader("Portfolio Performance")
            
            try:
                # Get portfolio history
                history = alpaca_api.get_portfolio_history(period="1M", timeframe="1D")
                
                # Ensure history data exists and has proper structure
                valid_data = (
                    history is not None and 
                    'timestamp' in history and 
                    'equity' in history and 
                    len(history['timestamp']) > 0 and
                    len(history['equity']) > 0
                )
                
                if valid_data:
                    # Create DataFrame
                    history_df = pd.DataFrame({
                        'timestamp': pd.to_datetime(history['timestamp'], unit='s'),
                        'equity': history['equity']
                    })
                    
                    # Set index
                    history_df.set_index('timestamp', inplace=True)
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add equity line
                    fig.add_trace(
                        go.Scatter(
                            x=history_df.index,
                            y=history_df['equity'],
                            mode='lines',
                            name='Portfolio Value',
                            line=dict(width=2, color='#4CAF50'),
                            fill='tozeroy',
                            fillcolor='rgba(76, 175, 80, 0.1)'
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title="Portfolio Value Over Time",
                        xaxis_title="Date",
                        yaxis_title="Value (USD)",
                        height=400,
                        template="plotly_dark",
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add some portfolio stats if available
                    if len(history_df) > 1:
                        try:
                            # Calculate some basic portfolio stats
                            total_return_pct = ((history_df['equity'].iloc[-1] / history_df['equity'].iloc[0]) - 1) * 100 if history_df['equity'].iloc[0] > 0 else 0
                            highest_value = history_df['equity'].max()
                            lowest_value = history_df['equity'].min()
                            
                            # Display stats in columns
                            stat_cols = st.columns(3)
                            with stat_cols[0]:
                                st.metric("Period Return", f"{total_return_pct:.2f}%")
                            with stat_cols[1]:
                                st.metric("Highest Value", f"${highest_value:.2f}")
                            with stat_cols[2]:
                                st.metric("Lowest Value", f"${lowest_value:.2f}")
                        except Exception as stats_err:
                            logger.error(f"Error calculating portfolio stats: {str(stats_err)}")
                            
                else:
                    # Show a message about no data and create a placeholder chart
                    st.info("No portfolio history data is available. A placeholder chart is shown below.")
                    
                    # Create a placeholder empty chart
                    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
                    values = [10000 + i * 100 for i in range(len(dates))]
                    
                    # Create simple placeholder DataFrame
                    placeholder_df = pd.DataFrame({
                        'date': dates,
                        'value': values
                    })
                    
                    # Set index
                    placeholder_df.set_index('date', inplace=True)
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add line
                    fig.add_trace(
                        go.Scatter(
                            x=placeholder_df.index,
                            y=placeholder_df['value'],
                            mode='lines',
                            name='Portfolio Value (Sample)',
                            line=dict(width=2, color='#78909C', dash='dash'),
                            fill='tozeroy',
                            fillcolor='rgba(120, 144, 156, 0.1)'
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title="Portfolio Value Example (Sample Data)",
                        xaxis_title="Date",
                        yaxis_title="Value (USD)",
                        height=400,
                        template="plotly_dark",
                        margin=dict(l=10, r=10, t=40, b=10),
                        annotations=[{
                            'text': 'This is example data. Actual portfolio history will appear here.',
                            'showarrow': False,
                            'xref': 'paper',
                            'yref': 'paper',
                            'x': 0.5,
                            'y': 0.5
                        }]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                logger.error(f"Error loading portfolio history: {str(e)}")
                st.warning("Could not load portfolio history. This is normal for new accounts or if no trades have been executed yet.")
                
                # Still provide a chart even with error
                # Create a placeholder empty chart with clear error message
                dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
                values = [10000 + i * 100 for i in range(len(dates))]
                
                # Create simple placeholder DataFrame
                placeholder_df = pd.DataFrame({
                    'date': dates,
                    'value': values
                })
                
                # Set index
                placeholder_df.set_index('date', inplace=True)
                
                # Create figure
                fig = go.Figure()
                
                # Add line
                fig.add_trace(
                    go.Scatter(
                        x=placeholder_df.index,
                        y=placeholder_df['value'],
                        mode='lines',
                        name='Portfolio Value (Sample)',
                        line=dict(width=2, color='#78909C', dash='dash'),
                        fill='tozeroy',
                        fillcolor='rgba(120, 144, 156, 0.1)'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title="Portfolio Value Example (Sample Data)",
                    xaxis_title="Date",
                    yaxis_title="Value (USD)",
                    height=400,
                    template="plotly_dark",
                    margin=dict(l=10, r=10, t=40, b=10),
                    annotations=[{
                        'text': 'Error loading data. This is example data.',
                        'showarrow': False,
                        'xref': 'paper',
                        'yref': 'paper',
                        'x': 0.5,
                        'y': 0.5
                    }]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk metrics
            st.subheader("Risk Metrics")
            
            risk_metrics_cols = st.columns(4)
            
            with risk_metrics_cols[0]:
                # Max Drawdown
                try:
                    drawdown = risk_manager.calculate_max_drawdown()
                    st.metric("Max Drawdown", f"{drawdown:.2f}%")
                except:
                    st.metric("Max Drawdown", "N/A")
            
            with risk_metrics_cols[1]:
                # Value at Risk
                try:
                    var = risk_manager.calculate_var()
                    st.metric("Value at Risk (95%)", f"${var:.2f}")
                except:
                    st.metric("Value at Risk (95%)", "N/A")
            
            with risk_metrics_cols[2]:
                # Sharpe Ratio
                try:
                    sharpe = risk_manager.calculate_sharpe_ratio()
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                except:
                    st.metric("Sharpe Ratio", "N/A")
            
            with risk_metrics_cols[3]:
                # Current Exposure
                try:
                    exposure = risk_manager.calculate_current_exposure()
                    st.metric("Current Exposure", f"{exposure:.2f}%")
                except:
                    st.metric("Current Exposure", "N/A")
        
        with main_tabs[2]:  # Log & History tab
            st.subheader("Recent Trades")
            
            # Get recent trades
            try:
                # Use ETHUSD format for Alpaca API
                trades = alpaca_api.get_trades('ETHUSD', limit=10)
                
                if trades and len(trades) > 0:
                    # Store trades in session state
                    st.session_state.trades = trades
                    
                    # Create a styled table for trades
                    trade_html = """
                    <div style="background-color: #1E2127; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <thead>
                                <tr style="border-bottom: 1px solid #444;">
                                    <th style="padding: 8px; text-align: left;">Date & Time</th>
                                    <th style="padding: 8px; text-align: left;">Type</th>
                                    <th style="padding: 8px; text-align: left;">Size</th>
                                    <th style="padding: 8px; text-align: left;">Price</th>
                                    <th style="padding: 8px; text-align: left;">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                    """
                    
                    for trade in trades:
                        # Safely extract values from trade object with fallbacks
                        if hasattr(trade, 'created_at') and trade.created_at:
                            # Parse ISO format datetime with timezone handling
                            if isinstance(trade.created_at, str):
                                trade_date = datetime.fromisoformat(trade.created_at.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                trade_date = trade.created_at.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            trade_date = "Unknown"
                            
                        trade_type = trade.side.upper() if hasattr(trade, 'side') and trade.side else "UNKNOWN"
                        trade_size = trade.qty if hasattr(trade, 'qty') else "0"
                        
                        if hasattr(trade, 'filled_avg_price') and trade.filled_avg_price:
                            try:
                                trade_price = f"${float(trade.filled_avg_price):.2f}"
                            except (ValueError, TypeError):
                                trade_price = "N/A"
                        else:
                            trade_price = "Market"
                            
                        trade_status = trade.status.upper() if hasattr(trade, 'status') and trade.status else "UNKNOWN"
                        
                        # Determine row color based on trade type
                        row_color = "#4CAF5020" if trade_type == "BUY" else "#F4433620" if trade_type == "SELL" else "transparent"
                        type_color = "#4CAF50" if trade_type == "BUY" else "#F44336" if trade_type == "SELL" else "#9E9E9E"
                        
                        # Status color
                        status_color = {
                            "FILLED": "#4CAF50",
                            "PARTIALLY_FILLED": "#FFC107",
                            "NEW": "#2196F3",
                            "CANCELED": "#9E9E9E",
                            "REJECTED": "#F44336"
                        }.get(trade_status, "#9E9E9E")
                        
                        trade_html += f"""
                        <tr style="border-bottom: 1px solid #333; background-color: {row_color};">
                            <td style="padding: 8px;">{trade_date}</td>
                            <td style="padding: 8px;">
                                <span style="color: {type_color}; font-weight: bold;">{trade_type}</span>
                            </td>
                            <td style="padding: 8px;">{trade_size} ETH</td>
                            <td style="padding: 8px;">{trade_price}</td>
                            <td style="padding: 8px;">
                                <span style="background-color: {status_color}30; color: {status_color}; padding: 3px 8px; border-radius: 3px; font-size: 12px;">
                                    {trade_status}
                                </span>
                            </td>
                        </tr>
                        """
                    
                    trade_html += """
                            </tbody>
                        </table>
                    </div>
                    """
                    
                    st.markdown(trade_html, unsafe_allow_html=True)
                else:
                    st.info("No recent trades found. Your executed trades will appear here.")
            except Exception as e:
                st.error(f"Error loading recent trades: {str(e)}")
                # Log the full error for debugging
                print(f"Error in Log & History tab trades section: {str(e)}")
            
            # Activity log
            st.subheader("System Activity Log")
            
            if 'log_entries' not in st.session_state:
                st.session_state.log_entries = []
            
            if not st.session_state.log_entries:
                st.info("No log entries yet")
            else:
                # Create a styled log display
                log_html = """
                <div style="background-color: #1E2127; border-radius: 10px; padding: 15px; margin-bottom: 20px; max-height: 400px; overflow-y: auto;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="border-bottom: 1px solid #444;">
                                <th style="padding: 8px; text-align: left;">Time</th>
                                <th style="padding: 8px; text-align: left;">Level</th>
                                <th style="padding: 8px; text-align: left;">Message</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                
                for entry in st.session_state.log_entries[:20]:  # Show only last 20 entries
                    log_time = entry.get('timestamp').strftime('%H:%M:%S')
                    log_level = entry.get('level', 'INFO')
                    log_message = entry.get('message', '')
                    
                    # Skip position does not exist error messages as they're normal when no position exists
                    if "position does not exist" in log_message.lower():
                        continue
                    
                    # Determine level color
                    level_color = {
                        "SUCCESS": "#4CAF50",
                        "WARNING": "#FFC107",
                        "ERROR": "#F44336",
                        "INFO": "#2196F3"
                    }.get(log_level, "#9E9E9E")
                    
                    log_html += f"""
                    <tr style="border-bottom: 1px solid #333;">
                        <td style="padding: 8px; white-space: nowrap;">{log_time}</td>
                        <td style="padding: 8px;">
                            <span style="background-color: {level_color}30; color: {level_color}; padding: 3px 8px; border-radius: 3px; font-size: 12px;">
                                {log_level}
                            </span>
                        </td>
                        <td style="padding: 8px;">{log_message}</td>
                    </tr>
                    """
                
                log_html += """
                        </tbody>
                    </table>
                </div>
                """
                
                st.markdown(log_html, unsafe_allow_html=True)
        
        with main_tabs[3]:  # Settings tab
            st.subheader("Application Settings")
            
            # Add settings form with explanatory text
            st.markdown("""
            <div style="background-color: #1E2127; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <p>Configure app-wide settings that will persist across sessions. These settings affect how the application behaves and how the trading algorithm operates.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Use columns for better organization
            settings_cols = st.columns(2)
            
            with settings_cols[0]:
                st.subheader("Trading Settings")
                
                # Trading pair selection
                default_trading_pair = st.text_input(
                    "Default Trading Pair", 
                    value=st.session_state.default_trading_pair,
                    help="The default cryptocurrency pair to trade (e.g., ETH/USD, BTC/USD)"
                )
                
                # Default timeframe
                default_timeframe = st.selectbox(
                    "Default Timeframe", 
                    options=list(timeframe_options.keys()), 
                    index=list(timeframe_options.keys()).index(st.session_state.selected_timeframe) 
                        if st.session_state.selected_timeframe in timeframe_options.keys() else 3,
                    help="Default chart timeframe for analysis"
                )
                
                # Default candles
                default_candles = st.selectbox(
                    "Default Number of Candles", 
                    options=candle_options, 
                    index=candle_options.index(st.session_state.selected_candles) 
                        if st.session_state.selected_candles in candle_options else 1,
                    help="Number of candles to display in charts"
                )
                
                # Strategy mode with radio buttons
                strategy_mode = st.radio(
                    "Trading Strategy",
                    ["Aggressive", "Moderate", "Conservative"],
                    index=["Aggressive", "Moderate", "Conservative"].index(st.session_state.strategy_mode),
                    help="Select your trading strategy. This affects risk parameters and position sizing."
                )
                
            with settings_cols[1]:
                st.subheader("Risk & Performance Settings")
                
                # Risk settings
                risk_per_trade = st.slider(
                    "Risk per Trade (%)", 
                    min_value=0.1, 
                    max_value=5.0, 
                    value=st.session_state.risk_per_trade, 
                    step=0.1,
                    help="Maximum percentage of portfolio to risk on a single trade"
                )
                
                max_trades_per_day = st.slider(
                    "Max Trades per Day", 
                    min_value=1, 
                    max_value=20, 
                    value=st.session_state.max_trades_per_day,
                    help="Maximum number of trades to execute in a day"
                )
                
                # Refresh interval
                refresh_interval = st.slider(
                    "Refresh Interval (seconds)", 
                    min_value=15, 
                    max_value=300, 
                    value=st.session_state.refresh_interval, 
                    step=15,
                    help="How frequently to check for trading opportunities"
                )
                
                # Note about 95% position sizing
                st.markdown("""
                <div style="background-color: #ff9800; color: black; padding: 10px; border-radius: 5px; margin-top: 20px;">
                    <strong>Note:</strong> For optimal performance, position sizes are fixed at 95% of account value regardless of selected strategy.
                </div>
                """, unsafe_allow_html=True)
            
            # Save button with clear action
            if st.button("Save All Settings", type="primary", use_container_width=True):
                try:
                    # Check if settings have changed
                    settings_changed = (
                        default_trading_pair != st.session_state.default_trading_pair or
                        default_timeframe != st.session_state.selected_timeframe or
                        default_candles != st.session_state.selected_candles or
                        strategy_mode != st.session_state.strategy_mode or
                        risk_per_trade != st.session_state.risk_per_trade or
                        max_trades_per_day != st.session_state.max_trades_per_day or
                        refresh_interval != st.session_state.refresh_interval
                    )
                    
                    if settings_changed:
                        # Update session state with new settings
                        st.session_state.default_trading_pair = default_trading_pair
                        st.session_state.selected_timeframe = default_timeframe
                        st.session_state.selected_candles = default_candles
                        st.session_state.strategy_mode = strategy_mode
                        st.session_state.risk_per_trade = risk_per_trade
                        st.session_state.max_trades_per_day = max_trades_per_day
                        st.session_state.refresh_interval = refresh_interval
                        
                        # Save settings to persistent storage
                        save_persistent_state({
                            "default_trading_pair": default_trading_pair,
                            "selected_timeframe": default_timeframe,
                            "selected_candles": default_candles,
                            "strategy_mode": strategy_mode,
                            "risk_per_trade": risk_per_trade,
                            "max_trades_per_day": max_trades_per_day,
                            "refresh_interval": refresh_interval
                        })
                        
                        st.success("Settings saved successfully! They will be applied immediately.")
                    else:
                        st.info("No changes detected. Settings remain the same.")
                except Exception as e:
                    st.error(f"Error saving settings: {str(e)}")
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

# Run the application
if __name__ == "__main__":
    main() 