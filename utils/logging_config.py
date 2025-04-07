"""
Centralized logging configuration for CryptoTrader application.
Provides a consistent logging setup with terminal and file output.
"""
import logging
import sys
from typing import Optional, Dict
import os
from datetime import datetime
import traceback
import logging.handlers

# Global logger name
LOGGER_NAME = "CryptoTrader"

# ANSI color codes for terminal coloring - expanded from both files
COLORS = {
    'RESET': '\033[0m',
    'BLACK': '\033[30m',
    'RED': '\033[31m',
    'GREEN': '\033[32m',
    'YELLOW': '\033[33m',
    'BLUE': '\033[34m',
    'MAGENTA': '\033[35m',
    'CYAN': '\033[36m',
    'WHITE': '\033[37m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
    'BRIGHT_RED': '\033[91m',
    'BRIGHT_GREEN': '\033[92m',
    'BRIGHT_YELLOW': '\033[93m',
    'BRIGHT_BLUE': '\033[94m',
    'BRIGHT_MAGENTA': '\033[95m',
    'BRIGHT_CYAN': '\033[96m',
    'BRIGHT_WHITE': '\033[97m',
    'BG_RED': '\033[41m',
    'BG_GREEN': '\033[42m',
    'BG_YELLOW': '\033[43m',
    'BG_BLUE': '\033[44m',
}

# Define logging profiles
LOGGING_PROFILES = {
    'default': {
        'root_level': logging.INFO,
        'api_level': logging.INFO,
        'background_level': logging.INFO,
        'technical_level': logging.INFO,
        'trading_level': logging.INFO,
        'console_log': True,
        'file_log': True
    },
    'verbose': {
        'root_level': logging.DEBUG,
        'api_level': logging.DEBUG,
        'background_level': logging.DEBUG,
        'technical_level': logging.DEBUG,
        'trading_level': logging.DEBUG,
        'console_log': True,
        'file_log': True
    },
    'cloud': {
        'root_level': logging.WARNING,
        'api_level': logging.WARNING,
        'background_level': logging.WARNING,
        'technical_level': logging.WARNING,
        'trading_level': logging.WARNING,
        'console_log': True,
        'file_log': False
    },
    'minimal': {
        'root_level': logging.ERROR,
        'api_level': logging.ERROR,
        'background_level': logging.WARNING,
        'technical_level': logging.ERROR,
        'trading_level': logging.WARNING,
        'console_log': True,
        'file_log': False
    }
}

class CloudOptimizedFilter(logging.Filter):
    """Filter that reduces log frequency for repeated similar messages"""
    
    def __init__(self):
        super().__init__()
        self.last_messages = {}
        self.message_counts = {}
    
    def filter(self, record):
        # Extract key information from the record
        msg_key = f"{record.levelname}:{record.getMessage()}"
        
        # Check if we've seen this message recently
        current_time = datetime.now()
        if msg_key in self.last_messages:
            last_time, count = self.last_messages[msg_key]
            time_diff = (current_time - last_time).total_seconds()
            
            # For repetitive messages, only log once per 30 seconds
            if time_diff < 30:
                self.message_counts[msg_key] = count + 1
                self.last_messages[msg_key] = (current_time, count + 1)
                return False
            else:
                # If it's been more than 30 seconds, log message and count of skipped messages
                if self.message_counts[msg_key] > 1:
                    record.getMessage = lambda: f"{record.getMessage()} (repeated {self.message_counts[msg_key]} times)"
                    self.message_counts[msg_key] = 0
        
        # Update tracking for this message
        self.last_messages[msg_key] = (current_time, 0)
        self.message_counts[msg_key] = 0
        return True

class ColoredFormatter(logging.Formatter):
    """Enhanced custom formatter that adds colors to log messages based on level"""
    
    # Define colors for different logging levels (merged from both files)
    LEVEL_COLORS = {
        logging.DEBUG: COLORS['BLUE'],
        logging.INFO: COLORS['GREEN'],
        logging.WARNING: COLORS['YELLOW'],
        logging.ERROR: COLORS['RED'],
        logging.CRITICAL: COLORS['BG_RED'] + COLORS['WHITE'] + COLORS['BOLD'],
    }
    
    def format(self, record):
        # Get the original formatted message
        formatted_msg = super().format(record)
        
        # Add color based on log level
        levelname = record.levelname
        levelno = record.levelno
        
        # Color the timestamp in cyan if it's in the message
        if hasattr(record, 'asctime'):
            formatted_msg = formatted_msg.replace(
                record.asctime,
                f"{COLORS['CYAN']}{record.asctime}{COLORS['RESET']}"
            )
            
        # Color the level name with appropriate color
        level_pattern = f" - {levelname} - "
        if level_pattern in formatted_msg:
            color_level = self.LEVEL_COLORS.get(levelno, COLORS['RESET'])
            formatted_msg = formatted_msg.replace(
                level_pattern,
                f" - {color_level}{levelname}{COLORS['RESET']} - "
            )
        
        # Apply colors based on content for keyword highlighting
        if 'ERROR' in formatted_msg or 'Error' in formatted_msg:
            return f"{COLORS['RED']}{formatted_msg}{COLORS['RESET']}"
        elif 'WARNING' in formatted_msg or 'Warning' in formatted_msg:
            return f"{COLORS['YELLOW']}{formatted_msg}{COLORS['RESET']}"
        elif 'SUCCESS' in formatted_msg or 'Success' in formatted_msg:
            return f"{COLORS['GREEN']}{formatted_msg}{COLORS['RESET']}"
        else:
            color = self.LEVEL_COLORS.get(record.levelno, '')
            if color:
                return f"{color}{formatted_msg}{COLORS['RESET']}"
        
        return formatted_msg

def setup_logging(
    profile: str = 'default',
    log_level: Optional[int] = None,
    enable_console_logging: Optional[bool] = None,
    enable_file_logging: Optional[bool] = None,
    cloud_optimized: bool = False
) -> logging.Logger:
    """
    Set up application logging with console handler and file handler.
    
    Args:
        profile: Logging profile to use ('default', 'verbose', 'cloud', or 'minimal')
        log_level: Override the profile's logging level (default: None)
        enable_console_logging: Override the profile's console logging setting (default: None)
        enable_file_logging: Override the profile's file logging setting (default: None)
        cloud_optimized: Enable cloud-specific optimizations to reduce log verbosity
    
    Returns:
        The configured root logger
    """
    # Get profile settings
    profile_settings = LOGGING_PROFILES.get(profile, LOGGING_PROFILES['default'])
    
    # Use provided values or fall back to profile settings
    actual_log_level = log_level if log_level is not None else profile_settings['root_level']
    actual_console_log = enable_console_logging if enable_console_logging is not None else profile_settings['console_log']
    actual_file_log = enable_file_logging if enable_file_logging is not None else profile_settings['file_log']
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(actual_log_level)
    
    # Clear existing handlers to avoid duplicates
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Create console handler with colored formatter if enabled
    if actual_console_log:
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(actual_log_level)
        
        # Add cloud-optimized filter if requested
        if cloud_optimized:
            console_handler.addFilter(CloudOptimizedFilter())
            
        root_logger.addHandler(console_handler)
    
    # Create file handler for logging to a file if enabled
    if actual_file_log:
        try:
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
                
            # Use a simpler formatter for log files (no colors)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Create a daily log file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(logs_dir, f'trading_{timestamp}.log')
            
            # Create the file handler with rotation support
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(actual_log_level)
            
            # Add cloud-optimized filter if requested
            if cloud_optimized:
                file_handler.addFilter(CloudOptimizedFilter())
                
            root_logger.addHandler(file_handler)
            
            # Log a message to confirm file logging is set up
            if actual_console_log:
                logging.info(f"Log file: {log_file}")
        except Exception as e:
            if actual_console_log:
                logging.warning(f"Could not set up file logging: {str(e)}")
    
    # Configure main application logger
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(actual_log_level)
    
    # Configure module-specific loggers with profile settings
    configure_module_loggers(
        root_level=actual_log_level,
        api_level=profile_settings['api_level'],
        background_level=profile_settings['background_level'],
        technical_level=profile_settings['technical_level'],
        trading_level=profile_settings['trading_level']
    )
    
    if actual_console_log:
        logging.info(f"Logging initialized at level {logging._levelToName[actual_log_level]}")
    
    return root_logger

def configure_module_loggers(
    root_level: int = logging.INFO,
    api_level: int = logging.INFO,
    background_level: int = logging.INFO,
    technical_level: int = logging.INFO,
    trading_level: int = logging.INFO
) -> None:
    """
    Configure specific loggers for different modules.
    
    Args:
        root_level: The base logging level
        api_level: Level for API loggers
        background_level: Level for background tasks
        technical_level: Level for technical analysis
        trading_level: Level for trading strategy and order management
    """
    # Configure API loggers
    for api_name in ['alpaca_api', 'finnhub_api', 'coinlore_api', 'openai_api']:
        api_logger = logging.getLogger(api_name)
        api_logger.setLevel(api_level)
    
    # Configure technical analysis loggers
    technical_logger = logging.getLogger('CryptoTrader.technical')
    technical_logger.setLevel(technical_level)
    
    # Configure technical sub-loggers
    for tech_name in ['indicators', 'signal_generator']:
        tech_sub_logger = logging.getLogger(f'CryptoTrader.technical.{tech_name}')
        tech_sub_logger.setLevel(technical_level)
    
    # Configure trading strategy and order management
    trading_logger = logging.getLogger('trading_strategy')
    trading_logger.setLevel(trading_level)
    
    order_logger = logging.getLogger('order_manager')
    order_logger.setLevel(trading_level)
    
    # Configure on-chain analysis loggers
    onchain_logger = logging.getLogger('CryptoTrader.onchain')
    onchain_logger.setLevel(root_level)
    
    # Background task loggers
    background_logger = logging.getLogger('CryptoTrader.background')
    background_logger.setLevel(background_level)

def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Configured logger for the module
    """
    if not module_name.startswith(LOGGER_NAME + '.') and module_name != LOGGER_NAME:
        # If module_name doesn't start with the logger name prefix, add it
        # But don't add it if it's exactly the logger name
        if '.' in module_name:
            # Keep the module hierarchy intact
            parts = module_name.split('.')
            if parts[0] != LOGGER_NAME:
                module_name = f"{LOGGER_NAME}.{module_name}"
        else:
            # Simple module name
            module_name = f"{LOGGER_NAME}.{module_name}"
    
    return logging.getLogger(module_name)

def is_cloud_environment():
    """
    Detect if running in a cloud/production environment.
    
    Returns:
        bool: True if running in cloud environment, False otherwise
    """
    # Check for various cloud environment indicators
    cloud_indicators = [
        os.environ.get('STREAMLIT_SHARING', ''),
        os.environ.get('STREAMLIT_CLOUD', ''),
        os.environ.get('AWS_LAMBDA_FUNCTION_NAME', ''),
        os.environ.get('GOOGLE_CLOUD_PROJECT', ''),
        os.environ.get('DYNO', ''),  # Heroku
        os.environ.get('KUBERNETES_SERVICE_HOST', '')
    ]
    
    return any(cloud_indicators) 