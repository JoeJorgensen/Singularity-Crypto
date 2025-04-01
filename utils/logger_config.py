"""
Logger configuration for CryptoTrader application.
Provides a flexible logging setup with console and file outputs.
"""
import os
import logging
import logging.handlers
from datetime import datetime

# ANSI color codes for terminal output
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

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output"""
    
    # Define colors for different logging levels
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
        
        # Add color based on the logging level
        levelname = record.levelname
        levelno = record.levelno
        
        # Add colors for specific parts
        color_level = self.LEVEL_COLORS.get(levelno, COLORS['RESET'])
        color_message = COLORS['RESET']
        
        # Color the timestamp in cyan
        formatted_msg = formatted_msg.replace(
            record.asctime,
            f"{COLORS['CYAN']}{record.asctime}{COLORS['RESET']}"
        )
        
        # Color the level name
        formatted_msg = formatted_msg.replace(
            f" - {levelname} - ",
            f" - {color_level}{levelname}{COLORS['RESET']} - "
        )
        
        return formatted_msg

def setup_logging(log_level=logging.INFO, log_to_file=True, console_output=True):
    """
    Set up application logging with console and file handlers.
    
    Args:
        log_level: The logging level (default: logging.INFO)
        log_to_file: Whether to log to file (default: True)
        console_output: Whether to log to console (default: True)
    
    Returns:
        The configured root logger
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log filename with timestamp
    log_filename = os.path.join(
        log_dir, 
        f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Use colored formatter for console output
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create and add file handler if requested
    if log_to_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_filename, 
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
    
    # Create and add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)
    
    # Configure module-specific loggers
    configure_module_loggers(log_level)
    
    logging.info(f"Logging initialized at level {logging._levelToName[log_level]}")
    if log_to_file:
        logging.info(f"Log file: {log_filename}")
    
    return root_logger

def configure_module_loggers(log_level):
    """
    Configure specific loggers for different modules.
    
    Args:
        log_level: The base logging level
    """
    # Trading strategy logger - full debug info
    trading_logger = logging.getLogger('trading_strategy')
    trading_logger.setLevel(log_level)
    
    # Order manager logger - full debug info  
    order_logger = logging.getLogger('order_manager')
    order_logger.setLevel(log_level)
    
    # API loggers - ensure they log at INFO level at minimum
    api_level = min(log_level, logging.INFO)  # Always log INFO or more verbose
    for api in ['alpaca_api', 'finnhub_api', 'coinlore_api', 'openai_api']:
        api_logger = logging.getLogger(api)
        api_logger.setLevel(api_level)
    
    # Technical analysis loggers
    technical_logger = logging.getLogger('CryptoTrader.technical')
    technical_logger.setLevel(log_level)

def get_logger(name):
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name) 