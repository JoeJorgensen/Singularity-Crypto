"""
Centralized logging configuration for CryptoTrader application.
Provides a consistent logging setup with terminal-only output.
"""
import logging
import sys
from typing import Optional
import os

# Global logger name
LOGGER_NAME = "CryptoTrader"

# ANSI color codes for terminal coloring
COLORS = {
    'RESET': '\033[0m',
    'RED': '\033[31m',
    'YELLOW': '\033[33m',
    'GREEN': '\033[32m',
    'BLUE': '\033[34m',
    'MAGENTA': '\033[35m',
    'CYAN': '\033[36m',
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages based on level"""
    
    LEVEL_COLORS = {
        logging.DEBUG: '',
        logging.INFO: '',
        logging.WARNING: COLORS['YELLOW'],
        logging.ERROR: COLORS['RED'],
        logging.CRITICAL: COLORS['RED'],
    }
    
    def format(self, record):
        # Add color based on log level
        levelname = record.levelname
        message = super().format(record)
        
        # Apply colors based on content
        if 'ERROR' in message or 'Error' in message:
            return f"{COLORS['RED']}{message}{COLORS['RESET']}"
        elif 'WARNING' in message or 'Warning' in message:
            return f"{COLORS['YELLOW']}{message}{COLORS['RESET']}"
        else:
            color = self.LEVEL_COLORS.get(record.levelno, '')
            if color:
                return f"{color}{message}{COLORS['RESET']}"
            return message

def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up application logging with console handler and file handler.
    
    Args:
        log_level: The logging level (default: logging.INFO)
    
    Returns:
        The configured root logger
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Create console handler with colored formatter
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Create file handler for logging to a file
    try:
        # Create logs directory if it doesn't exist
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            
        # Use a simpler formatter for log files (no colors)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create the file handler
        file_handler = logging.FileHandler(os.path.join(logs_dir, 'app.log'))
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
        
        # Log a message to confirm file logging is set up
        logging.info(f"File logging configured to {os.path.abspath(os.path.join(logs_dir, 'app.log'))}")
    except Exception as e:
        logging.warning(f"Could not set up file logging: {str(e)}")
    
    # Configure main application logger
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)
    
    # Configure module-specific loggers
    configure_module_loggers(log_level)
    
    logging.info(f"Logging initialized at level {logging._levelToName[log_level]}")
    
    return root_logger

def configure_module_loggers(log_level: int) -> None:
    """
    Configure specific loggers for different modules.
    
    Args:
        log_level: The base logging level
    """
    # Trading strategy logger
    trading_logger = logging.getLogger(f"{LOGGER_NAME}.trading_strategy")
    trading_logger.setLevel(log_level)
    
    # Order manager logger
    order_logger = logging.getLogger(f"{LOGGER_NAME}.order_manager")
    order_logger.setLevel(log_level)
    
    # Technical analysis loggers
    technical_logger = logging.getLogger(f"{LOGGER_NAME}.technical")
    technical_logger.setLevel(log_level)
    
    # API loggers - slightly less verbose
    api_level = max(log_level, logging.INFO)  # At least INFO level
    for api in ['alpaca_api', 'finnhub_api', 'coinlore_api', 'openai_api']:
        api_logger = logging.getLogger(f"{LOGGER_NAME}.{api}")
        api_logger.setLevel(api_level)

def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger with the specified module name.
    
    Args:
        module_name: Module name (without the LOGGER_NAME prefix)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"{LOGGER_NAME}.{module_name}") 