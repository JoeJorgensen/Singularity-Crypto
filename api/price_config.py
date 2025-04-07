"""
Price Provider Configuration.
Centralized settings for all price data providers and aggregation options.
"""
from typing import Dict, Any

# Default price provider to use (options: "alpaca", "coingecko", "alchemy", "aggregator")
DEFAULT_PROVIDER = "aggregator"

# Provider weights for aggregation (higher values = higher priority)
PROVIDER_WEIGHTS = {
    "alpaca": 1.0,     # Primary source for trading
    "alchemy": 0.8,    # Good reliability, limited to 300 req/hr
    "coingecko": 0.5   # Fallback source
}

# Cache settings (in seconds)
CACHE_TTL = {
    "current_price": 60,        # 1 minute for current prices
    "historical_price": 3600,   # 1 hour for historical data
    "bulk_prices": 300          # 5 minutes for bulk price lookups
}

# Rate limiting settings
RATE_LIMITS = {
    "alpaca": {
        "requests_per_minute": 200,  # Alpaca allows 200 requests per minute
        "min_interval": 0.3          # Minimum interval between requests (seconds)
    },
    "alchemy": {
        "requests_per_hour": 300,     # Alchemy allows 300 requests per hour
        "min_interval": 12.0          # Minimum interval to stay under limit (seconds)
    },
    "coingecko": {
        "requests_per_minute": 50,    # CoinGecko free tier limits
        "min_interval": 1.2           # Minimum interval between requests (seconds)
    }
}

# Aggregator settings
AGGREGATOR_CONFIG = {
    "default_strategy": "weighted",  # Default aggregation strategy
    "fallback_enabled": True,        # Enable fallback if primary source fails
    "timeout": 5.0                   # Timeout for provider requests (seconds)
}

# Test mode settings
TEST_MODE = {
    "enabled": False,                # Enable test mode (use mock data)
    "mock_data_file": "test_data/price_mock_data.json"  # Mock data source
}

def get_provider_config(provider_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific provider.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        Configuration dictionary for the provider
    """
    if provider_name == "alpaca":
        return {
            "cache_ttl": CACHE_TTL["current_price"],
            "historical_cache_ttl": CACHE_TTL["historical_price"],
            "rate_limit": RATE_LIMITS["alpaca"]
        }
    elif provider_name == "alchemy":
        return {
            "cache_ttl": CACHE_TTL["current_price"],
            "historical_cache_ttl": CACHE_TTL["historical_price"],
            "rate_limit": RATE_LIMITS["alchemy"]
        }
    elif provider_name == "coingecko":
        return {
            "cache_ttl": CACHE_TTL["current_price"],
            "historical_cache_ttl": CACHE_TTL["historical_price"],
            "rate_limit": RATE_LIMITS["coingecko"]
        }
    else:
        return {
            "cache_ttl": CACHE_TTL["current_price"],
            "historical_cache_ttl": CACHE_TTL["historical_price"]
        } 