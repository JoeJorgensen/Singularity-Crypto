"""
Price Provider Factory.
Creates and configures price data providers and aggregator based on configuration.
"""
from typing import Dict, Any, Optional, Union, List, Tuple

from api.price_provider import (
    PriceProvider, 
    AlpacaPriceProvider, 
    CoinGeckoPriceProvider,
    AlchemyPriceProvider
)
from api.price_aggregator import PriceAggregator
from api.price_config import PROVIDER_WEIGHTS, DEFAULT_PROVIDER, get_provider_config
from api.alpaca_api import AlpacaAPI
from api.coingecko_api import CoinGeckoAPI
from api.alchemy_price_api import AlchemyPriceAPI
from utils.logging_config import get_logger

# Get logger
logger = get_logger('price_factory')

def create_price_provider(provider_type: str, **kwargs) -> Optional[PriceProvider]:
    """
    Create a price provider of the specified type with configuration.
    
    Args:
        provider_type: Type of provider ('alpaca', 'coingecko', 'alchemy')
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured PriceProvider instance or None if creation fails
    """
    try:
        # Get base configuration for this provider type
        config = get_provider_config(provider_type)
        
        # Override with any provided kwargs
        if kwargs:
            config.update(kwargs)
            
        # Create the appropriate provider
        if provider_type == "alpaca":
            # Create AlpacaAPI if not provided
            if 'alpaca_api' not in kwargs:
                alpaca_api = AlpacaAPI(config)
                logger.info("Created new AlpacaAPI instance")
            else:
                alpaca_api = kwargs['alpaca_api']
                
            return AlpacaPriceProvider(alpaca_api)
            
        elif provider_type == "coingecko":
            # Create CoinGeckoAPI if not provided
            if 'coingecko_api' not in kwargs:
                coingecko_api = CoinGeckoAPI(config)
                logger.info("Created new CoinGeckoAPI instance")
            else:
                coingecko_api = kwargs['coingecko_api']
                
            return CoinGeckoPriceProvider(coingecko_api)
            
        elif provider_type == "alchemy":
            # Create AlchemyPriceAPI if not provided
            if 'alchemy_price_api' not in kwargs:
                alchemy_price_api = AlchemyPriceAPI(config)
                logger.info("Created new AlchemyPriceAPI instance")
            else:
                alchemy_price_api = kwargs['alchemy_price_api']
                
            return AlchemyPriceProvider(alchemy_price_api)
            
        else:
            logger.error(f"Unknown provider type: {provider_type}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating price provider of type {provider_type}: {e}")
        return None

def create_price_aggregator(provider_types: List[str] = None, 
                           custom_weights: Dict[str, float] = None,
                           **kwargs) -> PriceAggregator:
    """
    Create a price aggregator with the specified providers.
    
    Args:
        provider_types: List of provider types to include
        custom_weights: Optional custom weights for providers
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured PriceAggregator instance
    """
    # Use default providers if none specified
    if not provider_types:
        provider_types = list(PROVIDER_WEIGHTS.keys())
        
    # Use default weights if none specified
    weights = custom_weights or PROVIDER_WEIGHTS
    
    # Create the providers
    providers = []
    for provider_type in provider_types:
        if provider_type in weights:
            provider = create_price_provider(provider_type, **kwargs)
            if provider:
                providers.append((provider, weights[provider_type]))
                logger.info(f"Added {provider_type} provider with weight {weights[provider_type]}")
            else:
                logger.warning(f"Failed to create {provider_type} provider, skipping")
    
    # Create the aggregator
    aggregator = PriceAggregator(providers)
    logger.info(f"Created PriceAggregator with {len(providers)} providers")
    
    return aggregator

def get_default_price_provider(**kwargs) -> Union[PriceProvider, PriceAggregator]:
    """
    Get the default price provider based on configuration.
    
    Args:
        **kwargs: Additional configuration parameters
        
    Returns:
        Default configured price provider
    """
    if DEFAULT_PROVIDER == "aggregator":
        return create_price_aggregator(**kwargs)
    else:
        provider = create_price_provider(DEFAULT_PROVIDER, **kwargs)
        if provider:
            return provider
        
        # Fallback to aggregator if provider creation fails
        logger.warning(f"Failed to create {DEFAULT_PROVIDER} provider, falling back to aggregator")
        return create_price_aggregator(**kwargs) 