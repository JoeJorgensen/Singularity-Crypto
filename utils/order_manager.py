"""
OrderManager - Manages trading orders and execution.
"""
from typing import Dict, List, Optional
import logging
# Fix circular import by using type hints differently
# from api.alpaca_api import AlpacaAPI
from utils.logging_config import get_logger
from datetime import datetime
import time

# Get logger for order manager
logger = get_logger('order_manager')

class OrderManager:
    """
    Manages trading orders and execution.
    """
    
    def __init__(self, alpaca_api, config: Dict):
        """
        Initialize OrderManager.
        
        Args:
            alpaca_api: AlpacaAPI instance
            config: Configuration dictionary
        """
        self.alpaca = alpaca_api
        self.config = config
        
        # Load configuration settings
        self.risk_settings = config.get('trading', {}).get('risk_management', {})
        self.max_risk_per_trade = self.risk_settings.get('max_risk_per_trade', 0.02)
        self.trailing_stop_percent = self.risk_settings.get('trailing_stop_percent', 0.02)
        logger.info("OrderManager initialized with Alpaca API")
    
    def get_account_info(self) -> Dict:
        """
        Get account information.
        
        Returns:
            Account information dictionary
        """
        account_info = self.alpaca.get_account()
        logger.debug(f"Retrieved account info - Equity: {account_info.equity}, Buying Power: {account_info.buying_power}")
        return {
            'equity': account_info.equity,
            'buying_power': account_info.buying_power,
            'cash': account_info.cash,
            'portfolio_value': account_info.portfolio_value,
            'status': account_info.status
        }
    
    def get_positions(self) -> List[Dict]:
        """
        Get current ETH/USD position.
        
        Returns:
            List containing ETH/USD position dictionary if it exists
        """
        try:
            # Only check ETH/USD
            symbol = 'ETH/USD'
            positions = []
            
            try:
                position = self.alpaca.get_position(symbol)
                if position:
                    # Convert position object to dictionary
                    position_dict = {
                        'symbol': position.symbol,
                        'qty': position.qty,
                        'market_value': position.market_value,
                        'avg_entry_price': position.avg_entry_price,
                        'current_price': position.current_price,
                        'unrealized_pl': position.unrealized_pl,
                        'unrealized_plpc': position.unrealized_plpc
                    }
                    
                    # Calculate what percentage of portfolio this position represents
                    try:
                        account = self.alpaca.get_account()
                        portfolio_value = float(account.portfolio_value)
                        position_value = float(position.market_value)
                        position_dict['portfolio_percentage'] = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0
                    except:
                        position_dict['portfolio_percentage'] = 0
                        
                    positions.append(position_dict)
            except Exception as position_error:
                error_message = str(position_error).lower()
                # Only log at debug level for 'position does not exist' errors
                if 'position does not exist' in error_message:
                    logger.debug("No position exists for ETH/USD - this is normal")
                else:
                    # Log real errors at warning level
                    logger.warning(f"Error retrieving position for ETH/USD: {error_message}")
            
            if positions:
                logger.debug("Retrieved ETH/USD position")
            else:
                logger.debug("No active ETH/USD position found")
            return positions
        except Exception as e:
            error_message = str(e).lower()
            # Only log as error if it's not a 'position does not exist' error
            if 'position does not exist' not in error_message:
                logger.error(f"Error retrieving positions: {str(e)}", exc_info=True)
            else:
                # Log at debug level instead for 'position does not exist'
                logger.debug("No positions exist - this is normal when no trades have been executed")
            return []
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """
        Get recent trades.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries
        """
        # Use the get_trades method from AlpacaAPI
        trades = self.alpaca.get_trades(limit=limit)
        logger.debug(f"Retrieved {len(trades)} recent trades")
        
        # Convert trades to dictionary format with timestamp field
        trade_dicts = []
        for trade in trades:
            trade_dict = {
                'id': trade.id,
                'symbol': trade.symbol,
                'side': trade.side,
                'qty': trade.qty,
                'type': trade.type,
                'status': trade.status,
                'timestamp': trade.submitted_at.isoformat() if hasattr(trade, 'submitted_at') else datetime.now().isoformat()
            }
            if hasattr(trade, 'filled_avg_price'):
                trade_dict['filled_avg_price'] = trade.filled_avg_price
            trade_dicts.append(trade_dict)
            
        return trade_dicts
    
    def execute_trade(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Dict:
        """
        Execute a trade with the given parameters.
        
        Args:
            symbol: Trading pair symbol (e.g. ETH/USD)
            qty: Quantity to trade
            side: Trade side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', 'stop_limit')
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop limit orders
            
        Returns:
            Dictionary with order information
        """
        # Store original symbol before any modification
        original_symbol = symbol
        
        # Log the intended trade
        logger.info(f"Executing {order_type} {side} order for {qty} {original_symbol}")
        
        # Get current account info to double-check balances with forced refresh
        try:
            account = self.alpaca.get_account(force_refresh=True)
            buying_power = float(account.buying_power)
            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)
            
            # Log detailed account information for debugging
            logger.info(f"Account status before trade - Cash: ${cash:.2f}, Buying Power: ${buying_power:.2f}, Portfolio Value: ${portfolio_value:.2f}")
            
            # For buy orders, verify buying power again before submitting
            if side.lower() == 'buy':
                # Get current price if needed to estimate order value
                price = limit_price
                if not price:
                    # Try to get current price
                    try:
                        position = self.alpaca.get_position(original_symbol)
                        if position and hasattr(position, 'current_price'):
                            price = float(position.current_price)
                    except Exception as price_error:
                        # Fall back to last known price
                        logger.debug(f"Could not get current price for {original_symbol}: {str(price_error)}")
                        # Ensure price has a default value
                        price = None
                
                # If we have a price, calculate estimated order value
                if price:
                    order_value = qty * price
                    # Add safety margin for market orders
                    if order_type.lower() == 'market':
                        order_value *= 1.03  # Add 3% for market slippage (increased from 2%)
                    
                    # Check if order exceeds buying power
                    if order_value > buying_power:
                        logger.warning(f"Order value (${order_value:.2f}) exceeds buying power (${buying_power:.2f}). Adjusting quantity.")
                        # Calculate maximum quantity we can afford with a stronger safety margin
                        max_qty = (buying_power * 0.97) / price if price > 0 else 0  # Increased safety from 0.98 to 0.97
                        
                        # If max_qty is too small, abort the order
                        if max_qty < 0.001 or max_qty * price < 5:
                            logger.error(f"Insufficient buying power (${buying_power:.2f}) for minimum order.")
                            return {
                                "error": f"Insufficient buying power for minimum order.",
                                "status": "rejected"
                            }
                            
                        # Otherwise use adjusted quantity
                        qty = max_qty
                        logger.warning(f"Adjusted quantity to {qty:.6f} to fit buying power")
                else:
                    # If we couldn't determine a price, proceed with the order but log a warning
                    logger.warning(f"Could not determine current price for {original_symbol}. Proceeding with order without price validation.")
            
            # Validate order type and required parameters
            if order_type.lower() == 'limit' and limit_price is None:
                error_msg = "Limit price is required for limit orders"
                logger.error(error_msg)
                return {"error": error_msg, "status": "rejected"}
            
            if order_type.lower() == 'stop_limit':
                if stop_price is None:
                    error_msg = "Stop price is required for stop_limit orders"
                    logger.error(error_msg)
                    return {"error": error_msg, "status": "rejected"}
                
                if limit_price is None:
                    error_msg = "Limit price is required for stop_limit orders"
                    logger.error(error_msg)
                    return {"error": error_msg, "status": "rejected"}
        
            # Log the attempt
            price_info = ""
            if limit_price:
                price_info += f" at limit price {limit_price}"
            if stop_price:
                price_info += f" with stop price {stop_price}"
            
            logger.info(f"Submitting {order_type} {side} order for {qty} {original_symbol}{price_info}")
            
            # Set up retry logic for insufficient balance errors
            max_retries = 3
            current_qty = qty
            
            for attempt in range(max_retries):
                try:
                    # Submit order
                    order_params = {
                        'symbol': original_symbol,
                        'qty': current_qty,
                        'side': side,
                        'type': order_type
                    }
                    
                    # Only include optional parameters if they are provided
                    if limit_price is not None:
                        order_params['limit_price'] = limit_price
                        
                    if stop_price is not None:
                        order_params['stop_price'] = stop_price
                    
                    # Submit the order with only the relevant parameters
                    order = self.alpaca.submit_order(**order_params)
                    
                    # Convert order object to dictionary if needed
                    if not isinstance(order, dict):
                        order_dict = {
                            'id': order.id if hasattr(order, 'id') else None,
                            'status': order.status if hasattr(order, 'status') else 'unknown',
                            'symbol': order.symbol if hasattr(order, 'symbol') else original_symbol,
                            'side': order.side if hasattr(order, 'side') else side,
                            'qty': order.qty if hasattr(order, 'qty') else current_qty,
                            'filled_avg_price': order.filled_avg_price if hasattr(order, 'filled_avg_price') else None
                        }
                    else:
                        order_dict = order
                    
                    # Log the result
                    if order_dict.get('status') == 'filled':
                        logger.info(f"Order {order_dict.get('id')} filled: {side} {current_qty} {original_symbol} at {order_dict.get('filled_avg_price') or 'market price'}")
                    elif order_dict.get('status') == 'accepted' or order_dict.get('status') == 'new':
                        logger.info(f"Order {order_dict.get('id')} accepted: {side} {current_qty} {original_symbol} (awaiting fill)")
                    else:
                        logger.info(f"Order {order_dict.get('id')} status: {order_dict.get('status')}")
                    
                    return order_dict
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check for specific error messages
                    if "stop price cannot be zero" in error_msg or "limit price cannot be zero" in error_msg:
                        logger.error(f"Order validation failed: {error_msg}")
                        return {
                            "error": f"Price validation error: {error_msg}",
                            "status": "rejected"
                        }
                        
                    # Check for order type validation errors
                    if "unsupported order type" in error_msg:
                        # For crypto, we can only use market, limit, stop_limit
                        logger.error(f"Unsupported order type: {order_type}. For crypto, use 'market', 'limit', or 'stop_limit'")
                        return {
                            "error": f"Unsupported order type for crypto: {order_type}. Use 'market', 'limit', or 'stop_limit'",
                            "status": "rejected"
                        }
                        
                    # Check for price validation errors
                    if "stop price must be above" in error_msg or "stop price must be below" in error_msg:
                        logger.error(f"Stop price validation failed: {error_msg}")
                        return {
                            "error": f"Stop price validation error: {error_msg}",
                            "status": "rejected"
                        }
                    
                    # Check if this is an insufficient balance error
                    if "insufficient" in error_msg and attempt < max_retries - 1:
                        # Try to extract available balance from the error message
                        import re
                        available_match = re.search(r'available: \$([\d.]+)', error_msg)
                        requested_match = re.search(r'requested: \$([\d.]+)', error_msg)
                        
                        if available_match and requested_match:
                            available = float(available_match.group(1))
                            requested = float(requested_match.group(1))
                            
                            # Only retry if there's some actual available balance
                            if available > 5.0:  # Minimum order value of $5
                                # Calculate a conservative percentage of the available balance
                                new_size_ratio = (available * 0.95) / requested
                                current_qty = qty * new_size_ratio
                                
                                logger.warning(f"Insufficient balance error. Retrying with calculated quantity: {current_qty:.6f} {original_symbol} (attempt {attempt+1}/{max_retries})")
                                continue
                            else:
                                # If available balance is too low, provide clear error
                                logger.error(f"Order failed: requested ${requested:.2f} but only ${available:.2f} available")
                                return {
                                    "error": f"Insufficient balance: ${available:.2f} available, ${requested:.2f} required.",
                                    "status": "rejected"
                                }
                        else:
                            # If we can't extract the exact values, use a more aggressive reduction
                            reduction_factor = 0.65 * (0.5 ** attempt)
                            current_qty = qty * reduction_factor
                            
                            # Make sure we're not trying a tiny order
                            if current_qty < 0.001:
                                logger.error(f"Cannot reduce quantity further. Order aborted.")
                                return {
                                    "error": f"Insufficient balance for minimum order size.",
                                    "status": "rejected"
                                }
                                
                            logger.warning(f"Insufficient balance error. Retrying with reduced quantity: {current_qty:.6f} {original_symbol} (attempt {attempt+1}/{max_retries})")
                            continue
                    
                    # Log and return error for other errors or if we've exhausted retries
                    logger.error(f"Order execution failed for {side} {current_qty} {original_symbol}: {str(e)}", exc_info=True)
                    
                    # Special handling for price determination errors
                    if "could not determine current price" in error_msg.lower():
                        # If this is a market order, try again with a direct market order, bypassing buying power check
                        if order_type.lower() == 'market' and attempt < max_retries - 1:
                            logger.warning("Price data unavailable. Retrying market order with direct API approach...")
                            try:
                                # Try to execute a direct market order without the buying power check
                                trading_symbol = original_symbol.replace('/', '')
                                
                                # Use direct API call to Alpaca
                                base_url = "https://paper-api.alpaca.markets/v2/orders"
                                headers = {
                                    "APCA-API-KEY-ID": self.alpaca.api_key,
                                    "APCA-API-SECRET-KEY": self.alpaca.api_secret,
                                    "Content-Type": "application/json"
                                }
                                
                                # Build basic market order
                                order_data = {
                                    "symbol": trading_symbol,
                                    "qty": str(current_qty),
                                    "side": side.lower(),
                                    "type": "market",
                                    "time_in_force": "gtc"
                                }
                                
                                import requests
                                response = requests.post(base_url, headers=headers, json=order_data)
                                
                                if response.status_code == 200:
                                    order_dict = response.json()
                                    logger.info(f"Direct market order successful: {order_dict.get('id')}")
                                    return order_dict
                                else:
                                    logger.error(f"Direct market order failed: {response.text}")
                            except Exception as direct_error:
                                logger.error(f"Direct market order failed: {str(direct_error)}")
                    
                    return {
                        "error": str(e),
                        "status": "rejected" 
                    }
        except Exception as e:
            logger.warning(f"Could not verify account balance before order: {str(e)}")
            return {
                "error": f"Failed to verify account balance before order: {str(e)}",
                "status": "rejected"
            }
    
    def execute_trade_with_risk_management(
        self,
        symbol: str,
        qty: float,
        side: str,
        entry_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ) -> Dict:
        """
        Execute a trade with risk management (stop loss and take profit).
        
        Args:
            symbol: Symbol to trade
            qty: Quantity to trade
            side: 'buy' or 'sell' (where 'sell' means closing a long position, not short selling)
            entry_price: Entry price (None for market price)
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            
        Returns:
            Dictionary with order information
        """
        # Note: Only long positions are supported since crypto accounts are non-marginable 
        # and do not support short selling. 'sell' side here only means exiting an existing long position.
        
        logger.info(f"Executing {side} order for {qty} {symbol} with risk management")
        if stop_loss_price:
            logger.info(f"Stop loss set at {stop_loss_price}")
        if take_profit_price:
            logger.info(f"Take profit set at {take_profit_price}")
            
        # Execute main order
        order_type = 'limit' if entry_price else 'market'
        try:
            # Execute the main order
            main_order = self.execute_trade(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                limit_price=entry_price
            )
            
            # Only set up risk management orders if the main order was successful
            if main_order.get('status') in ['filled', 'accepted', 'new']:
                # Create supporting orders for risk management
                try:
                    # Handle stop loss order for long positions
                    if stop_loss_price and side == 'buy':
                        logger.info(f"Setting stop loss: sell {qty} {symbol} at {stop_loss_price}")
                        
                        try:
                            # Calculate limit price slightly lower than stop price for sell stop
                            # to ensure order execution
                            limit_price_offset = 0.99
                            stop_limit_price = stop_loss_price * limit_price_offset
                            
                            stop_order = self.execute_trade(
                                symbol=symbol,
                                qty=qty,
                                side='sell',
                                order_type='stop_limit',  # Use stop_limit instead of stop
                                stop_price=stop_loss_price,
                                limit_price=stop_limit_price  # Add limit price
                            )
                            logger.info(f"Stop loss order created with ID: {stop_order.get('id')}")
                        except Exception as stop_error:
                            logger.error(f"Failed to create stop loss order: {str(stop_error)}")
                    
                    # Handle take profit order for long positions
                    if take_profit_price and side == 'buy':
                        logger.info(f"Setting take profit: sell {qty} {symbol} at {take_profit_price}")
                        
                        try:
                            take_profit_order = self.execute_trade(
                                symbol=symbol,
                                qty=qty,
                                side='sell',
                                order_type='limit',
                                limit_price=take_profit_price
                            )
                            logger.info(f"Take profit order created with ID: {take_profit_order.get('id')}")
                        except Exception as tp_error:
                            logger.error(f"Failed to create take profit order: {str(tp_error)}")
                
                except Exception as risk_error:
                    logger.error(f"Error setting up risk management orders: {str(risk_error)}")
                    # We don't raise here because the main order was successful
            
            logger.info(f"Trade execution with risk management completed for {symbol}")
            return main_order
            
        except Exception as e:
            logger.error(f"Failed to execute trade with risk management for {symbol}: {str(e)}", exc_info=True)
            raise
    
    def close_position(self, symbol: str) -> Dict:
        """
        Close a position.
        
        Args:
            symbol: Symbol to close position for
            
        Returns:
            Order information dictionary
        """
        logger.info(f"Closing position for {symbol}")
        try:
            # Keep original symbol format with slash
            original_symbol = symbol
            
            # First, cancel all open orders for this symbol to prevent conflicts
            try:
                logger.info(f"Canceling all open orders for {original_symbol} before closing position")
                # Get all open orders for this symbol
                open_orders = self.alpaca.get_trades(original_symbol, limit=20, force_fresh=True)
                
                # Cancel any open orders to avoid conflicts
                canceled_orders = []
                for order in open_orders:
                    # Only cancel orders that are still open/active
                    if hasattr(order, 'status') and order.status in ['new', 'accepted', 'pending_new', 'accepted_for_bidding']:
                        try:
                            if hasattr(order, 'id') and order.id:
                                logger.info(f"Canceling order {order.id} for {original_symbol}")
                                self.alpaca.trading_client.cancel_order(order.id)
                                canceled_orders.append(order.id)
                        except Exception as cancel_error:
                            logger.warning(f"Could not cancel order {order.id}: {str(cancel_error)}")
                
                # Add a small delay to allow the cancellations to process
                if canceled_orders:
                    logger.info(f"Canceled {len(canceled_orders)} open orders for {original_symbol}, waiting for processing")
                    time.sleep(1.0)  # Increase delay to ensure cancellations process
            except Exception as cancel_error:
                logger.warning(f"Error canceling open orders for {original_symbol}: {str(cancel_error)}")
            
            # Try to get the current position to determine quantity
            try:
                position = self.alpaca.get_position(original_symbol)
                if position and hasattr(position, 'qty') and float(position.qty) > 0:
                    position_qty = float(position.qty)
                    logger.info(f"Current position size for {original_symbol}: {position_qty}")
                else:
                    logger.warning(f"No position found for {original_symbol}")
                    return {"status": "no_position", "message": f"No position exists for {original_symbol}"}
            except Exception as pos_error:
                logger.warning(f"Error retrieving position for {original_symbol}: {str(pos_error)}")
                position_qty = None
            
            # Try the standard way first
            try:
                # Now close the position
                result = self.alpaca.close_position(original_symbol)
                
                # Convert result to dictionary if needed
                if not isinstance(result, dict):
                    result_dict = {
                        'id': result.id if hasattr(result, 'id') else None,
                        'status': result.status if hasattr(result, 'status') else 'unknown',
                        'symbol': result.symbol if hasattr(result, 'symbol') else symbol,
                        'side': result.side if hasattr(result, 'side') else None,
                        'qty': result.qty if hasattr(result, 'qty') else None
                    }
                else:
                    result_dict = result
                
                logger.info(f"Position closed for {symbol}: {result_dict.get('status', 'unknown status')}")
                return result_dict
                
            except Exception as close_error:
                error_msg = str(close_error).lower()
                
                # Check if this is the insufficient balance error
                if "insufficient balance" in error_msg and position_qty:
                    logger.warning(f"Insufficient balance error when closing position, trying alternative method")
                    
                    # Try alternative approach: create a market sell order for the position quantity
                    try:
                        logger.info(f"Attempting to close {original_symbol} position with market order, qty: {position_qty}")
                        
                        # For crypto, we need to determine if this is a long or short position
                        # Assuming it's a long position (most common case)
                        close_side = 'sell'
                        
                        # Submit a market order to close the position
                        market_close = self.execute_trade(
                            symbol=original_symbol,
                            qty=position_qty,
                            side=close_side,
                            order_type='market'
                        )
                        
                        logger.info(f"Position closed via market order: {market_close.get('id')}")
                        return {
                            'id': market_close.get('id'),
                            'status': market_close.get('status', 'executed'),
                            'symbol': original_symbol,
                            'side': close_side,
                            'qty': position_qty,
                            'method': 'market_order'
                        }
                    except Exception as market_error:
                        logger.error(f"Failed to close position with market order: {str(market_error)}")
                        return {
                            "error": f"Failed to close position after multiple attempts: {str(market_error)}",
                            "status": "rejected"
                        }
                else:
                    # Re-raise other errors
                    raise close_error
                
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {str(e)}", exc_info=True)
            return {
                "error": f"Failed to close position: {str(e)}",
                "status": "rejected"
            }
    
    def close_all_positions(self) -> List[Dict]:
        """
        Close all positions.
        
        Returns:
            List of order information dictionaries
        """
        logger.info("Closing all positions")
        try:
            results = self.alpaca.close_all_positions()
            
            # Convert results to list of dictionaries if needed
            result_dicts = []
            if results:
                for result in results:
                    if not isinstance(result, dict):
                        result_dict = {
                            'id': result.id if hasattr(result, 'id') else None,
                            'status': result.status if hasattr(result, 'status') else 'unknown',
                            'symbol': result.symbol if hasattr(result, 'symbol') else None,
                            'side': result.side if hasattr(result, 'side') else None,
                            'qty': result.qty if hasattr(result, 'qty') else None
                        }
                    else:
                        result_dict = result
                    result_dicts.append(result_dict)
            
            logger.info(f"All positions closed - {len(result_dicts)} orders executed")
            return result_dicts
        except Exception as e:
            logger.error(f"Failed to close all positions: {str(e)}", exc_info=True)
            raise 