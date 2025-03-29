"""
Binance API interface for cryptocurrency trading.
"""
import os
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

class BinanceAPI:
    def __init__(self, config):
        """Initialize Binance API client."""
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = Client(self.api_key, self.api_secret)
        self.config = config
    
    def get_klines(self, symbol='ETHUSDT', interval='1h', limit=100):
        """Get candlestick data from Binance."""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df
            
        except BinanceAPIException as e:
            print(f"Error fetching klines: {e}")
            return pd.DataFrame()
    
    def get_account_info(self):
        """Get account information including balances."""
        try:
            account = self.client.get_account()
            balances = {}
            
            # Extract ETH and USDT balances
            for balance in account['balances']:
                if balance['asset'] in ['ETH', 'USDT']:
                    balances[balance['asset']] = float(balance['free'])
            
            return balances
            
        except BinanceAPIException as e:
            print(f"Error fetching account info: {e}")
            return {'ETH': 0, 'USDT': 0}
    
    def get_position(self, symbol='ETHUSDT'):
        """Get current position for a symbol."""
        try:
            # Get position information
            position = self.client.get_asset_balance(asset='ETH')
            if float(position['free']) > 0 or float(position['locked']) > 0:
                # Get average entry price
                trades = self.client.get_my_trades(symbol=symbol, limit=100)
                if trades:
                    # Calculate average entry price from recent trades
                    total_qty = sum(float(trade['qty']) for trade in trades if trade['isBuyer'])
                    total_cost = sum(float(trade['qty']) * float(trade['price']) for trade in trades if trade['isBuyer'])
                    avg_price = total_cost / total_qty if total_qty > 0 else 0
                    
                    # Get current price
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # Calculate unrealized PnL
                    position_size = float(position['free']) + float(position['locked'])
                    unrealized_pnl = (current_price - avg_price) * position_size
                    pnl_percentage = (unrealized_pnl / (avg_price * position_size)) * 100 if position_size > 0 else 0
                    
                    return {
                        'size': position_size,
                        'entry_price': avg_price,
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_percentage': pnl_percentage
                    }
            
            return None
            
        except BinanceAPIException as e:
            print(f"Error fetching position: {e}")
            return None
    
    def place_order(self, symbol='ETHUSDT', side='buy', order_type='market', quantity=0, price=None):
        """Place a new order."""
        try:
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'quantity': quantity
            }
            
            if order_type.lower() == 'limit':
                params['type'] = 'LIMIT'
                params['price'] = price
                params['timeInForce'] = 'GTC'
            else:
                params['type'] = 'MARKET'
            
            order = self.client.create_order(**params)
            
            return {
                'success': True,
                'order_id': order['orderId'],
                'message': 'Order placed successfully'
            }
            
        except BinanceAPIException as e:
            return {
                'success': False,
                'message': str(e)
            }
    
    def close_position(self, symbol='ETHUSDT'):
        """Close an open position."""
        try:
            position = self.get_position(symbol)
            if position and position['size'] > 0:
                result = self.place_order(
                    symbol=symbol,
                    side='sell',
                    order_type='market',
                    quantity=position['size']
                )
                return result
            return {'success': False, 'message': 'No position to close'}
            
        except BinanceAPIException as e:
            return {'success': False, 'message': str(e)}
    
    def close_all_positions(self):
        """Close all open positions."""
        try:
            # For this simple version, we only handle ETH/USDT
            result = self.close_position('ETHUSDT')
            return result
            
        except BinanceAPIException as e:
            return {'success': False, 'message': str(e)}
    
    def get_recent_trades(self, symbol='ETHUSDT', limit=5):
        """Get recent trades for a symbol."""
        try:
            trades = self.client.get_my_trades(symbol=symbol, limit=limit)
            formatted_trades = []
            
            for trade in trades:
                formatted_trades.append({
                    'side': 'buy' if trade['isBuyer'] else 'sell',
                    'amount': float(trade['qty']),
                    'price': float(trade['price']),
                    'value': float(trade['quoteQty']),
                    'timestamp': pd.to_datetime(trade['time'], unit='ms')
                })
            
            return formatted_trades
            
        except BinanceAPIException as e:
            print(f"Error fetching trades: {e}")
            return [] 