import hmac
import hashlib
import base64
import json
import time
import requests
from datetime import datetime
from config import Config
import logging

logger = logging.getLogger(__name__)

class BitgetAPI:
    def __init__(self):
        self.api_key = Config.BITGET_API_KEY
        self.secret_key = Config.BITGET_API_SECRET
        self.passphrase = Config.BITGET_PASSPHRASE
        self.base_url = Config.BITGET_BASE_URL
        self.paper_trading = Config.PAPER_TRADING
        
    def _generate_signature(self, timestamp, method, request_path, body=""):
        """Generate signature for Bitget API authentication"""
        message = timestamp + method.upper() + request_path + body
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        return signature
    
    def _get_headers(self, method, request_path, body=""):
        """Generate headers for API requests"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, request_path, body)
        
        headers = {
            'Content-Type': 'application/json',
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase
        }
        return headers
    
    def get_account_balance(self):
        """Get account balance - Enhanced to properly detect real money"""
        if self.paper_trading:
            # Return simulated balance for paper trading
            return {'success': True, 'data': {'available': 1000.0, 'total': 1000.0}}
        
        try:
            # Enhanced balance detection for real trading
            request_path = "/api/mix/v1/account/accounts"  # Updated endpoint
            url = f"{self.base_url}{request_path}"
            headers = self._get_headers('GET', request_path)
            
            # Add productType for futures
            params = {'productType': 'UMCBL'}
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            data = response.json()
            
            logger.info(f"Balance API Response: {data}")
            
            if data.get('code') == '00000' and data.get('data'):
                for account in data['data']:
                    if account.get('marginCoin') == 'USDT':
                        available = float(account.get('available', 0))
                        equity = float(account.get('equity', 0))
                        
                        logger.info(f"Real Balance Detected - Available: ${available}, Equity: ${equity}")
                        
                        return {
                            'success': True,
                            'data': {
                                'available': available,
                                'total': equity,
                                'margin_coin': 'USDT'
                            }
                        }
            
            # Fallback to older endpoint
            request_path_fallback = "/api/mix/v1/account/account"
            url_fallback = f"{self.base_url}{request_path_fallback}"
            headers_fallback = self._get_headers('GET', request_path_fallback)
            
            response_fallback = requests.get(url_fallback, headers=headers_fallback, timeout=15)
            data_fallback = response_fallback.json()
            
            if data_fallback.get('code') == '00000':
                balance_data = data_fallback.get('data', [])
                if balance_data:
                    available = float(balance_data[0].get('available', 0))
                    equity = float(balance_data[0].get('equity', 0))
                    
                    logger.info(f"Balance Fallback - Available: ${available}, Equity: ${equity}")
                    
                    return {
                        'success': True,
                        'data': {
                            'available': available,
                            'total': equity
                        }
                    }
            
            logger.error("Failed to get balance from both endpoints")
            return {'success': False, 'error': 'Failed to get balance from API'}
            
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_current_price(self, symbol="ETHUSDT"):
        """Get current price for symbol"""
        try:
            url = f"{self.base_url}/api/mix/v1/market/ticker?symbol={symbol}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('code') == '00000' and data.get('data'):
                return {
                    'success': True,
                    'price': float(data['data'][0]['last'])
                }
            
            return {'success': False, 'error': 'Failed to get price'}
            
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_klines(self, symbol="ETHUSDT", granularity="1m", limit=100):
        """Get kline data for technical analysis"""
        try:
            # Convert timeframe format
            granularity_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '1h': '1H', '4h': '4H', '1d': '1D'
            }
            
            bg_granularity = granularity_map.get(granularity, '1m')
            url = f"{self.base_url}/api/mix/v1/market/candles"
            params = {
                'symbol': symbol,
                'granularity': bg_granularity,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            if data.get('code') == '00000' and data.get('data'):
                klines = []
                for kline in data['data']:
                    klines.append({
                        'timestamp': int(kline[0]),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    })
                
                return {'success': True, 'data': sorted(klines, key=lambda x: x['timestamp'])}
            
            return {'success': False, 'error': 'Failed to get klines'}
            
        except Exception as e:
            logger.error(f"Error getting klines: {e}")
            return {'success': False, 'error': str(e)}
    
    def place_order(self, side, size, price=None, order_type='market'):
        """Place a trading order"""
        if self.paper_trading:
            # Simulate order placement for paper trading
            current_price_result = self.get_current_price()
            if current_price_result['success']:
                execution_price = current_price_result['price']
                return {
                    'success': True,
                    'data': {
                        'orderId': f"paper_{int(time.time())}",
                        'side': side,
                        'size': size,
                        'price': execution_price,
                        'status': 'filled'
                    }
                }
        
        try:
            request_path = "/api/mix/v1/order/placeOrder"
            url = f"{self.base_url}{request_path}"
            
            order_data = {
                'symbol': Config.SYMBOL,
                'marginCoin': 'USDT',
                'side': 'open_long' if side == 'buy' else 'open_short',
                'orderType': order_type,
                'size': str(size)
            }
            
            if price:
                order_data['price'] = str(price)
            
            body = json.dumps(order_data)
            headers = self._get_headers('POST', request_path, body)
            
            response = requests.post(url, data=body, headers=headers, timeout=10)
            data = response.json()
            
            if data.get('code') == '00000':
                return {'success': True, 'data': data.get('data')}
            
            return {'success': False, 'error': data.get('msg', 'Order failed')}
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'success': False, 'error': str(e)}
    
    def close_position(self, side, size):
        """Close a position"""
        if self.paper_trading:
            # Simulate position closing
            return {
                'success': True,
                'data': {
                    'orderId': f"paper_close_{int(time.time())}",
                    'status': 'filled'
                }
            }
        
        try:
            request_path = "/api/mix/v1/order/placeOrder"
            url = f"{self.base_url}{request_path}"
            
            close_side = 'close_long' if side == 'long' else 'close_short'
            
            order_data = {
                'symbol': Config.SYMBOL,
                'marginCoin': 'USDT',
                'side': close_side,
                'orderType': 'market',
                'size': str(size)
            }
            
            body = json.dumps(order_data)
            headers = self._get_headers('POST', request_path, body)
            
            response = requests.post(url, data=body, headers=headers, timeout=10)
            data = response.json()
            
            if data.get('code') == '00000':
                return {'success': True, 'data': data.get('data')}
            
            return {'success': False, 'error': data.get('msg', 'Close failed')}
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'error': str(e)}
    
    def set_leverage(self, leverage):
        """Set leverage for symbol"""
        if self.paper_trading:
            return {'success': True, 'data': {'leverage': leverage}}
        
        try:
            request_path = "/api/mix/v1/account/setLeverage"
            url = f"{self.base_url}{request_path}"
            
            leverage_data = {
                'symbol': Config.SYMBOL,
                'marginCoin': 'USDT',
                'leverage': str(leverage)
            }
            
            body = json.dumps(leverage_data)
            headers = self._get_headers('POST', request_path, body)
            
            response = requests.post(url, data=body, headers=headers, timeout=10)
            data = response.json()
            
            if data.get('code') == '00000':
                return {'success': True, 'data': data.get('data')}
            
            return {'success': False, 'error': data.get('msg', 'Leverage set failed')}
            
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return {'success': False, 'error': str(e)}
