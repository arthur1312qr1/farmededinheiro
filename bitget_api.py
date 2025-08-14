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
        
        # Paper trading state
        self.paper_balance = 1000.0
        self.paper_positions = []

    def _generate_signature(self, timestamp, method, request_path, body=""):
        """Generate signature for Bitget API authentication"""
        try:
            message = timestamp + method.upper() + request_path + body
            signature = base64.b64encode(
                hmac.new(
                    self.secret_key.encode('utf-8'),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')
            return signature
        except Exception as e:
            logger.error(f"Error generating signature: {e}")
            return ""

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
        """Get account balance with enhanced real money detection"""
        if self.paper_trading:
            logger.info(f"Paper Trading Mode - Balance: ${self.paper_balance}")
            return {
                'success': True, 
                'data': {
                    'available': self.paper_balance,
                    'total': self.paper_balance,
                    'mode': 'paper'
                }
            }

        try:
            # Try futures account endpoint first
            request_path = "/api/mix/v1/account/accounts"
            url = f"{self.base_url}{request_path}"
            headers = self._get_headers('GET', request_path)
            params = {'productType': 'UMCBL'}

            response = requests.get(url, headers=headers, params=params, timeout=15)
            data = response.json()
            
            logger.info(f"Balance API Response: {data}")

            if data.get('code') == '00000' and data.get('data'):
                for account in data['data']:
                    if account.get('marginCoin') == 'USDT':
                        available = float(account.get('available', 0))
                        equity = float(account.get('equity', 0))
                        
                        logger.info(f"✅ REAL BALANCE CONNECTED - Available: ${available:.2f}, Equity: ${equity:.2f}")
                        
                        return {
                            'success': True,
                            'data': {
                                'available': available,
                                'total': equity,
                                'margin_coin': 'USDT',
                                'mode': 'real'
                            }
                        }

            # Fallback endpoint
            request_path_fallback = "/api/mix/v1/account/account"
            url_fallback = f"{self.base_url}{request_path_fallback}"
            headers_fallback = self._get_headers('GET', request_path_fallback)
            
            response_fallback = requests.get(url_fallback, headers=headers_fallback, timeout=15)
            data_fallback = response_fallback.json()

            if data_fallback.get('code') == '00000' and data_fallback.get('data'):
                balance_data = data_fallback.get('data', [])
                if balance_data:
                    available = float(balance_data[0].get('available', 0))
                    equity = float(balance_data[0].get('equity', 0))
                    
                    logger.info(f"✅ REAL BALANCE (Fallback) - Available: ${available:.2f}, Equity: ${equity:.2f}")
                    
                    return {
                        'success': True,
                        'data': {
                            'available': available,
                            'total': equity,
                            'mode': 'real'
                        }
                    }

            logger.error(f"❌ Failed to get balance - Response: {data}")
            return {'success': False, 'error': 'Failed to connect to Bitget API'}

        except Exception as e:
            logger.error(f"❌ Error getting balance: {e}")
            return {'success': False, 'error': str(e)}

    def get_current_price(self, symbol="ETHUSDT_UMCBL"):
        """Get current price for symbol"""
        try:
            # Clean symbol format for API
            api_symbol = symbol.replace('_UMCBL', 'USDT_UMCBL')
            url = f"{self.base_url}/api/mix/v1/market/ticker?symbol={api_symbol}"
            
            response = requests.get(url, timeout=10)
            data = response.json()

            if data.get('code') == '00000' and data.get('data'):
                price = float(data['data'][0]['last'])
                return {'success': True, 'price': price}
            
            return {'success': False, 'error': 'Failed to get price'}
            
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return {'success': False, 'error': str(e)}

    def get_klines(self, symbol="ETHUSDT_UMCBL", granularity="1m", limit=100):
        """Get kline data for technical analysis"""
        try:
            # Granularity mapping
            granularity_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', 
                '1h': '1H', '4h': '4H', '1d': '1D'
            }
            bg_granularity = granularity_map.get(granularity, '1m')
            
            # Clean symbol format
            api_symbol = symbol.replace('_UMCBL', 'USDT_UMCBL')
            url = f"{self.base_url}/api/mix/v1/market/candles"
            
            params = {
                'symbol': api_symbol,
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
            # Paper trading simulation
            current_price_result = self.get_current_price()
            if current_price_result['success']:
                execution_price = current_price_result['price']
                
                # Simulate order execution
                order_id = f"paper_{int(time.time())}"
                
                return {
                    'success': True,
                    'data': {
                        'orderId': order_id,
                        'side': side,
                        'size': size,
                        'price': execution_price,
                        'status': 'filled',
                        'mode': 'paper'
                    }
                }
            return {'success': False, 'error': 'Failed to get price for paper trade'}

        try:
            request_path = "/api/mix/v1/order/placeOrder"
            url = f"{self.base_url}{request_path}"
            
            # Clean symbol format
            api_symbol = Config.SYMBOL.replace('_UMCBL', 'USDT_UMCBL')
            
            order_data = {
                'symbol': api_symbol,
                'marginCoin': 'USDT',
                'side': 'open_long' if side == 'buy' else 'open_short',
                'orderType': order_type,
                'size': str(size)
            }

            if price and order_type != 'market':
                order_data['price'] = str(price)

            body = json.dumps(order_data)
            headers = self._get_headers('POST', request_path, body)

            response = requests.post(url, data=body, headers=headers, timeout=10)
            data = response.json()

            if data.get('code') == '00000':
                return {'success': True, 'data': data.get('data'), 'mode': 'real'}
            
            return {'success': False, 'error': data.get('msg', 'Order failed')}

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'success': False, 'error': str(e)}

    def set_leverage(self, leverage):
        """Set leverage for symbol"""
        if self.paper_trading:
            logger.info(f"Paper Trading - Leverage set to {leverage}x")
            return {'success': True, 'data': {'leverage': leverage}}

        try:
            request_path = "/api/mix/v1/account/setLeverage"
            url = f"{self.base_url}{request_path}"
            
            # Clean symbol format
            api_symbol = Config.SYMBOL.replace('_UMCBL', 'USDT_UMCBL')
            
            leverage_data = {
                'symbol': api_symbol,
                'marginCoin': 'USDT',
                'leverage': str(leverage)
            }

            body = json.dumps(leverage_data)
            headers = self._get_headers('POST', request_path, body)

            response = requests.post(url, data=body, headers=headers, timeout=10)
            data = response.json()

            if data.get('code') == '00000':
                logger.info(f"✅ Leverage set to {leverage}x successfully")
                return {'success': True, 'data': data.get('data')}
            
            logger.error(f"❌ Failed to set leverage: {data.get('msg')}")
            return {'success': False, 'error': data.get('msg', 'Leverage set failed')}

        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return {'success': False, 'error': str(e)}
