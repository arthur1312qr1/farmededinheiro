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
        
        logger.info(f"🔥 BitgetAPI initialized - PAPER_TRADING={self.paper_trading}")
        
        # Paper trading state (only used if PAPER_TRADING=true)
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
        """Get account balance - CONNECTS TO REAL BITGET ACCOUNT"""
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
            logger.info("🔗 CONNECTING TO REAL BITGET ACCOUNT...")
            
            # Futures account endpoint
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
                                'mode': 'REAL'
                            }
                        }

            logger.error(f"❌ Failed to get balance - Response: {data}")
            return {'success': False, 'error': 'Failed to connect to Bitget API'}

        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {'success': False, 'error': str(e)}

    def get_current_price(self, symbol):
        """Get current price for symbol"""
        try:
            request_path = "/api/mix/v1/market/ticker"
            url = f"{self.base_url}{request_path}"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('code') == '00000' and data.get('data'):
                price = float(data['data']['last'])
                return {'success': True, 'price': price}
            
            return {'success': False, 'error': 'Failed to get price'}
            
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return {'success': False, 'error': str(e)}

    def get_klines(self, symbol, granularity, limit):
        """Get kline data"""
        try:
            request_path = "/api/mix/v1/market/candles"
            url = f"{self.base_url}{request_path}"
            params = {
                'symbol': symbol,
                'granularity': granularity,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            if data.get('code') == '00000' and data.get('data'):
                return {'success': True, 'data': data['data']}
            
            return {'success': False, 'error': 'Failed to get klines'}
            
        except Exception as e:
            logger.error(f"Error getting klines: {e}")
            return {'success': False, 'error': str(e)}

    def set_leverage(self, leverage):
        """Set leverage for the symbol"""
        if self.paper_trading:
            logger.info(f"Paper Trading - Leverage set to {leverage}x")
            return {'success': True}
            
        try:
            request_path = "/api/mix/v1/account/setLeverage"
            url = f"{self.base_url}{request_path}"
            
            body_data = {
                'symbol': Config.SYMBOL,
                'marginCoin': 'USDT',
                'leverage': str(leverage)
            }
            body = json.dumps(body_data)
            headers = self._get_headers('POST', request_path, body)
            
            response = requests.post(url, headers=headers, data=body, timeout=15)
            data = response.json()
            
            if data.get('code') == '00000':
                logger.info(f"✅ REAL LEVERAGE SET: {leverage}x")
                return {'success': True}
            
            logger.error(f"Failed to set leverage: {data}")
            return {'success': False, 'error': data.get('msg', 'Unknown error')}
            
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return {'success': False, 'error': str(e)}

    def place_real_order(self, side, size):
        """Place a REAL order on Bitget - NOT PAPER TRADING"""
        if self.paper_trading:
            logger.info(f"Paper Trading - Order placed: {side} {size}")
            return {
                'success': True,
                'data': {
                    'orderId': f'paper_{int(time.time())}',
                    'price': 2000.0,
                    'mode': 'paper'
                }
            }
        
        try:
            logger.info(f"🔥 PLACING REAL ORDER: {side} {size} on {Config.SYMBOL}")
            
            request_path = "/api/mix/v1/order/placeOrder"
            url = f"{self.base_url}{request_path}"
            
            # Convert side to Bitget format
            bitget_side = 'buy_long' if side == 'buy' else 'sell_short'
            
            body_data = {
                'symbol': Config.SYMBOL,
                'marginCoin': 'USDT',
                'side': bitget_side,
                'orderType': 'market',
                'size': str(size)
            }
            body = json.dumps(body_data)
            headers = self._get_headers('POST', request_path, body)
            
            response = requests.post(url, headers=headers, data=body, timeout=15)
            data = response.json()
            
            logger.info(f"REAL ORDER Response: {data}")
            
            if data.get('code') == '00000':
                logger.info(f"✅ REAL ORDER EXECUTED: {side} {size}")
                return {
                    'success': True,
                    'data': {
                        'orderId': data['data']['orderId'],
                        'price': data['data'].get('price', 0),
                        'mode': 'REAL'
                    }
                }
            
            logger.error(f"❌ REAL ORDER FAILED: {data}")
            return {'success': False, 'error': data.get('msg', 'Order failed')}
            
        except Exception as e:
            logger.error(f"Error placing real order: {e}")
            return {'success': False, 'error': str(e)}

    def place_stop_loss_order(self, position_id, stop_price):
        """Place stop loss order (2%)"""
        if self.paper_trading:
            return {'success': True}
            
        try:
            request_path = "/api/mix/v1/plan/placePlan"
            url = f"{self.base_url}{request_path}"
            
            body_data = {
                'symbol': Config.SYMBOL,
                'marginCoin': 'USDT',
                'planType': 'loss_plan',
                'triggerPrice': str(stop_price),
                'orderType': 'market'
            }
            body = json.dumps(body_data)
            headers = self._get_headers('POST', request_path, body)
            
            response = requests.post(url, headers=headers, data=body, timeout=15)
            data = response.json()
            
            if data.get('code') == '00000':
                logger.info(f"✅ STOP LOSS SET: ${stop_price:.4f}")
                return {'success': True}
            
            return {'success': False, 'error': data.get('msg', 'Failed to set stop loss')}
            
        except Exception as e:
            logger.error(f"Error setting stop loss: {e}")
            return {'success': False, 'error': str(e)}

    def place_take_profit_order(self, position_id, take_profit_price):
        """Place take profit order (5%)"""
        if self.paper_trading:
            return {'success': True}
            
        try:
            request_path = "/api/mix/v1/plan/placePlan"
            url = f"{self.base_url}{request_path}"
            
            body_data = {
                'symbol': Config.SYMBOL,
                'marginCoin': 'USDT',
                'planType': 'profit_plan',
                'triggerPrice': str(take_profit_price),
                'orderType': 'market'
            }
            body = json.dumps(body_data)
            headers = self._get_headers('POST', request_path, body)
            
            response = requests.post(url, headers=headers, data=body, timeout=15)
            data = response.json()
            
            if data.get('code') == '00000':
                logger.info(f"✅ TAKE PROFIT SET: ${take_profit_price:.4f}")
                return {'success': True}
            
            return {'success': False, 'error': data.get('msg', 'Failed to set take profit')}
            
        except Exception as e:
            logger.error(f"Error setting take profit: {e}")
            return {'success': False, 'error': str(e)}

    # Legacy method for compatibility
    def place_order(self, side, size):
        """Legacy method - redirects to place_real_order"""
        return self.place_real_order(side, size)
