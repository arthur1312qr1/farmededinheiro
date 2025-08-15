"""
Bitget API Integration for Trading Operations
Handles all cryptocurrency exchange interactions
"""

import logging
import requests
import hmac
import hashlib
import base64
import time
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__name__)

class BitgetAPI:
    def __init__(self, api_key: str, secret_key: str, passphrase: str, sandbox: bool = False):
        """Initialize Bitget API client"""
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.sandbox = sandbox
        
        # API endpoints
        self.base_url = "https://api.bitget.com" if not sandbox else "https://api.sandbox.bitget.com"
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'ACCESS-KEY': self.api_key,
            'ACCESS-PASSPHRASE': self.passphrase
        })
        
        logger.info(f"🔗 Bitget API inicializada - Sandbox: {sandbox}")
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Generate signature for API requests"""
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
            logger.error(f"❌ Erro ao gerar assinatura: {e}")
            raise
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Optional[Dict]:
        """Make authenticated request to Bitget API"""
        try:
            url = f"{self.base_url}{endpoint}"
            timestamp = str(int(time.time() * 1000))
            
            # Prepare request body
            body = json.dumps(data) if data else ""
            
            # Generate signature
            signature = self._generate_signature(timestamp, method, endpoint, body)
            
            # Set headers
            headers = {
                'ACCESS-TIMESTAMP': timestamp,
                'ACCESS-SIGN': signature
            }
            
            # Make request
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, headers=headers, timeout=10)
            else:
                raise ValueError(f"Método HTTP não suportado: {method}")
            
            # Check response
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Erro na API Bitget: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro de rede na API Bitget: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erro inesperado na API Bitget: {e}")
            return None
    
    def get_account_balance(self) -> float:
        """Get account balance in USDT"""
        try:
            if not self.api_key:
                # Return mock balance for demo
                return 10000.0
            
            endpoint = "/api/v2/mix/account/accounts"
            response = self._make_request("GET", endpoint)
            
            if response and response.get('code') == '00000':
                data = response.get('data', [])
                for account in data:
                    if account.get('marginCoin') == 'USDT':
                        return float(account.get('available', 0))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter saldo: {e}")
            return 0.0
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get current market data for symbol"""
        try:
            if not self.api_key:
                # Return mock data for demo
                import random
                base_price = 3847.32
                return {
                    'symbol': symbol,
                    'price': base_price + random.uniform(-50, 50),
                    'volume': random.uniform(1000000, 5000000),
                    'change_24h': random.uniform(-5, 5),
                    'high_24h': base_price + random.uniform(20, 80),
                    'low_24h': base_price - random.uniform(20, 80),
                    'timestamp': int(time.time() * 1000)
                }
            
            endpoint = f"/api/v2/mix/market/ticker"
            params = {'symbol': symbol}
            
            response = self._make_request("GET", endpoint, params=params)
            
            if response and response.get('code') == '00000':
                data = response.get('data', [])
                if data:
                    ticker = data[0]
                    return {
                        'symbol': symbol,
                        'price': float(ticker.get('lastPr', 0)),
                        'volume': float(ticker.get('baseVolume', 0)),
                        'change_24h': float(ticker.get('chgUtc', 0)),
                        'high_24h': float(ticker.get('high24h', 0)),
                        'low_24h': float(ticker.get('low24h', 0)),
                        'timestamp': int(ticker.get('ts', 0))
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter dados de mercado: {e}")
            return None
    
    def place_order(self, symbol: str, side: str, size: float, price: float, leverage: int = 10) -> Optional[Dict]:
        """Place a trading order"""
        try:
            if not self.api_key:
                # Return mock order for demo
                import random
                return {
                    'success': True,
                    'order_id': f"mock_order_{int(time.time())}_{random.randint(1000, 9999)}",
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'price': price,
                    'timestamp': int(time.time() * 1000)
                }
            
            endpoint = "/api/v2/mix/order/place-order"
            
            data = {
                'symbol': symbol,
                'productType': 'USDT-FUTURES',
                'marginMode': 'isolated',
                'marginCoin': 'USDT',
                'side': side,
                'orderType': 'market',  # Market order for scalping
                'size': str(size),
                'leverage': str(leverage)
            }
            
            response = self._make_request("POST", endpoint, data=data)
            
            if response and response.get('code') == '00000':
                return {
                    'success': True,
                    'order_id': response.get('data', {}).get('orderId'),
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'price': price,
                    'timestamp': int(time.time() * 1000)
                }
            else:
                logger.error(f"❌ Falha ao executar ordem: {response}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erro ao executar ordem: {e}")
            return None
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for symbol"""
        try:
            if not self.api_key:
                return None
            
            endpoint = "/api/v2/mix/position/all-position"
            params = {
                'symbol': symbol,
                'productType': 'USDT-FUTURES'
            }
            
            response = self._make_request("GET", endpoint, params=params)
            
            if response and response.get('code') == '00000':
                data = response.get('data', [])
                for position in data:
                    if float(position.get('total', 0)) > 0:
                        return {
                            'symbol': symbol,
                            'side': position.get('holdSide'),
                            'size': float(position.get('total', 0)),
                            'entry_price': float(position.get('averageOpenPrice', 0)),
                            'unrealized_pnl': float(position.get('unrealizedPL', 0)),
                            'leverage': position.get('leverage')
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter posição: {e}")
            return None
    
    def close_position(self, symbol: str) -> bool:
        """Close current position"""
        try:
            position = self.get_position(symbol)
            if not position:
                return True  # No position to close
            
            # Place opposite order to close position
            opposite_side = 'sell' if position['side'] == 'long' else 'buy'
            
            result = self.place_order(
                symbol=symbol,
                side=opposite_side,
                size=position['size'],
                price=0,  # Market order
                leverage=int(position['leverage'])
            )
            
            return result is not None and result.get('success', False)
            
        except Exception as e:
            logger.error(f"❌ Erro ao fechar posição: {e}")
            return False
    
    def get_order_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get recent order history"""
        try:
            if not self.api_key:
                # Return mock history for demo
                return []
            
            endpoint = "/api/v2/mix/order/history"
            params = {
                'symbol': symbol,
                'productType': 'USDT-FUTURES',
                'pageSize': str(limit)
            }
            
            response = self._make_request("GET", endpoint, params=params)
            
            if response and response.get('code') == '00000':
                orders = response.get('data', {}).get('orderList', [])
                return [
                    {
                        'order_id': order.get('orderId'),
                        'symbol': order.get('symbol'),
                        'side': order.get('side'),
                        'size': float(order.get('size', 0)),
                        'price': float(order.get('price', 0)),
                        'status': order.get('status'),
                        'timestamp': int(order.get('cTime', 0))
                    }
                    for order in orders
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter histórico: {e}")
            return []
