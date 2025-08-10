import os
import json
import time
import hmac
import hashlib
import base64
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class BitgetAPI:
    """Enhanced Bitget API client with proper Futures USDT-M support"""
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str, testnet: bool = False):
        """
        Initialize Bitget API client
        
        Args:
            api_key: Bitget API key
            api_secret: Bitget API secret  
            passphrase: Bitget API passphrase
            testnet: Use testnet environment
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = "https://api.bitget.com" if not testnet else "https://api.bitget.com"  # Same URL for both
        self.session = requests.Session()
        
        # Request timeout settings
        self.timeout = 30
        
        logger.info(f"Initialized Bitget API client (testnet: {testnet})")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in milliseconds"""
        return str(int(time.time() * 1000))
    
    def _sign_message(self, message: str) -> str:
        """Create signature for API request"""
        mac = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    def _build_headers(self, method: str, request_path: str, body: str = '') -> Dict[str, str]:
        """Build headers for authenticated requests"""
        timestamp = self._get_timestamp()
        message = timestamp + method.upper() + request_path + body
        signature = self._sign_message(message)
        
        return {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json',
            'User-Agent': 'Bitget-Trading-Bot/1.0'
        }
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                     body: Optional[Dict] = None, signed: bool = True) -> Dict[str, Any]:
        """
        Make HTTP request to Bitget API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            body: Request body
            signed: Whether request needs authentication
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        body_str = json.dumps(body) if body else ''
        
        headers = {}
        if signed:
            headers = self._build_headers(method, endpoint, body_str)
        else:
            headers = {'Content-Type': 'application/json'}
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=body_str if body else None,
                timeout=self.timeout
            )
            
            # Log request for debugging
            logger.debug(f"{method} {endpoint} - Status: {response.status_code}")
            
            response.raise_for_status()
            result = response.json()
            
            # Check Bitget API response format
            if isinstance(result, dict) and result.get('code') != '00000':
                error_msg = result.get('msg', 'Unknown API error')
                logger.error(f"Bitget API error: {error_msg} (code: {result.get('code')})")
                raise Exception(f"Bitget API error: {error_msg}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            raise Exception(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise Exception(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def get_futures_balance(self) -> Dict[str, Any]:
        """
        Get Futures USDT-M account balance
        
        Returns:
            Balance information for futures account
        """
        try:
            endpoint = "/api/mix/v1/account/accounts"
            params = {"productType": "USDT-FUTURES"}
            
            response = self._make_request("GET", endpoint, params=params)
            
            if response.get('data'):
                balance_data = response['data'][0] if isinstance(response['data'], list) else response['data']
                
                return {
                    'total_equity': float(balance_data.get('equity', 0)),
                    'available_balance': float(balance_data.get('available', 0)),
                    'used_margin': float(balance_data.get('locked', 0)),
                    'unrealized_pnl': float(balance_data.get('upl', 0)),
                    'margin_ratio': float(balance_data.get('marginRatio', 0)),
                    'currency': balance_data.get('marginCoin', 'USDT'),
                    'last_updated': datetime.now().isoformat()
                }
            else:
                logger.warning("No balance data received")
                return {
                    'total_equity': 0.0,
                    'available_balance': 0.0,
                    'used_margin': 0.0,
                    'unrealized_pnl': 0.0,
                    'margin_ratio': 0.0,
                    'currency': 'USDT',
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get futures balance: {e}")
            raise Exception(f"Balance fetch failed: {str(e)}")
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get current futures positions
        
        Args:
            symbol: Specific symbol to get positions for (optional)
            
        Returns:
            List of position information
        """
        try:
            endpoint = "/api/mix/v1/position/allPosition-v2"
            params = {"productType": "USDT-FUTURES"}
            
            if symbol:
                params["symbol"] = symbol
            
            response = self._make_request("GET", endpoint, params=params)
            
            positions = []
            if response.get('data'):
                for pos in response['data']:
                    if float(pos.get('size', 0)) > 0:  # Only include active positions
                        positions.append({
                            'symbol': pos.get('symbol'),
                            'side': pos.get('holdSide'),
                            'size': float(pos.get('size', 0)),
                            'entry_price': float(pos.get('averageOpenPrice', 0)),
                            'mark_price': float(pos.get('markPrice', 0)),
                            'unrealized_pnl': float(pos.get('upl', 0)),
                            'leverage': float(pos.get('leverage', 1)),
                            'margin': float(pos.get('margin', 0)),
                            'liquidation_price': float(pos.get('liqPx', 0)),
                            'position_id': pos.get('posId'),
                            'last_updated': datetime.now().isoformat()
                        })
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise Exception(f"Position fetch failed: {str(e)}")
    
    def place_market_order(self, symbol: str, side: str, size: float, 
                          leverage: int = 10, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Place a market order on Bitget Futures USDT-M
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            side: 'buy' or 'sell'  
            size: Order size in base currency
            leverage: Leverage multiplier
            reduce_only: Whether this is a reduce-only order
            
        Returns:
            Order placement result
        """
        try:
            endpoint = "/api/mix/v1/order/placeOrder"
            
            # Determine hold side based on order side
            hold_side = "long" if side.lower() == "buy" else "short"
            
            order_data = {
                "symbol": symbol,
                "productType": "USDT-FUTURES",
                "marginMode": "crossed",  # Use crossed margin
                "marginCoin": "USDT",
                "size": str(int(size)),  # Size as string, integer value
                "price": "",  # Empty for market orders
                "side": side.lower(),
                "orderType": "market",
                "force": "normal",  # Normal order execution
                "reduceOnly": reduce_only,
                "timeInForceValue": "normal",
                "clientOid": f"bot_{int(time.time() * 1000)}"  # Unique client order ID
            }
            
            # Set leverage if not reduce-only
            if not reduce_only:
                order_data["leverage"] = str(leverage)
            
            response = self._make_request("POST", endpoint, body=order_data)
            
            if response.get('data'):
                order_result = response['data']
                return {
                    'order_id': order_result.get('orderId'),
                    'client_order_id': order_result.get('clientOid'),
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'status': 'submitted',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                raise Exception("No order data in response")
                
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            raise Exception(f"Order placement failed: {str(e)}")
    
    def close_position(self, symbol: str, side: str) -> Dict[str, Any]:
        """
        Close a futures position
        
        Args:
            symbol: Trading symbol
            side: Position side to close ('long' or 'short')
            
        Returns:
            Close order result
        """
        try:
            # Get current position size
            positions = self.get_positions(symbol)
            position = None
            
            for pos in positions:
                if pos['side'].lower() == side.lower():
                    position = pos
                    break
            
            if not position:
                return {'success': False, 'message': f'No {side} position found for {symbol}'}
            
            # Place opposite market order to close position
            close_side = "sell" if side.lower() == "long" else "buy"
            
            result = self.place_market_order(
                symbol=symbol,
                side=close_side,
                size=position['size'],
                reduce_only=True
            )
            
            result['action'] = 'close_position'
            result['closed_side'] = side
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            raise Exception(f"Position close failed: {str(e)}")
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get current market data for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market data including price, volume, etc.
        """
        try:
            endpoint = "/api/mix/v1/market/ticker"
            params = {"symbol": symbol}
            
            response = self._make_request("GET", endpoint, params=params, signed=False)
            
            if response.get('data'):
                data = response['data'][0] if isinstance(response['data'], list) else response['data']
                
                return {
                    'symbol': data.get('symbol'),
                    'last_price': float(data.get('lastPr', 0)),
                    'bid_price': float(data.get('bidPr', 0)),
                    'ask_price': float(data.get('askPr', 0)),
                    'high_24h': float(data.get('high24h', 0)),
                    'low_24h': float(data.get('low24h', 0)),
                    'volume_24h': float(data.get('baseVolume', 0)),
                    'change_24h': float(data.get('change', 0)),
                    'change_percent_24h': float(data.get('changeUtc', 0)),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                raise Exception("No market data received")
                
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            raise Exception(f"Market data fetch failed: {str(e)}")
    
    def get_klines(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[List]:
        """
        Get candlestick/kline data
        
        Args:
            symbol: Trading symbol
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch
            
        Returns:
            List of OHLCV data
        """
        try:
            endpoint = "/api/mix/v1/market/candles"
            params = {
                "symbol": symbol,
                "granularity": interval,
                "limit": str(limit)
            }
            
            response = self._make_request("GET", endpoint, params=params, signed=False)
            
            if response.get('data'):
                return response['data']
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to get klines: {e}")
            return []
    
    def set_leverage(self, symbol: str, leverage: int, margin_mode: str = "crossed") -> bool:
        """
        Set leverage for a trading pair
        
        Args:
            symbol: Trading symbol
            leverage: Leverage multiplier
            margin_mode: 'crossed' or 'isolated'
            
        Returns:
            Success status
        """
        try:
            endpoint = "/api/mix/v1/account/setLeverage"
            
            data = {
                "symbol": symbol,
                "productType": "USDT-FUTURES",
                "marginCoin": "USDT",
                "leverage": str(leverage),
                "marginMode": margin_mode
            }
            
            response = self._make_request("POST", endpoint, body=data)
            
            if response.get('code') == '00000':
                logger.info(f"Leverage set to {leverage}x for {symbol}")
                return True
            else:
                logger.error(f"Failed to set leverage: {response.get('msg')}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False
