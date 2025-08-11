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
import urllib.parse

logger = logging.getLogger(__name__)

class BitgetAPIError(Exception):
    """Custom exception for Bitget API errors"""
    def __init__(self, message: str, code: Optional[str] = None, response_data: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.response_data = response_data

class BitgetAPI:
    """Enhanced Bitget API client with improved error handling and retry logic"""
    
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
        self.base_url = "https://api.bitget.com"
        self.session = requests.Session()
        
        # Enhanced request timeout and retry settings
        self.timeout = 30
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Set user agent
        self.session.headers.update({
            'User-Agent': 'Bitget-Trading-Bot/2.0',
            'Accept': 'application/json'
        })
        
        logger.info(f"Initialized Bitget API client (testnet: {testnet})")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in milliseconds"""
        return str(int(time.time() * 1000))
    
    def _sign_message(self, message: str) -> str:
        """Create signature for API request"""
        try:
            mac = hmac.new(
                self.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            )
            return base64.b64encode(mac.digest()).decode()
        except Exception as e:
            logger.error(f"Error creating signature: {e}")
            raise BitgetAPIError(f"Signature creation failed: {str(e)}")
    
    def _build_headers(self, method: str, request_path: str, body: str = '') -> Dict[str, str]:
        """Build headers for authenticated requests"""
        try:
            timestamp = self._get_timestamp()
            message = timestamp + method.upper() + request_path + body
            signature = self._sign_message(message)
            
            return {
                'ACCESS-KEY': self.api_key,
                'ACCESS-SIGN': signature,
                'ACCESS-TIMESTAMP': timestamp,
                'ACCESS-PASSPHRASE': self.passphrase,
                'Content-Type': 'application/json'
            }
        except Exception as e:
            logger.error(f"Error building headers: {e}")
            raise BitgetAPIError(f"Header creation failed: {str(e)}")
    
    def _make_request_with_retry(self, method: str, endpoint: str, params: Optional[Dict] = None,
                                body: Optional[Dict] = None, signed: bool = True) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            body: Request body
            signed: Whether request needs authentication
            
        Returns:
            API response as dictionary
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return self._make_request(method, endpoint, params, body, signed)
            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
                    
            except BitgetAPIError as e:
                # Don't retry API errors (authentication, invalid params, etc.)
                logger.error(f"API error on attempt {attempt + 1}: {e}")
                raise
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        # If we get here, all retries failed
        raise BitgetAPIError(f"Request failed after {self.max_retries} attempts: {str(last_exception)}")
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None,
                     body: Optional[Dict] = None, signed: bool = True) -> Dict[str, Any]:
        """
        Make single HTTP request to Bitget API
        
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
        body_str = json.dumps(body, separators=(',', ':')) if body else ''
        
        # Build query string properly
        query_string = ''
        if params:
            query_string = '?' + urllib.parse.urlencode(params)
            endpoint_with_params = endpoint + query_string
        else:
            endpoint_with_params = endpoint
        
        headers = {}
        if signed:
            headers = self._build_headers(method, endpoint_with_params, body_str)
        else:
            headers = {'Content-Type': 'application/json'}
        
        try:
            # Make the request
            response = self.session.request(
                method=method,
                url=url + query_string,
                headers=headers,
                data=body_str if body else None,
                timeout=self.timeout
            )
            
            # Log request details for debugging
            logger.debug(f"{method} {endpoint} - Status: {response.status_code}")
            
            # Handle HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response: {response.text}")
                raise BitgetAPIError(f"Invalid JSON response: {str(e)}")
            
            # Check Bitget API response format
            if isinstance(result, dict):
                code = result.get('code')
                if code and code != '00000':
                    error_msg = result.get('msg', 'Unknown API error')
                    logger.error(f"Bitget API error: {error_msg} (code: {code})")
                    raise BitgetAPIError(f"Bitget API error: {error_msg}", code, result)
            
            return result
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout: {e}")
            raise BitgetAPIError(f"Request timeout after {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise BitgetAPIError(f"Connection error: {str(e)}")
        except requests.exceptions.HTTPError as e:
            response_text = getattr(e.response, 'text', 'No response text available') if hasattr(e, 'response') else 'No response available'
            status_code = getattr(e.response, 'status_code', 'unknown') if hasattr(e, 'response') else 'unknown'
            logger.error(f"HTTP error: {e} - Response: {response_text}")
            raise BitgetAPIError(f"HTTP error {status_code}: {response_text}")
        except BitgetAPIError:
            raise  # Re-raise BitgetAPIError as-is
        except Exception as e:
            logger.error(f"Unexpected error in request: {e}")
            raise BitgetAPIError(f"Unexpected error: {str(e)}")
    
    def get_futures_balance(self) -> Dict[str, Any]:
        """
        Get Futures USDT-M account balance with enhanced error handling
        
        Returns:
            Balance information for futures account
        """
        try:
            endpoint = "/api/mix/v1/account/accounts"
            params = {"productType": "USDT-FUTURES"}
            
            logger.info("Requesting futures balance from Bitget API...")
            response = self._make_request_with_retry("GET", endpoint, params=params)
            
            if not response.get('data'):
                logger.warning("No balance data received from API")
                return {
                    'total_equity': 0.0,
                    'available_balance': 0.0,
                    'used_margin': 0.0,
                    'unrealized_pnl': 0.0,
                    'margin_ratio': 0.0,
                    'currency': 'USDT',
                    'last_updated': datetime.now().isoformat(),
                    'error': 'No balance data received'
                }
            
            # Handle both list and single object responses
            data = response['data']
            if isinstance(data, list):
                if len(data) == 0:
                    logger.warning("Empty balance data list received")
                    balance_data = {}
                else:
                    balance_data = data[0]
            else:
                balance_data = data
            
            # Safely extract balance values with defaults
            def safe_float(value, default=0.0):
                try:
                    return float(value) if value is not None else default
                except (ValueError, TypeError):
                    logger.warning(f"Invalid float value: {value}, using default: {default}")
                    return default
            
            result = {
                'total_equity': safe_float(balance_data.get('equity', 0)),
                'available_balance': safe_float(balance_data.get('available', 0)),
                'used_margin': safe_float(balance_data.get('locked', 0)),
                'unrealized_pnl': safe_float(balance_data.get('upl', 0)),
                'margin_ratio': safe_float(balance_data.get('marginRatio', 0)),
                'currency': balance_data.get('marginCoin', 'USDT'),
                'last_updated': datetime.now().isoformat(),
                'raw_data': balance_data  # Include raw data for debugging
            }
            
            logger.info(f"✓ Balance retrieved: {result['available_balance']} {result['currency']} available")
            return result
            
        except BitgetAPIError:
            raise  # Re-raise API errors
        except Exception as e:
            logger.error(f"Unexpected error getting futures balance: {e}")
            raise BitgetAPIError(f"Balance fetch failed: {str(e)}")
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get current futures positions with enhanced error handling
        
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
            
            logger.debug(f"Requesting positions for symbol: {symbol or 'all'}")
            response = self._make_request_with_retry("GET", endpoint, params=params)
            
            positions = []
            if response.get('data'):
                for pos in response['data']:
                    try:
                        size = float(pos.get('size', 0))
                        if size > 0:  # Only include active positions
                            positions.append({
                                'symbol': pos.get('symbol'),
                                'side': pos.get('holdSide'),
                                'size': size,
                                'entry_price': float(pos.get('averageOpenPrice', 0)),
                                'mark_price': float(pos.get('markPrice', 0)),
                                'unrealized_pnl': float(pos.get('upl', 0)),
                                'leverage': float(pos.get('leverage', 1)),
                                'margin': float(pos.get('margin', 0)),
                                'liquidation_price': float(pos.get('liqPx', 0)),
                                'position_id': pos.get('posId'),
                                'last_updated': datetime.now().isoformat()
                            })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing position data: {e}, skipping position")
                        continue
            
            logger.debug(f"Found {len(positions)} active positions")
            return positions
            
        except BitgetAPIError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting positions: {e}")
            raise BitgetAPIError(f"Position fetch failed: {str(e)}")
    
    def place_market_order(self, symbol: str, side: str, size: float,
                          leverage: int = 10, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Place a market order on Bitget Futures USDT-M with enhanced validation
        
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
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                raise BitgetAPIError("Invalid symbol provided")
            
            if side.lower() not in ['buy', 'sell']:
                raise BitgetAPIError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
            
            if size <= 0:
                raise BitgetAPIError(f"Invalid size: {size}. Must be positive")
            
            if leverage < 1 or leverage > 125:
                raise BitgetAPIError(f"Invalid leverage: {leverage}. Must be between 1-125")
            
            endpoint = "/api/mix/v1/order/placeOrder"
            
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
            
            logger.info(f"Placing {side} order: {size} {symbol} (leverage: {leverage}x, reduce_only: {reduce_only})")
            response = self._make_request_with_retry("POST", endpoint, body=order_data)
            
            if response.get('data'):
                order_result = response['data']
                result = {
                    'order_id': order_result.get('orderId'),
                    'client_order_id': order_result.get('clientOid'),
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'status': 'submitted',
                    'timestamp': datetime.now().isoformat(),
                    'raw_response': order_result
                }
                logger.info(f"✓ Order placed successfully: {result['order_id']}")
                return result
            else:
                raise BitgetAPIError("No order data in response")
                
        except BitgetAPIError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error placing market order: {e}")
            raise BitgetAPIError(f"Order placement failed: {str(e)}")
    
    def close_position(self, symbol: str, side: str) -> Dict[str, Any]:
        """
        Close a futures position with enhanced error handling
        
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
                return {
                    'success': False, 
                    'message': f'No {side} position found for {symbol}',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Place opposite market order to close position
            close_side = "sell" if side.lower() == "long" else "buy"
            
            logger.info(f"Closing {side} position: {position['size']} {symbol}")
            result = self.place_market_order(
                symbol=symbol,
                side=close_side,
                size=position['size'],
                reduce_only=True
            )
            
            result['action'] = 'close_position'
            result['closed_side'] = side
            result['success'] = True
            
            return result
            
        except BitgetAPIError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error closing position: {e}")
            raise BitgetAPIError(f"Position close failed: {str(e)}")
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get current market data for symbol with enhanced error handling
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market data including price, volume, etc.
        """
        try:
            if not symbol:
                raise BitgetAPIError("Symbol is required")
            
            endpoint = "/api/mix/v1/market/ticker"
            params = {"symbol": symbol}
            
            logger.debug(f"Requesting market data for {symbol}")
            response = self._make_request_with_retry("GET", endpoint, params=params, signed=False)
            
            if response.get('data'):
                # Handle both list and single object responses
                data = response['data']
                if isinstance(data, list):
                    if len(data) == 0:
                        raise BitgetAPIError(f"No market data found for {symbol}")
                    data = data[0]
                
                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value is not None else default
                    except (ValueError, TypeError):
                        return default
                
                result = {
                    'symbol': data.get('symbol'),
                    'last_price': safe_float(data.get('lastPr', 0)),
                    'bid_price': safe_float(data.get('bidPr', 0)),
                    'ask_price': safe_float(data.get('askPr', 0)),
                    'high_24h': safe_float(data.get('high24h', 0)),
                    'low_24h': safe_float(data.get('low24h', 0)),
                    'volume_24h': safe_float(data.get('baseVolume', 0)),
                    'change_24h': safe_float(data.get('change', 0)),
                    'change_percent_24h': safe_float(data.get('changeUtc', 0)),
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.debug(f"Market data for {symbol}: ${result['last_price']}")
                return result
            else:
                raise BitgetAPIError(f"No market data received for {symbol}")
                
        except BitgetAPIError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting market data: {e}")
            raise BitgetAPIError(f"Market data fetch failed: {str(e)}")
    
    def set_leverage(self, symbol: str, leverage: int, margin_mode: str = "crossed") -> bool:
        """
        Set leverage for a trading pair with enhanced validation
        
        Args:
            symbol: Trading symbol
            leverage: Leverage multiplier
            margin_mode: 'crossed' or 'isolated'
            
        Returns:
            Success status
        """
        try:
            if leverage < 1 or leverage > 125:
                logger.error(f"Invalid leverage: {leverage}. Must be between 1-125")
                return False
            
            if margin_mode not in ['crossed', 'isolated']:
                logger.error(f"Invalid margin mode: {margin_mode}")
                return False
            
            endpoint = "/api/mix/v1/account/setLeverage"
            data = {
                "symbol": symbol,
                "productType": "USDT-FUTURES",
                "marginCoin": "USDT",
                "leverage": str(leverage),
                "marginMode": margin_mode
            }
            
            logger.info(f"Setting leverage to {leverage}x for {symbol}")
            response = self._make_request_with_retry("POST", endpoint, body=data)
            
            if response.get('code') == '00000':
                logger.info(f"✓ Leverage set to {leverage}x for {symbol}")
                return True
            else:
                logger.error(f"Failed to set leverage: {response.get('msg')}")
                return False
                
        except BitgetAPIError as e:
            logger.error(f"API error setting leverage: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting leverage: {e}")
            return False

