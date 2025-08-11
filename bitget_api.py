import hashlib
import hmac
import base64
import json
import time
import logging
from datetime import datetime
import requests
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class BitgetAPI:
    """Simplified Bitget API client for Railway deployment"""
    
    def __init__(self, api_key, api_secret, passphrase):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = "https://api.bitget.com"
        
        # Create session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'TradingBot/1.0'
        })
        
        logger.info("Bitget API client initialized")
    
    def _generate_signature(self, timestamp, method, path, body=''):
        """Generate HMAC signature for API requests"""
        message = timestamp + method + path + body
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()
    
    def _get_headers(self, timestamp, method, path, body=''):
        """Generate request headers with authentication"""
        signature = self._generate_signature(timestamp, method, path, body)
        
        return {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
    
    def _make_request(self, method, path, params=None, data=None):
        """Make authenticated request to Bitget API"""
        try:
            timestamp = str(int(time.time() * 1000))
            
            # Prepare URL and body
            if params:
                query_string = urlencode(params)
                full_path = f"{path}?{query_string}"
            else:
                full_path = path
                query_string = ""
            
            body = json.dumps(data) if data else ''
            
            # Generate headers
            headers = self._get_headers(timestamp, method.upper(), full_path, body)
            
            # Make request
            url = f"{self.base_url}{full_path}"
            
            if method.upper() == 'GET':
                response = self.session.get(url, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                response = self.session.post(url, headers=headers, data=body, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Handle response
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text[:200]}"
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error in API request: {e}")
            return {
                'success': False,
                'error': f"Network error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            return {
                'success': False,
                'error': f"Request failed: {str(e)}"
            }
    
    def get_balance(self):
        """Get USDT futures account balance"""
        try:
            # Use the correct endpoint for futures balance
            path = '/api/v2/mix/account/accounts'
            params = {'productType': 'USDT-FUTURES'}
            
            response = self._make_request('GET', path, params=params)
            
            if response and not response.get('success') == False:
                # Check if response has the expected structure
                if response.get('code') == '00000' and response.get('data'):
                    balance_data = response['data'][0]  # First account
                    
                    return {
                        'available_balance': float(balance_data.get('available', 0)),
                        'total_equity': float(balance_data.get('equity', 0)),
                        'unrealized_pnl': float(balance_data.get('unrealizedPL', 0)),
                        'currency': 'USDT',
                        'source': 'bitget_api',
                        'last_updated': datetime.now().isoformat(),
                        'success': True
                    }
                else:
                    error_msg = response.get('msg', 'Unknown API error')
                    logger.error(f"Bitget API error: {error_msg}")
                    return {
                        'success': False,
                        'error': f"API Error: {error_msg}"
                    }
            else:
                return response  # Return the error response
                
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_ticker(self, symbol):
        """Get ticker information for a symbol"""
        try:
            path = '/api/v2/mix/market/ticker'
            params = {'symbol': symbol}
            
            response = self._make_request('GET', path, params=params)
            
            if response and response.get('code') == '00000':
                ticker_data = response['data'][0]
                return {
                    'symbol': symbol,
                    'price': float(ticker_data.get('lastPr', 0)),
                    'volume': float(ticker_data.get('baseVolume', 0)),
                    'change_24h': float(ticker_data.get('chgUtc', 0)),
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
            else:
                return {
                    'success': False,
                    'error': response.get('msg', 'Failed to get ticker')
                }
                
        except Exception as e:
            logger.error(f"Error getting ticker: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_connection(self):
        """Test API connection and authentication"""
        try:
            # Try to get balance as a connection test
            result = self.get_balance()
            return result.get('success', False)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
