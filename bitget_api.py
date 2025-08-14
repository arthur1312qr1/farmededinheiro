import aiohttp
import asyncio
import hashlib
import hmac
import base64
import json
import time
from typing import Dict, List, Optional
import logging

from config import Config

logger = logging.getLogger(__name__)

class BitgetAPI:
    def __init__(self):
        self.config = Config()
        self.base_url = "https://api.bitget.com"
        self.api_key = self.config.BITGET_API_KEY
        self.secret_key = self.config.BITGET_API_SECRET
        self.passphrase = self.config.BITGET_PASSPHRASE
        
        if not all([self.api_key, self.secret_key, self.passphrase]):
            if not self.config.PAPER_TRADING:
                raise ValueError("❌ API keys do Bitget são obrigatórias para trading real!")
        
        self.session = None
        logger.info("🔗 BitgetAPI inicializada")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """Gera assinatura para autenticação"""
        message = timestamp + method.upper() + request_path + body
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')
    
    def _get_headers(self, method: str, request_path: str, body: str = '') -> Dict:
        """Gera headers para requisição autenticada"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, request_path, body)
        
        return {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json',
            'locale': 'en-US'
        }
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, body: Dict = None) -> Optional[Dict]:
        """Faz requisição para API do Bitget"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}{endpoint}"
        request_body = json.dumps(body) if body else ''
        
        headers = self._get_headers(method, endpoint, request_body)
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=request_body if body else None,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    if data.get('code') == '00000':
                        return data.get('data')
                    else:
                        logger.error(f"❌ Erro da API Bitget: {data.get('msg')}")
                        return None
                else:
                    logger.error(f"❌ HTTP Error {response.status}: {await response.text()}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("❌ Timeout na requisição para Bitget")
            return None
        except Exception as e:
            logger.error(f"❌ Erro na requisição: {e}")
            return None
    
    async def get_account_info(self) -> Optional[Dict]:
        """Obtém informações da conta"""
        try:
            result = await self._make_request('GET', '/api/v2/mix/account/account')
            if result:
                # Encontrar conta USDT
                for account in result:
                    if account.get('marginCoin') == 'USDT':
                        logger.info(f"💰 Saldo USDT: {account.get('available', 0)}")
                        return account
            return None
        except Exception as e:
            logger.error(f"❌ Erro ao obter info da conta: {e}")
            return None
    
    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Obtém dados de mercado em tempo real"""
        try:
            # Ticker atual
            ticker = await self._make_request('GET', '/api/v2/mix/market/ticker', {'symbol': symbol})
            
            # Klines para análise técnica
            klines = await self._make_request('GET', '/api/v2/mix/market/candles', {
                'symbol': symbol,
                'granularity': '1m',
                'limit': 100
            })
            
            if ticker and klines:
                return {
                    'symbol': symbol,
                    'price': float(ticker[0]['last']),
                    'volume': float(ticker[0]['baseVolume']),
                    'change_24h': float(ticker[0]['chgUtc']),
                    'high_24h': float(ticker[0]['high24h']),
                    'low_24h': float(ticker[0]['low24h']),
                    'klines': klines,
                    'timestamp': int(time.time() * 1000)
                }
            return None
        except Exception as e:
            logger.error(f"❌ Erro ao obter dados de mercado: {e}")
            return None
    
    async def place_order(self, symbol: str, side: str, size: float, 
                         stop_loss: float = None, take_profit: float = None) -> Optional[Dict]:
        """Coloca ordem de trading"""
        try:
            # Dados da ordem
            order_data = {
                'symbol': symbol,
                'productType': 'UMCBL',
                'marginMode': 'cross',
                'marginCoin': 'USDT',
                'size': str(size),
                'side': side,  # 'buy' ou 'sell'
                'orderType': 'market',
                'timeInForceValue': 'IOC'
            }
            
            # Configurar leverage
            await self.set_leverage(symbol, self.config.LEVERAGE)
            
            logger.info(f"📤 Enviando ordem: {side.upper()} {size} {symbol}")
            
            # Colocar ordem principal
            result = await self._make_request('POST', '/api/v2/mix/order/place-order', body=order_data)
            
            if result:
                order_id = result.get('orderId')
                logger.info(f"✅ Ordem executada: ID {order_id}")
                
                # Configurar stop loss e take profit se fornecidos
                if stop_loss:
                    await self.set_stop_loss(symbol, order_id, stop_loss)
                
                if take_profit:
                    await self.set_take_profit(symbol, order_id, take_profit)
                
                return result
            else:
                logger.error("❌ Falha ao executar ordem")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erro ao colocar ordem: {e}")
            return None
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Define alavancagem"""
        try:
            result = await self._make_request('POST', '/api/v2/mix/account/set-leverage', body={
                'symbol': symbol,
                'productType': 'UMCBL',
                'marginCoin': 'USDT',
                'leverage': str(leverage)
            })
            
            if result:
                logger.info(f"📊 Alavancagem definida: {leverage}x")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro ao definir alavancagem: {e}")
            return False
    
    async def set_stop_loss(self, symbol: str, order_id: str, stop_price: float) -> bool:
        """Define stop loss"""
        try:
            result = await self._make_request('POST', '/api/v2/mix/order/place-plan-order', body={
                'symbol': symbol,
                'productType': 'UMCBL',
                'marginMode': 'cross',
                'marginCoin': 'USDT',
                'triggerPrice': str(stop_price),
                'orderType': 'market',
                'planType': 'loss_plan',
                'size': 'auto'
            })
            
            if result:
                logger.info(f"🛑 Stop Loss definido: {stop_price}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro ao definir stop loss: {e}")
            return False
    
    async def set_take_profit(self, symbol: str, order_id: str, take_price: float) -> bool:
        """Define take profit"""
        try:
            result = await self._make_request('POST', '/api/v2/mix/order/place-plan-order', body={
                'symbol': symbol,
                'productType': 'UMCBL',
                'marginMode': 'cross',
                'marginCoin': 'USDT',
                'triggerPrice': str(take_price),
                'orderType': 'market',
                'planType': 'profit_plan',
                'size': 'auto'
            })
            
            if result:
                logger.info(f"🎯 Take Profit definido: {take_price}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro ao definir take profit: {e}")
            return False
    
    async def get_open_positions(self) -> List[Dict]:
        """Obtém posições abertas"""
        try:
            result = await self._make_request('GET', '/api/v2/mix/position/all-position')
            return result or []
        except Exception as e:
            logger.error(f"❌ Erro ao obter posições: {e}")
            return []
    
    async def close_position(self, symbol: str) -> bool:
        """Fecha posição"""
        try:
            positions = await self.get_open_positions()
            for position in positions:
                if position.get('symbol') == symbol and float(position.get('size', 0)) > 0:
                    # Fechar posição
                    result = await self._make_request('POST', '/api/v2/mix/order/place-order', body={
                        'symbol': symbol,
                        'productType': 'UMCBL',
                        'marginMode': 'cross',
                        'marginCoin': 'USDT',
                        'size': position['size'],
                        'side': 'sell' if position['side'] == 'long' else 'buy',
                        'orderType': 'market',
                        'reduceOnly': 'true'
                    })
                    
                    if result:
                        logger.info(f"✅ Posição fechada: {symbol}")
                        return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro ao fechar posição: {e}")
            return False
