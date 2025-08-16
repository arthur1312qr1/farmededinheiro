from typing import Dict
import ccxt
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class BitgetAPI:
    def __init__(self, api_key: str, secret_key: str, passphrase: str, sandbox: bool = False):
        """Initialize Bitget API client"""
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.sandbox = sandbox
        
        try:
            self.exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': secret_key,  # CORREÇÃO: estava 'secret_key'
                'password': passphrase,
                'sandbox': sandbox,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',  # for futures trading
                }
            })
            
            # Test connection
            self.exchange.load_markets()
            logger.info("✅ Bitget API conectado com sucesso")
            
        except Exception as e:
            logger.error(f"❌ Erro ao conectar Bitget API: {e}")
            raise

    def validate_order_params(self, symbol: str, side: str, size: float, **kwargs) -> Dict:
        """Validate order parameters before placing"""
        errors = []
        
        # Check symbol - CORREÇÃO: Verificar se mercados foram carregados
        try:
            if not self.exchange.markets:
                self.exchange.load_markets()
            
            if symbol not in self.exchange.markets:
                # CORREÇÃO: Tentar símbolo alternativo
                alt_symbol = symbol.replace('_UMCBL', '/USDT:USDT')
                if alt_symbol in self.exchange.markets:
                    logger.info(f"🔄 Usando símbolo alternativo: {alt_symbol}")
                else:
                    errors.append(f"Símbolo inválido: {symbol}")
        except:
            logger.warning("⚠️ Não foi possível validar símbolo")
        
        # Check side
        if side not in ['buy', 'sell']:
            errors.append(f"Side inválido: {side}")
        
        # Check size
        if size <= 0:
            errors.append(f"Size deve ser positivo: {size}")
        
        # Check minimum order value (1 USDT)
        if 'price' in kwargs:
            order_value = size if side == 'buy' else size * kwargs['price']
            if order_value < 1.0:
                errors.append(f"Valor da ordem ${order_value:.2f} abaixo do mínimo 1 USDT")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def get_account_balance(self) -> float:
        """Get account balance in USDT"""
        try:
            balance = self.exchange.fetch_balance()
            
            # CORREÇÃO: Múltiplas tentativas de obter saldo USDT
            usdt_balance = 0.0
            
            # Tentar diferentes formas de obter o saldo
            if 'USDT' in balance and 'free' in balance['USDT']:
                usdt_balance = balance['USDT']['free']
            elif 'USDT' in balance:
                usdt_balance = balance['USDT'].get('available', 0.0)
            elif 'free' in balance and 'USDT' in balance['free']:
                usdt_balance = balance['free']['USDT']
            elif 'total' in balance and 'USDT' in balance['total']:
                usdt_balance = balance['total']['USDT']
            
            usdt_balance = float(usdt_balance) if usdt_balance else 0.0
            
            logger.info(f"💰 Saldo atual: ${usdt_balance:.2f} USDT")
            return usdt_balance
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter saldo: {e}")
            return 0.0

    def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for symbol"""
        try:
            # CORREÇÃO: Tentar diferentes formatos de símbolo
            symbols_to_try = [
                symbol,                           # ethusdt_UMCBL
                symbol.replace('_UMCBL', '/USDT:USDT'),  # ETH/USDT:USDT
                symbol.replace('_UMCBL', '/USDT'),       # ETH/USDT
                'ETH/USDT:USDT',                         # Direto
                'ETH/USDT'                               # Spot
            ]
            
            ticker = None
            used_symbol = None
            
            for test_symbol in symbols_to_try:
                try:
                    ticker = self.exchange.fetch_ticker(test_symbol)
                    used_symbol = test_symbol
                    logger.info(f"✅ Símbolo funcional: {used_symbol}")
                    break
                except Exception as e:
                    logger.debug(f"❌ Símbolo {test_symbol} falhou: {e}")
                    continue
            
            if not ticker:
                raise Exception(f"Nenhum símbolo funcional encontrado para {symbol}")
            
            return {
                'symbol': used_symbol,
                'price': float(ticker['last']),
                'bid': float(ticker['bid']) if ticker['bid'] else float(ticker['last']),
                'ask': float(ticker['ask']) if ticker['ask'] else float(ticker['last']),
                'volume': float(ticker['baseVolume']) if ticker['baseVolume'] else 0.0,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter dados de mercado para {symbol}: {e}")
            return None

    def place_order(self, symbol: str, side: str, size: float, price: float = None, leverage: int = 1) -> Dict:
        """Place trading order"""
        try:
            # CORREÇÃO: Usar símbolo correto primeiro
            corrected_symbol = symbol.replace('_UMCBL', '/USDT:USDT')
            
            # Validate parameters first
            validation = self.validate_order_params(corrected_symbol, side, size, price=price)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': '; '.join(validation['errors'])
                }
            
            # Set leverage if futures
            if ':USDT' in corrected_symbol:
                try:
                    self.exchange.set_leverage(leverage, corrected_symbol)
                    logger.info(f"🚨 Alavancagem definida: {leverage}x para {corrected_symbol}")
                except Exception as e:
                    logger.warning(f"⚠️ Não foi possível definir alavancagem: {e}")
            
            # Place order
            order_type = 'market' if price is None else 'limit'
            
            # For futures, size should be in USDT
            if ':USDT' in corrected_symbol:
                # Convert size to quote currency amount for futures
                if price is None:
                    ticker = self.exchange.fetch_ticker(corrected_symbol)
                    current_price = float(ticker['last'])
                else:
                    current_price = price
                
                # Size is already in USDT for futures
                quote_amount = size  # size já é em USDT
                base_amount = quote_amount / current_price
                
                logger.warning(f"🚨 ORDEM FUTURES:")
                logger.warning(f"💰 Valor USDT: ${quote_amount:.2f}")
                logger.warning(f"📊 Quantidade ETH: {base_amount:.6f}")
                logger.warning(f"💎 Preço: ${current_price:.2f}")
                logger.warning(f"🎯 Símbolo: {corrected_symbol}")
                
                order = self.exchange.create_order(
                    symbol=corrected_symbol,  # USAR SÍMBOLO CORRIGIDO
                    type=order_type,
                    side=side,
                    amount=base_amount,  # Use base amount for futures
                    price=current_price if order_type == 'limit' else None
                )
            else:
                # Spot trading
                order = self.exchange.create_order(
                    symbol=corrected_symbol,  # USAR SÍMBOLO CORRIGIDO
                    type=order_type,
                    side=side,
                    amount=size,
                    price=price if order_type == 'limit' else None
                )
            
            logger.info(f"✅ Ordem executada: {side} {size} {corrected_symbol}")
            
            return {
                'success': True,
                'order_id': order['id'],
                'order': order
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao executar ordem: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Get order status"""
        try:
            corrected_symbol = symbol.replace('_UMCBL', '/USDT:USDT')
            order = self.exchange.fetch_order(order_id, corrected_symbol)
            
            return {
                'id': order['id'],
                'status': order['status'],
                'filled': order['filled'],
                'remaining': order['remaining'],
                'price': order['price'],
                'average': order['average']
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter status da ordem {order_id}: {e}")
            return {}

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order"""
        try:
            corrected_symbol = symbol.replace('_UMCBL', '/USDT:USDT')
            self.exchange.cancel_order(order_id, corrected_symbol)
            logger.info(f"✅ Ordem cancelada: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao cancelar ordem {order_id}: {e}")
            return False

    def get_open_positions(self) -> list:
        """Get open positions"""
        try:
            positions = self.exchange.fetch_positions()
            open_positions = [pos for pos in positions if float(pos['contracts']) > 0]
            
            return open_positions
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter posições: {e}")
            return []
