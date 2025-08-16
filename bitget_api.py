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
                'secret': secret_key,
                'password': passphrase,
                'sandbox': sandbox,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',  # FUTURES com alavancagem
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
        
        # Check symbol - Para futures
        try:
            if not self.exchange.markets:
                self.exchange.load_markets()
            
            # Usar símbolo de futures
            futures_symbol = symbol.replace('_UMCBL', '/USDT:USDT')
            if futures_symbol not in self.exchange.markets:
                errors.append(f"Símbolo inválido: {futures_symbol}")
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
        """Get FUTURES account balance in USDT with 10x leverage"""
        try:
            # CORREÇÃO: Forçar tipo de conta para FUTURES
            balance = self.exchange.fetch_balance({'type': 'swap'})  # FUTURES/SWAP
            
            logger.info(f"🔍 Estrutura do saldo: {balance}")
            
            usdt_balance = 0.0
            
            # Tentar diferentes formas de obter o saldo de FUTURES
            if 'USDT' in balance and isinstance(balance['USDT'], dict):
                if 'free' in balance['USDT']:
                    usdt_balance = balance['USDT']['free']
                elif 'available' in balance['USDT']:
                    usdt_balance = balance['USDT']['available']
                elif 'total' in balance['USDT']:
                    usdt_balance = balance['USDT']['total']
            elif 'free' in balance and 'USDT' in balance['free']:
                usdt_balance = balance['free']['USDT']
            elif 'total' in balance and 'USDT' in balance['total']:
                usdt_balance = balance['total']['USDT']
            elif 'USDT' in balance:
                usdt_balance = balance['USDT']
            
            # Se for dict, pegar valor numérico
            if isinstance(usdt_balance, dict):
                usdt_balance = usdt_balance.get('available', usdt_balance.get('free', usdt_balance.get('total', 0)))
            
            usdt_balance = float(usdt_balance) if usdt_balance else 0.0
            
            logger.warning(f"💰 Saldo FUTURES: ${usdt_balance:.2f} USDT")
            logger.warning(f"🚨 Alavancagem 10x: Poder de compra ${usdt_balance * 10:.2f} USDT")
            
            return usdt_balance
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter saldo de FUTURES: {e}")
            # FALLBACK: Tentar sem especificar tipo
            try:
                balance = self.exchange.fetch_balance()
                usdt_balance = 0.0
                
                if 'USDT' in balance:
                    if isinstance(balance['USDT'], dict):
                        usdt_balance = balance['USDT'].get('free', balance['USDT'].get('total', 0))
                    else:
                        usdt_balance = balance['USDT']
                
                usdt_balance = float(usdt_balance) if usdt_balance else 0.0
                logger.warning(f"💰 Saldo GERAL: ${usdt_balance:.2f} USDT")
                return usdt_balance
                
            except Exception as e2:
                logger.error(f"❌ Erro total ao obter saldo: {e2}")
                return 0.0

    def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for FUTURES symbol"""
        try:
            # Usar símbolo de FUTURES
            futures_symbol = 'ETH/USDT:USDT'
            
            ticker = self.exchange.fetch_ticker(futures_symbol)
            logger.info(f"✅ Dados de mercado FUTURES obtidos: ETH @ ${ticker['last']:.2f}")
            
            return {
                'symbol': futures_symbol,
                'price': float(ticker['last']),
                'bid': float(ticker['bid']) if ticker['bid'] else float(ticker['last']),
                'ask': float(ticker['ask']) if ticker['ask'] else float(ticker['last']),
                'volume': float(ticker['baseVolume']) if ticker['baseVolume'] else 0.0,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter dados de mercado FUTURES: {e}")
            return None

    def place_order(self, symbol: str, side: str, size: float, price: float = None, leverage: int = 10) -> Dict:
        """Place FUTURES order with 10x leverage"""
        try:
            futures_symbol = 'ETH/USDT:USDT'  # Símbolo fixo para FUTURES
            
            # Definir alavancagem 10x
            try:
                self.exchange.set_leverage(10, futures_symbol)
                logger.warning(f"🚨 ALAVANCAGEM 10x DEFINIDA para {futures_symbol}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao definir alavancagem: {e}")
            
            # Obter preço atual se não especificado
            if price is None:
                ticker = self.exchange.fetch_ticker(futures_symbol)
                current_price = float(ticker['last'])
            else:
                current_price = price
            
            # Size já vem em USDT (80% do saldo)
            quote_amount = size  # Valor em USDT
            base_amount = quote_amount / current_price  # Quantidade ETH
            
            logger.warning(f"🚨 ORDEM FUTURES 10x:")
            logger.warning(f"💰 Valor USDT: ${quote_amount:.2f}")
            logger.warning(f"📊 Quantidade ETH: {base_amount:.6f}")
            logger.warning(f"💎 Preço ETH: ${current_price:.2f}")
            logger.warning(f"🎯 Símbolo: {futures_symbol}")
            logger.warning(f"⚡ Alavancagem: 10x")
            logger.warning(f"💥 Exposição Total: ${quote_amount * 10:.2f} USDT")
            
            # Executar ordem FUTURES
            order = self.exchange.create_order(
                symbol=futures_symbol,
                type='market',  # Ordem de mercado
                side=side,
                amount=base_amount,  # Quantidade em ETH
                price=None  # Market order não precisa de preço
            )
            
            logger.warning(f"✅ ORDEM FUTURES EXECUTADA: {side} {base_amount:.6f} ETH")
            
            return {
                'success': True,
                'order_id': order['id'],
                'order': order
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao executar ordem FUTURES: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Get order status"""
        try:
            futures_symbol = 'ETH/USDT:USDT'
            order = self.exchange.fetch_order(order_id, futures_symbol)
            
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
            futures_symbol = 'ETH/USDT:USDT'
            self.exchange.cancel_order(order_id, futures_symbol)
            logger.info(f"✅ Ordem cancelada: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao cancelar ordem {order_id}: {e}")
            return False

    def get_open_positions(self) -> list:
        """Get open FUTURES positions"""
        try:
            positions = self.exchange.fetch_positions(['ETH/USDT:USDT'])
            open_positions = [pos for pos in positions if float(pos['contracts']) > 0]
            
            logger.info(f"📊 Posições abertas: {len(open_positions)}")
            return open_positions
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter posições FUTURES: {e}")
            return []
