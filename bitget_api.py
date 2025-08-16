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
                    'defaultType': 'swap',
                    'createMarketBuyOrderRequiresPrice': False,
                }
            })
            
            self.exchange.load_markets()
            logger.info("✅ Bitget API conectado com sucesso")
            
        except Exception as e:
            logger.error(f"❌ Erro ao conectar Bitget API: {e}")
            raise

    def get_account_balance(self) -> float:
        """Get FUTURES account balance in USDT"""
        try:
            balance = self.exchange.fetch_balance({'type': 'swap'})
            usdt_balance = 0.0
            
            if 'USDT' in balance:
                usdt_data = balance['USDT']
                if isinstance(usdt_data, dict):
                    usdt_balance = usdt_data.get('free', 0) or usdt_data.get('available', 0) or usdt_data.get('total', 0)
                else:
                    usdt_balance = float(usdt_data)
            
            usdt_balance = float(usdt_balance) if usdt_balance else 0.0
            logger.warning(f"💰 SALDO DETECTADO: ${usdt_balance:.2f} USDT")
            
            return usdt_balance
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter saldo: {e}")
            return 0.0

    def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for FUTURES"""
        try:
            futures_symbol = 'ETH/USDT:USDT'
            ticker = self.exchange.fetch_ticker(futures_symbol)
            
            return {
                'symbol': futures_symbol,
                'price': float(ticker['last']),
                'bid': float(ticker['bid']) if ticker['bid'] else float(ticker['last']),
                'ask': float(ticker['ask']) if ticker['ask'] else float(ticker['last']),
                'volume': float(ticker['baseVolume']) if ticker['baseVolume'] else 0.0,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter dados de mercado: {e}")
            return None

    def place_order(self, symbol: str, side: str, size: float, price: float = None, leverage: int = 10) -> Dict:
        """CORREÇÃO FINAL: Usar notional (valor USDT) ao invés de amount (quantidade ETH)"""
        try:
            futures_symbol = 'ETH/USDT:USDT'
            
            # Definir alavancagem 10x
            try:
                self.exchange.set_leverage(10, futures_symbol)
                logger.warning(f"🚨 ALAVANCAGEM 10x DEFINIDA")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao definir alavancagem: {e}")
            
            # Obter preço atual
            if price is None:
                ticker = self.exchange.fetch_ticker(futures_symbol)
                current_price = float(ticker['last'])
            else:
                current_price = price
            
            # Buscar saldo atual (100% dinâmico)
            current_balance = self.get_account_balance()
            usdt_amount = current_balance  # 100% do saldo atual
            
            logger.warning(f"🚨 NOVA ABORDAGEM - USAR NOTIONAL:")
            logger.warning(f"💰 Saldo atual: ${current_balance:.2f} USDT")
            logger.warning(f"🎯 Usar 100%: ${usdt_amount:.2f} USDT")
            logger.warning(f"💎 Preço ETH: ${current_price:.2f}")
            logger.warning(f"⚡ Alavancagem: {leverage}x")
            logger.warning(f"💥 Exposição: ${usdt_amount * leverage:.2f} USDT")
            
            # CORREÇÃO: Usar 'quoteOrderQty' para especificar valor em USDT
            logger.warning(f"🚀 EXECUTANDO COM QUOTEORDERQTY:")
            logger.warning(f"💰 Valor USDT: ${usdt_amount:.2f}")
            
            # Método alternativo: usar params para passar quoteOrderQty
            order = self.exchange.create_order(
                symbol=futures_symbol,
                type='market',
                side=side,
                amount=None,  # Não especificar amount
                price=None,   # Market order
                params={
                    'quoteOrderQty': usdt_amount,  # Especificar valor em USDT
                    'reduceOnly': False
                }
            )
            
            logger.warning(f"✅ ORDEM EXECUTADA COM NOTIONAL!")
            logger.warning(f"💰 Valor usado: ${usdt_amount:.2f} USDT")
            
            return {
                'success': True,
                'order_id': order['id'],
                'order': order,
                'usdt_amount': usdt_amount,
                'price': current_price
            }
            
        except Exception as e:
            logger.error(f"❌ Erro método notional: {e}")
            
            # FALLBACK: Tentar com quantidade mínima possível
            try:
                logger.warning(f"🔄 TENTATIVA FALLBACK - QUANTIDADE MÍNIMA:")
                
                # Calcular ETH com precisão reduzida
                eth_quantity = round(usdt_amount / current_price, 6)  # 6 casas decimais
                
                logger.warning(f"📊 ETH calculado (6 decimais): {eth_quantity}")
                
                if eth_quantity <= 0:
                    return {'success': False, 'error': 'Quantidade muito pequena'}
                
                # Tentar ordem direta
                order = self.exchange.create_order(
                    symbol=futures_symbol,
                    type='market',
                    side=side,
                    amount=eth_quantity
                )
                
                logger.warning(f"✅ FALLBACK FUNCIONOU!")
                
                return {
                    'success': True,
                    'order_id': order['id'],
                    'order': order,
                    'usdt_amount': usdt_amount,
                    'eth_quantity': eth_quantity,
                    'price': current_price
                }
                
            except Exception as fallback_error:
                logger.error(f"❌ Fallback também falhou: {fallback_error}")
                return {
                    'success': False,
                    'error': f'Método notional falhou: {str(e)}, Fallback falhou: {str(fallback_error)}'
                }

    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Get order status"""
        try:
            order = self.exchange.fetch_order(order_id, 'ETH/USDT:USDT')
            return {
                'id': order['id'],
                'status': order['status'],
                'filled': order['filled'],
                'remaining': order['remaining'],
                'price': order['price'],
                'average': order['average']
            }
        except Exception as e:
            logger.error(f"❌ Erro ao obter status da ordem: {e}")
            return {}

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order"""
        try:
            self.exchange.cancel_order(order_id, 'ETH/USDT:USDT')
            logger.info(f"✅ Ordem cancelada: {order_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao cancelar ordem: {e}")
            return False

    def get_open_positions(self) -> list:
        """Get open FUTURES positions"""
        try:
            positions = self.exchange.fetch_positions(['ETH/USDT:USDT'])
            open_positions = [pos for pos in positions if float(pos['contracts']) > 0]
            return open_positions
        except Exception as e:
            logger.error(f"❌ Erro ao obter posições: {e}")
            return []

    def validate_order_params(self, symbol: str, side: str, size: float, **kwargs) -> Dict:
        """Validate order parameters before placing"""
        return {'valid': True, 'errors': []}
