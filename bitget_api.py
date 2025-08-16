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
            # CORREÇÃO: Usar as chaves EXATAS que o CCXT espera
            self.exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': secret_key,  # ← ESTE CAMPO É CRÍTICO
                'password': passphrase,
                'sandbox': sandbox,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',  # FUTURES
                }
            })
            
            # Test connection
            self.exchange.load_markets()
            logger.info("✅ Bitget API conectado com sucesso")
            
            # TESTE ESPECÍFICO das credenciais
            logger.warning(f"🔑 API Key presente: {bool(api_key)}")
            logger.warning(f"🔐 Secret presente: {bool(secret_key)}")
            logger.warning(f"🗝️ Passphrase presente: {bool(passphrase)}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao conectar Bitget API: {e}")
            logger.error(f"🔍 Debugging - API Key: {api_key[:10] if api_key else 'None'}...")
            logger.error(f"🔍 Debugging - Secret: {secret_key[:10] if secret_key else 'None'}...")
            logger.error(f"🔍 Debugging - Passphrase: {passphrase[:3] if passphrase else 'None'}...")
            raise

    def validate_order_params(self, symbol: str, side: str, size: float, **kwargs) -> Dict:
        """Validate order parameters before placing"""
        errors = []
        
        # Check symbol
        try:
            if not self.exchange.markets:
                self.exchange.load_markets()
            futures_symbol = 'ETH/USDT:USDT'
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
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def get_account_balance(self) -> float:
        """Get FUTURES account balance in USDT"""
        try:
            logger.warning("🔄 Tentando obter saldo FUTURES...")
            
            # MÉTODO 1: Saldo específico de FUTURES
            try:
                balance = self.exchange.fetch_balance({'type': 'swap'})
                logger.warning(f"✅ Saldo FUTURES obtido via type=swap")
            except Exception as e:
                logger.warning(f"❌ Método 1 falhou: {e}")
                # MÉTODO 2: Saldo geral
                balance = self.exchange.fetch_balance()
                logger.warning(f"✅ Saldo obtido via método geral")
            
            logger.warning(f"🔍 Estrutura completa do saldo: {balance}")
            
            usdt_balance = 0.0
            
            # Extrair saldo USDT
            if 'USDT' in balance:
                usdt_data = balance['USDT']
                logger.warning(f"💰 Dados USDT: {usdt_data}")
                if isinstance(usdt_data, dict):
                    usdt_balance = usdt_data.get('free', 0) or usdt_data.get('available', 0) or usdt_data.get('total', 0)
                else:
                    usdt_balance = float(usdt_data)
            
            # Se não encontrou, tentar outras formas
            if usdt_balance == 0:
                if 'free' in balance and 'USDT' in balance['free']:
                    usdt_balance = balance['free']['USDT']
                elif 'total' in balance and 'USDT' in balance['total']:
                    usdt_balance = balance['total']['USDT']
            
            usdt_balance = float(usdt_balance) if usdt_balance else 0.0
            
            logger.warning(f"💰 SALDO FINAL DETECTADO: ${usdt_balance:.2f} USDT")
            logger.warning(f"🚨 PODER DE COMPRA 10x: ${usdt_balance * 10:.2f} USDT")
            
            return usdt_balance
            
        except Exception as e:
            logger.error(f"❌ Erro crítico ao obter saldo: {e}")
            logger.error(f"🔍 Tipo do erro: {type(e)}")
            logger.error(f"🔍 Exchange configurado: {hasattr(self, 'exchange')}")
            
            # VERIFICAR se as credenciais estão sendo passadas
            if hasattr(self, 'exchange'):
                logger.error(f"🔑 API Key no exchange: {bool(self.exchange.apiKey)}")
                logger.error(f"🔐 Secret no exchange: {bool(self.exchange.secret)}")
                logger.error(f"🗝️ Password no exchange: {bool(self.exchange.password)}")
            
            return 0.0

    def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for FUTURES"""
        try:
            futures_symbol = 'ETH/USDT:USDT'
            ticker = self.exchange.fetch_ticker(futures_symbol)
            
            logger.info(f"✅ Preço ETH FUTURES: ${ticker['last']:.2f}")
            
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
        """Place FUTURES order with 10x leverage"""
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
            
            # Cálculos
            quote_amount = size  # Valor em USDT (80% do saldo)
            base_amount = quote_amount / current_price  # Quantidade ETH
            
            # CORREÇÃO: Validar quantidade mínima APENAS se o valor calculado for insuficiente
            MIN_ETH_AMOUNT = 0.01  # Mínimo exigido pela Bitget
            
            if base_amount < MIN_ETH_AMOUNT:
                logger.warning(f"⚠️ QUANTIDADE CALCULADA ABAIXO DO MÍNIMO:")
                logger.warning(f"📊 Calculado: {base_amount:.6f} ETH")
                logger.warning(f"📊 Mínimo: {MIN_ETH_AMOUNT:.6f} ETH")
                
                # Verificar se temos saldo suficiente para a quantidade mínima
                min_usdt_needed = MIN_ETH_AMOUNT * current_price
                current_balance = self.get_account_balance()
                
                if current_balance < min_usdt_needed:
                    logger.error(f"❌ SALDO INSUFICIENTE PARA QUANTIDADE MÍNIMA")
                    logger.error(f"💰 Necessário: ${min_usdt_needed:.2f} USDT")
                    logger.error(f"💰 Disponível: ${current_balance:.2f} USDT")
                    return {
                        'success': False,
                        'error': f'Saldo insuficiente. Necessário: ${min_usdt_needed:.2f} USDT'
                    }
                
                # Usar quantidade mínima apenas se necessário
                base_amount = MIN_ETH_AMOUNT
                quote_amount = base_amount * current_price
                logger.warning(f"⚡ AJUSTADO PARA QUANTIDADE MÍNIMA:")
                logger.warning(f"📊 Nova Quantidade ETH: {base_amount:.6f}")
                logger.warning(f"💰 Novo Valor USDT: ${quote_amount:.2f}")
            else:
                logger.warning(f"✅ USANDO QUANTIDADE CALCULADA (80% DO SALDO)")
            
            logger.warning(f"🚨 EXECUTANDO ORDEM FUTURES 10x:")
            logger.warning(f"💰 Valor USDT: ${quote_amount:.2f}")
            logger.warning(f"📊 Quantidade ETH: {base_amount:.6f}")
            logger.warning(f"💎 Preço: ${current_price:.2f}")
            logger.warning(f"⚡ Alavancagem: 10x")
            logger.warning(f"💥 Exposição: ${quote_amount * 10:.2f} USDT")
            
            # Executar ordem
            order = self.exchange.create_order(
                symbol=futures_symbol,
                type='market',
                side=side,
                amount=base_amount
            )
            
            logger.warning(f"✅ ORDEM FUTURES EXECUTADA!")
            
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
