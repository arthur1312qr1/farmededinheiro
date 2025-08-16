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
        """Place FUTURES order com cálculo dinâmico da quantidade ETH"""
        try:
            futures_symbol = 'ETH/USDT:USDT'
            
            # Definir alavancagem 10x
            try:
                self.exchange.set_leverage(10, futures_symbol)
                logger.warning(f"🚨 ALAVANCAGEM 10x DEFINIDA")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao definir alavancagem: {e}")
            
            # Obter preço atual do mercado
            if price is None:
                ticker = self.exchange.fetch_ticker(futures_symbol)
                current_price = float(ticker['last'])
            else:
                current_price = price
            
            # CORREÇÃO: Buscar saldo atual (100% dinâmico)
            current_balance = self.get_account_balance()
            usdt_amount = current_balance  # 100% do saldo atual
            
            # CÁLCULO DINÂMICO: Calcular quantidade ETH baseada no valor USDT
            eth_quantity = usdt_amount / current_price
            
            logger.warning(f"🚨 CÁLCULO DINÂMICO DA QUANTIDADE:")
            logger.warning(f"💰 Saldo Atual: ${current_balance:.2f} USDT")  
            logger.warning(f"🎯 Valor a usar: ${usdt_amount:.2f} USDT (100%)")
            logger.warning(f"💎 Preço ETH atual: ${current_price:.2f}")
            logger.warning(f"📊 ETH calculado: {eth_quantity:.8f} ETH")
            logger.warning(f"⚡ Alavancagem: {leverage}x")
            logger.warning(f"💥 Exposição total: ${usdt_amount * leverage:.2f} USDT")
            
            # Validar se a quantidade é positiva
            if eth_quantity <= 0:
                logger.error(f"❌ Quantidade ETH inválida: {eth_quantity}")
                return {
                    'success': False,
                    'error': f'Quantidade ETH calculada inválida: {eth_quantity:.8f}'
                }
            
            # Executar ordem com quantidade ETH calculada dinamicamente
            logger.warning(f"🚀 EXECUTANDO ORDEM COM VALORES DINÂMICOS:")
            logger.warning(f"📊 Quantidade: {eth_quantity:.8f} ETH")
            logger.warning(f"💰 Valor equivalente: ${usdt_amount:.2f} USDT")
            
            order = self.exchange.create_order(
                symbol=futures_symbol,
                type='market',
                side=side,
                amount=eth_quantity  # Quantidade ETH calculada dinamicamente
            )
            
            logger.warning(f"✅ ORDEM EXECUTADA COM CÁLCULO DINÂMICO!")
            logger.warning(f"💰 Valor usado: ${usdt_amount:.2f} USDT")
            logger.warning(f"📊 Quantidade: {eth_quantity:.8f} ETH")
            
            return {
                'success': True,
                'order_id': order['id'],
                'order': order,
                'usdt_amount': usdt_amount,
                'eth_quantity': eth_quantity,
                'price': current_price
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao executar ordem com cálculo dinâmico: {e}")
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
