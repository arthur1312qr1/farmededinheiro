import ccxt
import os
import time
from decimal import Decimal, ROUND_DOWN

class BitgetAPI:
    def __init__(self, api_key=None, secret_key=None, passphrase=None, sandbox=None, **kwargs):
        """Inicializa com todos os parâmetros possíveis"""
        # Mapear nomes corretos das credenciais
        self.api_key = api_key or os.getenv('BITGET_API_KEY')
        self.secret = secret_key or os.getenv('BITGET_SECRET')
        self.passphrase = passphrase or os.getenv('BITGET_PASSPHRASE')
        
        # Modo sandbox (sempre False para trading real)
        self.sandbox = False  # SEMPRE TRADING REAL
        
        if not all([self.api_key, self.secret, self.passphrase]):
            raise Exception("❌ Credenciais não encontradas")
        
        # Configurar exchange
        self.exchange = ccxt.bitget({
            'apiKey': self.api_key,
            'secret': self.secret,
            'password': self.passphrase,
            'sandbox': self.sandbox,  # TRADING REAL
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap'  # Futures
            }
        })
        
        print("✅ Conectado à Bitget - MODO REAL")

    def get_balance(self):
        """Pega saldo atual em USDT - CORRIGIDO"""
        try:
            balance = self.exchange.fetch_balance()
            
            # CORREÇÃO: Verificar se USDT existe no balance
            if 'USDT' not in balance:
                print("⚠️ USDT não encontrado no balance")
                return {'free': 0.0, 'used': 0.0, 'total': 0.0}
            
            usdt_balance = balance['USDT']
            usdt_free = float(usdt_balance.get('free', 0.0))
            usdt_used = float(usdt_balance.get('used', 0.0))
            usdt_total = float(usdt_balance.get('total', 0.0))
            
            print(f"💰 Saldo USDT:")
            print(f"   Livre: ${usdt_free:.4f}")
            print(f"   Usado: ${usdt_used:.4f}")
            print(f"   Total: ${usdt_total:.4f}")
            
            return {
                'free': usdt_free,
                'used': usdt_used,
                'total': usdt_total
            }
        except Exception as e:
            print(f"❌ Erro ao pegar saldo: {e}")
            return {'free': 0.0, 'used': 0.0, 'total': 0.0}

    def get_account_balance(self):
        """Alias para get_balance - compatibilidade"""
        balance_info = self.get_balance()
        if balance_info and 'free' in balance_info:
            return balance_info['free']
        return 0.0

    def get_eth_price(self):
        """Pega preço atual do ETH em tempo real - CORRIGIDO"""
        try:
            ticker = self.exchange.fetch_ticker('ETHUSDT')
            if not ticker or 'last' not in ticker:
                print("❌ Ticker inválido")
                return 0.0
                
            price = float(ticker['last'])
            if price <= 0:
                print("❌ Preço inválido")
                return 0.0
                
            print(f"📈 Preço ETH atual: ${price:.2f}")
            return price
        except Exception as e:
            print(f"❌ Erro ao pegar preço ETH: {e}")
            return 0.0

    def get_current_price(self, symbol='ETHUSDT'):
        """Alias para get_eth_price - compatibilidade"""
        return self.get_eth_price()

    def get_market_data(self, symbol='ETHUSDT'):
        """Dados de mercado - CORRIGIDO para tratar erros"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            if not ticker:
                print(f"❌ Ticker vazio para {symbol}")
                return self._empty_market_data(symbol)
            
            # Validar dados essenciais
            price = float(ticker.get('last', 0))
            if price <= 0:
                print(f"❌ Preço inválido: {price}")
                return self._empty_market_data(symbol)
            
            market_data = {
                'symbol': symbol,
                'price': price,
                'bid': float(ticker.get('bid', price)),
                'ask': float(ticker.get('ask', price)),
                'high': float(ticker.get('high', price)),
                'low': float(ticker.get('low', price)),
                'volume': float(ticker.get('baseVolume', 0)),
                'change': float(ticker.get('change', 0)),
                'percentage': float(ticker.get('percentage', 0)),
                'timestamp': int(ticker.get('timestamp', time.time() * 1000))
            }
            
            return market_data
            
        except Exception as e:
            print(f"❌ Erro ao pegar dados de mercado: {e}")
            return self._empty_market_data(symbol)

    def _empty_market_data(self, symbol):
        """Dados de mercado vazios para fallback"""
        return {
            'symbol': symbol, 'price': 0.0, 'bid': 0.0, 'ask': 0.0,
            'high': 0.0, 'low': 0.0, 'volume': 0.0, 'change': 0.0,
            'percentage': 0.0, 'timestamp': int(time.time() * 1000)
        }

    def calculate_eth_quantity(self, usdt_balance, eth_price):
        """CORREÇÃO DEFINITIVA - Calcula quantidade ETH com alavancagem"""
        try:
            if usdt_balance <= 0 or eth_price <= 0:
                print("❌ Parâmetros inválidos para cálculo")
                return 0.01
                
            leverage = 10  # Alavancagem 10x
            
            # Poder de compra total (100% do saldo * alavancagem)
            buying_power = usdt_balance * leverage
            
            # Quantidade ETH que pode comprar
            raw_quantity = buying_power / eth_price
            
            # Precisão da Bitget: 2 casas decimais
            eth_quantity = float(Decimal(str(raw_quantity)).quantize(Decimal('0.01'), rounding=ROUND_DOWN))
            
            # Garantir mínimo 0.01 ETH
            if eth_quantity < 0.01:
                eth_quantity = 0.01
            
            print(f"💪 Cálculo FINAL:")
            print(f"   Saldo: ${usdt_balance:.6f} USDT")
            print(f"   Alavancagem: {leverage}x")
            print(f"   Poder de compra: ${buying_power:.2f} USDT")
            print(f"   Preço ETH: ${eth_price:.2f}")
            print(f"   Quantidade ETH: {eth_quantity} ETH")
            print(f"   ✅ Acima do mínimo 0.01 ETH")
            
            return eth_quantity
            
        except Exception as e:
            print(f"❌ Erro no cálculo: {e}")
            return 0.01

    def place_buy_order(self):
        """Comprar ETH com 100% do saldo + alavancagem - CORRIGIDO"""
        try:
            print("🚀 Iniciando compra...")
            
            # Pegar saldo atual (sempre 100%)
            balance_info = self.get_balance()
            if not balance_info or balance_info.get('free', 0) <= 0:
                error_msg = "❌ Saldo insuficiente ou erro ao obter saldo"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
            usdt_balance = balance_info['free']
            if usdt_balance < 1:  # Mínimo $1 para trading
                error_msg = f"❌ Saldo insuficiente: ${usdt_balance:.4f}"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
            # Pegar preço atual
            eth_price = self.get_eth_price()
            if eth_price <= 0:
                error_msg = "❌ Erro ao pegar preço ETH ou preço inválido"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
            # Calcular quantidade (CORREÇÃO APLICADA)
            eth_quantity = self.calculate_eth_quantity(usdt_balance, eth_price)
            
            if eth_quantity <= 0:
                error_msg = "❌ Quantidade calculada inválida"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
            print(f"🎯 Executando ordem de compra...")
            print(f"   Símbolo: ETHUSDT")
            print(f"   Quantidade: {eth_quantity} ETH")
            print(f"   Alavancagem: 10x")
            
            # Fazer ordem de compra - CORRIGIDO COM TRATAMENTO DE ERRO
            try:
                order = self.exchange.create_market_buy_order(
                    symbol='ETHUSDT',
                    amount=eth_quantity,
                    params={
                        'leverage': 10
                    }
                )
                
                if not order or 'id' not in order:
                    raise Exception("Ordem retornou vazia ou sem ID")
                
                success_msg = f"✅ COMPRA EXECUTADA! ID: {order['id']}"
                print(success_msg)
                print(f"   Quantidade: {eth_quantity} ETH")
                print(f"   Preço: ${eth_price:.2f}")
                print(f"   Valor total: ${eth_quantity * eth_price:.2f}")
                
                return {
                    "success": True,
                    "order": order,
                    "quantity": eth_quantity,
                    "price": eth_price,
                    "message": success_msg
                }
                
            except Exception as order_error:
                error_msg = f"❌ Erro na execução da ordem: {str(order_error)}"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
        except Exception as e:
            error_msg = f"❌ Erro geral na compra: {str(e)}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    def place_sell_order(self, profit_target=0.01):
        """Vender ETH - CORRIGIDO para funcionar melhor"""
        try:
            print(f"🔄 Iniciando venda...")
            
            # Pegar posições atuais - CORRIGIDO
            try:
                positions = self.exchange.fetch_positions(['ETHUSDT'])
                if not positions:
                    print("❌ Nenhuma posição encontrada")
                    return {"success": False, "error": "Nenhuma posição encontrada"}
                    
            except Exception as pos_error:
                print(f"❌ Erro ao buscar posições: {pos_error}")
                return {"success": False, "error": f"Erro ao buscar posições: {pos_error}"}
            
            # Encontrar posição ETH ativa
            eth_position = None
            for pos in positions:
                if pos.get('symbol') == 'ETHUSDT' and pos.get('size', 0) != 0:
                    size = float(pos.get('size', 0))
                    if abs(size) > 0:
                        eth_position = pos
                        break
            
            if not eth_position:
                print("❌ Nenhuma posição ETH ativa encontrada")
                return {"success": False, "error": "Nenhuma posição ETH ativa"}
            
            entry_price = float(eth_position.get('entryPrice', 0))
            quantity = abs(float(eth_position.get('size', 0)))
            current_price = self.get_eth_price()
            
            if current_price <= 0:
                print("❌ Preço atual inválido")
                return {"success": False, "error": "Preço atual inválido"}
            
            # Calcular lucro atual
            if entry_price > 0:
                profit_pct = (current_price - entry_price) / entry_price
            else:
                profit_pct = 0
            
            print(f"📊 Posição atual:")
            print(f"   Quantidade: {quantity} ETH")
            print(f"   Preço entrada: ${entry_price:.2f}")
            print(f"   Preço atual: ${current_price:.2f}")
            print(f"   Lucro: {profit_pct * 100:.2f}%")
            
            # VENDER SEMPRE (para fechar posição) - CORRIGIDO
            print(f"🎯 Executando venda para fechar posição...")
            
            try:
                order = self.exchange.create_market_sell_order(
                    symbol='ETHUSDT',
                    amount=quantity
                )
                
                if not order:
                    raise Exception("Ordem de venda retornou vazia")
                
                success_msg = f"✅ VENDA EXECUTADA! Profit: {profit_pct * 100:.2f}%"
                print(success_msg)
                
                return {
                    "success": True,
                    "order": order,
                    "profit_pct": profit_pct * 100,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "message": success_msg
                }
                
            except Exception as sell_error:
                error_msg = f"❌ Erro na ordem de venda: {str(sell_error)}"
                print(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"❌ Erro geral na venda: {str(e)}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    def get_position_info(self):
        """Informações da posição atual - CORRIGIDO"""
        try:
            positions = self.exchange.fetch_positions(['ETHUSDT'])
            balance = self.get_balance()
            
            eth_position = None
            if positions:
                for pos in positions:
                    if pos.get('symbol') == 'ETHUSDT' and abs(float(pos.get('size', 0))) > 0:
                        eth_position = pos
                        break
            
            return {
                'balance': balance,
                'position': eth_position,
                'eth_price': self.get_eth_price()
            }
        except Exception as e:
            print(f"❌ Erro ao pegar informações: {e}")
            return {
                'balance': {'free': 0, 'used': 0, 'total': 0},
                'position': None,
                'eth_price': 0
            }

    def get_positions(self):
        """Alias para get_position_info - compatibilidade"""
        return self.get_position_info()

    def test_connection(self):
        """Testa conexão com a Bitget - CORRIGIDO"""
        try:
            balance = self.exchange.fetch_balance()
            if balance and isinstance(balance, dict):
                print("✅ Conexão com Bitget OK")
                return True
            else:
                print("❌ Resposta de balance inválida")
                return False
        except Exception as e:
            print(f"❌ Erro de conexão: {e}")
            return False

    def is_connected(self):
        """Alias para test_connection - compatibilidade"""
        return self.test_connection()

    # Métodos antigos para compatibilidade TOTAL
    def place_order(self, side='buy', **kwargs):
        """Compatibilidade com código antigo"""
        if side.lower() == 'buy':
            return self.place_buy_order()
        else:
            return self.place_sell_order()

    def get_ticker(self, symbol='ETHUSDT'):
        """Compatibilidade - alias para get_market_data"""
        return self.get_market_data(symbol)

    def fetch_balance(self):
        """Compatibilidade - alias para get_balance"""
        return self.get_balance()

    def fetch_ticker(self, symbol='ETHUSDT'):
        """Compatibilidade - acesso direto ao exchange"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            if not ticker:
                return {'last': 0, 'bid': 0, 'ask': 0}
            return ticker
        except Exception as e:
            print(f"❌ Erro fetch_ticker: {e}")
            return {'last': 0, 'bid': 0, 'ask': 0}

    def fetch_positions(self, symbols=None):
        """Compatibilidade - acesso direto ao exchange"""
        try:
            if symbols:
                # Corrigir o símbolo na lista
                corrected_symbols = []
                for s in symbols:
                    if 'ETH/USDT:USDT' in s:
                        corrected_symbols.append('ETHUSDT')
                    else:
                        corrected_symbols.append(s)
                return self.exchange.fetch_positions(corrected_symbols)
            else:
                return self.exchange.fetch_positions()
        except Exception as e:
            print(f"❌ Erro fetch_positions: {e}")
            return []

    def create_order(self, symbol, order_type, side, amount, price=None, params={}):
        """Compatibilidade - acesso direto ao exchange"""
        try:
            # Usar o símbolo corrigido
            corrected_symbol = symbol.replace('ETH/USDT:USDT', 'ETHUSDT')
            
            if order_type == 'market':
                if side.lower() == 'buy':
                    return self.exchange.create_market_buy_order(corrected_symbol, amount, None, params)
                else:
                    return self.exchange.create_market_sell_order(corrected_symbol, amount, None, params)
            else:
                return self.exchange.create_order(corrected_symbol, order_type, side, amount, price, params)
        except Exception as e:
            print(f"❌ Erro create_order: {e}")
            return None

# Função para compatibilidade total
def create_bitget_api():
    """Cria instância da BitgetAPI usando variáveis de ambiente"""
    return BitgetAPI()

# Testar se executado diretamente
if __name__ == "__main__":
    try:
        api = BitgetAPI()
        if api.test_connection():
            print("🔥 API funcionando!")
            
            # Teste básico
            balance = api.get_balance()
            price = api.get_eth_price()
            market = api.get_market_data('ETHUSDT')
            
            print(f"💰 Saldo livre: ${balance.get('free', 0):.4f}")
            print(f"📈 Preço ETH: ${price:.2f}")
            print(f"📊 Dados de mercado: OK")
            
        else:
            print("❌ Falha na conexão")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
