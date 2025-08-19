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
        """Pega saldo atual em USDT"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_free = balance['USDT']['free']
            usdt_used = balance['USDT']['used']
            usdt_total = balance['USDT']['total']
            
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
            return {'free': 0, 'used': 0, 'total': 0}

    def get_account_balance(self):
        """Alias para get_balance - compatibilidade"""
        balance_info = self.get_balance()
        if balance_info and 'free' in balance_info:
            return balance_info['free']
        return 0.0

    def get_eth_price(self):
        """Pega preço atual do ETH em tempo real"""
        try:
            ticker = self.exchange.fetch_ticker('ETHUSDT')
            price = ticker['last']
            print(f"📈 Preço ETH atual: ${price:.2f}")
            return price
        except Exception as e:
            print(f"❌ Erro ao pegar preço ETH: {e}")
            return 0

    def get_current_price(self, symbol='ETHUSDT'):
        """Alias para get_eth_price - compatibilidade"""
        return self.get_eth_price()

    def get_market_data(self, symbol='ETHUSDT'):
        """Dados de mercado - CORRIGIDO para aceitar symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            market_data = {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'high': ticker['high'],
                'low': ticker['low'],
                'volume': ticker['baseVolume'],
                'change': ticker['change'],
                'percentage': ticker['percentage'],
                'timestamp': ticker['timestamp']
            }
            
            print(f"📊 Dados de mercado {symbol}:")
            print(f"   Preço: ${market_data['price']:.2f}")
            return market_data
            
        except Exception as e:
            print(f"❌ Erro ao pegar dados de mercado: {e}")
            return {
                'symbol': symbol, 'price': 0, 'bid': 0, 'ask': 0,
                'high': 0, 'low': 0, 'volume': 0, 'change': 0,
                'percentage': 0, 'timestamp': int(time.time() * 1000)
            }

    def calculate_eth_quantity(self, usdt_balance, eth_price):
        """CORREÇÃO DEFINITIVA - Calcula quantidade ETH com alavancagem"""
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

    def place_buy_order(self):
        """Comprar ETH com 100% do saldo + alavancagem"""
        try:
            print("🚀 Iniciando compra...")
            
            # Pegar saldo atual (sempre 100%)
            balance_info = self.get_balance()
            if not balance_info:
                return {"success": False, "error": "Erro ao pegar saldo"}
            
            usdt_balance = balance_info['free']
            if usdt_balance < 1:  # Mínimo $1 para trading
                error_msg = f"❌ Saldo insuficiente: ${usdt_balance:.4f}"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
            # Pegar preço atual
            eth_price = self.get_eth_price()
            if not eth_price:
                return {"success": False, "error": "Erro ao pegar preço ETH"}
            
            # Calcular quantidade (CORREÇÃO APLICADA)
            eth_quantity = self.calculate_eth_quantity(usdt_balance, eth_price)
            
            print(f"🎯 Executando ordem de compra...")
            print(f"   Símbolo: ETHUSDT")
            print(f"   Quantidade: {eth_quantity} ETH")
            print(f"   Alavancagem: 10x")
            
            # Fazer ordem de compra
            order = self.exchange.create_market_buy_order(
                symbol='ETHUSDT',
                amount=eth_quantity,
                params={
                    'leverage': 10
                }
            )
            
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
            
        except Exception as e:
            error_msg = f"❌ Erro na compra: {str(e)}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    def place_sell_order(self, profit_target=0.01):  # 1% de lucro
        """Vender ETH quando atingir lucro"""
        try:
            # Pegar posição atual
            positions = self.exchange.fetch_positions(['ETHUSDT'])
            eth_position = None
            
            for pos in positions:
                if pos['symbol'] == 'ETHUSDT' and abs(pos['size']) > 0:
                    eth_position = pos
                    break
            
            if not eth_position:
                return {"success": False, "error": "Nenhuma posição ETH encontrada"}
            
            entry_price = eth_position['entryPrice']
            quantity = abs(eth_position['size'])
            current_price = self.get_eth_price()
            
            if not current_price:
                return {"success": False, "error": "Erro ao pegar preço atual"}
            
            # Calcular lucro atual
            profit_pct = (current_price - entry_price) / entry_price
            
            print(f"📊 Posição atual:")
            print(f"   Quantidade: {quantity} ETH")
            print(f"   Preço entrada: ${entry_price:.2f}")
            print(f"   Preço atual: ${current_price:.2f}")
            print(f"   Lucro: {profit_pct * 100:.2f}%")
            
            # Vender se atingiu o lucro alvo
            if profit_pct >= profit_target:
                print(f"🎯 Meta de {profit_target * 100}% atingida! Vendendo...")
                
                order = self.exchange.create_market_sell_order(
                    symbol='ETHUSDT',
                    amount=quantity
                )
                
                success_msg = f"✅ VENDA EXECUTADA! Lucro: {profit_pct * 100:.2f}%"
                print(success_msg)
                
                return {
                    "success": True,
                    "order": order,
                    "profit_pct": profit_pct * 100,
                    "message": success_msg
                }
            else:
                waiting_msg = f"⏳ Aguardando lucro de {profit_target * 100}%... Atual: {profit_pct * 100:.2f}%"
                print(waiting_msg)
                return {"success": False, "waiting": True, "message": waiting_msg}
                
        except Exception as e:
            error_msg = f"❌ Erro na venda: {str(e)}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    def get_position_info(self):
        """Informações da posição atual"""
        try:
            positions = self.exchange.fetch_positions(['ETHUSDT'])
            balance = self.get_balance()
            
            eth_position = None
            for pos in positions:
                if pos['symbol'] == 'ETHUSDT' and abs(pos['size']) > 0:
                    eth_position = pos
                    break
            
            return {
                'balance': balance,
                'position': eth_position,
                'eth_price': self.get_eth_price()
            }
        except Exception as e:
            print(f"❌ Erro ao pegar informações: {e}")
            return None

    def get_positions(self):
        """Alias para get_position_info - compatibilidade"""
        return self.get_position_info()

    def test_connection(self):
        """Testa conexão com a Bitget"""
        try:
            balance = self.exchange.fetch_balance()
            print("✅ Conexão com Bitget OK")
            return True
        except Exception as e:
            print(f"❌ Erro de conexão: {e}")
            return False

    def is_connected(self):
        """Alias para test_connection - compatibilidade"""
        return self.test_connection()

    # Métodos antigos para compatibilidade TOTAL
    def place_order(self, side='buy', **kwargs):
        """Compatibilidade com código antigo"""
        if side == 'buy':
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
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            print(f"❌ Erro fetch_ticker: {e}")
            return {'last': 0, 'bid': 0, 'ask': 0}

    def fetch_positions(self, symbols=None):
        """Compatibilidade - acesso direto ao exchange"""
        try:
            if symbols:
                # Corrigir o símbolo na lista
                corrected_symbols = [s.replace('ETH/USDT:USDT', 'ETHUSDT') for s in symbols]
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
                if side == 'buy':
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
            info = api.get_position_info()
            market = api.get_market_data('ETHUSDT')
            print("🔥 Bot funcionando corretamente!")
        else:
            print("❌ Falha na conexão")
    except Exception as e:
        print(f"❌ Erro: {e}")
