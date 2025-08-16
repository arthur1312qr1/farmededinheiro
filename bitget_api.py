import ccxt
import os
import time
from decimal import Decimal, ROUND_DOWN

class BitgetAPI:
    def __init__(self):
        # Credenciais do ambiente (render.com)
        api_key = os.getenv('BITGET_API_KEY')
        secret = os.getenv('BITGET_SECRET')
        passphrase = os.getenv('BITGET_PASSPHRASE')
        
        if not all([api_key, secret, passphrase]):
            raise Exception("❌ Credenciais não encontradas no ambiente")
        
        # Configurar exchange
        self.exchange = ccxt.bitget({
            'apiKey': api_key,
            'secret': secret,
            'password': passphrase,
            'sandbox': False,  # TRADING REAL
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
            return None

    def get_eth_price(self):
        """Pega preço atual do ETH em tempo real"""
        try:
            ticker = self.exchange.fetch_ticker('ETH/USDT:USDT')
            price = ticker['last']
            print(f"📈 Preço ETH atual: ${price:.2f}")
            return price
        except Exception as e:
            print(f"❌ Erro ao pegar preço ETH: {e}")
            return None

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
            
        print(f"💪 Cálculo:")
        print(f"   Saldo: ${usdt_balance:.4f} USDT")
        print(f"   Alavancagem: {leverage}x")
        print(f"   Poder de compra: ${buying_power:.2f} USDT")
        print(f"   Preço ETH: ${eth_price:.2f}")
        print(f"   Quantidade ETH: {eth_quantity} ETH")
        print(f"   ✅ Acima do mínimo 0.01 ETH")
        
        return eth_quantity

    def place_buy_order(self):
        """Comprar ETH com 100% do saldo + alavancagem"""
        try:
            # Pegar saldo atual (sempre 100%)
            balance_info = self.get_balance()
            if not balance_info:
                return None
                
            usdt_balance = balance_info['free']
            
            if usdt_balance < 1:  # Mínimo $1 para trading
                print(f"❌ Saldo insuficiente: ${usdt_balance:.4f}")
                return None
            
            # Pegar preço atual
            eth_price = self.get_eth_price()
            if not eth_price:
                return None
                
            # Calcular quantidade (CORREÇÃO APLICADA)
            eth_quantity = self.calculate_eth_quantity(usdt_balance, eth_price)
            
            print(f"🚀 Executando compra...")
            
            # Fazer ordem de compra
            order = self.exchange.create_market_buy_order(
                symbol='ETH/USDT:USDT',
                amount=eth_quantity,
                params={
                    'leverage': 10
                }
            )
            
            print(f"✅ Ordem de compra executada!")
            print(f"   ID: {order['id']}")
            print(f"   Quantidade: {eth_quantity} ETH")
            print(f"   Preço: ${eth_price:.2f}")
            
            return order
            
        except Exception as e:
            print(f"❌ Erro na compra: {e}")
            return None

    def place_sell_order(self, profit_target=0.01):  # 1% de lucro
        """Vender ETH quando atingir lucro"""
        try:
            # Pegar posição atual
            positions = self.exchange.fetch_positions(['ETH/USDT:USDT'])
            eth_position = None
            
            for pos in positions:
                if pos['symbol'] == 'ETH/USDT:USDT' and pos['size'] > 0:
                    eth_position = pos
                    break
            
            if not eth_position:
                print("❌ Nenhuma posição ETH encontrada")
                return None
            
            entry_price = eth_position['entryPrice']
            quantity = eth_position['size']
            current_price = self.get_eth_price()
            
            # Calcular lucro atual
            profit_pct = (current_price - entry_price) / entry_price
            
            print(f"📊 Posição atual:")
            print(f"   Quantidade: {quantity} ETH")
            print(f"   Preço entrada: ${entry_price:.2f}")
            print(f"   Preço atual: ${current_price:.2f}")
            print(f"   Lucro: {profit_pct*100:.2f}%")
            
            # Vender se atingiu o lucro alvo
            if profit_pct >= profit_target:
                print(f"🎯 Meta de {profit_target*100}% atingida! Vendendo...")
                
                order = self.exchange.create_market_sell_order(
                    symbol='ETH/USDT:USDT',
                    amount=quantity
                )
                
                print(f"✅ Ordem de venda executada!")
                print(f"   ID: {order['id']}")
                print(f"   Lucro: {profit_pct*100:.2f}%")
                
                return order
            else:
                print(f"⏳ Aguardando lucro de {profit_target*100}%...")
                return None
                
        except Exception as e:
            print(f"❌ Erro na venda: {e}")
            return None

    def get_position_info(self):
        """Informações da posição atual"""
        try:
            positions = self.exchange.fetch_positions(['ETH/USDT:USDT'])
            balance = self.get_balance()
            
            eth_position = None
            for pos in positions:
                if pos['symbol'] == 'ETH/USDT:USDT' and pos['size'] > 0:
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

# Testar conexão
if __name__ == "__main__":
    try:
        api = BitgetAPI()
        info = api.get_position_info()
        print("🔥 Bot funcionando corretamente!")
    except Exception as e:
        print(f"❌ Erro: {e}")
