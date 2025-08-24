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
        
        print("✅ Conectado à Bitget - MODO REAL COM SUPORTE A SHORT")

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
        """Comprar ETH (LONG) com 100% do saldo + alavancagem - CORRIGIDO"""
        try:
            print("🚀 Iniciando COMPRA (LONG)...")
            
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
            
            # Calcular quantidade
            eth_quantity = self.calculate_eth_quantity(usdt_balance, eth_price)
            
            if eth_quantity <= 0:
                error_msg = "❌ Quantidade calculada inválida"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
            print(f"🎯 Executando ordem de COMPRA (LONG)...")
            print(f"   Símbolo: ETHUSDT")
            print(f"   Quantidade: {eth_quantity} ETH")
            print(f"   Alavancagem: 10x")
            print(f"   Tipo: LONG (compra)")
            
            # Fazer ordem de compra
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
                
                success_msg = f"✅ COMPRA (LONG) EXECUTADA! ID: {order['id']}"
                print(success_msg)
                print(f"   Quantidade: {eth_quantity} ETH")
                print(f"   Preço: ${eth_price:.2f}")
                print(f"   Valor total: ${eth_quantity * eth_price:.2f}")
                
                return {
                    "success": True,
                    "order": order,
                    "quantity": eth_quantity,
                    "price": eth_price,
                    "side": "LONG",
                    "message": success_msg
                }
                
            except Exception as order_error:
                error_msg = f"❌ Erro na execução da ordem LONG: {str(order_error)}"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
        except Exception as e:
            error_msg = f"❌ Erro geral na compra LONG: {str(e)}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    def place_short_order(self):
        """Abrir posição SHORT - NOVO MÉTODO"""
        try:
            print("🔻 Iniciando VENDA (SHORT)...")
            
            # Pegar saldo atual
            balance_info = self.get_balance()
            if not balance_info or balance_info.get('free', 0) <= 0:
                error_msg = "❌ Saldo insuficiente para SHORT"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
            usdt_balance = balance_info['free']
            if usdt_balance < 1:
                error_msg = f"❌ Saldo insuficiente: ${usdt_balance:.4f}"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
            # Pegar preço atual
            eth_price = self.get_eth_price()
            if eth_price <= 0:
                error_msg = "❌ Erro ao pegar preço ETH"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
            # Calcular quantidade para SHORT
            eth_quantity = self.calculate_eth_quantity(usdt_balance, eth_price)
            
            print(f"🎯 Executando ordem de SHORT...")
            print(f"   Símbolo: ETHUSDT")
            print(f"   Quantidade: {eth_quantity} ETH")
            print(f"   Alavancagem: 10x")
            print(f"   Tipo: SHORT (venda)")
            
            # Fazer ordem de venda para SHORT
            try:
                order = self.exchange.create_market_sell_order(
                    symbol='ETHUSDT',
                    amount=eth_quantity,
                    params={
                        'leverage': 10
                    }
                )
                
                if not order or 'id' not in order:
                    raise Exception("Ordem SHORT retornou vazia ou sem ID")
                
                success_msg = f"✅ SHORT EXECUTADO! ID: {order['id']}"
                print(success_msg)
                print(f"   Quantidade: {eth_quantity} ETH")
                print(f"   Preço: ${eth_price:.2f}")
                print(f"   Valor total: ${eth_quantity * eth_price:.2f}")
                
                return {
                    "success": True,
                    "order": order,
                    "quantity": eth_quantity,
                    "price": eth_price,
                    "side": "SHORT",
                    "message": success_msg
                }
                
            except Exception as order_error:
                error_msg = f"❌ Erro na execução SHORT: {str(order_error)}"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
        except Exception as e:
            error_msg = f"❌ Erro geral no SHORT: {str(e)}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    def place_sell_order(self, profit_target=0.01):
        """Fechar posição LONG (vender ETH) - CRÍTICO CORRIGIDO"""
        try:
            print(f"🔄 FECHANDO POSIÇÃO LONG...")
            
            # 1. BUSCAR POSIÇÕES COM MÚLTIPLAS TENTATIVAS
            positions = None
            for attempt in range(3):
                try:
                    print(f"🔍 Tentativa {attempt+1}: Buscando posições...")
                    positions = self.exchange.fetch_positions(['ETHUSDT'])
                    if positions is not None:
                        print(f"✅ Posições encontradas: {len(positions)}")
                        break
                    else:
                        print(f"⚠️ Posições retornou None")
                        
                except Exception as pos_error:
                    print(f"❌ Erro ao buscar posições (tentativa {attempt+1}): {pos_error}")
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    else:
                        return {"success": False, "error": f"Falha ao buscar posições: {pos_error}"}
            
            if not positions:
                error_msg = "❌ Nenhuma posição encontrada após múltiplas tentativas"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
            # 2. ENCONTRAR POSIÇÃO ETH LONG ATIVA
            eth_position = None
            active_positions = 0
            
            print(f"🔍 Analisando {len(positions)} posições...")
            for i, pos in enumerate(positions):
                symbol = pos.get('symbol', 'UNKNOWN')
                size = float(pos.get('size', 0))
                side = pos.get('side', 'none')
                
                print(f"   Posição {i+1}: {symbol} | Size: {size} | Side: {side}")
                
                if symbol == 'ETHUSDT' and abs(size) > 0 and side == 'long':
                    eth_position = pos
                    active_positions += 1
                    print(f"   ✅ POSIÇÃO LONG ENCONTRADA!")
            
            if not eth_position:
                error_msg = f"❌ Nenhuma posição LONG ETH ativa encontrada"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
            # 3. EXTRAIR DADOS DA POSIÇÃO
            entry_price = float(eth_position.get('entryPrice', 0))
            quantity = abs(float(eth_position.get('size', 0)))
            unrealized_pnl = float(eth_position.get('unrealizedPnl', 0))
            
            # 4. PEGAR PREÇO ATUAL
            current_price = self.get_eth_price()
            if current_price <= 0:
                print("⚠️ Preço atual inválido, usando preço de entrada")
                current_price = entry_price
            
            # 5. CALCULAR PNL
            if entry_price > 0 and current_price > 0:
                profit_pct = (current_price - entry_price) / entry_price * 100
            else:
                profit_pct = 0
            
            print(f"📊 DADOS DA POSIÇÃO LONG:")
            print(f"   Quantidade: {quantity} ETH")
            print(f"   Preço entrada: ${entry_price:.2f}")
            print(f"   Preço atual: ${current_price:.2f}")
            print(f"   PnL: {profit_pct:.3f}%")
            print(f"   PnL não realizado: ${unrealized_pnl:.2f}")
            
            # 6. EXECUTAR VENDA PARA FECHAR LONG
            print(f"🎯 FECHANDO POSIÇÃO LONG (SELL)...")
            
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    print(f"🔄 TENTATIVA DE FECHAMENTO LONG {attempt+1}/{max_attempts}")
                    
                    # VENDER para fechar LONG
                    order = self.exchange.create_market_sell_order(
                        symbol='ETHUSDT',
                        amount=quantity,
                        params={'reduceOnly': True}  # CRÍTICO: Apenas fechar posição
                    )
                    
                    if order and 'id' in order:
                        order_id = order['id']
                        print(f"✅ LONG FECHADO! ID: {order_id}")
                        
                        # Verificar se foi fechado
                        time.sleep(1)
                        verification_success = self._verify_position_closed('ETHUSDT', 'long')
                        
                        success_msg = f"✅ POSIÇÃO LONG FECHADA! PnL: {profit_pct:.3f}%"
                        print(success_msg)
                        
                        return {
                            "success": True,
                            "order": order,
                            "order_id": order_id,
                            "profit_pct": profit_pct,
                            "quantity": quantity,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "position_side": "long",
                            "unrealized_pnl": unrealized_pnl,
                            "message": success_msg,
                            "verified_closed": verification_success
                        }
                    else:
                        raise Exception("Ordem LONG retornou vazia ou sem ID")
                        
                except Exception as sell_error:
                    print(f"❌ ERRO na tentativa LONG {attempt+1}: {sell_error}")
                    if attempt < max_attempts - 1:
                        sleep_time = (attempt + 1) * 2
                        print(f"⏳ Aguardando {sleep_time}s antes da próxima tentativa...")
                        time.sleep(sleep_time)
            
            # Se chegou aqui, todas as tentativas falharam
            error_msg = f"🚨 FALHA CRÍTICA: Não foi possível fechar posição LONG após {max_attempts} tentativas"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "attempts": max_attempts,
                "last_profit_pct": profit_pct,
                "position_data": eth_position
            }
                
        except Exception as e:
            error_msg = f"🚨 ERRO CRÍTICO no fechamento LONG: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {"success": False, "error": error_msg}

    def close_short_position(self):
        """Fechar posição SHORT (comprar ETH) - NOVO MÉTODO"""
        try:
            print(f"🔄 FECHANDO POSIÇÃO SHORT...")
            
            # 1. BUSCAR POSIÇÕES SHORT
            positions = None
            for attempt in range(3):
                try:
                    print(f"🔍 Tentativa {attempt+1}: Buscando posições SHORT...")
                    positions = self.exchange.fetch_positions(['ETHUSDT'])
                    if positions is not None:
                        print(f"✅ Posições encontradas: {len(positions)}")
                        break
                except Exception as pos_error:
                    print(f"❌ Erro ao buscar posições SHORT: {pos_error}")
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    else:
                        return {"success": False, "error": f"Falha ao buscar posições SHORT: {pos_error}"}
            
            if not positions:
                error_msg = "❌ Nenhuma posição SHORT encontrada"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
            # 2. ENCONTRAR POSIÇÃO ETH SHORT ATIVA
            eth_position = None
            
            for i, pos in enumerate(positions):
                symbol = pos.get('symbol', 'UNKNOWN')
                size = float(pos.get('size', 0))
                side = pos.get('side', 'none')
                
                print(f"   Posição {i+1}: {symbol} | Size: {size} | Side: {side}")
                
                if symbol == 'ETHUSDT' and abs(size) > 0 and side == 'short':
                    eth_position = pos
                    print(f"   ✅ POSIÇÃO SHORT ENCONTRADA!")
                    break
            
            if not eth_position:
                error_msg = f"❌ Nenhuma posição SHORT ETH ativa encontrada"
                print(error_msg)
                return {"success": False, "error": error_msg}
            
            # 3. EXTRAIR DADOS DA POSIÇÃO SHORT
            entry_price = float(eth_position.get('entryPrice', 0))
            quantity = abs(float(eth_position.get('size', 0)))
            unrealized_pnl = float(eth_position.get('unrealizedPnl', 0))
            
            # 4. PEGAR PREÇO ATUAL
            current_price = self.get_eth_price()
            if current_price <= 0:
                current_price = entry_price
            
            # 5. CALCULAR PNL PARA SHORT
            if entry_price > 0 and current_price > 0:
                profit_pct = (entry_price - current_price) / entry_price * 100
            else:
                profit_pct = 0
            
            print(f"📊 DADOS DA POSIÇÃO SHORT:")
            print(f"   Quantidade: {quantity} ETH")
            print(f"   Preço entrada: ${entry_price:.2f}")
            print(f"   Preço atual: ${current_price:.2f}")
            print(f"   PnL SHORT: {profit_pct:.3f}%")
            print(f"   PnL não realizado: ${unrealized_pnl:.2f}")
            
            # 6. EXECUTAR COMPRA PARA FECHAR SHORT
            print(f"🎯 FECHANDO POSIÇÃO SHORT (BUY)...")
            
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    print(f"🔄 TENTATIVA DE FECHAMENTO SHORT {attempt+1}/{max_attempts}")
                    
                    # COMPRAR para fechar SHORT
                    order = self.exchange.create_market_buy_order(
                        symbol='ETHUSDT',
                        amount=quantity,
                        params={'reduceOnly': True}  # CRÍTICO: Apenas fechar posição
                    )
                    
                    if order and 'id' in order:
                        order_id = order['id']
                        print(f"✅ SHORT FECHADO! ID: {order_id}")
                        
                        # Verificar se foi fechado
                        time.sleep(1)
                        verification_success = self._verify_position_closed('ETHUSDT', 'short')
                        
                        success_msg = f"✅ POSIÇÃO SHORT FECHADA! PnL: {profit_pct:.3f}%"
                        print(success_msg)
                        
                        return {
                            "success": True,
                            "order": order,
                            "order_id": order_id,
                            "profit_pct": profit_pct,
                            "quantity": quantity,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "position_side": "short",
                            "unrealized_pnl": unrealized_pnl,
                            "message": success_msg,
                            "verified_closed": verification_success
                        }
                    else:
                        raise Exception("Ordem SHORT retornou vazia ou sem ID")
                        
                except Exception as buy_error:
                    print(f"❌ ERRO na tentativa SHORT {attempt+1}: {buy_error}")
                    if attempt < max_attempts - 1:
                        sleep_time = (attempt + 1) * 2
                        print(f"⏳ Aguardando {sleep_time}s antes da próxima tentativa...")
                        time.sleep(sleep_time)
            
            # Se chegou aqui, todas as tentativas falharam
            error_msg = f"🚨 FALHA CRÍTICA: Não foi possível fechar posição SHORT após {max_attempts} tentativas"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "attempts": max_attempts,
                "last_profit_pct": profit_pct,
                "position_data": eth_position
            }
                
        except Exception as e:
            error_msg = f"🚨 ERRO CRÍTICO no fechamento SHORT: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {"success": False, "error": error_msg}

    def _verify_position_closed(self, symbol: str, side: str) -> bool:
        """Verificar se posição foi realmente fechada"""
        try:
            for verify_attempt in range(3):
                print(f"🔍 Verificando fechamento {side.upper()} (tentativa {verify_attempt+1})...")
                new_positions = self.exchange.fetch_positions([symbol])
                
                position_still_exists = False
                if new_positions:
                    for pos in new_positions:
                        if (pos.get('symbol') == symbol and 
                            abs(float(pos.get('size', 0))) > 0 and 
                            pos.get('side') == side):
                            position_still_exists = True
                            remaining_size = abs(float(pos.get('size', 0)))
                            print(f"⚠️ Posição {side.upper()} ainda existe! Size: {remaining_size}")
                            break
                
                if not position_still_exists:
                    print(f"✅ POSIÇÃO {side.upper()} TOTALMENTE FECHADA!")
                    return True
                else:
                    print(f"⚠️ Posição {side.upper()} ainda ativa, tentativa {verify_attempt+1}")
                    time.sleep(2)
                    
        except Exception as verify_error:
            print(f"❌ Erro na verificação: {verify_error}")
        
        return False

    def force_close_all_positions(self):
        """MÉTODO DE EMERGÊNCIA - Fechar TODAS as posições"""
        try:
            print("🚨 FECHAMENTO DE EMERGÊNCIA - TODAS AS POSIÇÕES")
            
            positions = self.exchange.fetch_positions()
            if not positions:
                print("✅ Nenhuma posição para fechar")
                return {"success": True, "message": "Nenhuma posição ativa"}
            
            closed_positions = []
            failed_positions = []
            
            for pos in positions:
                try:
                    symbol = pos.get('symbol', '')
                    size = float(pos.get('size', 0))
                    side = pos.get('side', '')
                    
                    if abs(size) > 0:
                        print(f"🔄 Fechando {symbol}: {size} {side}")
                        
                        if side == 'long':
                            order = self.exchange.create_market_sell_order(
                                symbol=symbol,
                                amount=abs(size),
                                params={'reduceOnly': True}
                            )
                        else:  # short
                            order = self.exchange.create_market_buy_order(
                                symbol=symbol,
                                amount=abs(size),
                                params={'reduceOnly': True}
                            )
                        
                        if order and 'id' in order:
                            closed_positions.append(f"{symbol}: {size} {side}")
                            print(f"✅ {symbol} {side} fechado")
                        else:
                            failed_positions.append(f"{symbol}: Ordem falhou")
                            
                except Exception as pos_error:
                    failed_positions.append(f"{pos.get('symbol', 'UNKNOWN')}: {str(pos_error)}")
                    print(f"❌ Erro fechando {pos.get('symbol', 'UNKNOWN')}: {pos_error}")
            
            return {
                "success": len(failed_positions) == 0,
                "closed_positions": closed_positions,
                "failed_positions": failed_positions,
                "message": f"Fechadas: {len(closed_positions)}, Falharam: {len(failed_positions)}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_position_info(self):
        """Informações da posição atual - CORRIGIDO"""
        try:
            positions = self.exchange.fetch_positions(['ETHUSDT'])
            balance = self.get_balance()
            
            eth_positions = []
            if positions:
                for pos in positions:
                    if pos.get('symbol') == 'ETHUSDT' and abs(float(pos.get('size', 0))) > 0:
                        eth_positions.append(pos)
            
            return {
                'balance': balance,
                'positions': eth_positions,  # Pode ter LONG e SHORT simultâneos
                'eth_price': self.get_eth_price()
            }
        except Exception as e:
            print(f"❌ Erro ao pegar informações: {e}")
            return {
                'balance': {'free': 0, 'used': 0, 'total': 0},
                'positions': [],
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
                print("✅ Conexão com Bitget OK - SUPORTE LONG/SHORT")
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
        """Compatibilidade com código antigo - SUPORTE LONG/SHORT"""
        if side.lower() == 'buy' or side.lower() == 'long':
            return self.place_buy_order()
        elif side.lower() == 'sell' or side.lower() == 'short':
            return self.place_short_order()
        else:
            return {"success": False, "error": f"Side inválido: {side}"}

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
            print("🔥 API funcionando com SUPORTE COMPLETO LONG/SHORT!")
            
            # Teste básico
            balance = api.get_balance()
            price = api.get_eth_price()
            market = api.get_market_data('ETHUSDT')
            
            print(f"💰 Saldo livre: ${balance.get('free', 0):.4f}")
            print(f"📈 Preço ETH: ${price:.2f}")
            print(f"📊 Dados de mercado: OK")
            
            # Teste de posições
            position_info = api.get_position_info()
            positions = position_info.get('positions', [])
            print(f"🔍 Posições ativas: {len(positions)}")
            
            for pos in positions:
                symbol = pos.get('symbol', 'UNKNOWN')
                side = pos.get('side', 'unknown')
                size = pos.get('size', 0)
                print(f"   {symbol}: {side} {size}")
            
            print("\n✅ RECURSOS DISPONÍVEIS:")
            print("   - place_buy_order() -> Abrir LONG")
            print("   - place_short_order() -> Abrir SHORT") 
            print("   - place_sell_order() -> Fechar LONG")
            print("   - close_short_position() -> Fechar SHORT")
            print("   - force_close_all_positions() -> Emergência")
            
        else:
            print("❌ Falha na conexão")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
