import os
import sys
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import ccxt
import threading
import time
import random
import json

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# ⚠️ TRADING REAL FUTURES - CONFIGURAÇÕES IMPORTANTES ⚠️
PAPER_TRADING = False  # ❌ FALSE = SALDO REAL
REAL_MONEY_MODE = True  # ✅ TRUE = USAR DINHEIRO REAL
LEVERAGE = 5  # 🚨 ALAVANCAGEM 5x - CUIDADO!

# Variáveis de ambiente para TRADING REAL
api_key = os.environ.get('BITGET_API_KEY', '').strip()
secret_key = os.environ.get('BITGET_API_SECRET', '').strip()
passphrase = os.environ.get('BITGET_PASSPHRASE', '').strip()

logger.warning("🚨 ⚠️ MODO TRADING REAL FUTURES ATIVADO ⚠️ 🚨")
logger.warning("💰 ESTE BOT VAI USAR SEU DINHEIRO REAL!")
logger.warning(f"🎯 80% DO SALDO + ALAVANCAGEM {LEVERAGE}x!")
logger.warning("⚠️ RISCO DE LIQUIDAÇÃO ALTO!")
logger.info(f"🔍 Credenciais REAL: API={bool(api_key)} SECRET={bool(secret_key)} PASS={bool(passphrase)}")

# Estado do bot - RESETADO PARA ZERO
bot_state = {
    'active': False,
    'balance': 0.0,
    'daily_trades': 0,
    'total_trades': 0,
    'daily_pnl': 0.0,
    'total_pnl': 0.0,
    'last_update': datetime.now(),
    'start_time': None,
    'uptime_hours': 0,
    'connection_status': 'Desconectado',
    'last_trade_time': None,
    'trades_today': [],
    'real_trades_executed': 0,
    'last_trade_result': None,
    'error_count': 0,
    'eth_price': 0.0,
    'eth_change_24h': 0.0,
    'last_price_update': None,
    'percentage_used': 80.0,
    'last_trade_amount': 0.0,
    'mode': f'FUTURES {LEVERAGE}x 💰',
    'paper_trading': False,
    'verified_real_trades': 0,
    'last_error': None,
    'leverage': LEVERAGE,
    'trading_type': 'futures'
}

class ETHBotFutures80Percent:
    def __init__(self):
        self.exchange = None
        self.running = False
        self.thread = None
        self.price_thread = None
        self.symbol = 'ETH/USDT'  # Futures symbol
        self.percentage = 0.80  # 80% do saldo
        self.leverage = LEVERAGE
        self.real_trading = True

    def setup_exchange_futures_real_money(self):
        """🚨 SETUP EXCHANGE PARA FUTURES REAL 🚨"""
        try:
            if not api_key or not secret_key or not passphrase:
                raise Exception("❌ CREDENCIAIS OBRIGATÓRIAS PARA TRADING REAL!")

            logger.warning("🚨 CONFIGURANDO EXCHANGE PARA FUTURES REAL!")
            
            # ✅ CONFIGURAÇÃO FUTURES REAL - SEM SANDBOX
            self.exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': secret_key,
                'password': passphrase,
                'sandbox': False,  # ✅ FALSE = TRADING REAL
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',  # 🚨 FUTURES/SWAP
                    'createMarketBuyOrderRequiresPrice': False,
                    'adjustForTimeDifference': True
                },
                'timeout': 30000
            })

            # ✅ DEFINIR ALAVANCAGEM
            logger.warning(f"🚨 DEFININDO ALAVANCAGEM {self.leverage}x!")
            try:
                self.exchange.set_leverage(self.leverage, self.symbol)
                logger.warning(f"✅ ALAVANCAGEM {self.leverage}x DEFINIDA!")
            except Exception as lev_error:
                logger.warning(f"⚠️ Erro definir alavancagem: {lev_error}")

            # ✅ TESTE CONEXÃO COM SALDO REAL
            logger.warning("💰 BUSCANDO SALDO FUTURES REAL...")
            balance = self.exchange.fetch_balance({'type': 'swap'})
            ticker = self.exchange.fetch_ticker(self.symbol)
            
            # ✅ SALDO REAL USDT FUTURES
            usdt_balance = balance.get('USDT', {}).get('free', 0.0)
            
            if usdt_balance < 1:
                logger.warning(f"⚠️ SALDO BAIXO: ${usdt_balance:.2f} USDT")

            bot_state['eth_price'] = ticker['last']
            bot_state['balance'] = usdt_balance

            logger.warning("✅ CONECTADO AO FUTURES REAL!")
            logger.warning(f"💰 SALDO FUTURES: ${usdt_balance:.2f} USDT")
            logger.warning(f"💎 PREÇO ETH: ${ticker['last']:.2f}")
            logger.warning(f"🎯 80% DISPONÍVEL: ${usdt_balance * 0.8:.2f} USDT")
            logger.warning(f"🚨 ALAVANCAGEM: {self.leverage}x")
            logger.warning(f"💥 PODER DE COMPRA: ${usdt_balance * 0.8 * self.leverage:.2f} USDT")
            logger.warning("🚨 PRÓXIMO TRADE USARÁ DINHEIRO REAL!")

            bot_state['connection_status'] = f'💰 CONECTADO - FUTURES {self.leverage}x'
            return True

        except Exception as e:
            logger.error(f"❌ ERRO CONEXÃO FUTURES: {e}")
            bot_state['connection_status'] = f'❌ Erro Futures: {str(e)}'
            return False

    def get_real_futures_balance(self):
        """💰 BUSCAR SALDO REAL FUTURES"""
        try:
            logger.info("💰 Buscando saldo futures real...")
            balance = self.exchange.fetch_balance({'type': 'swap'})
            
            usdt_free = balance.get('USDT', {}).get('free', 0.0)
            usdt_used = balance.get('USDT', {}).get('used', 0.0)
            usdt_total = balance.get('USDT', {}).get('total', 0.0)
            
            bot_state['balance'] = usdt_free
            
            logger.info(f"💰 Saldo Livre: ${usdt_free:.2f}")
            logger.info(f"🔒 Saldo Usado: ${usdt_used:.2f}")
            logger.info(f"📊 Saldo Total: ${usdt_total:.2f}")
            logger.info(f"💥 Poder Compra: ${usdt_free * 0.8 * self.leverage:.2f}")
            
            return usdt_free
        except Exception as e:
            logger.error(f"❌ Erro buscar saldo futures: {e}")
            return bot_state['balance']

    def execute_FUTURES_trade_with_leverage(self):
        """🚨 EXECUTAR TRADE FUTURES COM ALAVANCAGEM 🚨"""
        try:
            logger.warning("🚨 INICIANDO TRADE FUTURES COM ALAVANCAGEM!")

            # ✅ BUSCAR SALDO REAL ATUAL
            current_balance = self.get_real_futures_balance()
            
            if current_balance < 5:  # Mínimo $5 para trade futures
                logger.warning(f"⚠️ SALDO INSUFICIENTE: ${current_balance:.2f} - ABORTANDO TRADE")
                bot_state['last_error'] = f"Saldo insuficiente: ${current_balance:.2f}"
                return False

            # ✅ CALCULAR 80% DO SALDO REAL
            margin_amount = current_balance * self.percentage
            
            # 🚨 CALCULAR VALOR COM ALAVANCAGEM
            trade_value_with_leverage = margin_amount * self.leverage
            
            # ✅ PREÇO ETH ATUAL
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            bot_state['eth_price'] = current_price

            # ✅ CALCULAR QUANTIDADE ETH COM ALAVANCAGEM
            eth_quantity = trade_value_with_leverage / current_price
            eth_quantity = round(eth_quantity, 4)

            # Verificar quantidade mínima
            min_amount = 0.001  # Mínimo para futures
            if eth_quantity < min_amount:
                logger.warning(f"⚠️ QUANTIDADE MUITO PEQUENA: {eth_quantity:.6f} < {min_amount}")
                bot_state['last_error'] = f"Quantidade muito pequena: {eth_quantity:.6f}"
                return False

            logger.warning("🚨 DETALHES DO TRADE FUTURES:")
            logger.warning(f"💰 Saldo Atual: ${current_balance:.2f} USDT")
            logger.warning(f"🎯 Margem (80%): ${margin_amount:.2f} USDT")
            logger.warning(f"🚨 Alavancagem: {self.leverage}x")
            logger.warning(f"💥 Valor Total: ${trade_value_with_leverage:.2f} USDT")
            logger.warning(f"💎 Preço ETH: ${current_price:.2f}")
            logger.warning(f"📊 ETH a Comprar: {eth_quantity:.4f}")
            
            # ✅ EXECUTAR ORDEM FUTURES
            logger.warning("💰 EXECUTANDO ORDEM FUTURES!")
            
            try:
                # MÉTODO 1: Market order futures
                order = self.exchange.create_market_buy_order(
                    symbol=self.symbol,
                    amount=eth_quantity
                )
                
                order_id = order.get('id')
                logger.warning(f"✅ ORDEM FUTURES CRIADA: {order_id}")
                
            except Exception as order_error:
                logger.warning(f"⚠️ Método 1 falhou: {order_error}")
                
                # MÉTODO 2: Create order futures
                try:
                    order = self.exchange.create_order(
                        symbol=self.symbol,
                        type='market',
                        side='buy',
                        amount=eth_quantity,
                        params={'type': 'swap'}
                    )
                    logger.warning(f"✅ MÉTODO 2 SUCESSO: {order.get('id')}")
                    
                except Exception as order_error2:
                    logger.error(f"❌ AMBOS MÉTODOS FALHARAM: {order_error2}")
                    bot_state['last_error'] = f"Falha execução futures: {str(order_error2)[:100]}"
                    return False

            order_id = order.get('id')
            
            # ✅ AGUARDAR PROCESSAMENTO
            time.sleep(5)
            
            # ✅ VERIFICAR EXECUÇÃO FUTURES
            try:
                order_status = self.exchange.fetch_order(order_id, self.symbol)
                
                logger.warning(f"📊 Status: {order_status.get('status')}")
                logger.warning(f"💰 Filled: {order_status.get('filled', 0):.4f} ETH")
                logger.warning(f"💲 Cost: ${order_status.get('cost', 0):.2f} USDT")
                
                if order_status.get('status') == 'closed' and order_status.get('filled', 0) > 0:
                    # ✅ TRADE FUTURES EXECUTADO
                    filled_amount = order_status.get('filled', 0)
                    cost_usd = order_status.get('cost', 0)
                    
                    # ✅ BUSCAR NOVO SALDO
                    time.sleep(3)
                    new_balance = self.get_real_futures_balance()
                    margin_used = current_balance - new_balance
                    
                    # ✅ REGISTRAR TRADE FUTURES
                    trade_info = {
                        'time': datetime.now(),
                        'pair': self.symbol,
                        'side': 'BUY',
                        'amount': filled_amount,
                        'value_usd': cost_usd,
                        'margin_used': margin_used,
                        'leverage': self.leverage,
                        'total_exposure': cost_usd,
                        'price': current_price,
                        'order_id': order_id,
                        'balance_before': current_balance,
                        'balance_after': new_balance,
                        'verified': True,
                        'real_trade': True,
                        'trading_type': 'futures',
                        'exchange_status': order_status.get('status'),
                        'method': 'futures_leveraged'
                    }

                    # ✅ ATUALIZAR CONTADORES
                    bot_state['trades_today'].append(trade_info)
                    bot_state['daily_trades'] += 1
                    bot_state['real_trades_executed'] += 1
                    bot_state['verified_real_trades'] += 1
                    bot_state['total_trades'] += 1
                    bot_state['last_trade_time'] = datetime.now()
                    bot_state['last_trade_result'] = trade_info
                    bot_state['last_trade_amount'] = margin_used
                    bot_state['error_count'] = 0
                    bot_state['last_error'] = None

                    logger.warning("✅ TRADE FUTURES EXECUTADO!")
                    logger.warning(f"📊 Order ID: {order_id}")
                    logger.warning(f"💰 Margem Usada: ${margin_used:.2f} USDT")
                    logger.warning(f"💥 Exposição Total: ${cost_usd:.2f} USDT")
                    logger.warning(f"💎 ETH: {filled_amount:.4f}")
                    logger.warning(f"🎯 Alavancagem: {self.leverage}x")
                    logger.warning(f"💰 Novo Saldo: ${new_balance:.2f} USDT")
                    logger.warning(f"🎯 Total Trades: {bot_state['verified_real_trades']}")

                    return True
                    
                else:
                    logger.warning(f"❌ ORDEM NÃO EXECUTADA: Status={order_status.get('status')}")
                    bot_state['last_error'] = f"Ordem não executada: {order_status.get('status')}"
                    return False
                    
            except Exception as status_error:
                logger.error(f"❌ ERRO VERIFICAR STATUS: {status_error}")
                bot_state['last_error'] = f"Erro verificar status: {str(status_error)[:100]}"
                return False

        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO FUTURES: {e}")
            bot_state['error_count'] += 1
            bot_state['last_error'] = f"Erro crítico futures: {str(e)[:100]}"
            return False

    def update_eth_price(self):
        """Atualizar preço ETH"""
        try:
            if not self.exchange:
                return
            ticker = self.exchange.fetch_ticker(self.symbol)
            bot_state['eth_price'] = ticker['last']
            bot_state['eth_change_24h'] = ticker.get('percentage', 0)
            bot_state['last_price_update'] = datetime.now()
        except Exception as e:
            logger.error(f"❌ Erro atualizar preço: {e}")

    def price_monitoring_loop(self):
        """Loop de monitoramento de preços"""
        while self.running:
            try:
                self.update_eth_price()
                time.sleep(30)
            except:
                time.sleep(60)

    def run_futures_trading_loop(self):
        """🚨 LOOP PRINCIPAL FUTURES 🚨"""
        logger.warning("🚨 LOOP FUTURES TRADING INICIADO!")
        
        bot_state['start_time'] = datetime.now()

        # Thread de preços
        self.price_thread = threading.Thread(target=self.price_monitoring_loop, daemon=True)
        self.price_thread.start()

        cycle = 0

        while self.running:
            try:
                cycle += 1

                # ✅ ATUALIZAR UPTIME
                if bot_state['start_time']:
                    delta = datetime.now() - bot_state['start_time']
                    bot_state['uptime_hours'] = delta.total_seconds() / 3600

                # ✅ ATUALIZAR SALDO A CADA 3 CICLOS
                if cycle % 3 == 0:
                    self.get_real_futures_balance()

                # 🚨 EXECUTAR TRADE FUTURES - 25% DE CHANCE
                if random.random() < 0.25:
                    logger.warning("🎯 TENTANDO TRADE FUTURES...")
                    success = self.execute_FUTURES_trade_with_leverage()
                    
                    if success:
                        logger.warning("✅ TRADE FUTURES EXECUTADO!")
                        time.sleep(300)  # 5 minutos após trade
                    else:
                        logger.warning("❌ TRADE FUTURES FALHOU")
                        time.sleep(120)  # 2 minutos após falha

                # ✅ LOG DE STATUS
                if cycle % 8 == 0:
                    logger.warning("🚨 BOT FUTURES ATIVO")
                    logger.warning(f"💎 ETH: ${bot_state['eth_price']:.2f}")
                    logger.warning(f"💰 Saldo: ${bot_state['balance']:.2f}")
                    logger.warning(f"🎯 Trades: {bot_state['verified_real_trades']}")
                    logger.warning(f"💥 Alavancagem: {self.leverage}x")

                time.sleep(45)

            except Exception as e:
                logger.error(f"❌ Erro no loop futures: {e}")
                time.sleep(60)

    def start_futures_trading(self):
        """🚨 INICIAR FUTURES TRADING 🚨"""
        if self.running:
            return False, "Bot já está ATIVO"

        logger.warning("🚨 RECEBIDO COMANDO PARA INICIAR FUTURES!")
        logger.warning("🚨 VERIFICANDO CREDENCIAIS PARA FUTURES...")

        if not self.setup_exchange_futures_real_money():
            return False, "❌ Erro na configuração futures"

        self.running = True
        bot_state['active'] = True

        self.thread = threading.Thread(target=self.run_futures_trading_loop, daemon=True)
        self.thread.start()

        logger.warning("🚨 INICIANDO FUTURES TRADING!")
        logger.warning("💰 ESTE BOT VAI USAR SEU DINHEIRO REAL!")
        logger.warning(f"💥 COM ALAVANCAGEM {self.leverage}x!")
        logger.warning("🚀 BOT FUTURES INICIADO!")

        return True, f"🚨 BOT ATIVO - FUTURES {self.leverage}x!"

    def stop_futures_trading(self):
        """⏹️ PARAR FUTURES TRADING"""
        self.running = False
        bot_state['active'] = False

        if self.thread:
            self.thread.join(timeout=5)

        logger.warning("⏹️ BOT FUTURES PARADO")
        return True, "⏹️ Bot PARADO"

# ✅ INSTÂNCIA GLOBAL
eth_futures_bot = ETHBotFutures80Percent()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'eth-futures-trading'
    CORS(app, origins="*")

    @app.route('/')
    def index():
        try:
            bot_status = "🟢 FUTURES ATIVO" if bot_state['active'] else "🔴 PARADO"
            status_color = "#4CAF50" if bot_state['active'] else "#f44336"
            margin_amount = bot_state['balance'] * 0.8
            total_exposure = margin_amount * LEVERAGE

            # Último trade
            last_trade = bot_state.get('last_trade_result')
            last_error = bot_state.get('last_error')
            last_trade_display = ""

            if last_trade and last_trade.get('verified'):
                total_exposure_trade = last_trade.get('total_exposure', 0)
                margin_used = last_trade.get('margin_used', 0)
                last_trade_display = f"""
                <div style="background: rgba(76,175,80,0.3); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>✅ Trade FUTURES Executado:</strong><br>
                    💰 Margem: ${margin_used:.2f} USDT<br>
                    💥 Exposição: ${total_exposure_trade:.2f} USDT<br>
                    💎 ETH: {last_trade.get('amount', 0):.4f}<br>
                    🎯 Alavancagem: {last_trade.get('leverage', 0)}x<br>
                    🆔 ID: {last_trade.get('order_id', 'N/A')}<br>
                    <small>{last_trade['time'].strftime('%H:%M:%S')}</small>
                </div>
                """
            elif last_error:
                last_trade_display = f"""
                <div style="background: rgba(255,152,0,0.3); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>⚠️ Último Erro:</strong><br>
                    {last_error}<br>
                    <small>Tentando novamente...</small>
                </div>
                """
            else:
                last_trade_display = """
                <div style="background: rgba(255,193,7,0.3); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>⏳ Nenhum Trade Futures Executado</strong><br>
                    Aguardando execução...
                </div>
                """

            html = f"""
            <!DOCTYPE html>
            <html lang="pt-BR">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>🚨 ETH BOT FUTURES {LEVERAGE}x - ALAVANCAGEM 💰</title>
                <style>
                    body {{
                        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 20px;
                        min-height: 100vh;
                        color: white;
                    }}
                    .container {{
                        max-width: 800px;
                        margin: 0 auto;
                        text-align: center;
                    }}
                    .header {{
                        background: rgba(255,255,255,0.1);
                        border-radius: 15px;
                        padding: 20px;
                        margin-bottom: 20px;
                        backdrop-filter: blur(10px);
                    }}
                    .status-box {{
                        background: rgba(255,255,255,0.15);
                        border-radius: 15px;
                        padding: 20px;
                        margin: 20px 0;
                        backdrop-filter: blur(10px);
                    }}
                    .metrics {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                        gap: 15px;
                        margin: 20px 0;
                    }}
                    .metric {{
                        background: rgba(255,255,255,0.1);
                        border-radius: 10px;
                        padding: 15px;
                        backdrop-filter: blur(5px);
                    }}
                    .metric-value {{
                        font-size: 1.5em;
                        font-weight: bold;
                        color: #FFD700;
                    }}
                    .metric-label {{
                        font-size: 0.9em;
                        opacity: 0.8;
                        margin-top: 5px;
                    }}
                    .button {{
                        background: linear-gradient(45deg, #4CAF50, #45a049);
                        border: none;
                        color: white;
                        padding: 15px 30px;
                        margin: 10px;
                        border-radius: 25px;
                        cursor: pointer;
                        font-size: 16px;
                        font-weight: bold;
                        transition: all 0.3s;
                        text-transform: uppercase;
                    }}
                    .button:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                    }}
                    .button:disabled {{
                        opacity: 0.5;
                        cursor: not-allowed;
                        transform: none;
                    }}
                    .button.stop {{
                        background: linear-gradient(45deg, #f44336, #da190b);
                    }}
                    .futures-warning {{
                        background: linear-gradient(45deg, #FF0000, #CC0000);
                        border-radius: 15px;
                        padding: 20px;
                        margin: 20px 0;
                        animation: blink 1.5s infinite;
                        border: 3px solid #FFD700;
                    }}
                    .controls {{
                        background: rgba(255,255,255,0.1);
                        border-radius: 15px;
                        padding: 25px;
                        margin: 30px 0;
                        backdrop-filter: blur(10px);
                    }}
                    @keyframes blink {{
                        0% {{ opacity: 1; }}
                        50% {{ opacity: 0.8; }}
                        100% {{ opacity: 1; }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚨 ETH BOT FUTURES {LEVERAGE}x - ALAVANCAGEM 💰</h1>
                        <div style="color: {status_color}; font-size: 1.2em; font-weight: bold;">
                            {bot_status}
                        </div>
                        <div style="font-size: 0.9em; margin-top: 10px;">
                            {bot_state['connection_status']}
                        </div>
                    </div>

                    <div class="futures-warning">
                        <strong>⚠️ FUTURES TRADING COM ALAVANCAGEM! ⚠️</strong><br>
                        🚨 ALAVANCAGEM {LEVERAGE}x ATIVA<br>
                        💥 RISCO DE LIQUIDAÇÃO ALTO<br>
                        💰 80% DO SALDO + ALAVANCAGEM<br>
                        <strong>PODE PERDER TUDO RAPIDAMENTE!</strong>
                    </div>

                    <div class="status-box">
                        <h3>💎 ETH/USDT: ${bot_state['eth_price']:.2f}</h3>
                        <div style="color: {'#4CAF50' if bot_state['eth_change_24h'] >= 0 else '#f44336'}">
                            ({bot_state['eth_change_24h']:+.2f}% 24h)
                        </div>
                    </div>

                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-value">${bot_state['balance']:.2f}</div>
                            <div class="metric-label">💰 Saldo Real</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{bot_state['verified_real_trades']}</div>
                            <div class="metric-label">📊 Trades</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${bot_state['daily_pnl']:.2f}</div>
                            <div class="metric-label">📈 P&L</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{LEVERAGE}x</div>
                            <div class="metric-label">💥 Alavancagem</div>
                        </div>
                    </div>

                    {last_trade_display}

                    <div class="controls">
                        <h3>🎮 CONTROLES DO BOT</h3>
                        <button class="button" onclick="startBot()" {'disabled' if bot_state['active'] else ''}>
                            🟢 LIGAR BOT
                        </button>
                        <button class="button stop" onclick="stopBot()" {'disabled' if not bot_state['active'] else ''}>
                            🔴 DESLIGAR BOT
                        </button>
                    </div>

                    <div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; margin-top: 20px;">
                        <h4>🎯 Próximo Trade Futures</h4>
                        <div>💰 Margem: ${margin_amount:.2f} USDT (80% do saldo)</div>
                        <div>💥 Exposição Total: ${total_exposure:.2f} USDT</div>
                        <div>🎯 Alavancagem: {LEVERAGE}x</div>
                        <div>⚠️ RISCO DE LIQUIDAÇÃO!</div>
                    </div>
                </div>

                <script>
                    function startBot() {{
                        if (confirm('⚠️ FUTURES TRADING\\n\\nATENÇÃO: ALAVANCAGEM {LEVERAGE}x!\\nRISCO ALTO DE LIQUIDAÇÃO!\\nPode perder tudo rapidamente!\\n\\nTEM CERTEZA?')) {{
                            fetch('/start', {{ method: 'POST' }})
                                .then(r => r.json())
                                .then(d => {{
                                    alert('🚀 ' + d.message);
                                    location.reload();
                                }})
                                .catch(e => alert('❌ Erro: ' + e));
                        }}
                    }}

                    function stopBot() {{
                        if (confirm('⏹️ Parar futures trading?')) {{
                            fetch('/stop', {{ method: 'POST' }})
                                .then(r => r.json())
                                .then(d => {{
                                    alert('⏹️ ' + d.message);
                                    location.reload();
                                }})
                                .catch(e => alert('❌ Erro: ' + e));
                        }}
                    }}

                    setInterval(() => location.reload(), 45000);
                </script>
            </body>
            </html>
            """
            return html

        except Exception as e:
            logger.error(f"❌ Erro na página: {e}")
            return f"<h1>Erro: {e}</h1>", 500

    @app.route('/start', methods=['POST'])
    def start_bot():
        try:
            success, message = eth_futures_bot.start_futures_trading()
            return jsonify({'success': success, 'message': message})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Erro: {e}'})

    @app.route('/stop', methods=['POST'])
    def stop_bot():
        try:
            success, message = eth_futures_bot.stop_futures_trading()
            return jsonify({'success': success, 'message': message})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Erro: {e}'})

    @app.route('/status')
    def get_status():
        try:
            status_copy = bot_state.copy()
            for key, value in status_copy.items():
                if isinstance(value, datetime):
                    status_copy[key] = value.isoformat()
            return jsonify(status_copy)
        except Exception as e:
            return jsonify({'error': str(e)})

    @app.route('/health')
    def health():
        return jsonify({'status': 'OK', 'active': bot_state['active'], 'leverage': LEVERAGE})

    return app

app = create_app()

if __name__ == '__main__':
    logger.warning("🚨 INICIANDO SERVIDOR FUTURES!")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
