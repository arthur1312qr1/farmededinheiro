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

# ⚠️ TRADING REAL - CONFIGURAÇÕES IMPORTANTES ⚠️
PAPER_TRADING = False  # ❌ FALSE = SALDO REAL
REAL_MONEY_MODE = True  # ✅ TRUE = USAR DINHEIRO REAL

# Variáveis de ambiente para TRADING REAL
api_key = os.environ.get('BITGET_API_KEY', '').strip()
secret_key = os.environ.get('BITGET_API_SECRET', '').strip()
passphrase = os.environ.get('BITGET_PASSPHRASE', '').strip()

logger.warning("🚨 ⚠️ MODO TRADING REAL ATIVADO ⚠️ 🚨")
logger.warning("💰 ESTE BOT VAI USAR SEU DINHEIRO REAL!")
logger.warning("🎯 80% DO SALDO SERÁ USADO EM CADA TRADE!")
logger.info(f"🔍 Credenciais REAL: API={bool(api_key)} SECRET={bool(secret_key)} PASS={bool(passphrase)}")

# Estado do bot - RESETADO PARA ZERO
bot_state = {
    'active': False,
    'balance': 0.0,
    'daily_trades': 0,  # ✅ ZERADO - SÓ CONTA TRADES REAIS
    'total_trades': 0,  # ✅ ZERADO - SÓ CONTA TRADES REAIS
    'daily_pnl': 0.0,
    'total_pnl': 0.0,
    'last_update': datetime.now(),
    'start_time': None,
    'uptime_hours': 0,
    'connection_status': 'Desconectado',
    'last_trade_time': None,
    'trades_today': [],
    'real_trades_executed': 0,  # ✅ ZERADO - SÓ TRADES CONFIRMADOS
    'last_trade_result': None,
    'error_count': 0,
    'eth_price': 0.0,
    'eth_change_24h': 0.0,
    'last_price_update': None,
    'percentage_used': 80.0,
    'last_trade_amount': 0.0,
    'mode': 'REAL MONEY 💰',
    'paper_trading': False,
    'verified_real_trades': 0  # ✅ NOVO: TRADES VERIFICADOS PELA EXCHANGE
}

class ETHBotRealMoney80Percent:
    def __init__(self):
        self.exchange = None
        self.running = False
        self.thread = None
        self.price_thread = None
        self.symbol = 'ETH/USDT'
        self.percentage = 0.80  # 80% do saldo
        self.real_trading = True

    def setup_exchange_real_money(self):
        """🚨 SETUP EXCHANGE PARA TRADING REAL 🚨"""
        try:
            if not api_key or not secret_key or not passphrase:
                raise Exception("❌ CREDENCIAIS OBRIGATÓRIAS PARA TRADING REAL!")

            logger.warning("🚨 CONFIGURANDO EXCHANGE PARA DINHEIRO REAL!")
            
            # ✅ CONFIGURAÇÃO REAL - SEM SANDBOX
            self.exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': secret_key,
                'password': passphrase,
                'sandbox': False,  # ✅ FALSE = TRADING REAL
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'createMarketBuyOrderRequiresPrice': False,
                    'adjustForTimeDifference': True
                },
                'timeout': 30000
            })

            # ✅ TESTE CONEXÃO COM SALDO REAL
            logger.warning("💰 BUSCANDO SALDO REAL...")
            balance = self.exchange.fetch_balance()
            ticker = self.exchange.fetch_ticker(self.symbol)
            
            # ✅ SALDO REAL USDT
            usdt_balance = balance.get('USDT', {}).get('free', 0.0)
            
            if usdt_balance < 1:
                logger.warning(f"⚠️ SALDO BAIXO: ${usdt_balance:.2f} USDT")

            bot_state['eth_price'] = ticker['last']
            bot_state['balance'] = usdt_balance

            logger.warning("✅ CONECTADO AO TRADING REAL!")
            logger.warning(f"💰 SALDO REAL: ${usdt_balance:.2f} USDT")
            logger.warning(f"💎 PREÇO ETH: ${ticker['last']:.2f}")
            logger.warning(f"🎯 80% DISPONÍVEL: ${usdt_balance * 0.8:.2f} USDT")
            logger.warning("🚨 PRÓXIMO TRADE USARÁ DINHEIRO REAL!")

            bot_state['connection_status'] = '💰 CONECTADO - MODO REAL'
            return True

        except Exception as e:
            logger.error(f"❌ ERRO CONEXÃO REAL: {e}")
            bot_state['connection_status'] = f'❌ Erro Real: {str(e)}'
            return False

    def get_real_balance(self):
        """💰 BUSCAR SALDO REAL DA CONTA"""
        try:
            logger.info("💰 Buscando saldo real...")
            balance = self.exchange.fetch_balance()
            
            usdt_free = balance.get('USDT', {}).get('free', 0.0)
            usdt_used = balance.get('USDT', {}).get('used', 0.0)
            usdt_total = balance.get('USDT', {}).get('total', 0.0)
            
            bot_state['balance'] = usdt_free
            
            logger.info(f"💰 Saldo Livre: ${usdt_free:.2f}")
            logger.info(f"🔒 Saldo Usado: ${usdt_used:.2f}")
            logger.info(f"📊 Saldo Total: ${usdt_total:.2f}")
            
            return usdt_free
        except Exception as e:
            logger.error(f"❌ Erro buscar saldo real: {e}")
            return bot_state['balance']

    def verify_real_trade_execution(self, order_id):
        """🔍 VERIFICAR SE O TRADE FOI REALMENTE EXECUTADO NA EXCHANGE"""
        try:
            logger.warning(f"🔍 VERIFICANDO EXECUÇÃO REAL DA ORDEM: {order_id}")
            
            # Buscar ordem na exchange
            order = self.exchange.fetch_order(order_id, self.symbol)
            
            if order['status'] == 'closed' and order['filled'] > 0:
                logger.warning(f"✅ TRADE REAL CONFIRMADO!")
                logger.warning(f"💰 Valor Executado: ${order['cost']:.2f} USDT")
                logger.warning(f"💎 ETH Comprado: {order['filled']:.6f}")
                logger.warning(f"📊 Status: {order['status']}")
                return True, order
            else:
                logger.warning(f"❌ TRADE NÃO EXECUTADO: Status={order['status']}")
                return False, order
                
        except Exception as e:
            logger.error(f"❌ ERRO VERIFICAR TRADE: {e}")
            return False, None

    def execute_ONLY_real_trade_80_percent(self):
        """🚨 EXECUTAR APENAS TRADE REAL - SEM SIMULAÇÕES 🚨"""
        try:
            logger.warning("🚨 INICIANDO TRADE REAL - SEM SIMULAÇÕES!")

            # ✅ BUSCAR SALDO REAL ATUAL
            current_balance = self.get_real_balance()
            
            if current_balance < 5:  # Mínimo $5 para trade real
                logger.warning(f"⚠️ SALDO INSUFICIENTE: ${current_balance:.2f} - ABORTANDO TRADE")
                return False

            # ✅ CALCULAR 80% DO SALDO REAL
            trade_amount_usd = current_balance * self.percentage
            
            # ✅ PREÇO ETH ATUAL
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            bot_state['eth_price'] = current_price

            # ✅ CALCULAR QUANTIDADE ETH
            eth_quantity = trade_amount_usd / current_price
            eth_quantity = round(eth_quantity, 6)  # Precisão máxima

            # Verificar quantidade mínima
            if eth_quantity < 0.0001:
                logger.warning(f"⚠️ QUANTIDADE MUITO PEQUENA: {eth_quantity:.6f} ETH - ABORTANDO")
                return False

            logger.warning("🚨 DETALHES DO TRADE REAL:")
            logger.warning(f"💰 Saldo Atual: ${current_balance:.2f} USDT")
            logger.warning(f"🎯 Valor Trade (80%): ${trade_amount_usd:.2f} USDT")
            logger.warning(f"💎 Preço ETH: ${current_price:.2f}")
            logger.warning(f"📊 ETH a Comprar: {eth_quantity:.6f}")
            
            # ✅ EXECUTAR ORDEM REAL
            logger.warning("💰 EXECUTANDO ORDEM REAL NA BITGET!")
            
            order = self.exchange.create_market_buy_order(
                symbol=self.symbol,
                amount=eth_quantity,
                params={'quoteOrderQty': trade_amount_usd}
            )
            
            order_id = order.get('id')
            logger.warning(f"📝 ORDEM CRIADA: {order_id}")
            
            # ✅ AGUARDAR E VERIFICAR EXECUÇÃO
            time.sleep(3)  # Aguardar processamento
            
            # ✅ VERIFICAR SE FOI REALMENTE EXECUTADO
            trade_verified, verified_order = self.verify_real_trade_execution(order_id)
            
            if not trade_verified:
                logger.error("❌ TRADE NÃO FOI EXECUTADO REALMENTE!")
                return False

            # ✅ BUSCAR NOVO SALDO APÓS TRADE
            time.sleep(2)
            new_balance = self.get_real_balance()
            actual_spent = current_balance - new_balance
            
            if actual_spent <= 0:
                logger.error("❌ SALDO NÃO MUDOU - TRADE NÃO EXECUTADO!")
                return False

            # ✅ REGISTRAR TRADE REAL VERIFICADO
            trade_info = {
                'time': datetime.now(),
                'pair': self.symbol,
                'side': 'BUY',
                'amount': verified_order['filled'],
                'value_usd': verified_order['cost'],
                'actual_spent': actual_spent,
                'price': current_price,
                'order_id': order_id,
                'balance_before': current_balance,
                'balance_after': new_balance,
                'verified': True,
                'real_trade': True,
                'exchange_status': verified_order['status']
            }

            # ✅ ATUALIZAR CONTADORES APENAS PARA TRADES REAIS
            bot_state['trades_today'].append(trade_info)
            bot_state['daily_trades'] += 1  # ✅ SÓ INCREMENTA SE REAL
            bot_state['real_trades_executed'] += 1  # ✅ SÓ INCREMENTA SE REAL
            bot_state['verified_real_trades'] += 1  # ✅ SÓ INCREMENTA SE VERIFICADO
            bot_state['total_trades'] += 1  # ✅ SÓ INCREMENTA SE REAL
            bot_state['last_trade_time'] = datetime.now()
            bot_state['last_trade_result'] = trade_info
            bot_state['last_trade_amount'] = actual_spent
            bot_state['error_count'] = 0

            logger.warning("✅ TRADE REAL VERIFICADO E CONTABILIZADO!")
            logger.warning(f"📊 Order ID: {order_id}")
            logger.warning(f"💰 Gasto Real: ${actual_spent:.2f} USDT")
            logger.warning(f"💎 ETH Recebido: {verified_order['filled']:.6f}")
            logger.warning(f"💰 Novo Saldo: ${new_balance:.2f} USDT")
            logger.warning(f"🎯 Total Trades REAIS: {bot_state['verified_real_trades']}")

            return True

        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO NO TRADE: {e}")
            bot_state['error_count'] += 1
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

    def run_real_trading_loop(self):
        """🚨 LOOP PRINCIPAL - APENAS TRADES REAIS 🚨"""
        logger.warning("🚨 LOOP TRADING REAL INICIADO!")
        
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

                # ✅ ATUALIZAR SALDO A CADA CICLO
                if cycle % 3 == 0:
                    self.get_real_balance()

                # 🚨 EXECUTAR TRADE REAL - 20% DE CHANCE
                if random.random() < 0.20:
                    logger.warning("🎯 TENTANDO TRADE REAL 80%...")
                    success = self.execute_ONLY_real_trade_80_percent()
                    
                    if success:
                        logger.warning("✅ TRADE REAL EXECUTADO COM SUCESSO!")
                        time.sleep(300)  # 5 minutos após trade real
                    else:
                        logger.warning("❌ TRADE NÃO EXECUTADO")
                        time.sleep(120)  # 2 minutos após falha

                # ✅ LOG DE STATUS
                if cycle % 8 == 0:
                    logger.warning("🚨 BOT TRADING REAL ATIVO")
                    logger.warning(f"💎 ETH: ${bot_state['eth_price']:.2f}")
                    logger.warning(f"💰 Saldo: ${bot_state['balance']:.2f}")
                    logger.warning(f"🎯 Trades REAIS: {bot_state['verified_real_trades']}")

                time.sleep(45)  # 45 segundos entre ciclos

            except Exception as e:
                logger.error(f"❌ Erro no loop: {e}")
                time.sleep(60)

    def start_real_trading(self):
        """🚨 INICIAR TRADING REAL 🚨"""
        if self.running:
            return False, "Bot já está ATIVO"

        logger.warning("🚨 RECEBIDO COMANDO PARA INICIAR TRADING REAL!")
        logger.warning("🚨 VERIFICANDO CREDENCIAIS PARA TRADING REAL...")

        if not self.setup_exchange_real_money():
            return False, "❌ Erro na configuração"

        self.running = True
        bot_state['active'] = True

        self.thread = threading.Thread(target=self.run_real_trading_loop, daemon=True)
        self.thread.start()

        logger.warning("🚨 INICIANDO TRADING REAL!")
        logger.warning("💰 ESTE BOT VAI USAR SEU DINHEIRO REAL!")
        logger.warning("🚨 INICIANDO BOT DE TRADING REAL!")
        logger.warning("💰 ESTE BOT VAI USAR SEU DINHEIRO REAL!")
        logger.warning("🚀 BOT DE TRADING REAL INICIADO!")

        return True, "🚨 BOT ATIVO - TRADING REAL!"

    def stop_real_trading(self):
        """⏹️ PARAR TRADING REAL"""
        self.running = False
        bot_state['active'] = False

        if self.thread:
            self.thread.join(timeout=5)

        logger.warning("⏹️ BOT PARADO")
        return True, "⏹️ Bot PARADO"

# ✅ INSTÂNCIA GLOBAL
eth_real_bot = ETHBotRealMoney80Percent()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'eth-trading-real'
    CORS(app, origins="*")

    @app.route('/')
    def index():
        try:
            bot_status = "🟢 TRADING REAL ATIVO" if bot_state['active'] else "🔴 PARADO"
            status_color = "#4CAF50" if bot_state['active'] else "#f44336"
            next_trade = bot_state['balance'] * 0.8

            # Último trade REAL
            last_trade = bot_state.get('last_trade_result')
            last_trade_display = ""

            if last_trade and last_trade.get('verified'):
                actual_spent = last_trade.get('actual_spent', 0)
                last_trade_display = f"""
                <div style="background: rgba(76,175,80,0.3); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>✅ Último Trade REAL VERIFICADO:</strong><br>
                    💰 Gasto Real: ${actual_spent:.2f} USDT<br>
                    💎 ETH: {last_trade.get('amount', 0):.6f}<br>
                    🆔 ID: {last_trade.get('order_id', 'N/A')}<br>
                    📊 Status: {last_trade.get('exchange_status', 'EXECUTADO')}<br>
                    <small>{last_trade['time'].strftime('%H:%M:%S')}</small>
                </div>
                """
            elif not last_trade:
                last_trade_display = """
                <div style="background: rgba(255,193,7,0.3); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>⏳ Nenhum Trade Real Executado Ainda</strong><br>
                    Aguardando execução de trades reais...
                </div>
                """

            # HTML FINAL
            html = f"""
            <!DOCTYPE html>
            <html lang="pt-BR">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>🚨 ETH BOT REAL 80% - APENAS TRADES REAIS 💰</title>
                <style>
                    body {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
                    .real-warning {{
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
                        <h1>🚨 ETH BOT REAL 80% - APENAS TRADES REAIS 💰</h1>
                        <div style="color: {status_color}; font-size: 1.2em; font-weight: bold;">
                            {bot_status}
                        </div>
                        <div style="font-size: 0.9em; margin-top: 10px;">
                            {bot_state['connection_status']}
                        </div>
                    </div>

                    <div class="real-warning">
                        <strong>⚠️ AVISO: TRADING REAL ATIVO! ⚠️</strong><br>
                        🚨 SÓ CONTA TRADES REAIS VERIFICADOS<br>
                        💰 80% DO SALDO REAL | SEM SIMULAÇÕES<br>
                        <strong>ESTE BOT VAI USAR SEU DINHEIRO REAL!</strong>
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
                            <div class="metric-label">📊 Trades REAIS</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${bot_state['daily_pnl']:.2f}</div>
                            <div class="metric-label">📈 P&L</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{bot_state['uptime_hours']:.1f}h</div>
                            <div class="metric-label">⏱️ Uptime</div>
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
                        <h4>🎯 Próximo Trade Real</h4>
                        <div>💰 Valor: ${next_trade:.2f} USDT (80% do saldo)</div>
                        <div>📊 Método: APENAS TRADES REAIS VERIFICADOS</div>
                        <div>⚠️ VALOR SERÁ GASTO DO SEU SALDO REAL!</div>
                    </div>
                </div>

                <script>
                    function startBot() {{
                        if (confirm('⚠️ TRADING REAL\\n\\nEste bot vai usar DINHEIRO REAL!\\nCada trade = 80% do saldo!\\nSÓ EXECUTA TRADES REAIS!\\n\\nContinuar?')) {{
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
                        if (confirm('⏹️ Parar trading real?')) {{
                            fetch('/stop', {{ method: 'POST' }})
                                .then(r => r.json())
                                .then(d => {{
                                    alert('⏹️ ' + d.message);
                                    location.reload();
                                }})
                                .catch(e => alert('❌ Erro: ' + e));
                        }}
                    }}

                    // Auto-refresh
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
            success, message = eth_real_bot.start_real_trading()
            return jsonify({'success': success, 'message': message})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Erro: {e}'})

    @app.route('/stop', methods=['POST'])
    def stop_bot():
        try:
            success, message = eth_real_bot.stop_real_trading()
            return jsonify({'success': success, 'message': message})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Erro: {e}'})

    @app.route('/status')
    def get_status():
        try:
            status_copy = bot_state.copy()
            # Converter datetime para string
            for key, value in status_copy.items():
                if isinstance(value, datetime):
                    status_copy[key] = value.isoformat()
            return jsonify(status_copy)
        except Exception as e:
            return jsonify({'error': str(e)})

    @app.route('/health')
    def health():
        return jsonify({'status': 'OK', 'active': bot_state['active']})

    return app

# Instância para Gunicorn
app = create_app()

if __name__ == '__main__':
    logger.warning("🚀 INICIANDO SERVIDOR REAL TRADING!")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
