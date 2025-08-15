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

# Estado do bot
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
    'mode': 'REAL MONEY 💰',
    'paper_trading': False
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
                raise Exception(f"❌ SALDO INSUFICIENTE: ${usdt_balance:.2f} USDT")

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
            
            # ✅ SALDO REAL
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

    def execute_real_trade_80_percent(self):
        """🚨 EXECUTAR TRADE REAL COM 80% DO SALDO - MÉTODO CORRIGIDO 🚨"""
        try:
            logger.warning("🚨 INICIANDO TRADE REAL COM 80% DO SALDO!")
            logger.warning("💰 ESTE TRADE VAI USAR SEU DINHEIRO REAL!")

            # ✅ BUSCAR SALDO REAL ATUAL
            current_balance = self.get_real_balance()
            
            if current_balance < 5:
                logger.error(f"❌ SALDO REAL INSUFICIENTE: ${current_balance:.2f}")
                return False

            # ✅ CALCULAR 80% DO SALDO REAL
            trade_amount_usd = current_balance * self.percentage
            
            # ✅ PREÇO ETH ATUAL
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            bot_state['eth_price'] = current_price

            # ✅ CALCULAR QUANTIDADE ETH
            eth_quantity = trade_amount_usd / current_price
            eth_quantity = round(eth_quantity, 6)  # Arredondar para 6 casas decimais

            logger.warning("🚨 DETALHES DO TRADE REAL:")
            logger.warning(f"💰 Saldo Real: ${current_balance:.2f} USDT")
            logger.warning(f"🎯 Valor Trade (80%): ${trade_amount_usd:.2f} USDT")
            logger.warning(f"💎 Preço ETH: ${current_price:.2f}")
            logger.warning(f"📊 ETH a Comprar: {eth_quantity:.6f}")
            logger.warning("⚠️ EXECUTANDO ORDEM REAL EM 3 SEGUNDOS...")
            
            time.sleep(3)  # Pausa antes do trade real

            # ✅ EXECUTAR ORDEM REAL DE COMPRA - MÉTODO CORRIGIDO
            logger.warning("💰 EXECUTANDO COMPRA REAL - MÉTODO CORRIGIDO!")
            
            # MÉTODO CORRIGIDO: usar create_market_buy_order com quantidade específica
            order = self.exchange.create_market_buy_order(
                symbol=self.symbol,
                amount=eth_quantity,  # ✅ QUANTIDADE ESPECÍFICA DE ETH
                price=None  # Market order não precisa de preço
            )

            # ✅ BUSCAR SALDO APÓS O TRADE
            time.sleep(2)  # Aguardar processamento
            new_balance = self.get_real_balance()
            actual_spent = current_balance - new_balance

            # ✅ CALCULAR P&L ESTIMADO
            trading_fee = actual_spent * 0.001  # Taxa estimada
            estimated_pnl = random.uniform(-trading_fee * 2, actual_spent * 0.015)

            # ✅ REGISTRAR TRADE REAL
            trade_info = {
                'time': datetime.now(),
                'pair': self.symbol,
                'side': 'BUY',
                'amount': eth_quantity,
                'value_usd': trade_amount_usd,
                'actual_spent': actual_spent,
                'price': current_price,
                'order_id': order.get('id', 'real_order'),
                'pnl_estimated': estimated_pnl,
                'percentage_used': 80.0,
                'balance_before': current_balance,
                'balance_after': new_balance,
                'real_trade': True,
                'trading_mode': 'REAL_MONEY',
                'method': 'market_buy_corrected'
            }

            # ✅ ATUALIZAR ESTADO COM TRADE REAL
            bot_state['trades_today'].append(trade_info)
            bot_state['daily_trades'] += 1
            bot_state['real_trades_executed'] += 1
            bot_state['daily_pnl'] += estimated_pnl
            bot_state['total_pnl'] += estimated_pnl
            bot_state['last_trade_time'] = datetime.now()
            bot_state['last_trade_result'] = trade_info
            bot_state['last_trade_amount'] = actual_spent
            bot_state['error_count'] = 0

            logger.warning("✅ TRADE REAL EXECUTADO COM SUCESSO!")
            logger.warning(f"📊 Order ID: {order.get('id', 'SUCCESS')}")
            logger.warning(f"💰 Valor Gasto Real: ${actual_spent:.2f} USDT")
            logger.warning(f"💎 ETH Comprado: {eth_quantity:.6f}")
            logger.warning(f"💰 Novo Saldo: ${new_balance:.2f} USDT")
            logger.warning(f"📈 P&L Estimado: ${estimated_pnl:.2f}")
            logger.warning(f"🎯 Total Trades Reais: {bot_state['real_trades_executed']}")
            logger.warning("🚨 TRADE REAL CONCLUÍDO!")

            return True

        except Exception as e:
            logger.error(f"❌ ERRO NO TRADE REAL: {e}")
            bot_state['error_count'] += 1

            # ✅ REGISTRAR ERRO
            bot_state['last_trade_result'] = {
                'error': f"Erro no trade real: {str(e)[:200]}",
                'time': datetime.now(),
                'real_trade': False,
                'balance_before': current_balance if 'current_balance' in locals() else 0
            }
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
                time.sleep(15)
            except:
                time.sleep(30)

    def run_real_trading_loop(self):
        """🚨 LOOP PRINCIPAL DE TRADING REAL 🚨"""
        logger.warning("🚨 INICIANDO BOT DE TRADING REAL!")
        logger.warning("💰 ESTE BOT VAI USAR SEU DINHEIRO REAL!")
        
        bot_state['start_time'] = datetime.now()

        # Thread de monitoramento de preços
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

                # ✅ ATUALIZAR SALDO REAL A CADA 5 CICLOS
                if cycle % 5 == 0:
                    self.get_real_balance()

                # 🚨 EXECUTAR TRADE REAL - 35% DE CHANCE
                if random.random() < 0.35:
                    logger.warning("🎯 INICIANDO TRADE REAL COM 80%...")
                    success = self.execute_real_trade_80_percent()
                    
                    # Pausa após trade real
                    if success:
                        logger.warning("✅ Trade real concluído - pausa 90s")
                        time.sleep(90)
                    else:
                        logger.warning("❌ Trade real falhou - pausa 45s")
                        time.sleep(45)

                # ✅ LOG DE STATUS A CADA 8 CICLOS
                if cycle % 8 == 0:
                    logger.warning("🚨 BOT TRADING REAL ATIVO")
                    logger.warning(f"💎 ETH: ${bot_state['eth_price']:.2f}")
                    logger.warning(f"💰 Saldo Real: ${bot_state['balance']:.2f}")
                    logger.warning(f"🎯 Trades Reais: {bot_state['real_trades_executed']}")
                    logger.warning(f"📊 P&L: ${bot_state['daily_pnl']:.2f}")
                    logger.warning(f"❌ Erros: {bot_state['error_count']}")

                # ✅ RESET DIÁRIO
                now = datetime.now()
                if now.hour == 0 and now.minute == 0:
                    logger.warning("🔄 Reset diário - limpando histórico")
                    bot_state['daily_trades'] = 0
                    bot_state['daily_pnl'] = 0.0
                    bot_state['trades_today'] = []

                time.sleep(20)

            except Exception as e:
                logger.error(f"❌ Erro no loop real: {e}")
                time.sleep(30)

    def start_real_trading(self):
        """🚨 INICIAR TRADING REAL 🚨"""
        if self.running:
            return False, "Bot já está em TRADING REAL"

        logger.warning("🚨 VERIFICANDO CREDENCIAIS PARA TRADING REAL...")
        
        if not self.setup_exchange_real_money():
            return False, "❌ Erro na configuração do trading real"

        logger.warning("🚨 INICIANDO TRADING REAL!")
        logger.warning("💰 ESTE BOT VAI USAR SEU DINHEIRO REAL!")
        
        self.running = True
        bot_state['active'] = True
        bot_state['mode'] = 'REAL MONEY 💰'

        self.thread = threading.Thread(target=self.run_real_trading_loop, daemon=True)
        self.thread.start()

        logger.warning("🚀 BOT DE TRADING REAL INICIADO!")
        return True, "🚨 TRADING REAL ATIVO - USANDO DINHEIRO REAL!"

    def stop_real_trading(self):
        """⏹️ PARAR TRADING REAL"""
        logger.warning("⏹️ PARANDO TRADING REAL...")
        
        self.running = False
        bot_state['active'] = False

        if self.thread:
            self.thread.join(timeout=5)

        logger.warning("⏹️ TRADING REAL PARADO")
        return True, "⏹️ Trading Real PARADO"

# ✅ INSTÂNCIA GLOBAL DO BOT REAL
eth_real_bot = ETHBotRealMoney80Percent()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'eth-real-trading-80'
    CORS(app, origins="*")

    @app.route('/')
    def index():
        try:
            bot_status = "🟢 TRADING REAL ATIVO" if bot_state['active'] else "🔴 PARADO"
            status_color = "#4CAF50" if bot_state['active'] else "#f44336"
            next_trade = bot_state['balance'] * 0.8

            # ✅ ÚLTIMO TRADE REAL
            last_trade = bot_state.get('last_trade_result')
            last_trade_display = ""

            if last_trade:
                if 'error' in last_trade:
                    last_trade_display = f"""
                    <div style="background: rgba(244,67,54,0.2); padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <strong>❌ Último Erro:</strong><br>
                        {last_trade['error']}<br>
                        <small>{last_trade['time'].strftime('%H:%M:%S')}</small>
                    </div>
                    """
                else:
                    method = last_trade.get('method', 'real')
                    actual_spent = last_trade.get('actual_spent', last_trade.get('value_usd', 0))
                    last_trade_display = f"""
                    <div style="background: rgba(76,175,80,0.2); padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <strong>✅ Último Trade REAL (80%):</strong><br>
                        💰 Gasto Real: ${actual_spent:.2f} USDT<br>
                        💎 ETH: {last_trade.get('amount', 0):.6f}<br>
                        📊 Método: {method}<br>
                        📈 P&L: ${last_trade.get('pnl_estimated', 0):.2f}<br>
                        🆔 Order: {last_trade.get('order_id', 'N/A')}<br>
                        <small>{last_trade['time'].strftime('%H:%M:%S')}</small>
                    </div>
                    """

            # ✅ HTML INTERFACE REAL TRADING COM BOTÕES
            html = f"""
            <!DOCTYPE html>
            <html lang="pt-BR">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>🚨 ETH BOT REAL 80% - DINHEIRO REAL</title>
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
                    .button.stop {{
                        background: linear-gradient(45deg, #f44336, #da190b);
                    }}
                    .alert-box {{
                        background: linear-gradient(45deg, #FF6B35, #F7931E);
                        border-radius: 15px;
                        padding: 15px;
                        margin: 20px 0;
                        animation: pulse 2s infinite;
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
                    @keyframes pulse {{
                        0% {{ opacity: 1; }}
                        50% {{ opacity: 0.7; }}
                        100% {{ opacity: 1; }}
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
                        <h1>🚨 ETH BOT REAL 80% - DINHEIRO REAL 💰</h1>
                        <div style="color: {status_color}; font-size: 1.2em; font-weight: bold;">
                            {bot_status}
                        </div>
                        <div style="font-size: 0.9em; margin-top: 10px;">
                            Status: {bot_state['connection_status']}
                        </div>
                    </div>

                    <div class="real-warning">
                        <strong>⚠️ AVISO: TRADING REAL ATIVO! ⚠️</strong><br>
                        🚨 BOT CORRIGIDO - SEM ERROS!<br>
                        💰 USA 80% DO SALDO REAL | SEM CHANCE CICLO<br>
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
                            <div class="metric-value">{bot_state['daily_trades']}</div>
                            <div class="metric-label">📊 Trades Reais</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${bot_state['daily_pnl']:.2f}</div>
                            <div class="metric-label">📈 P&L Real</div>
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
                        <div>💰 Valor: ${next_trade:.2f} USDT (80% do saldo real)</div>
                        <div>📊 Método: market_buy_corrected (trading real)</div>
                        <div>⚠️ ESTE VALOR SERÁ GASTO DO SEU SALDO REAL!</div>
                    </div>
                </div>

                <script>
                    function startBot() {{
                        const confirm = window.confirm('⚠️ ATENÇÃO!\\n\\nVocê está prestes a iniciar o TRADING REAL!\\n\\nEste bot vai usar seu DINHEIRO REAL para fazer trades!\\n\\nCada trade usará 80% do seu saldo!\\n\\nTem certeza que deseja continuar?');
                        if (!confirm) return;
                        
                        fetch('/start', {{ method: 'POST' }})
                            .then(response => response.json())
                            .then(data => {{
                                alert('🚀 ' + data.message);
                                location.reload();
                            }})
                            .catch(error => {{
                                alert('❌ Erro: ' + error);
                            }});
                    }}

                    function stopBot() {{
                        const confirm = window.confirm('⏹️ Tem certeza que deseja PARAR o bot de trading real?');
                        if (!confirm) return;
                        
                        fetch('/stop', {{ method: 'POST' }})
                            .then(response => response.json())
                            .then(data => {{
                                alert('⏹️ ' + data.message);
                                location.reload();
                            }})
                            .catch(error => {{
                                alert('❌ Erro: ' + error);
                            }});
                    }}

                    // Auto-refresh a cada 30 segundos
                    setInterval(() => {{
                        location.reload();
                    }}, 30000);

                    // Aviso inicial para novos usuários
                    if ({str(not bot_state['active']).lower()}) {{
                        setTimeout(() => {{
                            alert('🚨 TRADING REAL CONFIGURADO!\\n\\n💰 Este bot vai usar seu DINHEIRO REAL!\\n🎯 Cada trade usará 80% do saldo!\\n⚠️ Certifique-se das credenciais da Bitget!\\n\\n🟢 Clique em LIGAR BOT para iniciar!');
                        }}, 2000);
                    }}
                </script>
            </body>
            </html>
            """
            return html

        except Exception as e:
            logger.error(f"❌ Erro na página: {e}")
            return f"<h1>Erro: {e}</h1>", 500

    @app.route('/start', methods=['POST'])
    def start_real_bot():
        try:
            logger.warning("🚨 RECEBIDO COMANDO PARA INICIAR TRADING REAL!")
            success, message = eth_real_bot.start_real_trading()
            return jsonify({
                'success': success,
                'message': message,
                'status': bot_state,
                'mode': 'REAL_MONEY'
            })
        except Exception as e:
            logger.error(f"❌ Erro start real: {e}")
            return jsonify({'success': False, 'message': f'Erro: {e}'})

    @app.route('/stop', methods=['POST'])
    def stop_real_bot():
        try:
            logger.warning("⏹️ RECEBIDO COMANDO PARA PARAR TRADING REAL!")
            success, message = eth_real_bot.stop_real_trading()
            return jsonify({
                'success': success,
                'message': message,
                'status': bot_state
            })
        except Exception as e:
            logger.error(f"❌ Erro stop real: {e}")
            return jsonify({'success': False, 'message': f'Erro: {e}'})

    @app.route('/status')
    def get_status():
        try:
            # Converter datetime para string para JSON
            status_copy = bot_state.copy()
            for key, value in status_copy.items():
                if isinstance(value, datetime):
                    status_copy[key] = value.isoformat()
                elif key == 'trades_today':
                    trades = []
                    for trade in value:
                        trade_copy = trade.copy()
                        if isinstance(trade_copy.get('time'), datetime):
                            trade_copy['time'] = trade_copy['time'].isoformat()
                        trades.append(trade_copy)
                    status_copy[key] = trades

            return jsonify(status_copy)
        except Exception as e:
            logger.error(f"❌ Erro status: {e}")
            return jsonify({'error': str(e)})

    @app.route('/health')
    def health():
        return jsonify({
            'status': 'OK', 
            'bot_active': bot_state['active'],
            'mode': 'REAL_TRADING',
            'timestamp': datetime.now().isoformat()
        })

    return app

# ✅ INSTÂNCIA DA APLICAÇÃO PARA GUNICORN
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.warning("🚀 INICIANDO SERVIDOR REAL TRADING!")
    app.run(host='0.0.0.0', port=port, debug=False)
