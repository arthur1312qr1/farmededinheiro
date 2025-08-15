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
LEVERAGE = 10  # 🚨 ALAVANCAGEM 10x - CUIDADO!

# Variáveis de ambiente para TRADING REAL
api_key = os.environ.get('BITGET_API_KEY', '').strip()
secret_key = os.environ.get('BITGET_API_SECRET', '').strip()
passphrase = os.environ.get('BITGET_PASSPHRASE', '').strip()

logger.warning("🚨 ⚠️ MODO TRADING REAL FUTURES ATIVADO ⚠️ 🚨")
logger.warning("💰 ESTE BOT VAI USAR SEU DINHEIRO REAL!")
logger.warning(f"🎯 QUANTIDADE FIXA + ALAVANCAGEM {LEVERAGE}x!")
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
                self.exchange.set_leverage(self.leverage, self.symbol, params={'marginCoin': 'USDT'})
                logger.warning(f"✅ ALAVANCAGEM {self.leverage}x DEFINIDA!")
            except Exception as lev_error:
                logger.warning(f"⚠️ Erro definir alavancagem: {lev_error}")

            # ✅ TESTE CONEXÃO COM SALDO REAL
            logger.warning("💰 BUSCANDO SALDO FUTURES REAL...")
            balance = self.exchange.fetch_balance({'type': 'swap'})
            ticker = self.exchange.fetch_ticker(self.symbol)

            # ✅ SALDO REAL USDT FUTURES
            usdt_balance = balance.get('USDT', {}).get('free', 0.0)

            if usdt_balance < 2:
                logger.warning(f"⚠️ SALDO BAIXO: ${usdt_balance:.2f} USDT")

            bot_state['eth_price'] = ticker['last']
            bot_state['balance'] = usdt_balance

            logger.warning("✅ CONECTADO AO FUTURES REAL!")
            logger.warning(f"💰 SALDO FUTURES: ${usdt_balance:.2f} USDT")
            logger.warning(f"💎 PREÇO ETH: ${ticker['last']:.2f}")
            logger.warning(f"🚨 ALAVANCAGEM: {self.leverage}x")
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

            return usdt_free
        except Exception as e:
            logger.error(f"❌ Erro buscar saldo futures: {e}")
            return bot_state['balance']

    def execute_FUTURES_trade_with_leverage(self):
        """🚨 EXECUTAR TRADE FUTURES COM QUANTIDADE FIXA 🚨"""
        try:
            logger.warning("🚨 INICIANDO TRADE FUTURES COM QUANTIDADE FIXA!")

            # ✅ BUSCAR SALDO REAL ATUAL
            current_balance = self.get_real_futures_balance()

            # 🚨 VERIFICAR SALDO MÍNIMO
            if current_balance < 1.5:
                logger.warning(f"❌ SALDO INSUFICIENTE: ${current_balance:.2f} USDT")
                bot_state['last_error'] = f"Saldo insuficiente: ${current_balance:.2f} USDT"
                return False

            # ✅ USAR QUANTIDADE FIXA DE ETH QUE ATENDE AOS MÍNIMOS
            eth_quantity = 0.01  # QUANTIDADE FIXA 0.01 ETH (mínimo da Bitget)
            
            # ✅ PREÇO ETH ATUAL
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            bot_state['eth_price'] = current_price

            # ✅ CALCULAR VALOR EM USDT DA ORDEM
            order_value_usdt = eth_quantity * current_price

            logger.warning("🚨 DETALHES DO TRADE FUTURES:")
            logger.warning(f"💰 Saldo Atual: ${current_balance:.2f} USDT")
            logger.warning(f"📊 ETH Fixo: {eth_quantity:.2f} ETH")
            logger.warning(f"💎 Preço ETH: ${current_price:.2f}")
            logger.warning(f"💲 Valor da Ordem: ${order_value_usdt:.2f} USDT")
            logger.warning(f"🚨 Alavancagem: {self.leverage}x")

            # ✅ EXECUTAR ORDEM FUTURES COM QUANTIDADE FIXA
            logger.warning("💰 EXECUTANDO ORDEM FUTURES!")

            try:
                # MÉTODO SIMPLES: Usar quantidade fixa de ETH
                order = self.exchange.create_market_buy_order(
                    symbol=self.symbol,
                    amount=eth_quantity,  # QUANTIDADE FIXA 0.01 ETH
                    params={
                        'type': 'swap',
                        'marginCoin': 'USDT'
                    }
                )

                order_id = order.get('id')
                logger.warning(f"✅ ORDEM FUTURES CRIADA: {order_id}")

            except Exception as order_error:
                logger.error(f"❌ ORDEM FUTURES FALHOU: {order_error}")
                bot_state['last_error'] = f"Falha execução futures: {str(order_error)[:100]}"
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
                        'method': 'fixed_quantity'
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
        logger.warning("🚨 INICIANDO FUTURES TRADING!")
        logger.warning("💰 ESTE BOT VAI USAR SEU DINHEIRO REAL!")
        logger.warning(f"💥 COM ALAVANCAGEM {self.leverage}x!")
        logger.warning("📊 USANDO QUANTIDADE FIXA 0.01 ETH!")

        bot_state['start_time'] = datetime.now()

        # Thread de preços
        self.price_thread = threading.Thread(target=self.price_monitoring_loop)
        self.price_thread.daemon = True
        self.price_thread.start()

        try:
            while self.running:
                logger.warning("🎯 TENTANDO TRADE FUTURES...")

                # ✅ EXECUTAR TRADE
                success = self.execute_FUTURES_trade_with_leverage()

                if success:
                    logger.warning("✅ TRADE EXECUTADO COM SUCESSO!")
                else:
                    logger.warning("❌ TRADE FUTURES FALHOU")

                # Aguardar próximo ciclo
                wait_time = random.randint(120, 300)  # 2-5 minutos
                logger.info(f"⏰ Aguardando {wait_time} segundos...")
                time.sleep(wait_time)

        except KeyboardInterrupt:
            logger.warning("🛑 TRADING INTERROMPIDO PELO USUÁRIO")
        except Exception as e:
            logger.error(f"❌ ERRO NO LOOP TRADING: {e}")
        finally:
            logger.warning("🚨 BOT FUTURES PARADO!")

    def start(self):
        """Iniciar bot"""
        if self.running:
            return False

        # ✅ SETUP EXCHANGE REAL
        if not self.setup_exchange_futures_real_money():
            return False

        self.running = True
        bot_state['active'] = True

        # Thread principal
        self.thread = threading.Thread(target=self.run_futures_trading_loop)
        self.thread.daemon = True
        self.thread.start()

        logger.warning("🚀 BOT FUTURES INICIADO!")
        return True

    def stop(self):
        """Parar bot"""
        self.running = False
        bot_state['active'] = False
        logger.warning("🛑 PARANDO BOT FUTURES...")


# Instância global
bot = ETHBotFutures80Percent()

# Flask app
app = Flask(__name__)
CORS(app)

logger.warning("🚨 INICIANDO SERVIDOR FUTURES!")

@app.route('/')
def index():
    """Dashboard"""
    try:
        # HTML do dashboard
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🚀 BOT FUTURES - QUANTIDADE FIXA</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    background: linear-gradient(135deg, #1e3c72, #2a5298);
                    color: white; 
                    margin: 0; 
                    padding: 20px;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto;
                }}
                .card {{ 
                    background: rgba(255,255,255,0.1); 
                    border-radius: 10px; 
                    padding: 20px; 
                    margin: 10px 0;
                    backdrop-filter: blur(10px);
                }}
                .grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px;
                }}
                .status-active {{ color: #00ff00; }}
                .status-inactive {{ color: #ff6b6b; }}
                .btn {{ 
                    background: #ff6b6b; 
                    color: white; 
                    border: none; 
                    padding: 10px 20px; 
                    border-radius: 5px; 
                    cursor: pointer;
                    font-size: 16px;
                    margin: 5px;
                }}
                .btn:hover {{ background: #ff5252; }}
                .btn-success {{ background: #4caf50; }}
                .btn-success:hover {{ background: #45a049; }}
                .warning {{ 
                    background: rgba(255,193,7,0.2); 
                    border-left: 4px solid #ffc107; 
                    padding: 15px; 
                    margin: 20px 0;
                }}
                .danger {{ 
                    background: rgba(220,53,69,0.2); 
                    border-left: 4px solid #dc3545; 
                    padding: 15px; 
                    margin: 20px 0;
                }}
                h1 {{ text-align: center; font-size: 2.5em; margin-bottom: 30px; }}
                h2 {{ color: #ffc107; }}
                .metric {{ font-size: 1.5em; font-weight: bold; }}
                .refresh {{ margin: 20px 0; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 BOT FUTURES - QUANTIDADE FIXA</h1>
                
                <div class="danger">
                    <h3>⚠️ AVISO IMPORTANTE - TRADING REAL ⚠️</h3>
                    <p>🚨 Este bot está configurado para TRADING REAL com ALAVANCAGEM {LEVERAGE}x</p>
                    <p>📊 O bot usa quantidade fixa de 0.01 ETH por trade</p>
                    <p>⚠️ RISCO DE LIQUIDAÇÃO MUITO ALTO!</p>
                </div>

                <div class="grid">
                    <div class="card">
                        <h2>📊 Status do Bot</h2>
                        <p>Status: <span class="{'status-active' if bot_state['active'] else 'status-inactive'}">{'🟢 ATIVO' if bot_state['active'] else '🔴 INATIVO'}</span></p>
                        <p>Conexão: {bot_state['connection_status']}</p>
                        <p>Modo: {bot_state['mode']}</p>
                        <p>Alavancagem: {bot_state['leverage']}x</p>
                        <p>Tipo: Futures Real Money</p>
                    </div>

                    <div class="card">
                        <h2>💰 Saldo & Trading</h2>
                        <p>Saldo: <span class="metric">${bot_state['balance']:.2f} USDT</span></p>
                        <p>Quantidade Fixa: <span class="metric">0.01 ETH</span></p>
                        <p>Preço ETH: <span class="metric">${bot_state['eth_price']:.2f}</span></p>
                        <p>Valor por Trade: <span class="metric">${bot_state['eth_price'] * 0.01:.2f} USDT</span></p>
                    </div>

                    <div class="card">
                        <h2>📈 Estatísticas</h2>
                        <p>Trades Hoje: <span class="metric">{bot_state['daily_trades']}</span></p>
                        <p>Trades Reais: <span class="metric">{bot_state['real_trades_executed']}</span></p>
                        <p>Total Trades: <span class="metric">{bot_state['total_trades']}</span></p>
                        <p>Última Atividade: {bot_state['last_trade_time'].strftime('%H:%M:%S') if bot_state['last_trade_time'] else 'Nenhuma'}</p>
                    </div>

                    <div class="card">
                        <h2>⚠️ Últimos Eventos</h2>
                        <p>Último Erro: {bot_state['last_error'] or 'Nenhum'}</p>
                        <p>Contagem Erros: {bot_state['error_count']}</p>
                        <p>Último Trade: ${bot_state['last_trade_amount']:.2f} USDT</p>
                    </div>
                </div>

                <div class="card">
                    <h2>🎮 Controles</h2>
                    <div style="text-align: center;">
                        <button class="btn btn-success" onclick="startBot()">🚀 INICIAR BOT</button>
                        <button class="btn" onclick="stopBot()">🛑 PARAR BOT</button>
                        <button class="btn" onclick="location.reload()">🔄 ATUALIZAR</button>
                    </div>
                </div>

                <div class="refresh">
                    <p>📊 Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                    <p>⚡ Auto-refresh em 30 segundos</p>
                </div>
            </div>

            <script>
                function startBot() {{
                    fetch('/start', {{method: 'POST'}})
                    .then(response => response.json())
                    .then(data => {{
                        alert(data.message);
                        setTimeout(() => location.reload(), 2000);
                    }});
                }}

                function stopBot() {{
                    fetch('/stop', {{method: 'POST'}})
                    .then(response => response.json())
                    .then(data => {{
                        alert(data.message);
                        setTimeout(() => location.reload(), 2000);
                    }});
                }}

                // Auto-refresh
                setTimeout(() => location.reload(), 30000);
            </script>
        </body>
        </html>
        """
        return html
    except Exception as e:
        logger.error(f"❌ Erro carregar dashboard: {e}")
        return f"❌ Erro: {e}"

@app.route('/start', methods=['POST'])
def start_bot():
    """Iniciar trading"""
    try:
        logger.warning("🚨 RECEBIDO COMANDO PARA INICIAR FUTURES!")
        logger.warning("🚨 VERIFICANDO CREDENCIAIS PARA FUTURES...")

        if bot.start():
            return jsonify({
                'success': True,
                'message': '🚀 Bot Futures iniciado com TRADING REAL!'
            })
        else:
            return jsonify({
                'success': False,
                'message': '❌ Falha ao iniciar bot futures'
            })
    except Exception as e:
        logger.error(f"❌ Erro iniciar bot: {e}")
        return jsonify({
            'success': False,
            'message': f'❌ Erro: {str(e)}'
        })

@app.route('/stop', methods=['POST'])
def stop_bot():
    """Parar trading"""
    try:
        bot.stop()
        return jsonify({
            'success': True,
            'message': '🛑 Bot futures parado'
        })
    except Exception as e:
        logger.error(f"❌ Erro parar bot: {e}")
        return jsonify({
            'success': False,
            'message': f'❌ Erro: {str(e)}'
        })

@app.route('/status')
def get_status():
    """Status do bot"""
    return jsonify(bot_state)

@app.route('/health')
def health_check():
    """Health check"""
    return jsonify({
        'status': 'OK',
        'timestamp': datetime.now().isoformat(),
        'bot_active': bot_state['active']
    })

if __name__ == '__main__':
    # Configurar para produção
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
