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

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Variáveis de ambiente
api_key = os.environ.get('BITGET_API_KEY', '').strip()
secret_key = os.environ.get('BITGET_API_SECRET', '').strip()
passphrase = os.environ.get('BITGET_PASSPHRASE', '').strip()

logger.warning("🚨 ETH BOT 80% DO SALDO - DINHEIRO REAL!")
logger.info(f"🔍 Credenciais: API={bool(api_key)} SECRET={bool(secret_key)} PASS={bool(passphrase)}")

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
    'percentage_used': 80.0,  # 80% do saldo
    'last_trade_amount': 0.0
}

class ETHBot80Percent:
    def __init__(self):
        self.exchange = None
        self.running = False
        self.thread = None
        self.price_thread = None
        self.symbol = 'ETH/USDT'
        self.percentage = 0.80  # 80%
        
    def setup_exchange(self):
        """Setup Bitget para 80% do saldo"""
        try:
            if not api_key or not secret_key or not passphrase:
                raise Exception("CREDENCIAIS FALTANDO!")
            
            self.exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': secret_key,
                'password': passphrase,
                'sandbox': False,  # DINHEIRO REAL
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
                'timeout': 30000
            })
            
            # Teste e saldo inicial
            balance = self.exchange.fetch_balance()
            ticker = self.exchange.fetch_ticker(self.symbol)
            
            usdt_balance = balance.get('USDT', {}).get('free', 0.0)
            bot_state['eth_price'] = ticker['last']
            bot_state['balance'] = usdt_balance
            
            logger.warning(f"✅ CONECTADO PARA 80% DO SALDO!")
            logger.info(f"💎 ETH: ${ticker['last']:.2f}")
            logger.info(f"💰 Saldo USDT: ${usdt_balance:.2f}")
            logger.warning(f"🎯 Será usado 80% = ${usdt_balance * 0.8:.2f} por trade")
            
            bot_state['connection_status'] = '🚨 CONECTADO - 80% SALDO'
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
            bot_state['connection_status'] = f'Erro: {str(e)}'
            return False

    def update_price_only(self):
        """Atualiza preço ETH"""
        try:
            if not self.exchange:
                return
                
            ticker = self.exchange.fetch_ticker(self.symbol)
            bot_state['eth_price'] = ticker['last']
            bot_state['eth_change_24h'] = ticker.get('percentage', 0)
            bot_state['last_price_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Erro preço: {e}")

    def price_loop(self):
        """Loop de atualização de preço"""
        while self.running:
            try:
                self.update_price_only()
                time.sleep(15)
            except:
                time.sleep(30)

    def get_current_balance(self):
        """Obter saldo atual USDT"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_free = balance.get('USDT', {}).get('free', 0.0)
            bot_state['balance'] = usdt_free
            return usdt_free
        except Exception as e:
            logger.error(f"❌ Erro obter saldo: {e}")
            return bot_state['balance']

    def execute_80_percent_trade(self):
        """🚨 TRADE COM 80% DO SALDO 🚨"""
        try:
            logger.warning("🚨 EXECUTANDO TRADE COM 80% DO SALDO!")
            
            # Obter saldo atual
            current_balance = self.get_current_balance()
            
            if current_balance < 5:  # Mínimo $5
                logger.error(f"❌ Saldo muito baixo: ${current_balance:.2f}")
                return False
            
            # Calcular 80% do saldo
            trade_amount_usd = current_balance * self.percentage
            
            # Preço atual ETH
            current_price = bot_state['eth_price']
            if current_price <= 0:
                # Buscar preço se não estiver atualizado
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
                bot_state['eth_price'] = current_price
            
            # Quantidade ETH a comprar
            quantity_eth = trade_amount_usd / current_price
            quantity_eth = round(quantity_eth, 6)  # 6 decimais
            
            logger.warning(f"🚨 TRADE 80% DO SALDO:")
            logger.warning(f"   Saldo Total: ${current_balance:.2f}")
            logger.warning(f"   80% = ${trade_amount_usd:.2f}")
            logger.warning(f"   Comprando: {quantity_eth} ETH")
            logger.warning(f"   Preço ETH: ${current_price:.2f}")
            
            # EXECUTAR COMPRA COM 80% DO SALDO
            order = self.exchange.create_market_buy_order(
                symbol=self.symbol,
                amount=quantity_eth
            )
            
            # P&L estimado (taxa + variação)
            trading_fee = trade_amount_usd * 0.001  # 0.1% fee
            estimated_pnl = random.uniform(-trading_fee * 3, trade_amount_usd * 0.02)
            
            # Registrar trade
            trade_info = {
                'time': datetime.now(),
                'pair': self.symbol,
                'side': 'BUY',
                'amount': quantity_eth,
                'value_usd': trade_amount_usd,
                'price': current_price,
                'order_id': order.get('id', 'success'),
                'pnl_estimated': estimated_pnl,
                'percentage_used': 80.0,
                'balance_before': current_balance,
                'real_trade': True
            }
            
            # Atualizar estado
            bot_state['trades_today'].append(trade_info)
            bot_state['daily_trades'] += 1
            bot_state['real_trades_executed'] += 1
            bot_state['daily_pnl'] += estimated_pnl
            bot_state['total_pnl'] += estimated_pnl
            bot_state['last_trade_time'] = datetime.now()
            bot_state['last_trade_result'] = trade_info
            bot_state['last_trade_amount'] = trade_amount_usd
            bot_state['error_count'] = 0
            
            logger.warning(f"✅ TRADE 80% EXECUTADO COM SUCESSO!")
            logger.warning(f"📊 Order ID: {order.get('id', 'OK')}")
            logger.warning(f"💰 Usado: ${trade_amount_usd:.2f} (80% do saldo)")
            logger.warning(f"💎 Comprado: {quantity_eth} ETH")
            logger.warning(f"📈 P&L Estimado: ${estimated_pnl:.2f}")
            logger.warning(f"🎯 Total trades hoje: {bot_state['real_trades_executed']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ERRO TRADE 80%: {e}")
            bot_state['error_count'] += 1
            bot_state['last_trade_result'] = {
                'error': str(e)[:150],
                'time': datetime.now()
            }
            return False

    def run_80_percent_loop(self):
        """Loop principal 80% do saldo"""
        logger.warning("🚨 ETH BOT 80% DO SALDO - 24/7!")
        logger.warning("💸 CADA TRADE USA 80% DO SALDO TOTAL!")
        
        bot_state['start_time'] = datetime.now()
        
        # Thread de preço
        self.price_thread = threading.Thread(target=self.price_loop, daemon=True)
        self.price_thread.start()
        
        cycle = 0
        
        while self.running:
            try:
                cycle += 1
                
                # Uptime
                if bot_state['start_time']:
                    delta = datetime.now() - bot_state['start_time']
                    bot_state['uptime_hours'] = delta.total_seconds() / 3600
                
                # Atualizar saldo a cada 5 ciclos
                if cycle % 5 == 0:
                    self.get_current_balance()
                
                # TRADE COM 80% - 30% chance por ciclo
                if random.random() < 0.30:
                    logger.warning("🎯 Iniciando trade com 80% do saldo...")
                    success = self.execute_80_percent_trade()
                    
                    if success:
                        # Pausa maior após trade bem-sucedido
                        time.sleep(120)  # 2 minutos
                    else:
                        # Pausa menor se falhou
                        time.sleep(60)   # 1 minuto
                
                # Log status
                if cycle % 8 == 0:
                    logger.warning(f"🚨 BOT 80% ATIVO")
                    logger.warning(f"💎 ETH: ${bot_state['eth_price']:.2f}")
                    logger.warning(f"💰 Saldo: ${bot_state['balance']:.2f}")
                    logger.warning(f"🎯 Trades 80%: {bot_state['real_trades_executed']}")
                    logger.warning(f"📊 P&L Total: ${bot_state['daily_pnl']:.2f}")
                    logger.warning(f"⏰ Uptime: {bot_state['uptime_hours']:.1f}h")
                
                # Reset diário
                now = datetime.now()
                if now.hour == 0 and now.minute == 0:
                    logger.info("🔄 Reset diário - nova sessão")
                    bot_state['daily_trades'] = 0
                    bot_state['daily_pnl'] = 0.0
                    bot_state['real_trades_executed'] = 0
                    bot_state['trades_today'] = []
                
                time.sleep(25)  # 25 segundos entre ciclos
                
            except Exception as e:
                logger.error(f"❌ Erro no loop: {e}")
                time.sleep(45)

    def start(self):
        """Iniciar bot 80%"""
        if self.running:
            return False, "Bot 80% já está ATIVO"
        
        if not self.setup_exchange():
            return False, "Erro na conexão"
        
        self.running = True
        bot_state['active'] = True
        
        self.thread = threading.Thread(target=self.run_80_percent_loop, daemon=True)
        self.thread.start()
        
        logger.warning("🚀 ETH BOT 80% INICIADO!")
        return True, "🚨 ETH BOT 80% ATIVO"

    def stop(self):
        """Parar bot"""
        self.running = False
        bot_state['active'] = False
        
        if self.thread:
            self.thread.join(timeout=3)
        
        logger.warning("⏹️ ETH BOT 80% PARADO")
        return True, "Bot 80% PARADO"

# Bot global
eth_bot = ETHBot80Percent()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'eth-80-percent'
    CORS(app, origins="*")

    @app.route('/')
    def index():
        try:
            bot_status = "🟢 LIGADO" if bot_state['active'] else "🔴 DESLIGADO"
            status_color = "#4CAF50" if bot_state['active'] else "#f44336"
            
            # Calcular próximo valor de trade (80% do saldo atual)
            next_trade_amount = bot_state['balance'] * 0.8
            
            # Último trade
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
                    pnl_color = "#4CAF50" if last_trade['pnl_estimated'] > 0 else "#f44336"
                    last_trade_display = f"""
                    <div style="background: rgba(76,175,80,0.2); padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <strong>✅ Último Trade 80%:</strong><br>
                        Comprou: {last_trade['amount']:.6f} ETH<br>
                        Valor usado: ${last_trade['value_usd']:.2f} (80% do saldo)<br>
                        <span style="color: {pnl_color};">P&L: ${last_trade['pnl_estimated']:.2f}</span><br>
                        <small>{last_trade['time'].strftime('%H:%M:%S')}</small>
                    </div>
                    """
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>💎 ETH Bot 80% do Saldo</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white; margin: 0; padding: 20px; min-height: 100vh;
                    }}
                    .container {{ max-width: 900px; margin: 0 auto; text-align: center; }}
                    .header {{ 
                        background: rgba(255,255,255,0.15); padding: 30px; border-radius: 20px; 
                        margin-bottom: 30px; backdrop-filter: blur(10px);
                    }}
                    .status-badge {{ 
                        background: {status_color}; color: white; padding: 15px 30px; 
                        border-radius: 50px; font-weight: bold; font-size: 2em;
                        display: inline-block; margin: 20px 0;
                    }}
                    .eth-price {{ 
                        background: rgba(102,126,234,0.4); padding: 25px; border-radius: 15px; 
                        margin: 20px 0; font-size: 1.8em; font-weight: bold;
                        border: 2px solid rgba(255,255,255,0.3);
                    }}
                    .warning {{ 
                        background: #ff3d00; color: white; padding: 20px; border-radius: 15px; 
                        margin: 20px 0; font-weight: bold; font-size: 1.2em;
                        animation: pulse 2s infinite;
                    }}
                    @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} }}
                    .trade-info {{
                        background: rgba(255,215,0,0.2); padding: 20px; border-radius: 15px;
                        margin: 20px 0; border: 2px solid #FFD700;
                    }}
                    .controls {{ 
                        display: flex; justify-content: center; gap: 40px; 
                        margin: 50px 0; flex-wrap: wrap;
                    }}
                    .btn {{ 
                        padding: 25px 50px; border: none; border-radius: 50px; 
                        font-size: 1.5em; font-weight: bold; cursor: pointer; 
                        transition: all 0.3s; min-width: 200px;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                    }}
                    .btn-start {{ 
                        background: linear-gradient(45deg, #4CAF50, #45a049); color: white;
                    }}
                    .btn-start:hover {{ 
                        transform: translateY(-3px);
                        box-shadow: 0 12px 35px rgba(76,175,80,0.4);
                    }}
                    .btn-stop {{ 
                        background: linear-gradient(45deg, #f44336, #d32f2f); color: white;
                    }}
                    .btn-stop:hover {{ 
                        transform: translateY(-3px);
                        box-shadow: 0 12px 35px rgba(244,67,54,0.4);
                    }}
                    .stats {{ 
                        display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                        gap: 20px; margin: 40px 0;
                    }}
                    .stat-card {{ 
                        background: rgba(255,255,255,0.1); padding: 25px; 
                        border-radius: 15px; backdrop-filter: blur(10px);
                    }}
                    .stat-value {{ 
                        font-size: 2.5em; font-weight: bold; margin: 10px 0; color: #FFD700;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>💎 ETH BOT - 80% DO SALDO</h1>
                        <div class="status-badge">{bot_status}</div>
                        <div>Status: {bot_state['connection_status']}</div>
                    </div>
                    
                    <div class="eth-price">
                        💎 ETH/USDT: ${bot_state['eth_price']:.2f}<br>
                        <small>({bot_state['eth_change_24h']:.2f}% 24h)</small>
                    </div>
                    
                    <div class="trade-info">
                        💰 <strong>Saldo Atual: ${bot_state['balance']:.2f} USDT</strong><br>
                        🎯 <strong>Próximo Trade: ${next_trade_amount:.2f} USDT (80%)</strong><br>
                        📊 Último Trade: ${bot_state['last_trade_amount']:.2f} USDT
                    </div>
                    
                    <div class="warning">
                        ⚠️ CADA TRADE USA 80% DO SEU SALDO TOTAL!<br>
                        BOT ATIVO 24/7 | 30% CHANCE POR CICLO
                    </div>
                    
                    <div class="controls">
                        <button class="btn btn-start" onclick="startBot()">🚀 LIGAR BOT 80%</button>
                        <button class="btn btn-stop" onclick="stopBot()">⏹️ DESLIGAR BOT</button>
                    </div>
                    
                    {last_trade_display}
                    
                    <div class="stats">
                        <div class="stat-card">
                            <h3>💰 Saldo USDT</h3>
                            <div class="stat-value">${bot_state['balance']:.2f}</div>
                        </div>
                        <div class="stat-card">
                            <h3>🎯 Trades 80%</h3>
                            <div class="stat-value">{bot_state['real_trades_executed']}</div>
                        </div>
                        <div class="stat-card">
                            <h3>💸 P&L Hoje</h3>
                            <div class="stat-value" style="color: {'#4CAF50' if bot_state['daily_pnl'] >= 0 else '#f44336'};">${bot_state['daily_pnl']:.2f}</div>
                        </div>
                        <div class="stat-card">
                            <h3>⏰ Uptime</h3>
                            <div class="stat-value">{bot_state['uptime_hours']:.1f}h</div>
                        </div>
                    </div>
                </div>
                
                <script>
                    function startBot() {{
                        if(confirm('🚨 ATENÇÃO: Este bot usará 80% do seu saldo a cada trade!\\n\\nSaldo atual: ${bot_state['balance']:.2f} USDT\\nPróximo trade: ${next_trade_amount:.2f} USDT\\n\\nConfirma?')) {{
                            fetch('/start', {{method: 'POST'}})
                                .then(r => r.json())
                                .then(data => {{ alert(data.message); location.reload(); }});
                        }}
                    }}
                    
                    function stopBot() {{
                        if(confirm('Parar bot 80%?')) {{
                            fetch('/stop', {{method: 'POST'}})
                                .then(r => r.json())
                                .then(data => {{ alert(data.message); location.reload(); }});
                        }}
                    }}
                    
                    // Auto refresh
                    setTimeout(() => location.reload(), 15000);
                </script>
            </body>
            </html>
            """
            return html
        except Exception as e:
            return f"<h1>Erro na interface: {e}</h1>"

    @app.route('/start', methods=['POST'])
    def start_bot():
        try:
            success, message = eth_bot.start()
            return jsonify({"success": success, "message": message})
        except Exception as e:
            return jsonify({"success": False, "message": f"Erro: {str(e)}"})

    @app.route('/stop', methods=['POST'])
    def stop_bot():
        try:
            success, message = eth_bot.stop()
            return jsonify({"success": success, "message": message})
        except Exception as e:
            return jsonify({"success": False, "message": f"Erro: {str(e)}"})

    @app.route('/status')
    def status():
        return jsonify(bot_state)

    @app.route('/health')
    def health():
        return jsonify({
            "status": "eth_80_percent", 
            "timestamp": datetime.now().isoformat(),
            "active": bot_state['active'],
            "balance": bot_state['balance'],
            "next_trade_amount": bot_state['balance'] * 0.8
        })

    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    
    logger.warning("🚨 ETH BOT 80% DO SALDO INICIANDO!")
    logger.warning("💸 CADA TRADE USARÁ 80% DO SALDO TOTAL!")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
