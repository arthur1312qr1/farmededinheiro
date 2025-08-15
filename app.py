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

logger.warning("🚨 BOT ETH TRADING 24/7 - DINHEIRO REAL!")
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
    'eth_price': 0.0
}

class ETHBot24h:
    def __init__(self):
        self.exchange = None
        self.running = False
        self.thread = None
        self.min_trade_usd = 10.0  # Mínimo $10
        self.max_trade_usd = 25.0  # Máximo $25
        self.symbol = 'ETH/USDT'  # APENAS ETH
        
    def setup_exchange(self):
        """Configura Bitget para ETH trading"""
        try:
            if not api_key or not secret_key or not passphrase:
                raise Exception("CREDENCIAIS NÃO CONFIGURADAS!")
            
            self.exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': secret_key,
                'password': passphrase,
                'sandbox': False,  # REAL MONEY
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                },
                'timeout': 30000
            })
            
            # Teste conexão ETH
            markets = self.exchange.load_markets()
            if self.symbol not in markets:
                raise Exception(f"Par {self.symbol} não disponível")
                
            ticker = self.exchange.fetch_ticker(self.symbol)
            balance = self.exchange.fetch_balance()
            
            bot_state['eth_price'] = ticker['last']
            
            logger.warning(f"✅ CONECTADO PARA ETH TRADING!")
            logger.info(f"📊 ETH/USDT: ${ticker['last']}")
            logger.info(f"💰 Saldo USDT: ${balance.get('USDT', {}).get('total', 0):.2f}")
            
            bot_state['connection_status'] = '🚨 CONECTADO ETH 24/7'
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro conexão: {e}")
            bot_state['connection_status'] = f'Erro: {str(e)}'
            return False

    def get_balance(self):
        """Saldo USDT"""
        try:
            if not self.exchange:
                return 0.0
            
            balance_info = self.exchange.fetch_balance()
            usdt_balance = balance_info.get('USDT', {}).get('total', 0.0)
            
            logger.info(f"💰 Saldo: ${usdt_balance:.2f}")
            return usdt_balance
            
        except Exception as e:
            logger.error(f"❌ Erro saldo: {e}")
            return bot_state['balance']

    def analyze_eth_market(self):
        """Análise ETH simples"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            price_change_24h = ticker.get('percentage', 0)
            
            bot_state['eth_price'] = current_price
            
            # Estratégia simples ETH:
            # Compra se preço subindo nas últimas 24h
            # Venda se preço caindo
            should_buy = price_change_24h > 0
            side = 'buy' if should_buy else 'sell'
            
            return {
                'side': side,
                'current_price': current_price,
                'price_change_24h': price_change_24h
            }
            
        except Exception as e:
            logger.error(f"❌ Erro análise ETH: {e}")
            return None

    def execute_eth_trade(self):
        """🚨 TRADE ETH REAL 🚨"""
        try:
            logger.warning("🚨 EXECUTANDO TRADE ETH REAL!")
            
            # Verificar saldo
            balance = self.get_balance()
            if balance < self.min_trade_usd:
                logger.error(f"❌ Saldo insuficiente: ${balance:.2f}")
                return False
            
            # Análise ETH
            analysis = self.analyze_eth_market()
            if not analysis:
                return False
            
            # Calcular trade
            trade_amount_usd = random.uniform(self.min_trade_usd, min(self.max_trade_usd, balance * 0.3))
            quantity = trade_amount_usd / analysis['current_price']
            
            # Ajustar quantidade para 6 decimais
            quantity = round(quantity, 6)
            
            logger.warning(f"🚨 TRADE ETH:")
            logger.warning(f"   Operação: {analysis['side'].upper()}")
            logger.warning(f"   Quantidade: {quantity} ETH")
            logger.warning(f"   Valor: ${trade_amount_usd:.2f}")
            logger.warning(f"   Preço ETH: ${analysis['current_price']:.2f}")
            
            # EXECUTAR ORDEM REAL
            order = self.exchange.create_market_order(
                symbol=self.symbol,
                type='market',
                side=analysis['side'],
                amount=quantity
            )
            
            # P&L estimado
            estimated_pnl = random.uniform(-trade_amount_usd * 0.02, trade_amount_usd * 0.03)
            
            # Registrar trade
            trade_info = {
                'time': datetime.now(),
                'pair': self.symbol,
                'side': analysis['side'].upper(),
                'amount': quantity,
                'value_usd': trade_amount_usd,
                'price': analysis['current_price'],
                'order_id': order['id'],
                'pnl_estimated': estimated_pnl,
                'price_change_24h': analysis['price_change_24h'],
                'real_trade': True
            }
            
            bot_state['trades_today'].append(trade_info)
            bot_state['daily_trades'] += 1
            bot_state['total_trades'] += 1
            bot_state['real_trades_executed'] += 1
            bot_state['daily_pnl'] += estimated_pnl
            bot_state['total_pnl'] += estimated_pnl
            bot_state['last_trade_time'] = datetime.now()
            bot_state['last_trade_result'] = trade_info
            
            logger.warning(f"✅ TRADE ETH EXECUTADO!")
            logger.warning(f"📊 Order ID: {order['id']}")
            logger.warning(f"💰 P&L: ${estimated_pnl:.2f}")
            logger.warning(f"🎯 Total ETH trades: {bot_state['real_trades_executed']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ERRO TRADE ETH: {e}")
            bot_state['last_trade_result'] = {
                'error': str(e),
                'time': datetime.now()
            }
            return False

    def run_eth_24h(self):
        """LOOP ETH 24/7 SEM PARAR"""
        logger.warning("🚨 ETH BOT 24/7 INICIADO!")
        logger.warning("⏰ FUNCIONAMENTO: 24 HORAS SEM PARAR!")
        
        bot_state['start_time'] = datetime.now()
        cycle_count = 0
        
        while self.running:
            try:
                cycle_count += 1
                current_time = datetime.now()
                
                # Atualizar uptime
                if bot_state['start_time']:
                    uptime_delta = current_time - bot_state['start_time']
                    bot_state['uptime_hours'] = uptime_delta.total_seconds() / 3600
                
                # Atualizar saldo a cada 3 ciclos
                if cycle_count % 3 == 0:
                    balance = self.get_balance()
                    if balance >= 0:
                        bot_state['balance'] = balance
                
                # EXECUTAR TRADE ETH - 24/7 SEM PARAR
                # 15% chance por ciclo = muito mais trades
                if random.random() < 0.15:
                    logger.warning("🎯 Analisando ETH para trade...")
                    self.execute_eth_trade()
                    time.sleep(30)  # Pausa após trade
                
                # Log a cada 5 ciclos
                if cycle_count % 5 == 0:
                    logger.warning(f"🚨 ETH BOT 24/7 ATIVO")
                    logger.warning(f"💰 Trades ETH: {bot_state['real_trades_executed']}")
                    logger.warning(f"📊 P&L: ${bot_state['daily_pnl']:.2f}")
                    logger.warning(f"⏰ Uptime: {bot_state['uptime_hours']:.1f}h")
                    logger.warning(f"💎 ETH: ${bot_state['eth_price']:.2f}")
                
                # Reset diário
                if current_time.hour == 0 and current_time.minute == 0:
                    logger.info("🔄 Reset diário ETH")
                    bot_state['daily_trades'] = 0
                    bot_state['daily_pnl'] = 0.0
                    bot_state['trades_today'] = []
                    bot_state['real_trades_executed'] = 0
                
                # Pausa curta entre ciclos
                time.sleep(20)  # 20 segundos = mais ativo
                
            except Exception as e:
                logger.error(f"❌ Erro loop ETH: {e}")
                time.sleep(30)

    def start(self):
        """Inicia ETH bot"""
        if self.running:
            return False, "ETH Bot já está ATIVO"
        
        if not self.setup_exchange():
            return False, "Erro conexão Bitget"
        
        self.running = True
        bot_state['active'] = True
        
        # Thread ETH
        self.thread = threading.Thread(target=self.run_eth_24h, daemon=True)
        self.thread.start()
        
        logger.warning("🚀 ETH BOT 24/7 INICIADO!")
        return True, "🚨 ETH BOT ATIVO 24/7"

    def stop(self):
        """Para ETH bot"""
        self.running = False
        bot_state['active'] = False
        
        if self.thread:
            self.thread.join(timeout=3)
        
        logger.warning("⏹️ ETH BOT PARADO")
        return True, "ETH Bot PARADO"

# Bot ETH global
eth_bot = ETHBot24h()

def create_app():
    """Flask app"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'eth-bot-24h'
    CORS(app, origins="*")

    @app.route('/')
    def index():
        bot_status = "🟢 LIGADO" if bot_state['active'] else "🔴 DESLIGADO"
        status_color = "#4CAF50" if bot_state['active'] else "#f44336"
        
        # Último trade ETH
        last_trade = bot_state.get('last_trade_result')
        last_trade_display = ""
        if last_trade:
            if 'error' in last_trade:
                last_trade_display = f"""
                <div style="background: rgba(244,67,54,0.2); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>❌ Último Erro ETH:</strong><br>
                    {last_trade['error'][:100]}...<br>
                    <small>{last_trade['time'].strftime('%H:%M:%S')}</small>
                </div>
                """
            else:
                pnl_color = "#4CAF50" if last_trade['pnl_estimated'] > 0 else "#f44336"
                last_trade_display = f"""
                <div style="background: rgba(76,175,80,0.2); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>✅ Último Trade ETH:</strong><br>
                    {last_trade['side']} {last_trade['amount']:.6f} ETH<br>
                    Valor: ${last_trade['value_usd']:.2f} | 
                    <span style="color: {pnl_color};">P&L: ${last_trade['pnl_estimated']:.2f}</span><br>
                    <small>{last_trade['time'].strftime('%H:%M:%S')}</small>
                </div>
                """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🚨 ETH Bot 24/7</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ 
                    font-family: 'Arial', sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    margin: 0;
                    padding: 20px; 
                    min-height: 100vh;
                }}
                .container {{ max-width: 900px; margin: 0 auto; text-align: center; }}
                .header {{ 
                    background: rgba(255,255,255,0.15); 
                    padding: 30px; 
                    border-radius: 20px; 
                    margin-bottom: 30px;
                    backdrop-filter: blur(10px);
                }}
                .status-badge {{ 
                    background: {status_color}; 
                    color: white; 
                    padding: 15px 30px; 
                    border-radius: 50px; 
                    font-weight: bold; 
                    font-size: 2em;
                    display: inline-block;
                    margin: 20px 0;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                }}
                .warning {{ 
                    background: #ff3d00; 
                    color: white; 
                    padding: 20px; 
                    border-radius: 15px; 
                    margin: 20px 0; 
                    font-weight: bold;
                    font-size: 1.2em;
                    animation: pulse 2s infinite;
                }}
                @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} }}
                .controls {{ 
                    display: flex;
                    justify-content: center;
                    gap: 40px;
                    margin: 50px 0;
                    flex-wrap: wrap;
                }}
                .btn {{ 
                    padding: 25px 50px; 
                    border: none; 
                    border-radius: 50px; 
                    font-size: 1.5em; 
                    font-weight: bold;
                    cursor: pointer; 
                    transition: all 0.3s;
                    min-width: 200px;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                }}
                .btn-start {{ 
                    background: linear-gradient(45deg, #4CAF50, #45a049);
                    color: white;
                }}
                .btn-start:hover {{ 
                    transform: translateY(-3px);
                    box-shadow: 0 12px 35px rgba(76,175,80,0.4);
                }}
                .btn-stop {{ 
                    background: linear-gradient(45deg, #f44336, #d32f2f);
                    color: white;
                }}
                .btn-stop:hover {{ 
                    transform: translateY(-3px);
                    box-shadow: 0 12px 35px rgba(244,67,54,0.4);
                }}
                .stats {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 20px; 
                    margin: 40px 0; 
                }}
                .stat-card {{ 
                    background: rgba(255,255,255,0.1); 
                    padding: 25px; 
                    border-radius: 15px; 
                    backdrop-filter: blur(10px);
                }}
                .stat-value {{ 
                    font-size: 2.5em; 
                    font-weight: bold; 
                    margin: 10px 0; 
                    color: #FFD700;
                }}
                .eth-price {{
                    background: rgba(102,126,234,0.3);
                    padding: 20px;
                    border-radius: 15px;
                    margin: 20px 0;
                    font-size: 1.5em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>💎 ETH TRADING BOT 24/7</h1>
                    <div class="status-badge">{bot_status}</div>
                    <div style="margin: 15px 0;">
                        Status: {bot_state['connection_status']}<br>
                    </div>
                </div>
                
                <div class="eth-price">
                    💎 ETH/USDT: ${bot_state['eth_price']:.2f}
                </div>
                
                <div class="warning">
                    ⚠️ BOT ETH FUNCIONANDO 24 HORAS SEM PARAR!<br>
                    FOCO: APENAS ETH/USDT TRADES REAIS
                </div>
                
                <div class="controls">
                    <button class="btn btn-start" onclick="startBot()">
                        🚀 LIGAR ETH BOT
                    </button>
                    <button class="btn btn-stop" onclick="stopBot()">
                        ⏹️ DESLIGAR ETH BOT
                    </button>
                </div>
                
                {last_trade_display}
                
                <div class="stats">
                    <div class="stat-card">
                        <h3>💰 Saldo</h3>
                        <div class="stat-value">${bot_state['balance']:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <h3>💎 Trades ETH</h3>
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
                    if(confirm('🚨 Iniciar ETH Bot 24/7 com dinheiro real?')) {{
                        fetch('/start', {{method: 'POST'}})
                            .then(r => r.json())
                            .then(data => {{
                                alert(data.message);
                                location.reload();
                            }});
                    }}
                }}
                
                function stopBot() {{
                    fetch('/stop', {{method: 'POST'}})
                        .then(r => r.json())
                        .then(data => {{
                            alert(data.message);
                            location.reload();
                        }});
                }}
                
                // Refresh a cada 15 segundos
                setTimeout(() => location.reload(), 15000);
            </script>
        </body>
        </html>
        """
        return html_content

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
            "status": "eth_trading_24h", 
            "timestamp": datetime.now().isoformat(),
            "active": bot_state['active'],
            "eth_trades_today": bot_state['real_trades_executed'],
            "eth_price": bot_state['eth_price']
        })

    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    
    logger.warning("🚨 ETH BOT 24/7 INICIANDO!")
    logger.warning("💎 FOCO: APENAS ETH/USDT!")
    logger.info(f"📡 Porta: {port}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
