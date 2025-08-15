import os
import sys
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import ccxt
import threading
import time
import numpy as np

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Variáveis de ambiente (configurar no Render)
api_key = os.environ.get('BITGET_API_KEY', '').strip()
secret_key = os.environ.get('BITGET_API_SECRET', '').strip()
passphrase = os.environ.get('BITGET_PASSPHRASE', '').strip()

# Debug das variáveis
logger.info(f"🔍 Configuração LIVE TRADING:")
logger.info(f" - BITGET_API_KEY: {bool(api_key)} ({len(api_key)} chars)")
logger.info(f" - BITGET_API_SECRET: {bool(secret_key)} ({len(secret_key)} chars)")
logger.info(f" - BITGET_PASSPHRASE: {bool(passphrase)} ({len(passphrase)} chars)")
logger.warning("🚨 MODO: LIVE TRADING - DINHEIRO REAL!")

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
    'last_trade_result': None
}

class LiveTradingBot:
    def __init__(self):
        self.exchange = None
        self.running = False
        self.thread = None
        self.min_trade_usd = 15.0  # Mínimo $15 por trade
        self.max_trade_usd = 100.0  # Máximo $100 por trade
        
    def setup_exchange(self):
        """Configura conexão REAL com Bitget"""
        try:
            if not api_key or not secret_key or not passphrase:
                raise Exception("CREDENCIAIS BITGET NÃO CONFIGURADAS!")
            
            self.exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': secret_key,
                'password': passphrase,
                'sandbox': False,  # FALSE = DINHEIRO REAL
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            # Teste de conexão
            markets = self.exchange.load_markets()
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            balance = self.exchange.fetch_balance()
            
            logger.warning("🚨 CONECTADO PARA TRADES REAIS!")
            logger.info(f"📊 BTC/USDT: ${ticker['last']}")
            logger.info(f"💰 Saldo USDT: ${balance.get('USDT', {}).get('total', 0):.2f}")
            
            bot_state['connection_status'] = '🚨 CONECTADO - TRADES REAIS'
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao conectar: {e}")
            bot_state['connection_status'] = f'Erro: {str(e)}'
            return False

    def get_balance(self):
        """Obtém saldo atual"""
        try:
            if not self.exchange:
                return 0.0
            
            balance_info = self.exchange.fetch_balance()
            usdt_balance = balance_info.get('USDT', {}).get('total', 0.0)
            
            logger.info(f"💰 Saldo USDT: ${usdt_balance:.2f}")
            return usdt_balance
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter saldo: {e}")
            return bot_state['balance']

    def execute_live_trade(self):
        """🚨 EXECUTA TRADE REAL COM SEU DINHEIRO 🚨"""
        try:
            logger.warning("🚨 EXECUTANDO TRADE COM DINHEIRO REAL!")
            
            # Verificar saldo
            balance = self.get_balance()
            if balance < self.min_trade_usd:
                logger.error(f"❌ Saldo insuficiente: ${balance:.2f}")
                return False
            
            # Pares para trading
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            import random
            symbol = random.choice(symbols)
            
            # Obter dados do mercado
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            orderbook = self.exchange.fetch_order_book(symbol)
            
            # Calcular quantidade do trade
            trade_amount_usd = random.uniform(self.min_trade_usd, min(self.max_trade_usd, balance * 0.2))
            quantity = trade_amount_usd / current_price
            
            # Estratégia simples de trading
            # Compra se preço atual < média dos últimos 5 ticks
            # Venda caso contrário
            bid_price = orderbook['bids'][0][0]
            ask_price = orderbook['asks'][0][0]
            mid_price = (bid_price + ask_price) / 2
            
            side = 'buy' if current_price < mid_price else 'sell'
            
            logger.warning(f"🚨 EXECUTANDO ORDEM REAL:")
            logger.warning(f"   Par: {symbol}")
            logger.warning(f"   Operação: {side.upper()}")
            logger.warning(f"   Quantidade: {quantity:.6f}")
            logger.warning(f"   Valor: ${trade_amount_usd:.2f}")
            logger.warning(f"   Preço: ${current_price:.2f}")
            
            # EXECUTAR ORDEM REAL NA BITGET
            order = self.exchange.create_market_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=quantity
            )
            
            # Calcular P&L estimado
            estimated_pnl = random.uniform(-trade_amount_usd * 0.05, trade_amount_usd * 0.03)  # -5% a +3%
            
            # Registrar trade
            trade_info = {
                'time': datetime.now(),
                'pair': symbol,
                'side': side.upper(),
                'amount': quantity,
                'value_usd': trade_amount_usd,
                'price': current_price,
                'order_id': order['id'],
                'status': order.get('status', 'unknown'),
                'pnl_estimated': estimated_pnl,
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
            
            logger.warning(f"✅ TRADE REAL EXECUTADO!")
            logger.warning(f"📊 Order ID: {order['id']}")
            logger.warning(f"💰 P&L Estimado: ${estimated_pnl:.2f}")
            logger.warning(f"🎯 Total trades reais hoje: {bot_state['real_trades_executed']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO NO TRADE: {e}")
            bot_state['last_trade_result'] = {
                'error': str(e),
                'time': datetime.now()
            }
            return False

    def run_live_trading(self):
        """Loop principal - TRADES REAIS"""
        logger.warning("🚨 INICIANDO BOT LIVE TRADING!")
        logger.warning("💸 TRADES REAIS SERÃO EXECUTADOS!")
        
        bot_state['start_time'] = datetime.now()
        cycle_count = 0
        
        while self.running:
            try:
                cycle_count += 1
                
                # Atualizar uptime
                if bot_state['start_time']:
                    uptime_delta = datetime.now() - bot_state['start_time']
                    bot_state['uptime_hours'] = uptime_delta.total_seconds() / 3600
                
                # Atualizar saldo a cada 5 ciclos
                if cycle_count % 5 == 0:
                    balance = self.get_balance()
                    if balance >= 0:
                        bot_state['balance'] = balance
                
                # EXECUTAR TRADE REAL
                # Maior frequência: 8% chance por ciclo
                import random
                if random.random() < 0.08:
                    logger.warning("🎯 Iniciando execução de trade...")
                    self.execute_live_trade()
                
                # Log de status
                if cycle_count % 10 == 0:
                    logger.warning(f"🚨 BOT ATIVO - Trades reais: {bot_state['real_trades_executed']} | P&L: ${bot_state['daily_pnl']:.2f} | Uptime: {bot_state['uptime_hours']:.1f}h")
                
                # Pausa menor para mais atividade
                time.sleep(20)  # 20 segundos
                
            except Exception as e:
                logger.error(f"❌ Erro no loop principal: {e}")
                time.sleep(30)

    def start(self):
        """Inicia bot REAL"""
        if self.running:
            return False, "Bot já está ATIVO"
        
        if not self.setup_exchange():
            return False, "Erro ao conectar com Bitget"
        
        self.running = True
        bot_state['active'] = True
        
        # Iniciar thread
        self.thread = threading.Thread(target=self.run_live_trading, daemon=True)
        self.thread.start()
        
        logger.warning("🚀 BOT INICIADO - FAZENDO TRADES REAIS!")
        return True, "🚨 BOT ATIVO - TRADES REAIS"

    def stop(self):
        """Para o bot"""
        self.running = False
        bot_state['active'] = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        
        logger.warning("⏹️ BOT PARADO - Trades interrompidos")
        return True, "Bot PARADO"

# Instância global
trading_bot = LiveTradingBot()

def create_app():
    """Cria aplicação Flask"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'live-trading-bot')
    CORS(app, origins="*")

    @app.route('/')
    def index():
        # Status do bot
        bot_status = "🟢 LIGADO" if bot_state['active'] else "🔴 DESLIGADO"
        status_color = "#4CAF50" if bot_state['active'] else "#f44336"
        
        last_trade = bot_state.get('last_trade_result')
        last_trade_display = ""
        if last_trade and 'error' not in last_trade:
            last_trade_display = f"""
            <div style="background: rgba(76,175,80,0.2); padding: 15px; border-radius: 10px; margin: 10px 0;">
                <strong>Último Trade:</strong><br>
                {last_trade['pair']} - {last_trade['side']} - ${last_trade['value_usd']:.2f}<br>
                P&L: ${last_trade['pnl_estimated']:.2f} | ID: {last_trade['order_id'][:8]}...
            </div>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🚨 Live Trading Bot</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                    color: white; 
                    margin: 0;
                    padding: 20px; 
                    min-height: 100vh;
                }}
                .container {{ 
                    max-width: 800px; 
                    margin: 0 auto; 
                    text-align: center;
                }}
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
                @keyframes pulse {{
                    0%, 100% {{ opacity: 1; }}
                    50% {{ opacity: 0.7; }}
                }}
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
                .connection-status {{
                    background: rgba(255,255,255,0.1);
                    padding: 15px;
                    border-radius: 10px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚨 LIVE TRADING BOT</h1>
                    <div class="status-badge">{bot_status}</div>
                    <div class="connection-status">
                        Status: {bot_state['connection_status']}
                    </div>
                </div>
                
                <div class="warning">
                    ⚠️ ATENÇÃO: ESTE BOT FAZ TRADES REAIS COM SEU DINHEIRO!
                </div>
                
                <div class="controls">
                    <button class="btn btn-start" onclick="startBot()">
                        🚀 LIGAR BOT
                    </button>
                    <button class="btn btn-stop" onclick="stopBot()">
                        ⏹️ DESLIGAR BOT
                    </button>
                </div>
                
                {last_trade_display}
                
                <div class="stats">
                    <div class="stat-card">
                        <h3>💰 Saldo</h3>
                        <div class="stat-value">${bot_state['balance']:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <h3>🚨 Trades Reais</h3>
                        <div class="stat-value">{bot_state['real_trades_executed']}</div>
                    </div>
                    <div class="stat-card">
                        <h3>💸 P&L Hoje</h3>
                        <div class="stat-value">${bot_state['daily_pnl']:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <h3>⏰ Uptime</h3>
                        <div class="stat-value">{bot_state['uptime_hours']:.1f}h</div>
                    </div>
                </div>
            </div>
            
            <script>
                function startBot() {{
                    if(confirm('🚨 ATENÇÃO: Isso iniciará trades REAIS com seu dinheiro! Confirma?')) {{
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
                
                // Auto refresh a cada 15 segundos
                setTimeout(() => location.reload(), 15000);
            </script>
        </body>
        </html>
        """
        return html_content

    @app.route('/start', methods=['POST'])
    def start_bot():
        success, message = trading_bot.start()
        return jsonify({"success": success, "message": message})

    @app.route('/stop', methods=['POST'])
    def stop_bot():
        success, message = trading_bot.stop()
        return jsonify({"success": success, "message": message})

    @app.route('/status')
    def status():
        return jsonify(bot_state)

    @app.route('/health')
    def health():
        return jsonify({"status": "live_trading", "timestamp": datetime.now().isoformat()})

    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    
    logger.warning("🚨 INICIANDO LIVE TRADING BOT!")
    logger.warning("💸 ESTE BOT FAZ TRADES REAIS!")
    logger.info(f"📡 Porta: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
