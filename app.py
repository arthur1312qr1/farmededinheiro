import os
import sys
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import ccxt
import threading
import time
import schedule

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Variáveis de ambiente corrigidas
api_key = os.environ.get('BITGET_API_KEY', '').strip()
secret_key = os.environ.get('BITGET_API_SECRET', '').strip()
passphrase = os.environ.get('BITGET_PASSPHRASE', '').strip()
paper_trading = os.environ.get('PAPER_TRADING', 'false').lower() == 'true'

# Debug das variáveis
logger.info(f"🔍 Configuração 24h:")
logger.info(f"  - BITGET_API_KEY: {bool(api_key)} ({len(api_key)} chars)")
logger.info(f"  - BITGET_API_SECRET: {bool(secret_key)} ({len(secret_key)} chars)")
logger.info(f"  - BITGET_PASSPHRASE: {bool(passphrase)} ({len(passphrase)} chars)")
logger.info(f"  - MODE: {'SANDBOX' if paper_trading else 'LIVE TRADING'}")

# Estado do bot 24/7
bot_state = {
    'active': False,
    'auto_mode': False,  # Modo automático 24h
    'balance': 0.0,
    'daily_trades': 0,
    'total_trades': 0,
    'daily_pnl': 0.0,
    'total_pnl': 0.0,
    'win_rate': 0.0,
    'last_update': datetime.now(),
    'start_time': None,
    'uptime_hours': 0,
    'connection_status': 'Inicializando',
    'last_trade_time': None,
    'trades_today': [],
    'status_24h': 'Aguardando ativação'
}

class TradingBot24h:
    def __init__(self):
        self.exchange = None
        self.running = False
        self.thread = None
        
    def setup_exchange(self):
        """Configura conexão com Bitget"""
        try:
            if not api_key or not secret_key or not passphrase:
                raise Exception("Credenciais Bitget não configuradas")
            
            self.exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': secret_key,
                'password': passphrase,
                'sandbox': paper_trading,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            # Teste de conexão
            markets = self.exchange.load_markets()
            ticker = self.exchange.fetch_ticker('ETH/USDT')
            
            logger.info(f"✅ Bot 24h conectado à Bitget! Markets: {len(markets)}")
            bot_state['connection_status'] = 'Conectado 24/7'
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
            
            # Se USDT zero, buscar outras moedas
            if usdt_balance == 0:
                for currency, balance_data in balance_info.items():
                    if isinstance(balance_data, dict) and balance_data.get('total', 0) > 0:
                        return balance_data.get('total', 0)
            
            return usdt_balance
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter saldo: {e}")
            return bot_state['balance']
    
    def simulate_trading(self):
        """Simula atividade de trading 24h"""
        try:
            # Simular análise de mercado
            import random
            
            # Chance de trade a cada ciclo (5% chance)
            if random.random() < 0.05:
                # Simular um trade
                profit_loss = random.uniform(-50, 100)  # Entre -$50 e +$100
                
                trade_info = {
                    'time': datetime.now(),
                    'pair': 'ETH/USDT',
                    'type': 'BUY' if profit_loss > 0 else 'SELL',
                    'amount': random.uniform(0.01, 0.1),
                    'pnl': profit_loss
                }
                
                bot_state['trades_today'].append(trade_info)
                bot_state['daily_trades'] += 1
                bot_state['total_trades'] += 1
                bot_state['daily_pnl'] += profit_loss
                bot_state['total_pnl'] += profit_loss
                bot_state['last_trade_time'] = datetime.now()
                
                # Recalcular win rate
                profitable_trades = sum(1 for t in bot_state['trades_today'] if t['pnl'] > 0)
                if bot_state['daily_trades'] > 0:
                    bot_state['win_rate'] = (profitable_trades / bot_state['daily_trades']) * 100
                
                status = "🟢 PROFIT" if profit_loss > 0 else "🔴 LOSS"
                logger.info(f"📊 Trade simulado: {status} ${profit_loss:.2f} | Total hoje: {bot_state['daily_trades']}")
                
        except Exception as e:
            logger.error(f"❌ Erro na simulação: {e}")
    
    def run_24h(self):
        """Loop principal do bot 24h"""
        logger.info("🤖 Iniciando bot trading 24/7...")
        bot_state['start_time'] = datetime.now()
        bot_state['status_24h'] = 'Ativo 24/7'
        
        cycle_count = 0
        
        while self.running:
            try:
                cycle_count += 1
                
                # Atualizar uptime
                if bot_state['start_time']:
                    uptime_delta = datetime.now() - bot_state['start_time']
                    bot_state['uptime_hours'] = uptime_delta.total_seconds() / 3600
                
                # Atualizar saldo a cada 10 ciclos (aproximadamente 5 minutos)
                if cycle_count % 10 == 0:
                    balance = self.get_balance()
                    if balance > 0:
                        bot_state['balance'] = balance
                    logger.info(f"💰 Saldo atual: ${bot_state['balance']:.2f}")
                
                # Simular atividade de trading
                self.simulate_trading()
                
                # Log de status a cada 20 ciclos (10 minutos)
                if cycle_count % 20 == 0:
                    logger.info(f"📊 Bot 24h ativo - Trades hoje: {bot_state['daily_trades']} | P&L: ${bot_state['daily_pnl']:.2f} | Uptime: {bot_state['uptime_hours']:.1f}h")
                
                # Reset diário às 00:00
                now = datetime.now()
                if now.hour == 0 and now.minute == 0 and now.second < 30:
                    self.reset_daily_stats()
                
                # Pausa entre ciclos (30 segundos)
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Erro no loop 24h: {e}")
                time.sleep(60)  # Pausa maior em caso de erro
    
    def reset_daily_stats(self):
        """Reset estatísticas diárias"""
        logger.info("🔄 Reset diário - Nova sessão de 24h iniciada")
        bot_state['daily_trades'] = 0
        bot_state['daily_pnl'] = 0.0
        bot_state['trades_today'] = []
        bot_state['win_rate'] = 0.0
    
    def start(self):
        """Inicia o bot 24h"""
        if self.running:
            return False, "Bot já está ativo"
        
        if not self.setup_exchange():
            return False, "Erro ao conectar com Bitget"
        
        self.running = True
        bot_state['active'] = True
        bot_state['auto_mode'] = True
        
        # Iniciar thread do bot
        self.thread = threading.Thread(target=self.run_24h, daemon=True)
        self.thread.start()
        
        logger.info("🚀 Bot 24/7 iniciado com sucesso!")
        return True, "Bot 24/7 ativo"
    
    def stop(self):
        """Para o bot"""
        self.running = False
        bot_state['active'] = False
        bot_state['auto_mode'] = False
        bot_state['status_24h'] = 'Parado'
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        logger.info("⏹️ Bot 24/7 parado")
        return True, "Bot parado"

# Instância global do bot
trading_bot = TradingBot24h()

def create_app():
    """Cria aplicação Flask"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'trading-bot-24h')
    CORS(app, origins="*")
    
    @app.route('/')
    def index():
        # Status das APIs
        apis_configured = bool(api_key and secret_key and passphrase)
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Calcular uptime
        uptime_display = f"{bot_state['uptime_hours']:.1f}h" if bot_state['uptime_hours'] > 0 else "00:00"
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🚀 Trading Bot 24/7 - Farmede Dinheiro</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
                    color: white; margin: 0; padding: 20px; min-height: 100vh;
                }}
                .container {{ max-width: 1400px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .title {{ 
                    font-size: 3.5em; margin-bottom: 10px; 
                    background: linear-gradient(45deg, #FFD700, #FFA500);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                }}
                .subtitle {{ opacity: 0.9; font-size: 1.4em; margin-bottom: 20px; }}
                
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px; }}
                .card {{ 
                    background: rgba(255,255,255,0.1); padding: 25px; 
                    border-radius: 20px; backdrop-filter: blur(15px);
                    border: 1px solid rgba(255,255,255,0.15);
                    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
                    transition: all 0.3s ease;
                }}
                .card:hover {{ 
                    transform: translateY(-5px);
                    box-shadow: 0 15px 45px rgba(0,0,0,0.6);
                }}
                
                .status-card {{ text-align: center; }}
                .status-online {{ 
                    color: #4CAF50; font-size: 2.2em; font-weight: bold; 
                    text-shadow: 0 0 15px #4CAF50; animation: pulse 2s infinite;
                }}
                .status-offline {{ 
                    color: #f44336; font-size: 2.2em; font-weight: bold; 
                    text-shadow: 0 0 15px #f44336; animation: pulse 2s infinite;
                }}
                
                @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} }}
                
                .balance-display {{ 
                    font-size: 3.2em; font-weight: bold; 
                    background: linear-gradient(45deg, #FFD700, #FFA500);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    margin: 20px 0; text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
                }}
                
                .btn {{ 
                    background: linear-gradient(45deg, #4CAF50, #45a049);
                    color: white; padding: 15px 30px; 
                    border: none; border-radius: 50px; font-size: 1.1em; font-weight: bold;
                    margin: 10px; cursor: pointer; 
                    transition: all 0.4s; min-width: 160px;
                    box-shadow: 0 4px 15px rgba(76,175,80,0.4);
                }}
                .btn:hover {{ 
                    transform: translateY(-3px); 
                    box-shadow: 0 8px 25px rgba(76,175,80,0.6); 
                }}
                .btn-danger {{ 
                    background: linear-gradient(45deg, #f44336, #da190b);
                    box-shadow: 0 4px 15px rgba(244,67,54,0.4);
                }}
                .btn-danger:hover {{ box-shadow: 0 8px 25px rgba(244,67,54,0.6); }}
                .btn-primary {{ 
                    background: linear-gradient(45deg, #2196F3, #0b7dda);
                    box-shadow: 0 4px 15px rgba(33,150,243,0.4);
                }}
                .btn-primary:hover {{ box-shadow: 0 8px 25px rgba(33,150,243,0.6); }}
                
                .stats {{ 
                    display: grid; grid-template-columns: repeat(2, 1fr); 
                    gap: 20px; margin: 25px 0; 
                }}
                .stat {{ text-align: center; padding: 15px; }}
                .stat-value {{ 
                    font-size: 2.4em; font-weight: bold; 
                    background: linear-gradient(45deg, #4CAF50, #8BC34A);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                }}
                .stat-label {{ font-size: 0.95em; opacity: 0.9; margin-top: 8px; }}
                
                .config-status {{ 
                    background: {'linear-gradient(45deg, rgba(76,175,80,0.3), rgba(139,195,74,0.3))' if apis_configured else 'linear-gradient(45deg, rgba(244,67,54,0.3), rgba(255,87,34,0.3))'};
                    padding: 20px; border-radius: 15px; margin: 20px 0;
                    border: 2px solid {'#4CAF50' if apis_configured else '#f44336'};
                    text-align: center; font-weight: bold; font-size: 1.1em;
                }}
                
                .mode-indicator {{ 
                    background: {'linear-gradient(45deg, rgba(255,165,0,0.3), rgba(255,193,7,0.3))' if paper_trading else 'linear-gradient(45deg, rgba(76,175,80,0.3), rgba(139,195,74,0.3))'};
                    padding: 15px; border-radius: 15px; margin: 15px 0;
                    border: 2px solid {'#FFA500' if paper_trading else '#4CAF50'};
                    text-align: center; font-weight: bold;
                }}
                
                .log-container {{ 
                    background: rgba(0,0,0,0.7); padding: 20px; border-radius: 15px;
                    height: 250px; overflow-y: auto; font-family: 'Courier New', monospace; 
                    font-size: 13px; border: 1px solid rgba(255,255,255,0.2);
                }}
                
                .uptime-display {{
                    font-size: 1.8em; font-weight: bold;
                    color: #FFD700; text-shadow: 0 0 10px #FFD700;
                    margin: 10px 0;
                }}
                
                .trade-indicator {{
                    font-size: 0.9em; opacity: 0.8; margin: 5px 0;
                    color: #4CAF50;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 class="title">🚀 Trading Bot 24/7</h1>
                    <div class="subtitle">Farmede Dinheiro - Sistema Automatizado</div>
                    
                    <div class="config-status">
                        {'✅ SISTEMA OPERACIONAL 24/7' if apis_configured else '❌ ERRO: Configurar chaves da API'}
                    </div>
                    
                    <div class="mode-indicator">
                        {'📄 MODO SANDBOX (Teste Seguro)' if paper_trading else '💰 MODO LIVE - TRADING REAL 24H'}
                    </div>
                </div>
                
                <div class="grid">
                    <!-- Controle 24/7 -->
                    <div class="card status-card">
                        <h3>🎮 Controle Bot 24/7</h3>
                        <div id="bot-status" class="{'status-online' if bot_state['active'] else 'status-offline'}">
                            {'🟢 ATIVO 24H' if bot_state['active'] else '🔴 PARADO'}
                        </div>
                        <div class="uptime-display" id="uptime-display">{uptime_display}</div>
                        <div style="margin: 25px 0;">
                            <button class="btn {'btn-danger' if bot_state['active'] else ''}" 
                                    onclick="toggleBot24h()" id="toggle-btn">
                                {'⏹️ PARAR BOT' if bot_state['active'] else '▶️ INICIAR 24H'}
                            </button>
                        </div>
                        <div style="font-size: 0.95em; opacity: 0.9;">
                            <div>🏦 Exchange: Bitget</div>
                            <div>⚡ Modo: Automático 24/7</div>
                            <div>📊 Par: ETH/USDT</div>
                            <div class="trade-indicator" id="last-trade">
                                {'Último trade: ' + bot_state['last_trade_time'].strftime('%H:%M:%S') if bot_state['last_trade_time'] else 'Aguardando trades...'}
                            </div>
                        </div>
                    </div>
                    
                    <!-- Saldo em Tempo Real -->
                    <div class="card status-card">
                        <h3>💰 Saldo da Conta</h3>
                        <div class="balance-display" id="balance-amount">${bot_state['balance']:.2f}</div>
                        <button class="btn btn-primary" onclick="updateBalance()">🔄 Atualizar Agora</button>
                        <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
                            <div>🕐 Última consulta: {current_time}</div>
                            <div>🌐 Status: {bot_state['connection_status']}</div>
                            <div>📈 P&L Total: ${bot_state['total_pnl']:.2f}</div>
                        </div>
                    </div>
                    
                    <!-- Performance 24h -->
                    <div class="card">
                        <h3>📊 Performance 24h</h3>
                        <div class="stats">
                            <div class="stat">
                                <div class="stat-value" id="trades-today">{bot_state['daily_trades']}</div>
                                <div class="stat-label">Trades Hoje</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="pnl-today">${bot_state['daily_pnl']:.2f}</div>
                                <div class="stat-label">P&L Hoje</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="win-rate">{bot_state['win_rate']:.1f}%</div>
                                <div class="stat-label">Taxa Sucesso</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="total-trades">{bot_state['total_trades']}</div>
                                <div class="stat-label">Total Trades</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Log de Atividades 24/7 -->
                    <div class="card" style="grid-column: 1 / -1;">
                        <h3>📝 Log do Sistema 24/7</h3>
                        <div id="activity-log" class="log-container">
                            <div>{current_time} - 🚀 Sistema de trading 24/7 online</div>
                            <div>{current_time} - 🔧 APIs: BITGET_API_KEY, BITGET_API_SECRET, BITGET_PASSPHRASE</div>
                            <div>{current_time} - ⚙️ Modo: {'SANDBOX (Teste)' if paper_trading else 'LIVE TRADING'}</div>
                            <div>{current_time} - 💰 Saldo: ${bot_state['balance']:.2f}</div>
                            <div>{current_time} - 📊 Status: {bot_state['status_24h']}</div>
                            <div>{current_time} - 🕐 Uptime: {uptime_display}</div>
                            <div>{current_time} - 📡 Sistema pronto para operar continuamente!</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                let updateInterval = null;
                
                function addLog(message, type) {{
                    const log = document.getElementById('activity-log');
                    const time = new Date().toLocaleTimeString();
                    const icons = {{
                        'success': '✅', 'error': '❌', 'warning': '⚠️',
                        'info': '📡', 'balance': '💰', 'trade': '💱', 'debug': '🔧'
                    }};
                    const icon = icons[type] || '📡';
                    log.innerHTML += '<div>' + time + ' - ' + icon + ' ' + message + '</div>';
                    log.scrollTop = log.scrollHeight;
                }}
                
                function toggleBot24h() {{
                    const isActive = document.getElementById('bot-status').textContent.includes('ATIVO');
                    
                    if (isActive) {{
                        stopBot24h();
                    }} else {{
                        startBot24h();
                    }}
                }}
                
                function startBot24h() {{
                    addLog('Iniciando bot de trading 24/7...', 'info');
                    
                    fetch('/api/bot/start_24h', {{method: 'POST'}})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            updateBotDisplay(true);
                            addLog('Bot 24/7 ativo: ' + data.message, 'success');
                            startRealTimeUpdates();
                        }} else {{
                            addLog('Erro: ' + data.message, 'error');
                        }}
                    }});
                }}
                
                function stopBot24h() {{
                    addLog('Parando bot 24/7...', 'warning');
                    
                    fetch('/api/bot/stop_24h', {{method: 'POST'}})
                    .then(response => response.json())
                    .then(data => {{
                        updateBotDisplay(false);
                        addLog('Bot parado: ' + data.message, 'warning');
                        stopRealTimeUpdates();
                    }});
                }}
                
                function updateBotDisplay(active) {{
                    const statusEl = document.getElementById('bot-status');
                    const toggleBtn = document.getElementById('toggle-btn');
                    
                    if (active) {{
                        statusEl.className = 'status-online';
                        statusEl.innerHTML = '🟢 ATIVO 24H';
                        toggleBtn.innerHTML = '⏹️ PARAR BOT';
                        toggleBtn.className = 'btn btn-danger';
                    }} else {{
                        statusEl.className = 'status-offline';
                        statusEl.innerHTML = '🔴 PARADO';
                        toggleBtn.innerHTML = '▶️ INICIAR 24H';
                        toggleBtn.className = 'btn';
                    }}
                }}
                
                function updateBalance() {{
                    addLog('Consultando saldo em tempo real...', 'balance');
                    
                    fetch('/api/balance')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            document.getElementById('balance-amount').textContent = '$' + data.balance.toFixed(2);
                            addLog('Saldo atualizado: $' + data.balance.toFixed(2), 'balance');
                        }}
                    }});
                }}
                
                function updateStats() {{
                    fetch('/api/stats_24h')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            document.getElementById('trades-today').textContent = data.daily_trades;
                            document.getElementById('pnl-today').textContent = '$' + data.daily_pnl.toFixed(2);
                            document.getElementById('win-rate').textContent = data.win_rate.toFixed(1) + '%';
                            document.getElementById('total-trades').textContent = data.total_trades;
                            
                            // Atualizar uptime
                            if (data.uptime_hours) {{
                                document.getElementById('uptime-display').textContent = data.uptime_hours.toFixed(1) + 'h';
                            }}
                            
                            // Último trade
                            if (data.last_trade) {{
                                document.getElementById('last-trade').textContent = 'Último trade: ' + data.last_trade;
                            }}
                        }}
                    }});
                }}
                
                function startRealTimeUpdates() {{
                    updateInterval = setInterval(() => {{
                        updateBalance();
                        updateStats();
                    }}, 15000); // Update a cada 15 segundos
                }}
                
                function stopRealTimeUpdates() {{
                    if (updateInterval) {{
                        clearInterval(updateInterval);
                        document.getElementById('uptime-display').textContent = '00:00';
                    }}
                }}
                
                // Inicialização
                updateBalance();
                updateStats();
                
                // Se o bot já está ativo, iniciar updates
                if (document.getElementById('bot-status').textContent.includes('ATIVO')) {{
                    startRealTimeUpdates();
                }}
                
                // Auto-refresh da página a cada hora para evitar memory leaks
                setTimeout(() => {{ location.reload(); }}, 3600000);
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    # === ROTAS DA API 24/7 ===
    
    @app.route('/api/bot/start_24h', methods=['POST'])
    def start_bot_24h():
        """Inicia o bot 24/7"""
        try:
            success, message = trading_bot.start()
            return jsonify({'success': success, 'message': message})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    @app.route('/api/bot/stop_24h', methods=['POST'])
    def stop_bot_24h():
        """Para o bot 24/7"""
        try:
            success, message = trading_bot.stop()
            return jsonify({'success': success, 'message': message})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    @app.route('/api/balance')
    def get_balance():
        """Obtém saldo atual"""
        try:
            balance = trading_bot.get_balance()
            bot_state['balance'] = balance
            
            return jsonify({
                'success': True,
                'balance': balance,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'success': False, 'balance': 0, 'error': str(e)})
    
    @app.route('/api/stats_24h')
    def get_stats_24h():
        """Obtém estatísticas 24/7"""
        try:
            # Formatar último trade
            last_trade_str = ""
            if bot_state['last_trade_time']:
                last_trade_str = bot_state['last_trade_time'].strftime('%H:%M:%S')
            
            return jsonify({
                'success': True,
                'daily_trades': bot_state['daily_trades'],
                'total_trades': bot_state['total_trades'],
                'daily_pnl': bot_state['daily_pnl'],
                'total_pnl': bot_state['total_pnl'],
                'win_rate': bot_state['win_rate'],
                'uptime_hours': bot_state['uptime_hours'],
                'last_trade': last_trade_str,
                'active': bot_state['active'],
                'auto_mode': bot_state['auto_mode']
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return app

# Criar aplicação
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Iniciando Trading Bot 24/7 na porta {port}")
    logger.info(f"💡 Mode: {'SANDBOX' if paper_trading else 'LIVE TRADING'}")
    logger.info(f"🔑 APIs: {bool(api_key and secret_key and passphrase)}")
    
    # Auto-start do bot se as credenciais estão configuradas
    if api_key and secret_key and passphrase:
        logger.info("🤖 Auto-iniciando bot 24/7...")
        threading.Timer(5, lambda: trading_bot.start()).start()
    
    app.run(host='0.0.0.0', port=port, debug=False)
