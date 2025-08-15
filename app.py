import os
import sys
import logging
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time

# Importar o bot existente
try:
    from trading_bot import TradingBot
    from bitget_api import BitgetAPI
    BOT_AVAILABLE = True
except ImportError:
    BOT_AVAILABLE = False
    logging.warning("⚠️ Módulos do bot não encontrados - funcionando em modo demo")

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Variáveis globais do bot
bot_instance = None
bot_config = {
    'BITGET_API_KEY': os.environ.get('BITGET_API_KEY', ''),
    'BITGET_SECRET_KEY': os.environ.get('BITGET_SECRET_KEY', ''),
    'BITGET_PASSPHRASE': os.environ.get('BITGET_PASSPHRASE', ''),
    'GEMINI_API_KEY': os.environ.get('GEMINI_API_KEY', ''),
    'PAPER_TRADING': os.environ.get('PAPER_TRADING', 'true').lower() == 'true',
    'SYMBOL': 'ethusdt_UMCBL',
    'LEVERAGE': 10,
    'TARGET_TRADES_PER_DAY': 200,
    'BASE_CURRENCY': 'USDT'
}

def create_app():
    """Cria aplicação Flask com controle do bot"""
    app = Flask(__name__)
    
    # Configurações básicas
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-seguro')
    
    # CORS
    CORS(app, origins="*")
    
    # Rota principal com painel de controle
    @app.route('/')
    def index():
        return f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🚀 Trading Bot - Farmede Dinheiro</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                    color: white; margin: 0; padding: 20px; min-height: 100vh;
                }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .title {{ font-size: 3em; margin-bottom: 10px; }}
                .subtitle {{ opacity: 0.9; font-size: 1.2em; }}
                
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .card {{ 
                    background: rgba(255,255,255,0.1); padding: 25px; 
                    border-radius: 15px; backdrop-filter: blur(10px);
                    border: 1px solid rgba(255,255,255,0.2);
                }}
                
                .status-card {{ text-align: center; }}
                .status-online {{ color: #4CAF50; font-size: 1.8em; font-weight: bold; }}
                .status-offline {{ color: #f44336; font-size: 1.8em; font-weight: bold; }}
                
                .balance-display {{ 
                    font-size: 2.5em; font-weight: bold; color: #4CAF50; 
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                }}
                
                .btn {{ 
                    background: #4CAF50; color: white; padding: 15px 30px; 
                    border: none; border-radius: 25px; font-size: 1.1em; 
                    margin: 10px; cursor: pointer; text-decoration: none;
                    display: inline-block; transition: all 0.3s;
                    min-width: 140px;
                }}
                .btn:hover {{ background: #45a049; transform: translateY(-2px); }}
                .btn-danger {{ background: #f44336; }}
                .btn-danger:hover {{ background: #da190b; }}
                .btn-primary {{ background: #2196F3; }}
                .btn-primary:hover {{ background: #0b7dda; }}
                .btn:disabled {{ background: #666; cursor: not-allowed; transform: none; }}
                
                .pulse {{ animation: pulse 2s infinite; }}
                @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} 100% {{ opacity: 1; }} }}
                
                .log-container {{ 
                    background: rgba(0,0,0,0.4); padding: 15px; border-radius: 10px;
                    height: 150px; overflow-y: auto; font-family: monospace; font-size: 12px;
                    border: 1px solid rgba(255,255,255,0.2);
                }}
                
                .stats {{ display: flex; justify-content: space-between; text-align: center; margin: 20px 0; }}
                .stat {{ flex: 1; }}
                .stat-value {{ font-size: 1.8em; font-weight: bold; color: #4CAF50; }}
                .stat-label {{ font-size: 0.9em; opacity: 0.8; }}
                
                .mode-indicator {{ 
                    background: {'rgba(255,165,0,0.3)' if bot_config['PAPER_TRADING'] else 'rgba(255,0,0,0.3)'};
                    padding: 10px; border-radius: 10px; margin: 10px 0;
                    border: 1px solid {'#FFA500' if bot_config['PAPER_TRADING'] else '#FF0000'};
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 class="title">🚀 Trading Bot</h1>
                    <div class="subtitle">Farmede Dinheiro - Painel de Controle</div>
                    <div class="mode-indicator">
                        {'📄 MODO PAPER TRADING (Seguro)' if bot_config['PAPER_TRADING'] else '💰 MODO REAL (Cuidado!)'}
                    </div>
                </div>
                
                <div class="grid">
                    <!-- Status e Controle do Bot -->
                    <div class="card status-card">
                        <h3>🤖 Controle do Bot</h3>
                        <div id="bot-status" class="status-offline pulse">🔴 OFFLINE</div>
                        <div style="margin: 20px 0;">
                            <button class="btn" onclick="startBot()" id="start-btn">▶️ INICIAR BOT</button>
                            <button class="btn btn-danger" onclick="stopBot()" id="stop-btn" disabled>⏹️ PARAR BOT</button>
                        </div>
                        <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
                            <div>Símbolo: ETH/USDT</div>
                            <div>Alavancagem: 10x</div>
                            <div>Estratégia: Scalping</div>
                        </div>
                    </div>
                    
                    <!-- Saldo da Conta -->
                    <div class="card status-card">
                        <h3>💰 Saldo da Conta</h3>
                        <div class="balance-display" id="balance-amount">$0.00</div>
                        <div style="margin-top: 15px;">
                            <button class="btn btn-primary" onclick="refreshBalance()">🔄 Atualizar Saldo</button>
                        </div>
                        <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
                            <div>Moeda Base: USDT</div>
                            <div id="last-update">Última atualização: --</div>
                        </div>
                    </div>
                    
                    <!-- Estatísticas de Performance -->
                    <div class="card">
                        <h3>📊 Performance Hoje</h3>
                        <div class="stats">
                            <div class="stat">
                                <div class="stat-value" id="daily-trades">0</div>
                                <div class="stat-label">Trades</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="daily-pnl">$0.00</div>
                                <div class="stat-label">P&L</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="win-rate">0%</div>
                                <div class="stat-label">Taxa de Acerto</div>
                            </div>
                        </div>
                        <button class="btn btn-primary" onclick="refreshStats()">📈 Atualizar Stats</button>
                    </div>
                    
                    <!-- Log de Atividades -->
                    <div class="card" style="grid-column: 1 / -1;">
                        <h3>📝 Log de Atividades</h3>
                        <div id="activity-log" class="log-container">
                            <div>{datetime.now().strftime('%H:%M:%S')} - Sistema iniciado - Pronto para operar</div>
                            <div>{datetime.now().strftime('%H:%M:%S')} - Modo: {'Paper Trading' if bot_config['PAPER_TRADING'] else 'Trading Real'}</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                let botActive = false;
                let updateInterval = null;
                
                function addLog(message, type = 'info') {{
                    const log = document.getElementById('activity-log');
                    const time = new Date().toLocaleTimeString();
                    const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️';
                    log.innerHTML += '<div>' + time + ' - ' + icon + ' ' + message + '</div>';
                    log.scrollTop = log.scrollHeight;
                }}
                
                function updateBotStatus(active) {{
                    const statusElement = document.getElementById('bot-status');
                    const startBtn = document.getElementById('start-btn');
                    const stopBtn = document.getElementById('stop-btn');
                    
                    if (active) {{
                        statusElement.className = 'status-online pulse';
                        statusElement.innerHTML = '🟢 ONLINE';
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                        botActive = true;
                        startAutoRefresh();
                    }} else {{
                        statusElement.className = 'status-offline pulse';
                        statusElement.innerHTML = '🔴 OFFLINE';
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                        botActive = false;
                        stopAutoRefresh();
                    }}
                }}
                
                function startAutoRefresh() {{
                    if (updateInterval) clearInterval(updateInterval);
                    updateInterval = setInterval(() => {{
                        if (botActive) {{
                            refreshBalance();
                            refreshStats();
                        }}
                    }}, 30000); // Atualiza a cada 30 segundos
                }}
                
                function stopAutoRefresh() {{
                    if (updateInterval) {{
                        clearInterval(updateInterval);
                        updateInterval = null;
                    }}
                }}
                
                function startBot() {{
                    addLog('Iniciando bot de trading...', 'info');
                    fetch('/api/bot/start', {{method: 'POST'}})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            updateBotStatus(true);
                            addLog('Bot iniciado com sucesso!', 'success');
                            refreshBalance();
                        }} else {{
                            addLog('Erro ao iniciar bot: ' + data.message, 'error');
                        }}
                    }})
                    .catch(error => {{
                        addLog('Erro de conexão ao iniciar bot', 'error');
                    }});
                }}
                
                function stopBot() {{
                    addLog('Parando bot de trading...', 'warning');
                    fetch('/api/bot/stop', {{method: 'POST'}})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            updateBotStatus(false);
                            addLog('Bot parado com sucesso', 'warning');
                        }}
                    }});
                }}
                
                function refreshBalance() {{
                    fetch('/api/balance')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            document.getElementById('balance-amount').textContent = '$' + data.balance.toFixed(2);
                            document.getElementById('last-update').textContent = 'Última atualização: ' + new Date().toLocaleTimeString();
                        }}
                    }});
                }}
                
                function refreshStats() {{
                    fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            document.getElementById('daily-trades').textContent = data.daily_trades;
                            document.getElementById('daily-pnl').textContent = '$' + data.daily_pnl.toFixed(2);
                            document.getElementById('win-rate').textContent = data.win_rate.toFixed(1) + '%';
                        }}
                    }});
                }}
                
                // Verificar status inicial
                fetch('/api/bot/status')
                .then(response => response.json())
                .then(data => {{
                    updateBotStatus(data.active);
                }});
                
                // Carregar saldo inicial
                refreshBalance();
                refreshStats();
            </script>
        </body>
        </html>
        """
    
    # === APIs PARA CONTROLAR O BOT ===
    
    @app.route('/api/bot/start', methods=['POST'])
    def start_bot():
        global bot_instance
        try:
            if bot_instance and bot_instance.is_running:
                return jsonify({'success': False, 'message': 'Bot já está rodando'})
            
            if BOT_AVAILABLE:
                bot_instance = TradingBot(bot_config)
                bot_instance.start()
                logger.info("🤖 Trading bot iniciado!")
                return jsonify({'success': True, 'message': 'Bot iniciado com sucesso'})
            else:
                logger.info("🤖 Bot demo iniciado!")
                return jsonify({'success': True, 'message': 'Bot demo iniciado'})
                
        except Exception as e:
            logger.error(f"Erro ao iniciar bot: {e}")
            return jsonify({'success': False, 'message': str(e)})
    
    @app.route('/api/bot/stop', methods=['POST'])
    def stop_bot():
        global bot_instance
        try:
            if bot_instance:
                bot_instance.stop()
                bot_instance = None
                logger.info("⏹️ Trading bot parado!")
            return jsonify({'success': True, 'message': 'Bot parado'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    @app.route('/api/bot/status')
    def bot_status():
        active = bot_instance.is_running if bot_instance else False
        return jsonify({
            'active': active,
            'paper_trading': bot_config['PAPER_TRADING'],
            'symbol': bot_config['SYMBOL']
        })
    
    @app.route('/api/balance')
    def get_balance():
        try:
            if BOT_AVAILABLE and bot_config.get('BITGET_API_KEY'):
                api = BitgetAPI(
                    bot_config['BITGET_API_KEY'],
                    bot_config['BITGET_SECRET_KEY'], 
                    bot_config['BITGET_PASSPHRASE'],
                    bot_config['PAPER_TRADING']
                )
                balance = api.get_account_balance()
            else:
                # Saldo demo
                balance = 10000.0
            
            return jsonify({
                'success': True,
                'balance': balance,
                'currency': 'USDT',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {e}")
            return jsonify({'success': False, 'balance': 0, 'error': str(e)})
    
    @app.route('/api/stats')
    def get_stats():
        try:
            if bot_instance:
                return jsonify({
                    'success': True,
                    'daily_trades': bot_instance.daily_trades,
                    'daily_pnl': bot_instance.daily_pnl,
                    'win_rate': bot_instance.win_rate,
                    'total_trades': bot_instance.total_trades
                })
            else:
                return jsonify({
                    'success': True,
                    'daily_trades': 0,
                    'daily_pnl': 0.0,
                    'win_rate': 0.0,
                    'total_trades': 0
                })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    # APIs originais
    @app.route('/api/status')
    def status():
        return jsonify({
            'status': 'online',
            'timestamp': datetime.now().isoformat(),
            'message': 'Trading Bot funcionando!',
            'version': '2.0.0',
            'bot_available': BOT_AVAILABLE,
            'paper_trading': bot_config['PAPER_TRADING']
        })
    
    @app.route('/api/test')
    def test():
        return jsonify({
            'success': True,
            'message': 'API funcionando 100%!',
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/api/ccxt')
    def ccxt_info():
        try:
            import ccxt
            exchanges = list(ccxt.exchanges)[:10]
            return jsonify({
                'success': True,
                'total_exchanges': len(ccxt.exchanges),
                'sample_exchanges': exchanges,
                'message': 'CCXT funcionando'
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return app

# Criar aplicação
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Iniciando Trading Bot na porta {port}")
    logger.info(f"📄 Paper Trading: {bot_config['PAPER_TRADING']}")
    logger.info(f"🤖 Bot disponível: {BOT_AVAILABLE}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
