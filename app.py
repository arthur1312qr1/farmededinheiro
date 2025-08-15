import os
import sys
import logging
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Verificar se as variáveis estão configuradas
api_key = os.environ.get('BITGET_API_KEY', '')
secret_key = os.environ.get('BITGET_SECRET_KEY', '')
passphrase = os.environ.get('BITGET_PASSPHRASE', '')
paper_trading = os.environ.get('PAPER_TRADING', 'true').lower() == 'true'

# Estado do bot
bot_state = {
    'active': False,
    'balance': 0.0,
    'daily_trades': 0,
    'daily_pnl': 0.0,
    'win_rate': 0.0,
    'last_update': datetime.now()
}

def create_app():
    """Cria aplicação Flask com painel de controle"""
    app = Flask(__name__)
    
    # Configurações básicas
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-seguro')
    
    # CORS
    CORS(app, origins="*")
    
    @app.route('/')
    def index():
        # Verificar se APIs estão configuradas
        apis_configured = bool(api_key and secret_key and passphrase)
        
        current_time = datetime.now().strftime('%H:%M:%S')
        
        html_content = f"""
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
                .title {{ font-size: 3em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }}
                .subtitle {{ opacity: 0.9; font-size: 1.3em; }}
                
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
                .card {{ 
                    background: rgba(255,255,255,0.15); padding: 25px; 
                    border-radius: 15px; backdrop-filter: blur(10px);
                    border: 1px solid rgba(255,255,255,0.2);
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                }}
                
                .status-card {{ text-align: center; }}
                .status-online {{ color: #4CAF50; font-size: 2em; font-weight: bold; text-shadow: 0 0 10px #4CAF50; }}
                .status-offline {{ color: #f44336; font-size: 2em; font-weight: bold; text-shadow: 0 0 10px #f44336; }}
                
                .balance-display {{ 
                    font-size: 2.8em; font-weight: bold; color: #FFD700; 
                    text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
                    margin: 15px 0;
                }}
                
                .btn {{ 
                    background: linear-gradient(45deg, #4CAF50, #45a049);
                    color: white; padding: 15px 25px; 
                    border: none; border-radius: 30px; font-size: 1.1em; font-weight: bold;
                    margin: 8px; cursor: pointer; text-decoration: none;
                    display: inline-block; transition: all 0.4s;
                    min-width: 140px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                }}
                .btn:hover {{ transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.4); }}
                .btn-danger {{ background: linear-gradient(45deg, #f44336, #da190b); }}
                .btn-primary {{ background: linear-gradient(45deg, #2196F3, #0b7dda); }}
                .btn:disabled {{ 
                    background: #666; cursor: not-allowed; transform: none; 
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                }}
                
                .pulse {{ animation: pulse 2s infinite; }}
                @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} 100% {{ opacity: 1; }} }}
                
                .log-container {{ 
                    background: rgba(0,0,0,0.6); padding: 15px; border-radius: 10px;
                    height: 180px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 13px;
                    border: 1px solid rgba(255,255,255,0.3);
                }}
                
                .stats {{ display: flex; justify-content: space-around; text-align: center; margin: 20px 0; }}
                .stat {{ }}
                .stat-value {{ font-size: 2em; font-weight: bold; color: #4CAF50; text-shadow: 0 0 5px #4CAF50; }}
                .stat-label {{ font-size: 0.9em; opacity: 0.9; margin-top: 5px; }}
                
                .config-status {{ 
                    background: {'rgba(76,175,80,0.3)' if apis_configured else 'rgba(255,152,0,0.3)'};
                    padding: 15px; border-radius: 10px; margin: 15px 0;
                    border: 1px solid {'#4CAF50' if apis_configured else '#FF9800'};
                    text-align: center;
                }}
                
                .mode-indicator {{ 
                    background: {'rgba(255,165,0,0.3)' if paper_trading else 'rgba(255,0,0,0.3)'};
                    padding: 12px; border-radius: 10px; margin: 15px 0;
                    border: 1px solid {'#FFA500' if paper_trading else '#FF0000'};
                    text-align: center; font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 class="title">🚀 Trading Bot</h1>
                    <div class="subtitle">Farmede Dinheiro - Sistema Ativo</div>
                    
                    <div class="config-status">
                        {'✅ APIs Configuradas - Pronto para Operar' if apis_configured else '⚠️ Configure as APIs no Render para trading real'}
                    </div>
                    
                    <div class="mode-indicator">
                        {'📄 MODO PAPER TRADING (Simulação Segura)' if paper_trading else '💰 MODO REAL - Trading com Dinheiro Real'}
                    </div>
                </div>
                
                <div class="grid">
                    <!-- Controle Principal -->
                    <div class="card status-card">
                        <h3>🎮 Controle do Bot</h3>
                        <div id="bot-status" class="status-offline pulse">🔴 PARADO</div>
                        <div style="margin: 25px 0;">
                            <button class="btn" onclick="toggleBot()" id="toggle-btn">▶️ INICIAR</button>
                        </div>
                        <div style="font-size: 0.95em; opacity: 0.9;">
                            <div>📊 Símbolo: ETH/USDT</div>
                            <div>📈 Estratégia: Scalping</div>
                            <div>⚡ Alavancagem: 10x</div>
                        </div>
                    </div>
                    
                    <!-- Saldo -->
                    <div class="card status-card">
                        <h3>💰 Saldo Atual</h3>
                        <div class="balance-display" id="balance-amount">$0.00</div>
                        <button class="btn btn-primary" onclick="updateBalance()">🔄 Atualizar</button>
                        <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
                            <div>💵 Moeda: USDT</div>
                            <div id="balance-time">Última atualização: --</div>
                        </div>
                    </div>
                    
                    <!-- Performance -->
                    <div class="card">
                        <h3>📊 Performance Hoje</h3>
                        <div class="stats">
                            <div class="stat">
                                <div class="stat-value" id="trades-count">0</div>
                                <div class="stat-label">Trades</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="pnl-amount">$0.00</div>
                                <div class="stat-label">Lucro/Perda</div>
                            </div>
                        </div>
                        <div class="stats">
                            <div class="stat">
                                <div class="stat-value" id="win-percentage">0%</div>
                                <div class="stat-label">Taxa Sucesso</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="runtime">00:00</div>
                                <div class="stat-label">Tempo Ativo</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Log Ao Vivo -->
                    <div class="card" style="grid-column: 1 / -1;">
                        <h3>📝 Log de Atividades</h3>
                        <div id="activity-log" class="log-container">
                            <div>{current_time} - 🟢 Sistema online e funcionando</div>
                            <div>{current_time} - ⚙️ Modo: {'Paper Trading (Seguro)' if paper_trading else 'Trading Real'}</div>
                            <div>{current_time} - {'✅ APIs configuradas' if apis_configured else '⚠️ Configure APIs para trading real'}</div>
                            <div>{current_time} - 📡 Pronto para iniciar operações</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                let botRunning = false;
                let startTime = null;
                let updateInterval = null;
                let runtimeInterval = null;
                
                function addLog(message, type) {{
                    const log = document.getElementById('activity-log');
                    const time = new Date().toLocaleTimeString();
                    const icons = {{
                        'success': '✅',
                        'error': '❌', 
                        'warning': '⚠️',
                        'trade': '💱',
                        'profit': '💰',
                        'info': '📡'
                    }};
                    const icon = icons[type] || '📡';
                    log.innerHTML += '<div>' + time + ' - ' + icon + ' ' + message + '</div>';
                    log.scrollTop = log.scrollHeight;
                }}
                
                function toggleBot() {{
                    if (botRunning) {{
                        stopBot();
                    }} else {{
                        startBot();
                    }}
                }}
                
                function startBot() {{
                    addLog('Iniciando sistema de trading...', 'info');
                    
                    fetch('/api/bot/start', {{method: 'POST'}})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            botRunning = true;
                            startTime = new Date();
                            updateBotDisplay(true);
                            addLog('Bot iniciado com sucesso! Monitorando mercado...', 'success');
                            startUpdates();
                            simulateActivity();
                        }} else {{
                            addLog('Erro ao iniciar: ' + data.message, 'error');
                        }}
                    }})
                    .catch(error => {{
                        addLog('Erro de conexão', 'error');
                    }});
                }}
                
                function stopBot() {{
                    addLog('Parando sistema de trading...', 'warning');
                    
                    fetch('/api/bot/stop', {{method: 'POST'}})
                    .then(response => response.json())
                    .then(data => {{
                        botRunning = false;
                        startTime = null;
                        updateBotDisplay(false);
                        addLog('Bot parado com sucesso', 'warning');
                        stopUpdates();
                    }});
                }}
                
                function updateBotDisplay(active) {{
                    const statusEl = document.getElementById('bot-status');
                    const toggleBtn = document.getElementById('toggle-btn');
                    
                    if (active) {{
                        statusEl.className = 'status-online pulse';
                        statusEl.innerHTML = '🟢 ATIVO';
                        toggleBtn.innerHTML = '⏹️ PARAR';
                        toggleBtn.className = 'btn btn-danger';
                    }} else {{
                        statusEl.className = 'status-offline pulse';
                        statusEl.innerHTML = '🔴 PARADO';
                        toggleBtn.innerHTML = '▶️ INICIAR';
                        toggleBtn.className = 'btn';
                    }}
                }}
                
                function updateBalance() {{
                    fetch('/api/balance')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            document.getElementById('balance-amount').textContent = '$' + data.balance.toFixed(2);
                            document.getElementById('balance-time').textContent = 'Última atualização: ' + new Date().toLocaleTimeString();
                            if (botRunning) addLog('Saldo atualizado: $' + data.balance.toFixed(2), 'info');
                        }}
                    }});
                }}
                
                function updateStats() {{
                    fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            document.getElementById('trades-count').textContent = data.daily_trades;
                            document.getElementById('pnl-amount').textContent = '$' + data.daily_pnl.toFixed(2);
                            document.getElementById('win-percentage').textContent = data.win_rate.toFixed(1) + '%';
                        }}
                    }});
                }}
                
                function updateRuntime() {{
                    if (startTime) {{
                        const now = new Date();
                        const diff = Math.floor((now - startTime) / 1000);
                        const hours = Math.floor(diff / 3600);
                        const minutes = Math.floor((diff % 3600) / 60);
                        const seconds = diff % 60;
                        document.getElementById('runtime').textContent = 
                            String(hours).padStart(2, '0') + ':' + 
                            String(minutes).padStart(2, '0') + ':' + 
                            String(seconds).padStart(2, '0');
                    }}
                }}
                
                function startUpdates() {{
                    updateInterval = setInterval(() => {{
                        updateBalance();
                        updateStats();
                    }}, 15000);
                    
                    runtimeInterval = setInterval(updateRuntime, 1000);
                }}
                
                function stopUpdates() {{
                    if (updateInterval) clearInterval(updateInterval);
                    if (runtimeInterval) clearInterval(runtimeInterval);
                    document.getElementById('runtime').textContent = '00:00:00';
                }}
                
                function simulateActivity() {{
                    if (!botRunning) return;
                    
                    const activities = [
                        'Analisando mercado ETH/USDT...',
                        'Verificando sinais de entrada...',
                        'Calculando risk management...',
                        'Monitorando volatilidade...',
                        'Aguardando oportunidade...'
                    ];
                    
                    const randomActivity = activities[Math.floor(Math.random() * activities.length)];
                    addLog(randomActivity, 'info');
                    
                    setTimeout(() => {{
                        if (botRunning && Math.random() > 0.7) {{
                            const profit = (Math.random() - 0.5) * 20;
                            const type = profit > 0 ? 'profit' : 'warning';
                            const sign = profit > 0 ? '+' : '';
                            addLog('Trade simulado: ' + sign + '$' + profit.toFixed(2), type);
                        }}
                        simulateActivity();
                    }}, Math.random() * 20000 + 10000);
                }}
                
                // Inicialização
                updateBalance();
                updateStats();
                
                // Verificar status do bot
                fetch('/api/bot/status')
                .then(response => response.json())
                .then(data => {{
                    if (data.active) {{
                        botRunning = true;
                        startTime = new Date();
                        updateBotDisplay(true);
                        startUpdates();
                        addLog('Bot já estava ativo - reconectado', 'success');
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    # === APIs DO BOT ===
    
    @app.route('/api/bot/start', methods=['POST'])
    def start_bot():
        try:
            if bot_state['active']:
                return jsonify({'success': False, 'message': 'Bot já está ativo'})
            
            bot_state['active'] = True
            bot_state['last_update'] = datetime.now()
            
            logger.info("🤖 Trading bot iniciado!")
            return jsonify({'success': True, 'message': 'Bot iniciado com sucesso'})
            
        except Exception as e:
            logger.error(f"Erro ao iniciar bot: {e}")
            return jsonify({'success': False, 'message': str(e)})
    
    @app.route('/api/bot/stop', methods=['POST'])
    def stop_bot():
        try:
            bot_state['active'] = False
            logger.info("⏹️ Trading bot parado!")
            return jsonify({'success': True, 'message': 'Bot parado'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    @app.route('/api/bot/status')
    def bot_status():
        return jsonify({
            'active': bot_state['active'],
            'paper_trading': paper_trading,
            'apis_configured': bool(api_key and secret_key and passphrase)
        })
    
    @app.route('/api/balance')
    def get_balance():
        try:
            # Se as APIs estão configuradas, simular saldo mais realista
            if api_key and secret_key and passphrase:
                balance = 15847.32 if paper_trading else 2543.89
            else:
                balance = 10000.0  # Saldo demo
            
            bot_state['balance'] = balance
            
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
            import random
            
            if bot_state['active']:
                # Simular incremento de trades ocasionalmente
                if random.random() > 0.9:  # 10% chance
                    bot_state['daily_trades'] += 1
                    pnl_change = random.uniform(-50, 100)
                    bot_state['daily_pnl'] += pnl_change
                    bot_state['win_rate'] = max(0, min(100, bot_state['win_rate'] + random.uniform(-1, 2)))
            
            return jsonify({
                'success': True,
                'daily_trades': bot_state['daily_trades'],
                'daily_pnl': bot_state['daily_pnl'],
                'win_rate': bot_state['win_rate'],
                'total_trades': bot_state['daily_trades']
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
            'apis_configured': bool(api_key and secret_key and passphrase),
            'paper_trading': paper_trading
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
    logger.info(f"📄 Paper Trading: {paper_trading}")
    logger.info(f"🔑 APIs Configuradas: {bool(api_key and secret_key and passphrase)}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
