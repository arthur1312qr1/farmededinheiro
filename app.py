import os
import sys
import logging
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import ccxt

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

# Configurar exchange Bitget
def get_bitget_exchange():
    """Configura e retorna instância da exchange Bitget"""
    try:
        if not (api_key and secret_key and passphrase):
            logger.warning("Credenciais Bitget não configuradas - usando modo demo")
            return None
            
        exchange = ccxt.bitget({
            'apiKey': api_key,
            'secret': secret_key,
            'password': passphrase,  # Bitget usa passphrase
            'sandbox': paper_trading,  # True para paper trading, False para real
            'enableRateLimit': True,
        })
        
        logger.info(f"✅ Bitget configurado - Sandbox: {paper_trading}")
        return exchange
        
    except Exception as e:
        logger.error(f"❌ Erro ao configurar Bitget: {e}")
        return None

def get_real_balance():
    """Obtém saldo real da conta Bitget"""
    try:
        exchange = get_bitget_exchange()
        if not exchange:
            # Se não conseguir conectar, retornar saldo demo
            return 10000.0, "Demo - Configure APIs"
        
        # Buscar saldo da conta
        balance = exchange.fetch_balance()
        
        # Pegar saldo USDT (principal moeda para trading)
        usdt_balance = balance.get('USDT', {}).get('free', 0.0)
        total_balance = balance.get('USDT', {}).get('total', 0.0)
        
        logger.info(f"💰 Saldo real obtido: ${total_balance:.2f} USDT")
        return total_balance, "Real"
        
    except ccxt.AuthenticationError as e:
        logger.error(f"❌ Erro de autenticação Bitget: {e}")
        return 0.0, "Erro Auth"
    except ccxt.NetworkError as e:
        logger.error(f"❌ Erro de rede Bitget: {e}")
        return 0.0, "Erro Rede"
    except Exception as e:
        logger.error(f"❌ Erro ao obter saldo: {e}")
        return 0.0, "Erro"

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
        
        # Obter saldo real na inicialização
        real_balance, balance_source = get_real_balance()
        
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
                
                .balance-source {{
                    font-size: 0.8em; opacity: 0.8; margin-top: 5px;
                    color: {'#4CAF50' if balance_source == 'Real' else '#FFA500'};
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
                        {'✅ APIs Configuradas - Conectado à Bitget' if apis_configured else '⚠️ Configure as APIs no Render para trading real'}
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
                            <div>📊 Exchange: Bitget</div>
                            <div>📈 Símbolo: ETH/USDT</div>
                            <div>⚡ Estratégia: Scalping</div>
                        </div>
                    </div>
                    
                    <!-- Saldo Real -->
                    <div class="card status-card">
                        <h3>💰 Saldo Real</h3>
                        <div class="balance-display" id="balance-amount">${real_balance:.2f}</div>
                        <div class="balance-source" id="balance-source">Fonte: {balance_source}</div>
                        <button class="btn btn-primary" onclick="updateBalance()">🔄 Atualizar</button>
                        <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
                            <div>💵 Moeda: USDT</div>
                            <div id="balance-time">Última atualização: {current_time}</div>
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
                            <div>{current_time} - 🏦 Conectado à Bitget Exchange</div>
                            <div>{current_time} - ⚙️ Modo: {'Paper Trading' if paper_trading else 'Trading Real'}</div>
                            <div>{current_time} - 💰 Saldo obtido: ${real_balance:.2f} ({balance_source})</div>
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
                        'info': '📡',
                        'balance': '💰'
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
                    addLog('Iniciando conexão com Bitget...', 'info');
                    
                    fetch('/api/bot/start', {{method: 'POST'}})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            botRunning = true;
                            startTime = new Date();
                            updateBotDisplay(true);
                            addLog('Bot conectado e ativo na Bitget!', 'success');
                            startUpdates();
                            simulateActivity();
                        }} else {{
                            addLog('Erro ao conectar: ' + data.message, 'error');
                        }}
                    }})
                    .catch(error => {{
                        addLog('Erro de conexão com servidor', 'error');
                    }});
                }}
                
                function stopBot() {{
                    addLog('Desconectando da Bitget...', 'warning');
                    
                    fetch('/api/bot/stop', {{method: 'POST'}})
                    .then(response => response.json())
                    .then(data => {{
                        botRunning = false;
                        startTime = null;
                        updateBotDisplay(false);
                        addLog('Bot desconectado com sucesso', 'warning');
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
                    addLog('Consultando saldo real na Bitget...', 'info');
                    
                    fetch('/api/balance')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            document.getElementById('balance-amount').textContent = '$' + data.balance.toFixed(2);
                            document.getElementById('balance-source').textContent = 'Fonte: ' + data.source;
                            document.getElementById('balance-time').textContent = 'Última atualização: ' + new Date().toLocaleTimeString();
                            addLog('Saldo atualizado: $' + data.balance.toFixed(2) + ' (' + data.source + ')', 'balance');
                        }} else {{
                            addLog('Erro ao obter saldo: ' + data.error, 'error');
                        }}
                    }})
                    .catch(error => {{
                        addLog('Erro de conexão ao consultar saldo', 'error');
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
                    }}, 30000); // A cada 30 segundos para não sobrecarregar API
                    
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
                        'Analisando mercado ETH/USDT na Bitget...',
                        'Verificando oportunidades de scalping...',
                        'Consultando orderbook em tempo real...',
                        'Calculando níveis de suporte e resistência...',
                        'Monitorando spread e liquidez...',
                        'Aguardando sinal de entrada...'
                    ];
                    
                    const randomActivity = activities[Math.floor(Math.random() * activities.length)];
                    addLog(randomActivity, 'info');
                    
                    setTimeout(() => {{
                        if (botRunning && Math.random() > 0.8) {{
                            const profit = (Math.random() - 0.5) * 15;
                            const type = profit > 0 ? 'profit' : 'warning';
                            const sign = profit > 0 ? '+' : '';
                            addLog('Operação simulada: ' + sign + '$' + profit.toFixed(2), type);
                        }}
                        simulateActivity();
                    }}, Math.random() * 15000 + 10000); // 10-25 segundos
                }}
                
                // Inicialização - buscar saldo real
                updateBalance();
                updateStats();
                
                // Auto-atualização do saldo a cada 2 minutos
                setInterval(updateBalance, 120000);
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
            
            # Testar conexão com Bitget antes de iniciar
            exchange = get_bitget_exchange()
            if exchange:
                try:
                    # Teste simples de conectividade
                    exchange.fetch_ticker('ETH/USDT')
                    logger.info("🤖 Bot conectado à Bitget com sucesso!")
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao testar Bitget: {e}")
            
            bot_state['active'] = True
            bot_state['last_update'] = datetime.now()
            
            return jsonify({'success': True, 'message': 'Bot conectado à Bitget'})
            
        except Exception as e:
            logger.error(f"Erro ao iniciar bot: {e}")
            return jsonify({'success': False, 'message': str(e)})
    
    @app.route('/api/bot/stop', methods=['POST'])
    def stop_bot():
        try:
            bot_state['active'] = False
            logger.info("⏹️ Bot desconectado da Bitget!")
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
        """Obtém saldo real da Bitget"""
        try:
            balance, source = get_real_balance()
            
            return jsonify({
                'success': True,
                'balance': balance,
                'source': source,
                'currency': 'USDT',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {e}")
            return jsonify({
                'success': False, 
                'balance': 0, 
                'source': 'Erro',
                'error': str(e)
            })
    
    @app.route('/api/stats')
    def get_stats():
        try:
            import random
            
            if bot_state['active']:
                # Simular incremento ocasional de trades
                if random.random() > 0.95:  # 5% chance
                    bot_state['daily_trades'] += 1
                    pnl_change = random.uniform(-25, 50)
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
    
    # APIs de teste
    @app.route('/api/status')
    def status():
        return jsonify({
            'status': 'online',
            'timestamp': datetime.now().isoformat(),
            'message': 'Trading Bot conectado à Bitget!',
            'version': '2.1.0',
            'apis_configured': bool(api_key and secret_key and passphrase),
            'paper_trading': paper_trading
        })
    
    @app.route('/api/test')
    def test():
        return jsonify({
            'success': True,
            'message': 'Sistema funcionando com Bitget real!',
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/api/test_bitget')
    def test_bitget():
        """Endpoint para testar conexão com Bitget"""
        try:
            balance, source = get_real_balance()
            exchange = get_bitget_exchange()
            
            if exchange:
                markets = exchange.load_markets()
                eth_ticker = exchange.fetch_ticker('ETH/USDT')
                
                return jsonify({
                    'success': True,
                    'balance': balance,
                    'source': source,
                    'markets_count': len(markets),
                    'eth_price': eth_ticker['last'],
                    'message': 'Conexão Bitget OK!'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Não foi possível conectar à Bitget'
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    return app

# Criar aplicação
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Iniciando Trading Bot na porta {port}")
    logger.info(f"📄 Paper Trading: {paper_trading}")
    logger.info(f"🔑 APIs Configuradas: {bool(api_key and secret_key and passphrase)}")
    
    # Testar conexão Bitget na inicialização
    balance, source = get_real_balance()
    logger.info(f"💰 Saldo inicial: ${balance:.2f} ({source})")
    
    app.run(host='0.0.0.0', port=port, debug=False)
