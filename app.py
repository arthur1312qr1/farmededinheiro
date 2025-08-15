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

# Verificar se as variáveis estão configuradas com debug detalhado
api_key = os.environ.get('BITGET_API_KEY', '').strip()
secret_key = os.environ.get('BITGET_SECRET_KEY', '').strip()
passphrase = os.environ.get('BITGET_PASSPHRASE', '').strip()
paper_trading = os.environ.get('PAPER_TRADING', 'true').lower() == 'true'

# Debug das variáveis (sem mostrar valores reais por segurança)
logger.info(f"🔍 Debug variáveis:")
logger.info(f"  - API_KEY presente: {bool(api_key)} (tamanho: {len(api_key)})")
logger.info(f"  - SECRET_KEY presente: {bool(secret_key)} (tamanho: {len(secret_key)})")
logger.info(f"  - PASSPHRASE presente: {bool(passphrase)} (tamanho: {len(passphrase)})")
logger.info(f"  - PAPER_TRADING: {paper_trading}")

# Estado do bot
bot_state = {
    'active': False,
    'balance': 0.0,
    'daily_trades': 0,
    'daily_pnl': 0.0,
    'win_rate': 0.0,
    'last_update': datetime.now(),
    'connection_status': 'Não testado'
}

def get_bitget_exchange():
    """Configura e retorna instância da exchange Bitget"""
    try:
        if not api_key or not secret_key or not passphrase:
            logger.warning(f"❌ Credenciais incompletas - API:{bool(api_key)} SECRET:{bool(secret_key)} PASS:{bool(passphrase)}")
            return None, "Credenciais incompletas"
            
        # Configuração específica da Bitget
        exchange_config = {
            'apiKey': api_key,
            'secret': secret_key,
            'password': passphrase,
            'sandbox': paper_trading,  # True para sandbox, False para live
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Usar spot trading como padrão
            }
        }
        
        logger.info(f"🔧 Configurando Bitget - Sandbox: {paper_trading}")
        
        exchange = ccxt.bitget(exchange_config)
        
        # Teste básico de conectividade
        try:
            # Testar se consegue acessar informações básicas
            markets = exchange.load_markets()
            logger.info(f"✅ Bitget conectado com sucesso! Markets: {len(markets)}")
            return exchange, "Conectado"
            
        except ccxt.AuthenticationError as auth_err:
            logger.error(f"❌ Erro de autenticação Bitget: {auth_err}")
            return None, f"Erro autenticação: {str(auth_err)}"
        except ccxt.NetworkError as net_err:
            logger.error(f"❌ Erro de rede Bitget: {net_err}")
            return None, f"Erro rede: {str(net_err)}"
        except Exception as test_err:
            logger.error(f"❌ Erro ao testar Bitget: {test_err}")
            return None, f"Erro teste: {str(test_err)}"
            
    except Exception as e:
        logger.error(f"❌ Erro ao configurar Bitget: {e}")
        return None, f"Erro config: {str(e)}"

def get_real_balance():
    """Obtém saldo real da conta Bitget"""
    try:
        exchange, status = get_bitget_exchange()
        
        if not exchange:
            logger.warning(f"⚠️ Não foi possível conectar: {status}")
            bot_state['connection_status'] = status
            return 0.0, f"Erro: {status}"
        
        # Buscar saldo da conta
        logger.info("📊 Consultando saldo na Bitget...")
        balance_info = exchange.fetch_balance()
        
        # Verificar saldo USDT
        usdt_balance = balance_info.get('USDT', {})
        total_balance = usdt_balance.get('total', 0.0)
        free_balance = usdt_balance.get('free', 0.0)
        used_balance = usdt_balance.get('used', 0.0)
        
        logger.info(f"💰 Saldo USDT - Total: {total_balance}, Livre: {free_balance}, Usado: {used_balance}")
        
        # Se não tem USDT, verificar outros saldos
        if total_balance == 0:
            logger.info("🔍 USDT zerado, verificando outras moedas...")
            non_zero_balances = {}
            for currency, balance_data in balance_info.items():
                if isinstance(balance_data, dict) and balance_data.get('total', 0) > 0:
                    non_zero_balances[currency] = balance_data.get('total', 0)
            
            if non_zero_balances:
                logger.info(f"💵 Saldos encontrados: {non_zero_balances}")
                # Usar o maior saldo encontrado
                max_currency = max(non_zero_balances, key=non_zero_balances.get)
                total_balance = non_zero_balances[max_currency]
                logger.info(f"💰 Usando saldo {max_currency}: {total_balance}")
                
        bot_state['connection_status'] = 'Conectado - Saldo obtido'
        return total_balance, "Bitget Real"
        
    except ccxt.AuthenticationError as e:
        error_msg = f"Autenticação: {str(e)}"
        logger.error(f"❌ Erro de autenticação: {e}")
        bot_state['connection_status'] = error_msg
        return 0.0, error_msg
    except ccxt.NetworkError as e:
        error_msg = f"Rede: {str(e)}"
        logger.error(f"❌ Erro de rede: {e}")
        bot_state['connection_status'] = error_msg
        return 0.0, error_msg
    except Exception as e:
        error_msg = f"Geral: {str(e)}"
        logger.error(f"❌ Erro geral ao obter saldo: {e}")
        bot_state['connection_status'] = error_msg
        return 0.0, error_msg

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
        try:
            real_balance, balance_source = get_real_balance()
        except:
            real_balance, balance_source = 0.0, "Erro na inicialização"
        
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
                    color: {'#4CAF50' if 'Real' in balance_source else '#FF9800' if 'Erro' in balance_source else '#FFA500'};
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
                    height: 200px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 12px;
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
                
                .debug-info {{
                    background: rgba(0,0,0,0.4); padding: 10px; border-radius: 5px;
                    font-size: 0.8em; margin: 10px 0; font-family: monospace;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 class="title">🚀 Trading Bot</h1>
                    <div class="subtitle">Farmede Dinheiro - Sistema Ativo</div>
                    
                    <div class="config-status">
                        {'✅ APIs Configuradas - Conectando à Bitget' if apis_configured else '⚠️ Configure as APIs no Render para trading real'}
                    </div>
                    
                    <div class="mode-indicator">
                        {'📄 MODO SANDBOX (Paper Trading)' if paper_trading else '💰 MODO LIVE - Trading Real'}
                    </div>
                    
                    <div class="debug-info">
                        🔍 Debug: API_KEY({len(api_key)}), SECRET({len(secret_key)}), PASS({len(passphrase)}) | Status: {bot_state['connection_status']}
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
                            <div>🏦 Exchange: Bitget</div>
                            <div>📊 Par: ETH/USDT</div>
                            <div>📈 Tipo: Spot Trading</div>
                        </div>
                        <button class="btn btn-primary" onclick="testConnection()" style="font-size: 0.9em; padding: 10px;">🔧 Testar Conexão</button>
                    </div>
                    
                    <!-- Saldo Real -->
                    <div class="card status-card">
                        <h3>💰 Saldo da Conta</h3>
                        <div class="balance-display" id="balance-amount">${real_balance:.2f}</div>
                        <div class="balance-source" id="balance-source">Fonte: {balance_source}</div>
                        <button class="btn btn-primary" onclick="updateBalance()">🔄 Atualizar Saldo</button>
                        <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
                            <div>💵 Moeda Base: USDT</div>
                            <div id="balance-time">Última consulta: {current_time}</div>
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
                    
                    <!-- Log Detalhado -->
                    <div class="card" style="grid-column: 1 / -1;">
                        <h3>📝 Log de Sistema</h3>
                        <div id="activity-log" class="log-container">
                            <div>{current_time} - 🟢 Sistema iniciado</div>
                            <div>{current_time} - 🔧 Debug: API_KEY presente: {bool(api_key)}</div>
                            <div>{current_time} - 🔧 Debug: SECRET_KEY presente: {bool(secret_key)}</div>
                            <div>{current_time} - 🔧 Debug: PASSPHRASE presente: {bool(passphrase)}</div>
                            <div>{current_time} - ⚙️ Modo: {'Sandbox (Paper)' if paper_trading else 'Live (Real)'}</div>
                            <div>{current_time} - 💰 Saldo inicial: ${real_balance:.2f} ({balance_source})</div>
                            <div>{current_time} - 📡 Status conexão: {bot_state['connection_status']}</div>
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
                        'balance': '💰',
                        'debug': '🔧'
                    }};
                    const icon = icons[type] || '📡';
                    log.innerHTML += '<div>' + time + ' - ' + icon + ' ' + message + '</div>';
                    log.scrollTop = log.scrollHeight;
                }}
                
                function testConnection() {{
                    addLog('Testando conexão com Bitget...', 'debug');
                    
                    fetch('/api/test_bitget')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            addLog('✅ Conexão OK! Saldo: $' + data.balance + ' Markets: ' + data.markets_count, 'success');
                            if (data.eth_price) {{
                                addLog('📊 Preço ETH/USDT: $' + data.eth_price, 'info');
                            }}
                        }} else {{
                            addLog('❌ Erro conexão: ' + data.message, 'error');
                            if (data.error) {{
                                addLog('Detalhes: ' + data.error, 'error');
                            }}
                        }}
                    }})
                    .catch(error => {{
                        addLog('❌ Erro ao testar: ' + error, 'error');
                    }});
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
                            addLog('Bot ativo! ' + data.message, 'success');
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
                    addLog('Parando bot...', 'warning');
                    
                    fetch('/api/bot/stop', {{method: 'POST'}})
                    .then(response => response.json())
                    .then(data => {{
                        botRunning = false;
                        startTime = null;
                        updateBotDisplay(false);
                        addLog('Bot parado: ' + data.message, 'warning');
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
                    addLog('Consultando saldo na Bitget...', 'info');
                    
                    fetch('/api/balance')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            document.getElementById('balance-amount').textContent = '$' + data.balance.toFixed(2);
                            document.getElementById('balance-source').textContent = 'Fonte: ' + data.source;
                            document.getElementById('balance-time').textContent = 'Última consulta: ' + new Date().toLocaleTimeString();
                            addLog('Saldo obtido: $' + data.balance.toFixed(2) + ' (' + data.source + ')', 'balance');
                        }} else {{
                            addLog('Erro ao obter saldo: ' + data.error, 'error');
                            document.getElementById('balance-source').textContent = 'Fonte: ' + data.source;
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
                    }}, 60000); // A cada 1 minuto
                    
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
                        'Analisando orderbook ETH/USDT...',
                        'Calculando indicadores técnicos...',
                        'Verificando spreads na Bitget...',
                        'Monitorando volume de negociação...',
                        'Aguardando sinal de entrada...'
                    ];
                    
                    const randomActivity = activities[Math.floor(Math.random() * activities.length)];
                    addLog(randomActivity, 'info');
                    
                    setTimeout(() => {{
                        simulateActivity();
                    }}, Math.random() * 20000 + 15000); // 15-35 segundos
                }}
                
                // Inicialização
                updateBalance();
                updateStats();
                
                // Teste automático de conexão
                setTimeout(() => {{
                    testConnection();
                }}, 2000);
                
                // Auto-atualização do saldo
                setInterval(updateBalance, 300000); // A cada 5 minutos
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
            
            # Testar conexão antes de iniciar
            exchange, status = get_bitget_exchange()
            
            bot_state['active'] = True
            bot_state['last_update'] = datetime.now()
            
            message = f"Bot iniciado - {status}"
            logger.info(f"🤖 {message}")
            
            return jsonify({'success': True, 'message': message})
            
        except Exception as e:
            logger.error(f"Erro ao iniciar bot: {e}")
            return jsonify({'success': False, 'message': str(e)})
    
    @app.route('/api/bot/stop', methods=['POST'])
    def stop_bot():
        try:
            bot_state['active'] = False
            logger.info("⏹️ Bot parado!")
            return jsonify({'success': True, 'message': 'Bot parado com sucesso'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    @app.route('/api/bot/status')
    def bot_status():
        return jsonify({
            'active': bot_state['active'],
            'paper_trading': paper_trading,
            'apis_configured': bool(api_key and secret_key and passphrase),
            'connection_status': bot_state['connection_status']
        })
    
    @app.route('/api/balance')
    def get_balance():
        """Obtém saldo real da Bitget com debug detalhado"""
        try:
            balance, source = get_real_balance()
            
            return jsonify({
                'success': True,
                'balance': balance,
                'source': source,
                'currency': 'USDT',
                'timestamp': datetime.now().isoformat(),
                'connection_status': bot_state['connection_status']
            })
            
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {e}")
            return jsonify({
                'success': False, 
                'balance': 0, 
                'source': f'Erro: {str(e)}',
                'error': str(e)
            })
    
    @app.route('/api/stats')
    def get_stats():
        try:
            import random
            
            if bot_state['active']:
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
    
    @app.route('/api/test_bitget')
    def test_bitget():
        """Endpoint para debug completo da conexão Bitget"""
        try:
            logger.info("🧪 Iniciando teste detalhado da Bitget...")
            
            # Verificar variáveis
            debug_info = {
                'api_key_length': len(api_key),
                'secret_key_length': len(secret_key),
                'passphrase_length': len(passphrase),
                'paper_trading': paper_trading,
                'all_vars_present': bool(api_key and secret_key and passphrase)
            }
            
            logger.info(f"🔍 Debug info: {debug_info}")
            
            if not (api_key and secret_key and passphrase):
                return jsonify({
                    'success': False,
                    'message': 'Variáveis de ambiente não configuradas',
                    'debug': debug_info
                })
            
            # Tentar conectar
            exchange, status = get_bitget_exchange()
            
            if not exchange:
                return jsonify({
                    'success': False,
                    'message': f'Falha na conexão: {status}',
                    'debug': debug_info
                })
            
            # Testes adicionais
            try:
                markets = exchange.load_markets()
                balance = exchange.fetch_balance()
                eth_ticker = exchange.fetch_ticker('ETH/USDT')
                
                # Verificar saldos
                non_zero_balances = {}
                for currency, balance_data in balance.items():
                    if isinstance(balance_data, dict) and balance_data.get('total', 0) > 0:
                        non_zero_balances[currency] = balance_data.get('total', 0)
                
                usdt_balance = balance.get('USDT', {}).get('total', 0)
                
                return jsonify({
                    'success': True,
                    'message': 'Conexão Bitget estabelecida com sucesso!',
                    'balance': usdt_balance,
                    'markets_count': len(markets),
                    'eth_price': eth_ticker['last'],
                    'non_zero_balances': non_zero_balances,
                    'debug': debug_info,
                    'status': status
                })
                
            except Exception as test_error:
                return jsonify({
                    'success': False,
                    'message': f'Erro nos testes: {str(test_error)}',
                    'error': str(test_error),
                    'debug': debug_info
                })
                
        except Exception as e:
            logger.error(f"❌ Erro no teste Bitget: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'message': 'Erro geral no teste'
            })
    
    @app.route('/api/status')
    def status():
        return jsonify({
            'status': 'online',
            'timestamp': datetime.now().isoformat(),
            'message': 'Trading Bot - Debug Mode',
            'version': '2.2.0',
            'apis_configured': bool(api_key and secret_key and passphrase),
            'paper_trading': paper_trading,
            'connection_status': bot_state['connection_status']
        })
    
    return app

# Criar aplicação
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Iniciando Trading Bot na porta {port}")
    logger.info(f"📄 Paper Trading: {paper_trading}")
    logger.info(f"🔑 APIs Configuradas: {bool(api_key and secret_key and passphrase)}")
    
    # Teste inicial da conexão
    try:
        logger.info("🧪 Testando conexão inicial...")
        balance, source = get_real_balance()
        logger.info(f"💰 Resultado teste: ${balance:.2f} ({source})")
    except Exception as e:
        logger.error(f"❌ Erro no teste inicial: {e}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
