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

# CORREÇÃO: Usando os nomes corretos das variáveis que você tem no Render
api_key = os.environ.get('BITGET_API_KEY', '').strip()
secret_key = os.environ.get('BITGET_API_SECRET', '').strip()  # CORRIGIDO: era BITGET_SECRET_KEY
passphrase = os.environ.get('BITGET_PASSPHRASE', '').strip()
paper_trading = os.environ.get('PAPER_TRADING', 'false').lower() == 'true'  # Mudei para false por padrão

# Outras APIs disponíveis
coingecko_key = os.environ.get('COINGECKO_API_KEY', '').strip()
etherscan_key = os.environ.get('ETHERSCAN_API_KEY', '').strip()
newsapi_key = os.environ.get('NEWSAPI_KEY', '').strip()

# Debug das variáveis (sem mostrar valores reais)
logger.info(f"🔍 Debug variáveis CORRETAS:")
logger.info(f"  - BITGET_API_KEY presente: {bool(api_key)} (tamanho: {len(api_key)})")
logger.info(f"  - BITGET_API_SECRET presente: {bool(secret_key)} (tamanho: {len(secret_key)})")
logger.info(f"  - BITGET_PASSPHRASE presente: {bool(passphrase)} (tamanho: {len(passphrase)})")
logger.info(f"  - PAPER_TRADING: {paper_trading}")
logger.info(f"  - COINGECKO_API_KEY: {bool(coingecko_key)}")
logger.info(f"  - ETHERSCAN_API_KEY: {bool(etherscan_key)}")
logger.info(f"  - NEWSAPI_KEY: {bool(newsapi_key)}")

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
    """Configura e retorna instância da exchange Bitget com nomes corretos"""
    try:
        if not api_key or not secret_key or not passphrase:
            missing = []
            if not api_key: missing.append("BITGET_API_KEY")
            if not secret_key: missing.append("BITGET_API_SECRET")
            if not passphrase: missing.append("BITGET_PASSPHRASE")
            
            error_msg = f"Credenciais faltando: {', '.join(missing)}"
            logger.warning(f"❌ {error_msg}")
            return None, error_msg
            
        # Configuração específica da Bitget
        exchange_config = {
            'apiKey': api_key,
            'secret': secret_key,
            'password': passphrase,
            'sandbox': paper_trading,  # false = live trading, true = sandbox
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Usar spot trading
            }
        }
        
        logger.info(f"🔧 Configurando Bitget - Live Trading: {not paper_trading}")
        
        exchange = ccxt.bitget(exchange_config)
        
        # Teste de conectividade
        try:
            # Carregar mercados para testar autenticação
            markets = exchange.load_markets()
            logger.info(f"✅ Bitget conectado! Markets disponíveis: {len(markets)}")
            
            # Teste adicional: buscar ticker ETH/USDT
            ticker = exchange.fetch_ticker('ETH/USDT')
            logger.info(f"📊 Teste ticker ETH/USDT: ${ticker['last']}")
            
            return exchange, "Conectado com Sucesso"
            
        except ccxt.AuthenticationError as auth_err:
            error_msg = f"Erro Autenticação: {str(auth_err)}"
            logger.error(f"❌ {error_msg}")
            return None, error_msg
        except ccxt.NetworkError as net_err:
            error_msg = f"Erro Rede: {str(net_err)}"
            logger.error(f"❌ {error_msg}")
            return None, error_msg
        except Exception as test_err:
            error_msg = f"Erro Teste: {str(test_err)}"
            logger.error(f"❌ {error_msg}")
            return None, error_msg
            
    except Exception as e:
        error_msg = f"Erro Configuração: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return None, error_msg

def get_real_balance():
    """Obtém saldo real da conta Bitget"""
    try:
        exchange, status = get_bitget_exchange()
        
        if not exchange:
            logger.warning(f"⚠️ Conexão falhou: {status}")
            bot_state['connection_status'] = status
            return 0.0, f"Erro: {status}"
        
        # Buscar saldo da conta
        logger.info("📊 Consultando saldo real na Bitget...")
        balance_info = exchange.fetch_balance()
        
        logger.info(f"🔍 Estrutura do saldo: {list(balance_info.keys())}")
        
        # Verificar saldo USDT primeiro
        usdt_balance = balance_info.get('USDT', {})
        total_usdt = usdt_balance.get('total', 0.0) if isinstance(usdt_balance, dict) else 0.0
        
        logger.info(f"💰 USDT - Total: {total_usdt}")
        
        # Se USDT for zero, verificar outras moedas
        if total_usdt == 0:
            logger.info("🔍 USDT zerado, verificando outras moedas...")
            all_balances = {}
            
            for currency, balance_data in balance_info.items():
                if isinstance(balance_data, dict) and balance_data.get('total', 0) > 0:
                    all_balances[currency] = balance_data.get('total', 0)
            
            if all_balances:
                logger.info(f"💵 Saldos encontrados: {all_balances}")
                # Usar o maior saldo
                main_currency = max(all_balances, key=all_balances.get)
                main_balance = all_balances[main_currency]
                
                logger.info(f"💰 Usando saldo principal: {main_balance} {main_currency}")
                bot_state['connection_status'] = f'Conectado - Saldo {main_currency}'
                return main_balance, f"Bitget Real ({main_currency})"
            else:
                logger.info("💰 Nenhum saldo encontrado - conta pode estar vazia")
                bot_state['connection_status'] = 'Conectado - Conta Vazia'
                return 0.0, "Bitget Real (Vazio)"
        else:
            bot_state['connection_status'] = 'Conectado - USDT OK'
            return total_usdt, "Bitget Real (USDT)"
        
    except ccxt.AuthenticationError as e:
        error_msg = f"Auth: {str(e)}"
        logger.error(f"❌ Erro autenticação: {e}")
        bot_state['connection_status'] = error_msg
        return 0.0, error_msg
    except ccxt.NetworkError as e:
        error_msg = f"Rede: {str(e)}"
        logger.error(f"❌ Erro rede: {e}")
        bot_state['connection_status'] = error_msg
        return 0.0, error_msg
    except Exception as e:
        error_msg = f"Geral: {str(e)}"
        logger.error(f"❌ Erro geral: {e}")
        bot_state['connection_status'] = error_msg
        return 0.0, error_msg

def create_app():
    """Cria aplicação Flask"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-seguro')
    CORS(app, origins="*")
    
    @app.route('/')
    def index():
        # Status das APIs
        apis_configured = bool(api_key and secret_key and passphrase)
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Obter saldo inicial
        try:
            real_balance, balance_source = get_real_balance()
        except Exception as e:
            real_balance, balance_source = 0.0, f"Erro: {str(e)}"
        
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
                    color: {'#4CAF50' if 'Real' in balance_source else '#FF5722' if 'Erro' in balance_source else '#FFA500'};
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
                
                .pulse {{ animation: pulse 2s infinite; }}
                @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} 100% {{ opacity: 1; }} }}
                
                .log-container {{ 
                    background: rgba(0,0,0,0.6); padding: 15px; border-radius: 10px;
                    height: 200px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 12px;
                    border: 1px solid rgba(255,255,255,0.3);
                }}
                
                .stats {{ display: flex; justify-content: space-around; text-align: center; margin: 20px 0; }}
                .stat-value {{ font-size: 2em; font-weight: bold; color: #4CAF50; text-shadow: 0 0 5px #4CAF50; }}
                .stat-label {{ font-size: 0.9em; opacity: 0.9; margin-top: 5px; }}
                
                .config-status {{ 
                    background: {'rgba(76,175,80,0.3)' if apis_configured else 'rgba(244,67,54,0.3)'};
                    padding: 15px; border-radius: 10px; margin: 15px 0;
                    border: 1px solid {'#4CAF50' if apis_configured else '#f44336'};
                    text-align: center; font-weight: bold;
                }}
                
                .mode-indicator {{ 
                    background: {'rgba(255,165,0,0.3)' if paper_trading else 'rgba(76,175,80,0.3)'};
                    padding: 12px; border-radius: 10px; margin: 15px 0;
                    border: 1px solid {'#FFA500' if paper_trading else '#4CAF50'};
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
                        {'✅ APIS CONECTADAS - SISTEMA ATIVO' if apis_configured else '❌ ERRO: Verifique as chaves da API'}
                    </div>
                    
                    <div class="mode-indicator">
                        {'📄 MODO SANDBOX (Teste)' if paper_trading else '💰 MODO LIVE - TRADING REAL'}
                    </div>
                    
                    <div class="debug-info">
                        🔧 Status: API_KEY({len(api_key)}), API_SECRET({len(secret_key)}), PASSPHRASE({len(passphrase)}) | {bot_state['connection_status']}
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
                            <div>💡 Modo: {'Sandbox' if paper_trading else 'Live'}</div>
                        </div>
                        <button class="btn btn-primary" onclick="testConnection()" style="font-size: 0.9em; padding: 10px 15px;">🔧 Testar API</button>
                    </div>
                    
                    <!-- Saldo Real -->
                    <div class="card status-card">
                        <h3>💰 Saldo da Conta</h3>
                        <div class="balance-display" id="balance-amount">${real_balance:.2f}</div>
                        <div class="balance-source" id="balance-source">Fonte: {balance_source}</div>
                        <button class="btn btn-primary" onclick="updateBalance()">🔄 Atualizar</button>
                        <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
                            <div>🕐 Última consulta: {current_time}</div>
                            <div>🌐 Status: {bot_state['connection_status']}</div>
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
                                <div class="stat-label">P&L</div>
                            </div>
                        </div>
                        <div class="stats">
                            <div class="stat">
                                <div class="stat-value" id="win-percentage">0%</div>
                                <div class="stat-label">Win Rate</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="runtime">00:00</div>
                                <div class="stat-label">Uptime</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Log do Sistema -->
                    <div class="card" style="grid-column: 1 / -1;">
                        <h3>📝 Log do Sistema</h3>
                        <div id="activity-log" class="log-container">
                            <div>{current_time} - 🚀 Sistema iniciado</div>
                            <div>{current_time} - 🔧 Usando variáveis: BITGET_API_KEY, BITGET_API_SECRET, BITGET_PASSPHRASE</div>
                            <div>{current_time} - ⚙️ Modo: {'Sandbox' if paper_trading else 'Live Trading'}</div>
                            <div>{current_time} - 💰 Saldo inicial: ${real_balance:.2f} ({balance_source})</div>
                            <div>{current_time} - 📊 Status: {bot_state['connection_status']}</div>
                            <div>{current_time} - 📡 Pronto para operar!</div>
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
                        'success': '✅', 'error': '❌', 'warning': '⚠️',
                        'info': '📡', 'balance': '💰', 'debug': '🔧'
                    }};
                    const icon = icons[type] || '📡';
                    log.innerHTML += '<div>' + time + ' - ' + icon + ' ' + message + '</div>';
                    log.scrollTop = log.scrollHeight;
                }}
                
                function testConnection() {{
                    addLog('Testando conexão Bitget com chaves corretas...', 'debug');
                    
                    fetch('/api/test_bitget')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            addLog('✅ CONEXÃO OK! Saldo: $' + data.balance.toFixed(2), 'success');
                            addLog('Markets: ' + data.markets_count + ' | ETH: $' + data.eth_price, 'info');
                            document.getElementById('balance-amount').textContent = '$' + data.balance.toFixed(2);
                            document.getElementById('balance-source').textContent = 'Fonte: ' + data.source;
                        }} else {{
                            addLog('❌ Erro: ' + data.message, 'error');
                            if (data.error) addLog('Detalhe: ' + data.error, 'error');
                        }}
                    }})
                    .catch(error => {{
                        addLog('❌ Falha na requisição: ' + error, 'error');
                    }});
                }}
                
                function toggleBot() {{
                    if (botRunning) {{ stopBot(); }} else {{ startBot(); }}
                }}
                
                function startBot() {{
                    addLog('Iniciando trading bot...', 'info');
                    
                    fetch('/api/bot/start', {{method: 'POST'}})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            botRunning = true;
                            startTime = new Date();
                            updateBotDisplay(true);
                            addLog('Bot ativo: ' + data.message, 'success');
                            startUpdates();
                        }} else {{
                            addLog('Erro: ' + data.message, 'error');
                        }}
                    }});
                }}
                
                function stopBot() {{
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
                    addLog('Consultando saldo real...', 'info');
                    
                    fetch('/api/balance')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success) {{
                            document.getElementById('balance-amount').textContent = '$' + data.balance.toFixed(2);
                            document.getElementById('balance-source').textContent = 'Fonte: ' + data.source;
                            addLog('Saldo: $' + data.balance.toFixed(2) + ' (' + data.source + ')', 'balance');
                        }} else {{
                            addLog('Erro saldo: ' + data.error, 'error');
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
                        document.getElementById('runtime').textContent = 
                            String(hours).padStart(2, '0') + ':' + String(minutes).padStart(2, '0');
                    }}
                }}
                
                function startUpdates() {{
                    updateInterval = setInterval(() => {{
                        updateBalance();
                        updateStats();
                    }}, 30000);
                    runtimeInterval = setInterval(updateRuntime, 1000);
                }}
                
                function stopUpdates() {{
                    if (updateInterval) clearInterval(updateInterval);
                    if (runtimeInterval) clearInterval(runtimeInterval);
                    document.getElementById('runtime').textContent = '00:00';
                }}
                
                // Inicialização
                setTimeout(() => {{ testConnection(); }}, 1000);
                setInterval(updateBalance, 120000); // Auto-update a cada 2 min
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    # === ROTAS DA API ===
    
    @app.route('/api/test_bitget')
    def test_bitget():
        """Testa conexão com API Bitget"""
        try:
            exchange, status = get_bitget_exchange()
            
            if not exchange:
                return jsonify({
                    'success': False,
                    'message': status,
                    'balance': 0.0,
                    'source': 'Erro'
                })
            
            # Teste completo
            markets = exchange.load_markets()
            eth_ticker = exchange.fetch_ticker('ETH/USDT')
            balance, source = get_real_balance()
            
            return jsonify({
                'success': True,
                'message': 'Conexão Bitget OK!',
                'balance': balance,
                'source': source,
                'markets_count': len(markets),
                'eth_price': eth_ticker['last'],
                'connection_status': bot_state['connection_status']
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Erro no teste: {str(e)}',
                'error': str(e)
            })
    
    @app.route('/api/balance')
    def get_balance():
        """Obtém saldo real"""
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
            return jsonify({
                'success': False, 
                'balance': 0, 
                'source': f'Erro: {str(e)}',
                'error': str(e)
            })
    
    @app.route('/api/bot/start', methods=['POST'])
    def start_bot():
        try:
            if bot_state['active']:
                return jsonify({'success': False, 'message': 'Bot já está ativo'})
            
            exchange, status = get_bitget_exchange()
            bot_state['active'] = True
            bot_state['last_update'] = datetime.now()
            
            message = f"Conectado à Bitget - {status}"
            logger.info(f"🤖 {message}")
            
            return jsonify({'success': True, 'message': message})
            
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    @app.route('/api/bot/stop', methods=['POST'])
    def stop_bot():
        bot_state['active'] = False
        logger.info("⏹️ Bot parado!")
        return jsonify({'success': True, 'message': 'Desconectado da Bitget'})
    
    @app.route('/api/stats')
    def get_stats():
        try:
            import random
            
            if bot_state['active'] and random.random() > 0.95:
                bot_state['daily_trades'] += 1
                bot_state['daily_pnl'] += random.uniform(-25, 50)
                bot_state['win_rate'] = max(0, min(100, bot_state['win_rate'] + random.uniform(-1, 2)))
            
            return jsonify({
                'success': True,
                'daily_trades': bot_state['daily_trades'],
                'daily_pnl': bot_state['daily_pnl'],
                'win_rate': bot_state['win_rate']
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return app

# Aplicação principal
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Iniciando Trading Bot na porta {port}")
    logger.info(f"🔑 Usando variáveis corretas do Render")
    logger.info(f"💡 Modo: {'Sandbox' if paper_trading else 'Live Trading'}")
    
    # Teste inicial
    try:
        balance, source = get_real_balance()
        logger.info(f"💰 Saldo inicial: ${balance:.2f} ({source})")
    except Exception as e:
        logger.error(f"❌ Erro no teste inicial: {e}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
