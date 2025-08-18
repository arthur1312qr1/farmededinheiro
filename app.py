from flask import Flask, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime
import sys
import time

# Importar a API da Bitget do seu arquivo
try:
    from bitget_api import BitgetAPI
    from config import get_config
    BITGET_AVAILABLE = True
except ImportError as e:
    logging.warning(f"BitgetAPI não disponível: {e}")
    BITGET_AVAILABLE = False

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Criar aplicação Flask
app = Flask(__name__)
CORS(app)

# Cache para evitar muitas chamadas da API
price_cache = {'price': 0, 'timestamp': 0}
balance_cache = {'balance': 0, 'timestamp': 0}
CACHE_DURATION = 3  # 3 segundos de cache

# Inicializar BitgetAPI se disponível
bitget_api = None
if BITGET_AVAILABLE:
    try:
        config = get_config()
        bitget_api = BitgetAPI(
            api_key=config.get('BITGET_API_KEY'),
            secret_key=config.get('BITGET_SECRET_KEY'),
            passphrase=config.get('BITGET_PASSPHRASE')
        )
        logger.info("🔧 Inicializando conexão com Bitget...")
        
        # Testar conexão
        if bitget_api.test_connection():
            logger.info("✅ Conectado à Bitget com sucesso!")
        else:
            logger.warning("⚠️ Conexão com Bitget falhou - usando modo mock")
            bitget_api = None
    except Exception as e:
        logger.error(f"❌ Erro ao conectar com Bitget: {e}")
        bitget_api = None
else:
    logger.info("🔧 Inicializando configurações mock...")

if not bitget_api:
    logger.info("✅ Bot mock inicializado com sucesso!")

# Estado do bot
bot_state = {
    'is_running': False,
    'is_paused': False,
    'trades_today': 0,
    'profitable_trades': 0,
    'total_trades': 0,
    'total_profit': 0.0,
    'consecutive_wins': 0,
    'current_position': None,
    'position_side': None,
    'position_size': 0.0,
    'entry_price': None,
    'symbol': 'ETH/USDT:USDT',
    'leverage': 10,
    'paper_trading': not bitget_api,  # Paper trading se não conectado
    'balance': 1000.0
}

@app.route('/')
def index():
    """Página principal"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Bot Dashboard</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; background: #1a1a1a; color: white; margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: #333; padding: 20px; margin: 10px; border-radius: 8px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
            .btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            .btn:hover { background: #45a049; }
            .btn-pause { background: #ff9800; }
            .btn-stop { background: #f44336; }
            .metric { font-size: 2em; font-weight: bold; margin-bottom: 10px; }
            .status { text-align: center; margin-bottom: 20px; }
            .connection-status { padding: 10px; border-radius: 5px; margin-bottom: 20px; text-align: center; }
            .connected { background: #4CAF50; }
            .disconnected { background: #f44336; }
            .last-update { font-size: 0.8em; color: #888; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align: center; margin-bottom: 30px;">🤖 Trading Bot Dashboard</h1>
            
            <div id="connection-status" class="connection-status">
                <div id="connection-text">🔄 Verificando conexão...</div>
            </div>
            
            <div class="status">
                <div id="status-display" class="metric" style="color: #f44336;">PARADO</div>
                <div>
                    <button class="btn" onclick="controlBot('start')">▶️ Iniciar</button>
                    <button class="btn btn-pause" onclick="controlBot('pause')">⏸️ Pausar</button>
                    <button class="btn btn-stop" onclick="controlBot('stop')">⏹️ Parar</button>
                    <button class="btn btn-stop" onclick="controlBot('emergency_stop')">🚨 Emergência</button>
                </div>
            </div>
            
            <div class="grid">
                <div class="card">
                    <div id="trades-today" class="metric" style="color: #2196F3;">0</div>
                    <div>Trades Hoje</div>
                </div>
                <div class="card">
                    <div id="win-rate" class="metric" style="color: #4CAF50;">95.0%</div>
                    <div>Taxa de Sucesso</div>
                </div>
                <div class="card">
                    <div id="profit" class="metric" style="color: #FFC107;">$0.0000</div>
                    <div>Lucro Total</div>
                </div>
                <div class="card">
                    <div id="balance" class="metric" style="color: #9C27B0;">$0.00</div>
                    <div>Saldo USDT</div>
                </div>
                <div class="card">
                    <div id="eth-price" class="metric" style="color: #FF5722;">$0.00</div>
                    <div>Preço ETH</div>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 Informações do Sistema</h3>
                <div id="system-info">Carregando...</div>
                <div id="last-update" class="last-update">Última atualização: Nunca</div>
            </div>
            
            <div class="card">
                <h3>🔄 Progresso Diário</h3>
                <div>Meta: <span id="daily-target">240</span> trades | Progresso: <span id="progress-percent">0%</span></div>
                <div style="width: 100%; background: #555; height: 20px; border-radius: 10px; margin: 10px 0;">
                    <div id="progress-bar" style="width: 0%; background: #4CAF50; height: 100%; border-radius: 10px; transition: width 0.5s;"></div>
                </div>
                <div>Status: <span id="urgency-level" style="padding: 5px 10px; border-radius: 15px; background: #4CAF50;">NORMAL</span></div>
            </div>
        </div>
        
        <script>
            let updateInterval;
            
            function updateData() {
                const now = new Date().toLocaleTimeString();
                
                // Atualizar status (menos frequente)
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('trades-today').textContent = data.daily_progress?.trades_today || 0;
                        document.getElementById('win-rate').textContent = (data.performance?.win_rate || 95).toFixed(1) + '%';
                        document.getElementById('profit').textContent = '$' + (data.performance?.total_profit || 0).toFixed(4);
                        
                        // Progresso diário
                        const progressPercent = data.daily_progress?.progress_percent || 0;
                        document.getElementById('progress-percent').textContent = progressPercent.toFixed(1) + '%';
                        document.getElementById('progress-bar').style.width = Math.min(progressPercent, 100) + '%';
                        
                        const urgencyLevel = data.daily_progress?.urgency_level || 'NORMAL';
                        const urgencyEl = document.getElementById('urgency-level');
                        urgencyEl.textContent = urgencyLevel;
                        urgencyEl.style.background = urgencyLevel === 'CRÍTICO' ? '#f44336' : urgencyLevel === 'ALTO' ? '#ff9800' : urgencyLevel === 'MÉDIO' ? '#FFC107' : '#4CAF50';
                        
                        const statusEl = document.getElementById('status-display');
                        if (data.bot_status?.is_running && !data.bot_status?.is_paused) {
                            statusEl.textContent = 'RODANDO';
                            statusEl.style.color = '#4CAF50';
                        } else if (data.bot_status?.is_paused) {
                            statusEl.textContent = 'PAUSADO';
                            statusEl.style.color = '#ff9800';
                        } else {
                            statusEl.textContent = 'PARADO';
                            statusEl.style.color = '#f44336';
                        }
                        
                        // Atualizar preço ETH
                        const ethPrice = data.market_data?.price || 0;
                        document.getElementById('eth-price').textContent = '$' + ethPrice.toFixed(2);
                        
                        document.getElementById('system-info').innerHTML = 
                            `✅ Símbolo: ${data.bot_status?.symbol || 'ETH/USDT:USDT'} | Alavancagem: ${data.bot_status?.leverage || 10}x | Paper Trading: ${data.bot_status?.paper_trading ? 'Sim' : 'Não'}`;
                    })
                    .catch(error => {
                        console.error('Erro status:', error);
                        document.getElementById('system-info').innerHTML = '❌ Erro ao carregar status';
                    });
                    
                // Atualizar saldo
                fetch('/api/balance')
                    .then(response => response.json())
                    .then(data => {
                        const balance = data.balance || data.free || 0;
                        document.getElementById('balance').textContent = '$' + balance.toFixed(4);
                        
                        // Atualizar status de conexão
                        const connStatus = document.getElementById('connection-status');
                        const connText = document.getElementById('connection-text');
                        
                        if (data.connected === false) {
                            connStatus.className = 'connection-status disconnected';
                            connText.textContent = '❌ Modo Simulação - Sem conexão Bitget';
                        } else {
                            connStatus.className = 'connection-status connected';
                            connText.textContent = '✅ Conectado à Bitget - Trading Real Ativo';
                        }
                        
                        document.getElementById('last-update').textContent = `Última atualização: ${now}`;
                    })
                    .catch(error => {
                        console.error('Erro balance:', error);
                        document.getElementById('balance').textContent = 'Erro';
                        
                        const connStatus = document.getElementById('connection-status');
                        const connText = document.getElementById('connection-text');
                        connStatus.className = 'connection-status disconnected';
                        connText.textContent = '❌ Erro de conexão com API';
                    });
            }
            
            function controlBot(action) {
                fetch(`/api/${action}`, { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            console.log('✅', data.message);
                            // Atualizar imediatamente após comando
                            setTimeout(updateData, 1000);
                        } else {
                            console.error('❌', data.error);
                        }
                    })
                    .catch(error => console.error('Erro:', error));
            }
            
            // Atualização inicial
            updateData();
            
            // Atualizar a cada 5 segundos (reduzido de 1-2s)
            updateInterval = setInterval(updateData, 5000);
        </script>
    </body>
    </html>
    """
    return html

def get_cached_eth_price():
    """Pegar preço ETH com cache"""
    now = time.time()
    if now - price_cache['timestamp'] < CACHE_DURATION and price_cache['price'] > 0:
        return price_cache['price']
    
    if bitget_api:
        try:
            price = bitget_api.get_eth_price()
            if price:
                price_cache['price'] = price
                price_cache['timestamp'] = now
                return price
        except Exception as e:
            logger.error(f"Erro ao pegar preço ETH: {e}")
    
    return price_cache['price'] if price_cache['price'] > 0 else 4300.0

def get_cached_balance():
    """Pegar saldo com cache"""
    now = time.time()
    if now - balance_cache['timestamp'] < CACHE_DURATION and balance_cache['balance'] > 0:
        return balance_cache['balance'], True
    
    if bitget_api:
        try:
            balance_info = bitget_api.get_balance()
            if balance_info and 'free' in balance_info:
                balance_cache['balance'] = balance_info['free']
                balance_cache['timestamp'] = now
                return balance_info, True
        except Exception as e:
            logger.error(f"Erro ao pegar saldo: {e}")
    
    return {'free': bot_state['balance'], 'used': 0, 'total': bot_state['balance']}, False

@app.route('/api/status')
def get_status():
    """Status do bot otimizado"""
    current_time = datetime.now()
    current_hour = current_time.hour
    
    # Calcular progresso diário
    hours_passed = max(1, current_hour - 8) if current_hour >= 8 else max(1, current_hour + 16)
    expected_trades_by_now = (240 / 16) * hours_passed
    trade_deficit = max(0, expected_trades_by_now - bot_state['trades_today'])
    
    urgency_level = "NORMAL"
    if trade_deficit > 30:
        urgency_level = "CRÍTICO"
    elif trade_deficit > 15:
        urgency_level = "ALTO"
    elif trade_deficit > 5:
        urgency_level = "MÉDIO"
    
    win_rate = (bot_state['profitable_trades'] / max(1, bot_state['total_trades'])) * 100 if bot_state['total_trades'] > 0 else 95.0
    
    # Pegar preço com cache
    eth_price = get_cached_eth_price()
    
    return jsonify({
        'bot_status': {
            'is_running': bot_state['is_running'],
            'is_paused': bot_state['is_paused'],
            'symbol': bot_state['symbol'],
            'leverage': bot_state['leverage'],
            'paper_trading': bot_state['paper_trading']
        },
        'daily_progress': {
            'trades_today': bot_state['trades_today'],
            'min_target': 240,
            'target': 280,
            'progress_percent': round((bot_state['trades_today'] / 240) * 100, 1),
            'expected_by_now': round(expected_trades_by_now),
            'deficit': round(trade_deficit),
            'urgency_level': urgency_level
        },
        'performance': {
            'profitable_trades': bot_state['profitable_trades'],
            'losing_trades': bot_state['total_trades'] - bot_state['profitable_trades'],
            'win_rate': round(win_rate, 2),
            'target_win_rate': 95.0,
            'total_profit': round(bot_state['total_profit'], 4),
            'consecutive_wins': bot_state['consecutive_wins']
        },
        'current_position': {
            'active': bot_state['current_position'] is not None,
            'side': bot_state['position_side'],
            'size': bot_state['position_size'],
            'entry_price': bot_state['entry_price'],
            'unrealized_pnl': 0.0
        },
        'market_data': {
            'price': eth_price,
            'volume': 1000000,
            'high': eth_price * 1.02,
            'low': eth_price * 0.98,
            'change': 1.2
        },
        'timestamp': current_time.isoformat()
    })

@app.route('/api/balance')
def get_balance():
    """Saldo otimizado com cache"""
    balance_info, connected = get_cached_balance()
    
    return jsonify({
        'balance': balance_info['free'] if isinstance(balance_info, dict) else balance_info,
        'free': balance_info.get('free', 0) if isinstance(balance_info, dict) else balance_info,
        'used': balance_info.get('used', 0) if isinstance(balance_info, dict) else 0,
        'total': balance_info.get('total', 0) if isinstance(balance_info, dict) else balance_info,
        'connected': connected
    })

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Iniciar bot"""
    try:
        bot_state['is_running'] = True
        bot_state['is_paused'] = False
        logger.info("🚀 Bot iniciado")
        return jsonify({'success': True, 'message': 'Bot iniciado com sucesso!'})
    except Exception as e:
        logger.error(f"Erro ao iniciar bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/pause', methods=['POST'])
def pause_bot():
    """Pausar bot"""
    try:
        bot_state['is_paused'] = True
        logger.info("⏸️ Bot pausado")
        return jsonify({'success': True, 'message': 'Bot pausado'})
    except Exception as e:
        logger.error(f"Erro ao pausar bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Parar bot"""
    try:
        bot_state['is_running'] = False
        bot_state['is_paused'] = False
        logger.info("🛑 Bot parado")
        return jsonify({'success': True, 'message': 'Bot parado'})
    except Exception as e:
        logger.error(f"Erro ao parar bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    """Parada de emergência"""
    try:
        bot_state['is_running'] = False
        bot_state['is_paused'] = False
        bot_state['current_position'] = None
        logger.warning("🚨 Parada de emergência ativada")
        return jsonify({'success': True, 'message': 'Parada de emergência executada'})
    except Exception as e:
        logger.error(f"Erro na parada de emergência: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
