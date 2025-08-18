from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import os
import logging
import json
from datetime import datetime
import sys

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

# HTML como string (versão simplificada)
HTML_CONTENT = '''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen p-8">
    <div class="max-w-6xl mx-auto">
        <h1 class="text-4xl font-bold mb-8 text-center">🤖 Trading Bot Dashboard</h1>
        
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-gray-800 p-6 rounded-lg text-center">
                <div class="text-2xl font-bold text-blue-400" id="trades-today">0</div>
                <div class="text-sm text-gray-400">Trades Hoje</div>
            </div>
            <div class="bg-gray-800 p-6 rounded-lg text-center">
                <div class="text-2xl font-bold text-green-400" id="win-rate">95%</div>
                <div class="text-sm text-gray-400">Taxa Sucesso</div>
            </div>
            <div class="bg-gray-800 p-6 rounded-lg text-center">
                <div class="text-2xl font-bold text-yellow-400" id="profit">$0</div>
                <div class="text-sm text-gray-400">Lucro</div>
            </div>
            <div class="bg-gray-800 p-6 rounded-lg text-center">
                <div class="text-2xl font-bold" id="status">PARADO</div>
                <div class="text-sm text-gray-400">Status</div>
            </div>
        </div>
        
        <div class="flex gap-4 justify-center mb-8">
            <button onclick="controlBot('start')" class="bg-green-600 hover:bg-green-700 px-6 py-3 rounded-lg font-semibold">
                ▶️ Iniciar
            </button>
            <button onclick="controlBot('pause')" class="bg-yellow-600 hover:bg-yellow-700 px-6 py-3 rounded-lg font-semibold">
                ⏸️ Pausar
            </button>
            <button onclick="controlBot('stop')" class="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg font-semibold">
                ⏹️ Parar
            </button>
        </div>
        
        <div class="bg-gray-800 p-6 rounded-lg">
            <h2 class="text-xl font-bold mb-4">📋 Status do Sistema</h2>
            <div id="system-status" class="text-green-400">✅ Sistema funcionando normalmente</div>
        </div>
    </div>
    
    <script>
        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('trades-today').textContent = data.daily_progress?.trades_today || 0;
                    document.getElementById('win-rate').textContent = (data.performance?.win_rate || 95).toFixed(1) + '%';
                    document.getElementById('profit').textContent = '$' + (data.performance?.total_profit || 0).toFixed(2);
                    
                    const statusEl = document.getElementById('status');
                    if (data.bot_status?.is_running && !data.bot_status?.is_paused) {
                        statusEl.textContent = 'RODANDO';
                        statusEl.className = 'text-2xl font-bold text-green-400';
                    } else if (data.bot_status?.is_paused) {
                        statusEl.textContent = 'PAUSADO';
                        statusEl.className = 'text-2xl font-bold text-yellow-400';
                    } else {
                        statusEl.textContent = 'PARADO';
                        statusEl.className = 'text-2xl font-bold text-red-400';
                    }
                })
                .catch(error => {
                    console.error('Erro:', error);
                    document.getElementById('system-status').innerHTML = '❌ Erro de conexão: ' + error.message;
                    document.getElementById('system-status').className = 'text-red-400';
                });
        }
        
        function controlBot(action) {
            fetch(`/api/${action}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('system-status').innerHTML = '✅ ' + data.message;
                        document.getElementById('system-status').className = 'text-green-400';
                        updateDashboard();
                    } else {
                        document.getElementById('system-status').innerHTML = '❌ Erro: ' + (data.error || 'Erro desconhecido');
                        document.getElementById('system-status').className = 'text-red-400';
                    }
                })
                .catch(error => {
                    document.getElementById('system-status').innerHTML = '❌ Erro de rede: ' + error.message;
                    document.getElementById('system-status').className = 'text-red-400';
                });
        }
        
        // Atualizar a cada 5 segundos
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
'''

# Variáveis globais para simular estado do bot
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
    'paper_trading': True,
    'balance': 1000.0
}

@app.route('/')
def index():
    """Servir página principal"""
    return render_template_string(HTML_CONTENT)

@app.route('/api/status')
def get_status():
    """Retornar status atual do bot"""
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
            'price': 2500.0 + (hash(str(datetime.now().second)) % 100 - 50),
            'volume': 1000000,
            'high': 2550.0,
            'low': 2450.0,
            'change': 1.2
        },
        'timestamp': current_time.isoformat()
    })

@app.route('/api/balance')
def get_balance():
    """Retornar saldo da conta"""
    return jsonify({
        'balance': bot_state['balance'],
        'free': bot_state['balance'],
        'used': 0.0,
        'total': bot_state['balance']
    })

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Iniciar o bot"""
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
    """Pausar o bot"""
    try:
        bot_state['is_paused'] = True
        logger.info("⏸️ Bot pausado")
        return jsonify({'success': True, 'message': 'Bot pausado'})
    except Exception as e:
        logger.error(f"Erro ao pausar bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Parar o bot"""
    try:
        bot_state['is_running'] = False
        bot_state['is_paused'] = False
        logger.info("🛑 Bot parado")
        return jsonify({'success': True, 'message': 'Bot parado'})
    except Exception as e:
        logger.error(f"Erro ao parar bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
