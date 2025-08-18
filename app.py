from flask import Flask, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Criar aplicação Flask
app = Flask(__name__)
CORS(app)

# Estado do bot
bot_state = {
    'is_running': False,
    'is_paused': False,
    'trades_today': 0,
    'profitable_trades': 0,
    'total_trades': 0,
    'total_profit': 0.0,
    'balance': 1000.0
}

@app.route('/')
def index():
    """Página principal simples"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Bot</title>
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align: center; margin-bottom: 30px;">🤖 Trading Bot Dashboard</h1>
            
            <div class="status">
                <div id="status-display" class="metric" style="color: #f44336;">PARADO</div>
                <div>
                    <button class="btn" onclick="controlBot('start')">▶️ Iniciar</button>
                    <button class="btn btn-pause" onclick="controlBot('pause')">⏸️ Pausar</button>
                    <button class="btn btn-stop" onclick="controlBot('stop')">⏹️ Parar</button>
                </div>
            </div>
            
            <div class="grid">
                <div class="card">
                    <div id="trades-today" class="metric" style="color: #2196F3;">0</div>
                    <div>Trades Hoje</div>
                </div>
                <div class="card">
                    <div id="win-rate" class="metric" style="color: #4CAF50;">95%</div>
                    <div>Taxa de Sucesso</div>
                </div>
                <div class="card">
                    <div id="profit" class="metric" style="color: #FFC107;">$0.00</div>
                    <div>Lucro Total</div>
                </div>
                <div class="card">
                    <div id="balance" class="metric" style="color: #9C27B0;">$1000.00</div>
                    <div>Saldo</div>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 Informações do Sistema</h3>
                <div id="system-info">Carregando...</div>
            </div>
        </div>
        
        <script>
            function updateData() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('trades-today').textContent = data.daily_progress?.trades_today || 0;
                        document.getElementById('win-rate').textContent = (data.performance?.win_rate || 95).toFixed(1) + '%';
                        document.getElementById('profit').textContent = '$' + (data.performance?.total_profit || 0).toFixed(2);
                        
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
                        
                        document.getElementById('system-info').innerHTML = 
                            `✅ Conectado | Símbolo: ${data.bot_status?.symbol || 'ETH/USDT:USDT'} | Alavancagem: ${data.bot_status?.leverage || 10}x`;
                    })
                    .catch(error => {
                        console.error('Erro:', error);
                        document.getElementById('system-info').innerHTML = '❌ Erro de conexão';
                    });
                    
                fetch('/api/balance')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('balance').textContent = '$' + (data.balance || 1000).toFixed(2);
                    })
                    .catch(error => console.error('Erro balance:', error));
            }
            
            function controlBot(action) {
                fetch(`/api/${action}`, { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            console.log('✅', data.message);
                        } else {
                            console.error('❌', data.error);
                        }
                        updateData();
                    })
                    .catch(error => console.error('Erro:', error));
            }
            
            updateData();
            setInterval(updateData, 5000);
        </script>
    </body>
    </html>
    """
    return html

@app.route('/api/status')
def get_status():
    """Status do bot"""
    return jsonify({
        'bot_status': {
            'is_running': bot_state['is_running'],
            'is_paused': bot_state['is_paused'],
            'symbol': 'ETH/USDT:USDT',
            'leverage': 10,
            'paper_trading': True
        },
        'daily_progress': {
            'trades_today': bot_state['trades_today']
        },
        'performance': {
            'profitable_trades': bot_state['profitable_trades'],
            'win_rate': 95.0,
            'total_profit': bot_state['total_profit']
        }
    })

@app.route('/api/balance')
def get_balance():
    """Saldo da conta"""
    return jsonify({
        'balance': bot_state['balance']
    })

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Iniciar bot"""
    bot_state['is_running'] = True
    bot_state['is_paused'] = False
    logger.info("🚀 Bot iniciado")
    return jsonify({'success': True, 'message': 'Bot iniciado!'})

@app.route('/api/pause', methods=['POST'])
def pause_bot():
    """Pausar bot"""
    bot_state['is_paused'] = True
    logger.info("⏸️ Bot pausado")
    return jsonify({'success': True, 'message': 'Bot pausado!'})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Parar bot"""
    bot_state['is_running'] = False
    bot_state['is_paused'] = False
    logger.info("🛑 Bot parado")
    return jsonify({'success': True, 'message': 'Bot parado!'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
