import os
import sys
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import json

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Importar nosso sistema de trading corrigido
from trading_bot import TradingBot
from bitget_api import BitgetAPI

def create_app():
    """Factory function para criar app Flask"""
    app = Flask(__name__)
    CORS(app)
    
    # Configurações do bot - CORRIGIDAS
    config = {
        'BITGET_API_KEY': os.environ.get('BITGET_API_KEY', '').strip(),
        'BITGET_SECRET_KEY': os.environ.get('BITGET_API_SECRET', '').strip(),
        'BITGET_PASSPHRASE': os.environ.get('BITGET_PASSPHRASE', '').strip(),
        'GEMINI_API_KEY': os.environ.get('GEMINI_API_KEY', '').strip(),
        'PAPER_TRADING': os.environ.get('PAPER_TRADING', 'true').lower() == 'true',
        'SYMBOL': 'ethusdt_UMCBL',
        'LEVERAGE': int(os.environ.get('LEVERAGE', '10')),
        'TARGET_TRADES_PER_DAY': int(os.environ.get('TARGET_TRADES', '200')),
        'BASE_CURRENCY': 'USDT'
    }
    
    # Inicializar bot com configuração corrigida
    try:
        trading_bot = TradingBot(config)
        logger.info("✅ Trading Bot inicializado com configuração corrigida")
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar Trading Bot: {e}")
        trading_bot = None
    
    # Estado global do sistema - SIMPLIFICADO
    app_state = {
        'bot_active': False,
        'last_update': datetime.now(),
        'trades_today': 0,
        'current_balance': 0.0,
        'daily_pnl': 0.0,
        'eth_price': 0.0,
        'last_trade_time': None,
        'error_count': 0,
        'connection_status': 'Desconectado'
    }
    
    @app.route('/')
    def dashboard():
        """Dashboard principal - SIMPLIFICADO"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ETH Bot 80% - Corrigido</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 20px; 
                    background: #1a1a1a; 
                    color: #fff; 
                }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .status { 
                    padding: 20px; 
                    border-radius: 10px; 
                    margin: 10px 0; 
                    background: #2d2d2d;
                }
                .active { background: #0d5f2a !important; }
                .inactive { background: #5f0d0d !important; }
                .controls { text-align: center; margin: 20px 0; }
                button { 
                    padding: 15px 30px; 
                    margin: 0 10px; 
                    border: none; 
                    border-radius: 5px; 
                    font-size: 16px; 
                    cursor: pointer; 
                }
                .start-btn { background: #28a745; color: white; }
                .stop-btn { background: #dc3545; color: white; }
                .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
                .metric { background: #2d2d2d; padding: 15px; border-radius: 8px; text-align: center; }
                .logs { 
                    background: #000; 
                    padding: 15px; 
                    border-radius: 8px; 
                    height: 300px; 
                    overflow-y: auto; 
                    font-family: monospace; 
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 ETH Trading Bot - 80% Dinâmico</h1>
                    <p>Sistema Corrigido - Valor Mínimo Validado</p>
                </div>
                
                <div class="status inactive" id="status">
                    <h3>Status: Bot Parado</h3>
                    <p>Aguardando início do trading...</p>
                </div>
                
                <div class="controls">
                    <button class="start-btn" onclick="startBot()">🚀 Iniciar Bot</button>
                    <button class="stop-btn" onclick="stopBot()">🛑 Parar Bot</button>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <h4>💰 Saldo</h4>
                        <div id="balance">$0.00 USDT</div>
                    </div>
                    <div class="metric">
                        <h4>📊 Trades Hoje</h4>
                        <div id="trades">0</div>
                    </div>
                    <div class="metric">
                        <h4>💎 Preço ETH</h4>
                        <div id="eth-price">$0.00</div>
                    </div>
                    <div class="metric">
                        <h4>📈 P&L Diário</h4>
                        <div id="pnl">$0.00</div>
                    </div>
                </div>
                
                <div class="logs" id="logs">
                    <div>Sistema inicializado - Aguardando comandos...</div>
                </div>
            </div>
            
            <script>
                function startBot() {
                    fetch('/start', { method: 'POST' })
                        .then(r => r.json())
                        .then(data => {
                            if (data.success) {
                                document.getElementById('status').className = 'status active';
                                document.getElementById('status').innerHTML = '<h3>Status: Bot Ativo</h3><p>' + data.message + '</p>';
                                addLog('✅ ' + data.message);
                            } else {
                                addLog('❌ ' + data.message);
                            }
                        });
                }
                
                function stopBot() {
                    fetch('/stop', { method: 'POST' })
                        .then(r => r.json())
                        .then(data => {
                            document.getElementById('status').className = 'status inactive';
                            document.getElementById('status').innerHTML = '<h3>Status: Bot Parado</h3><p>' + data.message + '</p>';
                            addLog('🛑 ' + data.message);
                        });
                }
                
                function addLog(message) {
                    const logs = document.getElementById('logs');
                    const time = new Date().toLocaleTimeString();
                    logs.innerHTML += '<div>' + time + ' - ' + message + '</div>';
                    logs.scrollTop = logs.scrollHeight;
                }
                
                function updateStatus() {
                    fetch('/status')
                        .then(r => r.json())
                        .then(data => {
                            document.getElementById('balance').textContent = '$' + (data.balance || 0).toFixed(2) + ' USDT';
                            document.getElementById('trades').textContent = data.trades_today || 0;
                            document.getElementById('eth-price').textContent = '$' + (data.eth_price || 0).toFixed(2);
                            document.getElementById('pnl').textContent = '$' + (data.daily_pnl || 0).toFixed(2);
                        })
                        .catch(e => console.error('Erro ao atualizar status:', e));
                }
                
                // Atualizar status a cada 5 segundos
                setInterval(updateStatus, 5000);
                updateStatus();
            </script>
        </body>
        </html>
        """
        return html

    @app.route('/start', methods=['POST'])
    def start_bot():
        """Iniciar trading usando nosso bot corrigido"""
        try:
            if trading_bot is None:
                return jsonify({
                    'success': False,
                    'message': '❌ Bot não inicializado corretamente'
                })
            
            if not trading_bot.is_running:
                trading_bot.start()
                app_state['bot_active'] = True
                logger.info("🚀 Trading Bot iniciado via web interface")
                
                return jsonify({
                    'success': True,
                    'message': '🚀 Bot iniciado com sistema de 80% corrigido!'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': '⚠️ Bot já está em execução'
                })
                
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar bot: {e}")
            return jsonify({
                'success': False,
                'message': f'❌ Erro: {str(e)}'
            })

    @app.route('/stop', methods=['POST'])
    def stop_bot():
        """Parar trading"""
        try:
            if trading_bot and trading_bot.is_running:
                trading_bot.stop()
                app_state['bot_active'] = False
                logger.info("🛑 Trading Bot parado via web interface")
                
                return jsonify({
                    'success': True,
                    'message': '🛑 Bot parado com segurança'
                })
            else:
                return jsonify({
                    'success': True,
                    'message': '🛑 Bot já estava parado'
                })
                
        except Exception as e:
            logger.error(f"❌ Erro ao parar bot: {e}")
            return jsonify({
                'success': False,
                'message': f'❌ Erro: {str(e)}'
            })

    @app.route('/status')
    def get_status():
        """Status do sistema"""
        try:
            if trading_bot:
                # Obter dados do bot
                bitget_api = BitgetAPI(
                    api_key=config['BITGET_API_KEY'],
                    secret_key=config['BITGET_SECRET_KEY'],
                    passphrase=config['BITGET_PASSPHRASE'],
                    sandbox=config['PAPER_TRADING']
                )
                
                # Tentar obter saldo atual
                try:
                    balance = bitget_api.get_account_balance()
                    app_state['current_balance'] = balance
                except:
                    pass
                
                # Tentar obter preço ETH
                try:
                    market_data = bitget_api.get_market_data(config['SYMBOL'])
                    if market_data:
                        app_state['eth_price'] = market_data.get('price', 0)
                except:
                    pass
                
                # Dados do bot
                app_state.update({
                    'bot_active': trading_bot.is_running,
                    'trades_today': trading_bot.daily_trades,
                    'daily_pnl': trading_bot.daily_pnl,
                    'last_update': datetime.now().isoformat(),
                    'connection_status': 'Conectado' if trading_bot.is_running else 'Desconectado'
                })
            
            return jsonify(app_state)
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter status: {e}")
            return jsonify({
                'error': str(e),
                'bot_active': False,
                'balance': 0.0
            })

    @app.route('/health')
    def health_check():
        """Health check para monitoring"""
        return jsonify({
            'status': 'OK',
            'timestamp': datetime.now().isoformat(),
            'bot_initialized': trading_bot is not None,
            'bot_active': app_state['bot_active']
        })

    return app

# Criar aplicação
app = create_app()

if __name__ == '__main__':
    # Configurar para produção
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Iniciando servidor Flask na porta {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
