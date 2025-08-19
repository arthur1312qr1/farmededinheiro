from flask import Flask
from flask_cors import CORS
import os
import logging
from datetime import datetime
import sys
import time
import threading
import traceback

# Importar seus arquivos existentes
try:
    from bitget_api import BitgetAPI
    from config import get_config
    from trading_bot import TradingBot
    
    config = get_config()
    
    # Inicializar API
    bitget_api = BitgetAPI(
        api_key=config.get('BITGET_API_KEY'),
        secret_key=config.get('BITGET_SECRET_KEY'),
        passphrase=config.get('BITGET_PASSPHRASE')
    )
    
    # Inicializar TradingBot
    trading_bot = TradingBot(
        api=bitget_api,
        symbol=config.get('SYMBOL', 'ETH/USDT:USDT'),
        leverage=int(config.get('LEVERAGE', 10)),
        paper_trading=config.get('PAPER_TRADING', 'true').lower() == 'true'
    )
    
    API_CONNECTED = True
    print("✅ Sistema inicializado com sucesso!")
    
except Exception as e:
    print(f"❌ Erro na inicialização: {e}")
    traceback.print_exc()
    bitget_api = None
    trading_bot = None
    API_CONNECTED = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    'thread': None,
    'last_error': None,
    'last_analysis': None,
    'current_prediction': None
}

def run_trading_loop():
    """Loop principal que executa o trading_bot"""
    print("🚀 Iniciando loop de trading...")
    
    while bot_state['is_running']:
        if bot_state['is_paused']:
            print("⏸️ Bot pausado")
            time.sleep(5)
            continue
            
        try:
            if trading_bot and API_CONNECTED:
                print("🔄 Executando trading_bot...")
                
                # CHAMAR SEU TRADING_BOT.PY - ele faz tudo!
                if hasattr(trading_bot, 'execute'):
                    result = trading_bot.execute()
                elif hasattr(trading_bot, 'run'):
                    result = trading_bot.run()
                else:
                    # Métodos individuais
                    analysis = None
                    prediction = None
                    trade_result = None
                    
                    if hasattr(trading_bot, 'analyze_market'):
                        analysis = trading_bot.analyze_market()
                        bot_state['last_analysis'] = str(analysis)
                    
                    if hasattr(trading_bot, 'predict'):
                        prediction = trading_bot.predict()
                        bot_state['current_prediction'] = str(prediction)
                    
                    if hasattr(trading_bot, 'execute_trade'):
                        trade_result = trading_bot.execute_trade()
                    
                    result = trade_result
                
                # Processar resultado
                if result:
                    bot_state['trades_today'] += 1
                    bot_state['total_trades'] += 1
                    print(f"✅ Trade executado! Total: {bot_state['trades_today']}")
                
                bot_state['last_error'] = None
                
            else:
                bot_state['last_error'] = "API ou TradingBot não disponível"
                
        except Exception as e:
            error_msg = f"Erro no trading: {str(e)}"
            print(f"❌ {error_msg}")
            bot_state['last_error'] = error_msg
            time.sleep(5)
        
        time.sleep(2)  # Pausa entre execuções
    
    print("🛑 Loop finalizado")

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial; background: #1a1a1a; color: white; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: #333; padding: 20px; margin: 10px; border-radius: 8px; }
        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
        .metric { text-align: center; }
        .metric-value { font-size: 2em; margin-bottom: 10px; }
        .btn { 
            padding: 15px 25px; 
            margin: 5px; 
            border: none; 
            border-radius: 5px; 
            color: white; 
            font-size: 16px; 
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn:hover { transform: scale(1.05); }
        .btn-start { background: #4CAF50; }
        .btn-pause { background: #ff9800; }
        .btn-stop { background: #f44336; }
        .status-running { color: #4CAF50; }
        .status-paused { color: #ff9800; }
        .status-stopped { color: #f44336; }
        .error { color: #ff6b6b; }
        .success { color: #4CAF50; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Trading Bot Dashboard</h1>
        
        <div class="card">
            <h2>Status: <span id="status" class="status-stopped">PARADO</span></h2>
            <button id="btn-start" class="btn btn-start">🚀 Iniciar</button>
            <button id="btn-pause" class="btn btn-pause">⏸️ Pausar</button>
            <button id="btn-stop" class="btn btn-stop">🛑 Parar</button>
        </div>
        
        <div class="grid">
            <div class="card metric">
                <div id="trades" class="metric-value" style="color: #2196F3;">0</div>
                <div>Trades Hoje</div>
            </div>
            <div class="card metric">
                <div id="balance" class="metric-value" style="color: #9C27B0;">$0.00</div>
                <div>Saldo USDT</div>
            </div>
            <div class="card metric">
                <div id="price" class="metric-value" style="color: #FF5722;">$0.00</div>
                <div>Preço ETH</div>
            </div>
            <div class="card metric">
                <div id="profit" class="metric-value" style="color: #FFC107;">$0.00</div>
                <div>Lucro Total</div>
            </div>
        </div>
        
        <div class="card">
            <h3>📊 Análise: <span id="analysis">Aguardando...</span></h3>
            <h3>🔮 Previsão: <span id="prediction">Aguardando...</span></h3>
        </div>
        
        <div class="card">
            <h3>🔧 Sistema</h3>
            <div id="info">Carregando...</div>
            <div id="debug" style="font-size: 0.8em; color: #888; margin-top: 10px;"></div>
        </div>
        
        <div class="card">
            <h3>⚠️ Status e Erros</h3>
            <div id="errors" class="success">Sistema funcionando</div>
        </div>
    </div>

    <script>
        // Event listeners para os botões
        document.getElementById('btn-start').addEventListener('click', function() {
            console.log('🚀 Clicou em Iniciar');
            this.disabled = true;
            this.textContent = '🔄 Iniciando...';
            
            fetch('/api/start', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    console.log('Resposta start:', data);
                    this.disabled = false;
                    this.textContent = '🚀 Iniciar';
                    updateData();
                })
                .catch(e => {
                    console.error('Erro:', e);
                    this.disabled = false;
                    this.textContent = '🚀 Iniciar';
                });
        });
        
        document.getElementById('btn-pause').addEventListener('click', function() {
            console.log('⏸️ Clicou em Pausar');
            fetch('/api/pause', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    console.log('Resposta pause:', data);
                    updateData();
                });
        });
        
        document.getElementById('btn-stop').addEventListener('click', function() {
            console.log('🛑 Clicou em Parar');
            fetch('/api/stop', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    console.log('Resposta stop:', data);
                    updateData();
                });
        });
        
        function updateData() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    // Atualizar métricas
                    document.getElementById('trades').textContent = data.trades || 0;
                    document.getElementById('balance').textContent = '$' + (data.balance || 0).toFixed(4);
                    document.getElementById('price').textContent = '$' + (data.price || 0).toFixed(2);
                    document.getElementById('profit').textContent = '$' + (data.profit || 0).toFixed(4);
                    
                    // Status
                    const statusEl = document.getElementById('status');
                    statusEl.textContent = data.status || 'PARADO';
                    statusEl.className = 'status-' + (data.status === 'RODANDO' ? 'running' : 
                                                     data.status === 'PAUSADO' ? 'paused' : 'stopped');
                    
                    // Análise e previsão
                    document.getElementById('analysis').textContent = data.analysis || 'Aguardando...';
                    document.getElementById('prediction').textContent = data.prediction || 'Aguardando...';
                    
                    // Info
                    document.getElementById('info').textContent = data.info || 'Sistema ativo';
                    document.getElementById('debug').textContent = 'Atualizado: ' + new Date().toLocaleTimeString();
                    
                    // Erros
                    const errorEl = document.getElementById('errors');
                    if (data.error) {
                        errorEl.textContent = '❌ ERRO: ' + data.error;
                        errorEl.className = 'error';
                    } else {
                        errorEl.textContent = '✅ Sistema funcionando - Trading ativo';
                        errorEl.className = 'success';
                    }
                })
                .catch(e => {
                    console.error('Erro:', e);
                    document.getElementById('errors').textContent = '❌ Erro de conexão';
                    document.getElementById('errors').className = 'error';
                });
        }
        
        // Inicializar
        updateData();
        setInterval(updateData, 3000);
    </script>
</body>
</html>
'''

@app.route('/api/data')
def get_data():
    """Endpoint para dados do sistema"""
    try:
        # Buscar saldo
        balance = 0
        if bitget_api and API_CONNECTED:
            try:
                balance_info = bitget_api.get_balance()
                if balance_info:
                    balance = balance_info.get('free', 0)
            except:
                pass
        
        # Buscar preço
        price = 0
        if bitget_api and API_CONNECTED:
            try:
                price = bitget_api.get_eth_price() or 0
            except:
                pass
        
        # Status do bot
        if bot_state['is_running'] and not bot_state['is_paused']:
            status = 'RODANDO'
        elif bot_state['is_paused']:
            status = 'PAUSADO'
        else:
            status = 'PARADO'
        
        # Thread ativa?
        thread_active = bot_state['thread'] and bot_state['thread'].is_alive()
        
        return {
            'trades': bot_state['trades_today'],
            'balance': balance,
            'price': price,
            'profit': bot_state['total_profit'],
            'status': status,
            'analysis': bot_state['last_analysis'] or 'Aguardando análise...',
            'prediction': bot_state['current_prediction'] or 'Aguardando previsão...',
            'info': f'API: {API_CONNECTED}, Thread: {"Ativa" if thread_active else "Inativa"}',
            'error': bot_state['last_error']
        }
        
    except Exception as e:
        return {
            'trades': 0, 'balance': 0, 'price': 0, 'profit': 0,
            'status': 'ERRO', 'error': str(e),
            'analysis': 'Erro', 'prediction': 'Erro'
        }

@app.route('/api/start', methods=['POST'])
def start():
    """Iniciar o bot"""
    try:
        if not trading_bot:
            return {'success': False, 'error': 'TradingBot não disponível'}
        
        bot_state['is_running'] = True
        bot_state['is_paused'] = False
        bot_state['last_error'] = None
        
        # Iniciar thread se necessário
        if not bot_state['thread'] or not bot_state['thread'].is_alive():
            bot_state['thread'] = threading.Thread(target=run_trading_loop, daemon=True)
            bot_state['thread'].start()
            print("🚀 Thread iniciada!")
        
        logger.info("🚀 Bot iniciado")
        return {'success': True, 'message': 'Bot iniciado com sucesso'}
        
    except Exception as e:
        error_msg = f"Erro ao iniciar: {e}"
        bot_state['last_error'] = error_msg
        return {'success': False, 'error': error_msg}

@app.route('/api/pause', methods=['POST'])
def pause():
    """Pausar o bot"""
    bot_state['is_paused'] = True
    logger.info("⏸️ Bot pausado")
    return {'success': True, 'message': 'Bot pausado'}

@app.route('/api/stop', methods=['POST'])
def stop():
    """Parar o bot"""
    bot_state['is_running'] = False
    bot_state['is_paused'] = False
    logger.info("🛑 Bot parado")
    return {'success': True, 'message': 'Bot parado'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🚀 Iniciando Trading Bot Dashboard...")
    print(f"🔗 API: {'Conectada' if API_CONNECTED else 'Desconectada'}")
    print(f"🤖 TradingBot: {'Disponível' if trading_bot else 'Indisponível'}")
    
    app.run(host='0.0.0.0', port=port, debug=True)
